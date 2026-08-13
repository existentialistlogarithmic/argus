"""Graph analytics and financial-crime typologies.

Two families of technique live here.

*Structural* measures ask where an entity sits in the network: how central it
is, which community it belongs to, how close it is to known-bad seeds. They are
domain-agnostic and would work on any graph.

*Typologies* ask whether a specific, recognizable pattern of behaviour is
present: value moving in a closed loop, payments sized to sit under a reporting
threshold, unrelated companies sharing one registered address. These encode
what an investigator would actually look for, and each one produces evidence
that can be shown rather than a number that must be trusted.

Neither family is meaningful before resolution. A cycle through five accounts
is invisible if those accounts are still five hundred unlinked records.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field

from .graph import KnowledgeGraph


@dataclass
class Finding:
    """One machine-generated observation, with the evidence behind it."""

    kind: str
    severity: float           # 0..1
    entities: list[str]
    title: str
    detail: str
    evidence: dict = field(default_factory=dict)
    # Entities pulled in by following control links, mapped to how many hops
    # away they were. Implication weakens with distance.
    attributed: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "severity": round(self.severity, 3),
            "entities": self.entities,
            "title": self.title,
            "detail": self.detail,
            "evidence": self.evidence,
        }


# --------------------------------------------------------------------------
# structural measures
# --------------------------------------------------------------------------

def degree_centrality(graph: KnowledgeGraph) -> dict[str, float]:
    if not graph.entities:
        return {}
    scale = max(1, len(graph.entities) - 1)
    return {eid: graph.degree(eid) / scale for eid in graph.entities}


def betweenness_centrality(graph: KnowledgeGraph, samples: int | None = None,
                           seed: int = 11) -> dict[str, float]:
    """Brandes' algorithm, optionally estimated from sampled source nodes.

    Betweenness is the measure that finds brokers: entities that sit on the
    only route between otherwise separate parts of a network. Those are rarely
    the loudest nodes, which is exactly why degree alone misses them.

    Exact computation is O(VE) and gets expensive fast, so beyond a few hundred
    nodes it is estimated from a random sample of sources and rescaled -- the
    ranking stabilizes long before the absolute values do.
    """
    nodes = list(graph.entities)
    count = len(nodes)
    if count < 3:
        return {node: 0.0 for node in nodes}

    if samples is None:
        samples = count if count <= 600 else 300
    samples = min(samples, count)
    sources = nodes if samples == count else random.Random(seed).sample(nodes, samples)

    betweenness = dict.fromkeys(nodes, 0.0)
    for source in sources:
        stack = []
        predecessors: dict[str, list[str]] = defaultdict(list)
        sigma = dict.fromkeys(nodes, 0.0)
        distance = dict.fromkeys(nodes, -1)
        sigma[source] = 1.0
        distance[source] = 0
        queue = deque([source])

        while queue:
            node = queue.popleft()
            stack.append(node)
            for neighbour in graph.neighbours(node):
                other = neighbour.other
                if distance[other] < 0:
                    distance[other] = distance[node] + 1
                    queue.append(other)
                if distance[other] == distance[node] + 1:
                    sigma[other] += sigma[node]
                    predecessors[other].append(node)

        delta = dict.fromkeys(nodes, 0.0)
        while stack:
            node = stack.pop()
            for predecessor in predecessors[node]:
                if sigma[node]:
                    delta[predecessor] += (sigma[predecessor] / sigma[node]) * (1 + delta[node])
            if node != source:
                betweenness[node] += delta[node]

    scale = (count - 1) * (count - 2)
    correction = (count / samples) if samples else 1.0
    if scale <= 0:
        return betweenness
    return {node: (value * correction) / scale for node, value in betweenness.items()}


def personalized_pagerank(graph: KnowledgeGraph, seeds: dict[str, float] | None = None,
                          damping: float = 0.85, iterations: int = 40,
                          tolerance: float = 1e-9) -> dict[str, float]:
    """Random-walk-with-restart, weighted by link type.

    With seeds, this answers "how close is everything to these known-bad
    entities", propagating suspicion along relationships in proportion to how
    much each relationship type actually implies about association. Being a
    company's director carries more than sharing an employer, and the ontology
    already says so via each link type's weight.
    """
    nodes = list(graph.entities)
    if not nodes:
        return {}

    if seeds:
        total = sum(seeds.values()) or 1.0
        restart = {node: seeds.get(node, 0.0) / total for node in nodes}
    else:
        restart = {node: 1.0 / len(nodes) for node in nodes}

    weights: dict[str, list[tuple[str, float]]] = {}
    out_strength: dict[str, float] = {}
    for node in nodes:
        entries = []
        for neighbour in graph.neighbours(node):
            weight = graph.ontology.links[neighbour.edge.link_type].weight
            # Repeated observations of the same relationship strengthen it, but
            # with diminishing returns -- 200 payments are not 200x one payment.
            entries.append((neighbour.other, weight * (1.0 + math.log1p(neighbour.edge.count))))
        weights[node] = entries
        out_strength[node] = sum(w for _, w in entries)

    rank = dict(restart)
    for _ in range(iterations):
        nxt = {node: (1 - damping) * restart[node] for node in nodes}
        leaked = 0.0
        for node in nodes:
            mass = rank[node]
            if not mass:
                continue
            strength = out_strength[node]
            if strength <= 0:
                leaked += damping * mass
                continue
            share = damping * mass / strength
            for other, weight in weights[node]:
                nxt[other] += share * weight
        if leaked:
            for node in nodes:
                nxt[node] += leaked * restart[node]
        delta = sum(abs(nxt[n] - rank[n]) for n in nodes)
        rank = nxt
        if delta < tolerance:
            break
    return rank


def weighted_adjacency(graph: KnowledgeGraph) -> dict[str, dict[str, float]]:
    """Undirected weighted projection of the graph, for clustering."""
    adjacency: dict[str, dict[str, float]] = {node: {} for node in graph.entities}
    for edge in graph.edges.values():
        if edge.src not in adjacency or edge.dst not in adjacency:
            continue
        weight = graph.ontology.links[edge.link_type].weight * (1.0 + math.log1p(edge.count))
        adjacency[edge.src][edge.dst] = adjacency[edge.src].get(edge.dst, 0.0) + weight
        adjacency[edge.dst][edge.src] = adjacency[edge.dst].get(edge.src, 0.0) + weight
    return adjacency


def _louvain_level(adjacency, strength, total, resolution, rng):
    """One pass of local moving: shift nodes to whichever community gains most."""
    community = {node: node for node in adjacency}
    sigma_total = dict(strength)
    nodes = list(adjacency)
    rng.shuffle(nodes)
    improved = False

    for _ in range(30):
        moved = 0
        for node in nodes:
            current = community[node]
            incident: dict[str, float] = defaultdict(float)
            for neighbour, weight in adjacency[node].items():
                if neighbour != node:
                    incident[community[neighbour]] += weight

            sigma_total[current] -= strength[node]
            best_community = current
            best_gain = incident.get(current, 0.0) - resolution * sigma_total[current] * strength[node] / (2 * total)
            for candidate, weight in incident.items():
                if candidate == current:
                    continue
                gain = weight - resolution * sigma_total[candidate] * strength[node] / (2 * total)
                if gain > best_gain + 1e-12:
                    best_gain, best_community = gain, candidate
            sigma_total[best_community] += strength[node]

            if best_community != current:
                community[node] = best_community
                moved += 1
                improved = True
        if not moved:
            break
    return community, improved


def communities(graph: KnowledgeGraph, resolution: float = 1.0, seed: int = 5) -> dict[str, int]:
    """Louvain modularity-maximizing community detection.

    Label propagation was the obvious cheap choice here and it fails badly on
    this graph: with a dense giant component every node adopts its neighbours'
    label and the whole population collapses into a single community with
    modularity ~0, which is worse than useless because it looks like an answer.

    Louvain instead greedily moves nodes between communities to maximize
    modularity, then contracts each community into a node and repeats. That
    hierarchical contraction is what lets a small tightly-knit group survive
    inside a large loosely-connected one.
    """
    rng = random.Random(seed)
    adjacency = weighted_adjacency(graph)
    if not adjacency:
        return {}

    strength = {
        node: sum(edges.values()) + edges.get(node, 0.0)
        for node, edges in adjacency.items()
    }
    total = sum(strength.values()) / 2.0
    if total <= 0:
        return {node: index for index, node in enumerate(adjacency)}

    membership = {node: node for node in adjacency}
    for _ in range(10):
        community, improved = _louvain_level(adjacency, strength, total, resolution, rng)
        if not improved:
            break
        membership = {node: community[membership[node]] for node in membership}

        # Contract: every community becomes a single node whose self-loop
        # carries the weight that was internal to it.
        contracted: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for node, edges in adjacency.items():
            src = community[node]
            for neighbour, weight in edges.items():
                dst = community[neighbour]
                if neighbour == node:
                    contracted[src][dst] += weight          # pre-existing self-loop
                elif src == dst:
                    # An internal edge is visited from both endpoints; the
                    # self-loop must hold its weight once, not twice.
                    contracted[src][dst] += weight / 2.0
                else:
                    contracted[src][dst] += weight
        adjacency = {node: dict(edges) for node, edges in contracted.items()}
        strength = {
            node: sum(edges.values()) + edges.get(node, 0.0)
            for node, edges in adjacency.items()
        }
        if len(adjacency) <= 1:
            break

    remap: dict[str, int] = {}
    labels = {}
    for node in graph.entities:
        key = membership.get(node, node)
        if key not in remap:
            remap[key] = len(remap)
        labels[node] = remap[key]
    return labels


def modularity(graph: KnowledgeGraph, labels: dict[str, int]) -> float:
    """Newman modularity of a partition; a quick sanity check on communities."""
    total = len(graph.edges)
    if not total:
        return 0.0
    internal: dict[int, int] = defaultdict(int)
    degrees: dict[int, int] = defaultdict(int)
    for edge in graph.edges.values():
        src_label, dst_label = labels.get(edge.src), labels.get(edge.dst)
        if src_label is None or dst_label is None:
            continue
        degrees[src_label] += 1
        degrees[dst_label] += 1
        if src_label == dst_label:
            internal[src_label] += 1
    return sum(
        internal[label] / total - (degrees[label] / (2 * total)) ** 2
        for label in degrees
    )


# --------------------------------------------------------------------------
# typologies
# --------------------------------------------------------------------------

def circular_flows(graph: KnowledgeGraph, link_type: str = "transacted",
                   min_observations: int = 3, max_length: int = 8,
                   limit: int = 20) -> list[Finding]:
    """Find value that leaves an account and comes back.

    Money moving in a closed loop has no commercial purpose -- the point is to
    put distance and paperwork between an origin and a destination that are the
    same place. Only repeatedly-used edges are considered: a single payment
    that happens to complete a ring is coincidence, a standing pattern is not.
    """
    adjacency: dict[str, list[tuple[str, int, float]]] = defaultdict(list)
    for edge in graph.edges.values():
        if edge.link_type != link_type or edge.count < min_observations:
            continue
        adjacency[edge.src].append((edge.dst, edge.count, edge.total("amount")))

    findings: list[Finding] = []
    seen_cycles: set[frozenset] = set()

    for start in sorted(adjacency):
        stack = [(start, [start], 0.0, 0)]
        while stack and len(findings) < limit:
            node, path, value, hops = stack.pop()
            if hops >= max_length:
                continue
            for other, count, amount in adjacency.get(node, ()):
                if other == start and len(path) >= 3:
                    signature = frozenset(path)
                    if signature in seen_cycles:
                        continue
                    seen_cycles.add(signature)
                    total = value + amount
                    findings.append(Finding(
                        kind="circular_flow",
                        severity=min(1.0, 0.55 + 0.08 * len(path)),
                        entities=list(path),
                        title=f"Circular value flow across {len(path)} accounts",
                        detail=(
                            f"Funds traverse {len(path)} accounts and return to the "
                            f"origin. Total observed value {total:,.2f} over repeated transfers."
                        ),
                        evidence={
                            "cycle": [graph.label(node_id) for node_id in path],
                            "length": len(path),
                            "total_amount": round(total, 2),
                        },
                    ))
                    break
                # Only extend through nodes greater than the start, so each
                # cycle is discovered once rather than once per member.
                if other not in path and other > start:
                    stack.append((other, path + [other], value + amount, hops + 1))
    findings.sort(key=lambda f: -f.severity)
    return findings[:limit]


def structuring(graph: KnowledgeGraph, threshold: float = 10000.0,
                band: float = 0.15, min_count: int = 4,
                limit: int = 20) -> list[Finding]:
    """Detect payments deliberately sized to stay under a reporting threshold.

    Legitimate payment amounts are spread across orders of magnitude. A stream
    of transfers clustered tightly in the band just below a round reporting
    limit, and never above it, is a signature of someone who knows where the
    limit is.
    """
    floor = threshold * (1 - band)
    per_account: dict[str, list[dict]] = defaultdict(list)
    totals: dict[str, int] = defaultdict(int)

    for edge in graph.edges.values():
        if edge.link_type != "transacted":
            continue
        for observation in edge.observations:
            amount = observation.get("amount")
            if amount is None:
                continue
            amount = float(amount)
            totals[edge.src] += 1
            if floor <= amount < threshold:
                per_account[edge.src].append({
                    "amount": amount,
                    "to": edge.dst,
                    "date": observation.get("date"),
                })

    findings = []
    for account, hits in per_account.items():
        if len(hits) < min_count:
            continue
        total_payments = totals[account] or len(hits)
        concentration = len(hits) / total_payments
        # A handful of near-threshold payments out of hundreds is noise. The
        # signal is when they dominate the account's outgoing activity.
        if concentration < 0.35:
            continue
        amounts = [h["amount"] for h in hits]
        findings.append(Finding(
            kind="structuring",
            severity=min(1.0, 0.4 + 0.6 * concentration),
            entities=[account] + sorted({h["to"] for h in hits}),
            title=f"{len(hits)} payments just below the {threshold:,.0f} threshold",
            detail=(
                f"{graph.label(account)} sent {len(hits)} of {total_payments} outgoing "
                f"payments ({concentration:.0%}) in the band {floor:,.0f}-{threshold:,.0f}, "
                f"none above it."
            ),
            evidence={
                "account": graph.label(account),
                "count": len(hits),
                "outgoing_total": total_payments,
                "min": round(min(amounts), 2),
                "max": round(max(amounts), 2),
                "mean": round(sum(amounts) / len(amounts), 2),
                "counterparties": [graph.label(c) for c in sorted({h["to"] for h in hits})][:8],
            },
        ))
    findings.sort(key=lambda f: -f.severity)
    return findings[:limit]


def shared_registration(graph: KnowledgeGraph, min_orgs: int = 3,
                        limit: int = 20) -> list[Finding]:
    """Find addresses hosting an implausible number of companies.

    Company formation agents legitimately register many firms at one address,
    so this is a lead rather than a conclusion. It becomes interesting when the
    companies at that address also share officers, incorporation dates, or
    counterparties -- which the correlation step checks.
    """
    findings = []
    for location in graph.of_type("location"):
        registrants = [
            n.other for n in graph.neighbours(location.id, {"registered_at"})
            if graph.entities.get(n.other) and graph.entities[n.other].entity_type == "organization"
        ]
        if len(registrants) < min_orgs:
            continue

        officers: dict[str, set[str]] = defaultdict(set)
        for org_id in registrants:
            for neighbour in graph.neighbours(org_id, {"officer_of"}):
                officers[neighbour.other].add(org_id)
        shared_officers = {p: orgs for p, orgs in officers.items() if len(orgs) > 1}

        incorporations = sorted(
            str(graph.entities[o].props.get("incorporated"))
            for o in registrants if graph.entities[o].props.get("incorporated")
        )
        severity = min(1.0, 0.3 + 0.1 * len(registrants) + 0.12 * len(shared_officers))
        findings.append(Finding(
            kind="shared_registration",
            severity=severity,
            entities=[location.id] + registrants,
            title=f"{len(registrants)} companies registered at one address",
            detail=(
                f"{location.label()} is the registered office of {len(registrants)} "
                f"companies, {len(shared_officers)} of whose officers sit on more than one board."
            ),
            evidence={
                "address": location.label(),
                "companies": [graph.label(o) for o in registrants],
                "shared_officers": [graph.label(p) for p in shared_officers],
                "incorporation_dates": incorporations,
            },
        ))
    findings.sort(key=lambda f: -f.severity)
    return findings[:limit]


def _as_date(value):
    from datetime import date as _date

    if isinstance(value, _date):
        return value
    if isinstance(value, str):
        try:
            return _date.fromisoformat(value)
        except ValueError:
            return None
    return None


def rapid_incorporation(graph: KnowledgeGraph, window_days: int = 90,
                        min_orgs: int = 3, limit: int = 10) -> list[Finding]:
    """Companies sharing officers *and* incorporated in a tight window.

    The cluster is what matters, not any single officer. A network of shells is
    typically built by spreading directorships thinly -- each person sits on
    two boards, nobody sits on five -- precisely so that no one name links the
    whole structure. So companies are joined pairwise when they share any
    officer and were incorporated close together, and the connected components
    of that relation are the clusters.
    """
    by_officer: dict[str, list[str]] = defaultdict(list)
    for edge in graph.edges.values():
        if edge.link_type == "officer_of":
            by_officer[edge.src].append(edge.dst)

    incorporated = {
        org.id: _as_date(org.props.get("incorporated")) for org in graph.of_type("organization")
    }

    # Union-find over companies linked by a shared officer and close dates.
    parent: dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    pair_officers: dict[tuple[str, str], set[str]] = defaultdict(set)
    for officer, orgs in by_officer.items():
        unique = sorted(set(orgs))
        for i, left in enumerate(unique):
            for right in unique[i + 1:]:
                left_date, right_date = incorporated.get(left), incorporated.get(right)
                if not left_date or not right_date:
                    continue
                if abs((left_date - right_date).days) > window_days:
                    continue
                pair_officers[(left, right)].add(officer)
                parent[find(left)] = find(right)

    clusters: dict[str, set[str]] = defaultdict(set)
    for org_id in parent:
        clusters[find(org_id)].add(org_id)

    grouped: dict[frozenset, set[str]] = {}
    for members in clusters.values():
        if len(members) < min_orgs:
            continue
        officers = set()
        for (left, right), shared in pair_officers.items():
            if left in members and right in members:
                officers |= shared
        grouped[frozenset(members)] = officers

    findings = []
    for orgs, officers in grouped.items():
        findings.append(Finding(
            kind="rapid_incorporation",
            severity=min(1.0, 0.45 + 0.1 * len(orgs) + 0.1 * len(officers)),
            entities=sorted(orgs) + sorted(officers),
            title=f"{len(orgs)} companies incorporated within {window_days} days by shared officers",
            detail=(
                f"{', '.join(graph.label(o) for o in sorted(orgs))} were incorporated inside a "
                f"{window_days}-day window and share {len(officers)} officer(s)."
            ),
            evidence={
                "companies": [graph.label(o) for o in sorted(orgs)],
                "officers": [graph.label(p) for p in sorted(officers)],
            },
        ))
    findings.sort(key=lambda f: -f.severity)
    return findings[:limit]


# --------------------------------------------------------------------------
# risk
# --------------------------------------------------------------------------

CONTROL_LINKS = {"owns_account", "controls_account", "officer_of"}


def principals(graph: KnowledgeGraph, entity_id: str, depth: int = 2) -> dict[str, int]:
    """Walk control relationships from an entity back to the humans behind it.

    A typology fires on whatever object carries the behaviour, which for
    financial patterns is an account number. An account number is not a
    suspect. Following ownership and directorship back to people and companies
    is what turns a pattern into something an investigator can act on -- and it
    is also what lets findings on different object types be recognized as
    describing the same network.
    """
    found: dict[str, int] = {}
    frontier = {entity_id}
    for hop in range(1, depth + 1):
        nxt: set[str] = set()
        for node in frontier:
            entity = graph.entities.get(node)
            if entity is None:
                continue
            if entity.entity_type == "account":
                for neighbour in graph.neighbours(node, {"owns_account", "controls_account"}):
                    nxt.add(neighbour.other)
            elif entity.entity_type == "organization":
                for neighbour in graph.neighbours(node, {"officer_of"}):
                    other = graph.entities.get(neighbour.other)
                    if other is not None and other.entity_type == "person":
                        nxt.add(neighbour.other)
        nxt -= set(found) | {entity_id}
        if not nxt:
            break
        for node in nxt:
            found[node] = hop
        frontier = nxt
    return found


def attribute_findings(graph: KnowledgeGraph, findings: list[Finding]) -> list[Finding]:
    """Expand every finding to include the parties who control its entities."""
    for finding in findings:
        core = set(finding.entities)
        attributed: dict[str, int] = {}
        for entity_id in list(finding.entities):
            for owner, hops in principals(graph, entity_id).items():
                if owner in core:
                    continue
                # Keep the shortest route to each party.
                if owner not in attributed or hops < attributed[owner]:
                    attributed[owner] = hops
        if attributed:
            ordered = sorted(attributed, key=lambda e: (attributed[e], e))
            finding.entities = list(finding.entities) + ordered
            finding.attributed = attributed
            finding.evidence["controlled_by"] = [
                f"{graph.label(e)} [{graph.entities[e].entity_type}, {attributed[e]} hop(s)]"
                for e in ordered if e in graph.entities
            ][:12]
    return findings


def risk_scores(graph: KnowledgeGraph, seeds: list[str], findings: list[Finding],
                seed_weight: float = 1.0) -> dict[str, float]:
    """Blend seeded proximity with typology hits into one score per entity.

    Two independent signals. Proximity says an entity is closely tied to
    something already known to be bad. Typologies say the entity is itself
    behaving in a recognized way. Combining them means neither a well-connected
    innocent nor an isolated oddity gets to the top on its own.
    """
    if not graph.entities:
        return {}

    proximity = personalized_pagerank(graph, {s: seed_weight for s in seeds} if seeds else None)
    ordered = sorted(proximity.values(), reverse=True)
    # Normalize against a high percentile rather than the max: one dominant
    # seed would otherwise flatten everything else to zero.
    reference = ordered[max(0, len(ordered) // 100)] or 1.0

    scores: dict[str, float] = {}
    for entity_id in graph.entities:
        scores[entity_id] = min(1.0, proximity.get(entity_id, 0.0) / reference) * 0.6

    # Typology evidence accumulates as noisy-OR rather than a running sum.
    # Adding severities pins everything the case touches at 1.0 within three
    # findings, which erases the ranking; combining them as independent
    # probabilities keeps later evidence meaningful but never saturating.
    for finding in findings:
        for entity_id in finding.entities:
            if entity_id not in scores:
                continue
            hops = finding.attributed.get(entity_id, 0)
            contribution = 0.45 * finding.severity / (1 + hops)
            scores[entity_id] = 1 - (1 - scores[entity_id]) * (1 - contribution)

    # Seeds are deliberately *not* pinned to 1.0. Doing so ties every known
    # subject at the top of the ranking and destroys exactly the comparison the
    # ranking exists to support: which not-yet-known entities look worst.
    return scores


def run_all(graph: KnowledgeGraph, seeds: list[str] | None = None) -> dict:
    """Run every typology and structural measure; returns findings and scores."""
    seeds = seeds or []
    findings: list[Finding] = []
    findings.extend(circular_flows(graph))
    findings.extend(structuring(graph))
    findings.extend(shared_registration(graph))
    findings.extend(rapid_incorporation(graph))
    attribute_findings(graph, findings)

    scores = risk_scores(graph, seeds, findings)
    labels = communities(graph)
    return {
        "findings": findings,
        "risk": scores,
        "communities": labels,
        "modularity": modularity(graph, labels),
        "community_sizes": Counter(labels.values()),
    }


def correlate(graph: KnowledgeGraph, findings: list[Finding],
              risk: dict[str, float], min_overlap: int = 2) -> list[dict]:
    """Group findings that concern overlapping sets of entities.

    A single typology hit is a lead. Several independent typologies landing on
    the same cluster of entities is a case, and presenting them separately
    leaves the analyst to notice the overlap by eye.
    """
    groups: list[dict] = []
    used: set[int] = set()

    for i, finding in enumerate(findings):
        if i in used:
            continue
        members = set(finding.entities)
        cluster = [finding]
        used.add(i)
        changed = True
        while changed:
            changed = False
            for j, other in enumerate(findings):
                if j in used:
                    continue
                if len(members & set(other.entities)) >= min_overlap:
                    members |= set(other.entities)
                    cluster.append(other)
                    used.add(j)
                    changed = True
        if len(cluster) < 2:
            continue

        people = [e for e in members if graph.entities.get(e)
                  and graph.entities[e].entity_type == "person"]
        orgs = [e for e in members if graph.entities.get(e)
                and graph.entities[e].entity_type == "organization"]
        groups.append({
            "kinds": sorted({f.kind for f in cluster}),
            "findings": [f.to_dict() for f in cluster],
            "entities": sorted(members),
            "people": sorted(people, key=lambda e: -risk.get(e, 0)),
            "organizations": sorted(orgs, key=lambda e: -risk.get(e, 0)),
            "score": round(sum(f.severity for f in cluster) / len(cluster)
                           * min(1.0, 0.6 + 0.2 * len({f.kind for f in cluster})), 3),
        })
    groups.sort(key=lambda g: -g["score"])
    return groups
