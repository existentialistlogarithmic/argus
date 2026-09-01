"""Entity resolution.

Two ways to get this wrong. Under-merging leaves one person as five entities,
so the network never appears; its edges are spread across the fragments.
Over-merging fuses unrelated people, which invents relationships that do not
exist. The second is much worse.

Scoring is Fellegi-Sunter probabilistic linkage: each property contributes a
log-likelihood ratio, the ratios sum, and the total is compared against a
threshold. A property's weight comes from how often it agrees for true matches
against how often it collides by chance, so a national ID outweighs a first
name by orders of magnitude with nothing to tune.

Clustering is greedy and conflict-aware rather than connected components.
Transitive closure over pairwise matches is how over-merging happens: A~B and
B~C does not make A~C, and naive union-find will chain a whole population into
one blob.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

from .model import Entity, MatchDecision, Record, merge_props
from .ontology import EntityType, Ontology

# Below this u_prob a property counts as an exclusive identifier: two
# different non-empty values are evidence of difference, enough to veto a
# merge outright.
EXCLUSIVE_U_PROB = 1e-4


@dataclass
class ResolutionStats:
    records: int = 0
    candidate_pairs: int = 0
    scored_pairs: int = 0
    matches: int = 0
    review: int = 0
    conflicts_blocked: int = 0
    entities: int = 0
    blocks: int = 0
    largest_block: int = 0
    skipped_blocks: int = 0

    def as_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class ResolutionResult:
    entities: dict[str, Entity] = field(default_factory=dict)
    assignment: dict[str, str] = field(default_factory=dict)  # record key -> entity id
    review_queue: list[MatchDecision] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def entity_for(self, record_key: str) -> Entity | None:
        entity_id = self.assignment.get(record_key)
        return self.entities.get(entity_id) if entity_id else None


class Weights:
    """Precomputed Fellegi-Sunter weights for one entity type."""

    def __init__(self, entity_type: EntityType):
        self.entity_type = entity_type
        self.agree: dict[str, float] = {}
        self.disagree: dict[str, float] = {}
        self.exclusive: set[str] = set()
        for name, prop in entity_type.properties.items():
            m = min(max(prop.m_prob, 1e-9), 1 - 1e-9)
            u = min(max(prop.u_prob, 1e-12), 1 - 1e-9)
            self.agree[name] = math.log2(m / u)
            self.disagree[name] = math.log2((1 - m) / (1 - u))
            if u < EXCLUSIVE_U_PROB and not prop.multi:
                self.exclusive.add(name)

    def evidence(self, name: str, similarity: float) -> float:
        """Turn a similarity into a signed weight.

        Full agreement earns the agreement weight, clear disagreement the
        negative one. In between it interpolates, so a near-miss on a date of
        birth is not scored as though the records asserted unrelated dates.
        """
        prop = self.entity_type.properties[name]
        threshold = prop.threshold
        if similarity >= threshold:
            return self.agree[name]
        floor = prop.partial_floor
        if similarity <= floor:
            return self.disagree[name]
        span = threshold - floor
        fraction = (similarity - floor) / span if span else 0.0
        return self.disagree[name] + fraction * (self.agree[name] - self.disagree[name])


class Resolver:
    def __init__(self, ontology: Ontology, max_block_size: int = 400):
        self.ontology = ontology
        self.max_block_size = max_block_size
        self._weights: dict[str, Weights] = {}

    def weights(self, type_name: str) -> Weights:
        if type_name not in self._weights:
            self._weights[type_name] = Weights(self.ontology.entity(type_name))
        return self._weights[type_name]

    # -- scoring -----------------------------------------------------------

    def score(self, left: Record, right: Record) -> MatchDecision:
        """Score one candidate pair and explain the verdict."""
        entity_type = self.ontology.entity(left.entity_type)
        weights = self.weights(left.entity_type)
        total = 0.0
        evidence = []

        for name, prop in entity_type.properties.items():
            a, b = left.props.get(name), right.props.get(name)
            # Missing data is not disagreement. A source that never collects
            # a field should not be penalised for its silence.
            if a in (None, "", []) or b in (None, "", []):
                continue
            similarity = prop.similarity(a, b)
            weight = weights.evidence(name, similarity)
            total += weight
            evidence.append((name, similarity, weight))

        if total >= entity_type.match_threshold:
            verdict = "match"
        elif total >= entity_type.review_threshold:
            verdict = "review"
        else:
            verdict = "no-match"
        return MatchDecision(left.key, right.key, total, verdict, evidence)

    # -- blocking ----------------------------------------------------------

    def _blocks(self, records: list[Record], entity_type: EntityType, stats: ResolutionStats):
        index: dict[str, list[int]] = defaultdict(list)
        for position, record in enumerate(records):
            for key in entity_type.blocking_keys(record.props):
                index[key].append(position)

        stats.blocks = len(index)
        for key, members in index.items():
            size = len(members)
            stats.largest_block = max(stats.largest_block, size)
            if size < 2:
                continue
            # An oversized block means the key lost its selectivity, usually
            # a default value or a blank that slipped through. Scoring it is
            # quadratic and rarely productive.
            if size > self.max_block_size:
                stats.skipped_blocks += 1
                continue
            yield members

    def _candidate_pairs(self, records: list[Record], entity_type: EntityType,
                         stats: ResolutionStats):
        seen: set[tuple[int, int]] = set()
        for members in self._blocks(records, entity_type, stats):
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    pair = (members[i], members[j]) if members[i] < members[j] else (members[j], members[i])
                    if pair not in seen:
                        seen.add(pair)
                        yield pair
        stats.candidate_pairs = len(seen)

    # -- clustering --------------------------------------------------------

    def _cluster(self, records: list[Record], decisions: list[tuple[int, int, MatchDecision]],
                 entity_type: EntityType, stats: ResolutionStats) -> list[list[int]]:
        """Greedy agglomeration, strongest evidence first.

        A merge is refused when the two clusters hold different values for an
        exclusive identifier. Without that check, one strong name match chains
        together two people who demonstrably hold different passports.
        """
        parent = list(range(len(records)))
        exclusive_values: list[dict[str, set]] = []
        for record in records:
            values = {}
            for name in self.weights(entity_type.name).exclusive:
                value = record.props.get(name)
                if value not in (None, "", []):
                    values[name] = {value}
            exclusive_values.append(values)

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        ordered = sorted(decisions, key=lambda item: -item[2].score)
        for i, j, decision in ordered:
            root_i, root_j = find(i), find(j)
            if root_i == root_j:
                continue
            left, right = exclusive_values[root_i], exclusive_values[root_j]
            conflict = any(
                name in right and not (left[name] & right[name])
                for name in left
            )
            if conflict:
                stats.conflicts_blocked += 1
                continue
            parent[root_j] = root_i
            for name, values in right.items():
                left.setdefault(name, set()).update(values)

        clusters: dict[int, list[int]] = defaultdict(list)
        for position in range(len(records)):
            clusters[find(position)].append(position)
        return list(clusters.values())

    # -- driver ------------------------------------------------------------

    def resolve(self, records_by_type: dict[str, list[Record]],
                progress=None) -> ResolutionResult:
        result = ResolutionResult()
        overall = ResolutionStats()
        per_type = {}

        for type_name, records in sorted(records_by_type.items()):
            entity_type = self.ontology.entity(type_name)
            stats = ResolutionStats(records=len(records))

            if not entity_type.resolvable:
                clusters = [[i] for i in range(len(records))]
            else:
                decisions = []
                for i, j in self._candidate_pairs(records, entity_type, stats):
                    decision = self.score(records[i], records[j])
                    stats.scored_pairs += 1
                    if decision.verdict == "match":
                        decisions.append((i, j, decision))
                        stats.matches += 1
                    elif decision.verdict == "review":
                        stats.review += 1
                        if len(result.review_queue) < 500:
                            result.review_queue.append(decision)
                clusters = self._cluster(records, decisions, entity_type, stats)

            for index, members in enumerate(sorted(clusters, key=lambda c: min(c))):
                entity_id = f"{type_name[:3].upper()}-{index:06d}"
                member_records = [records[position] for position in members]
                entity = Entity(
                    id=entity_id,
                    entity_type=type_name,
                    props=merge_props(entity_type, member_records),
                    members=[record.key for record in member_records],
                    sources=sorted({record.source for record in member_records}),
                    confidence=min(1.0, 0.55 + 0.15 * len(member_records)),
                )
                result.entities[entity_id] = entity
                for record in member_records:
                    result.assignment[record.key] = entity_id

            stats.entities = len(clusters)
            per_type[type_name] = stats.as_dict()
            for attribute in overall.__dict__:
                setattr(overall, attribute,
                        getattr(overall, attribute) + getattr(stats, attribute))
            if progress:
                progress(type_name, stats)

        result.stats = {"total": overall.as_dict(), "by_type": per_type}
        return result


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------

def _pair_count(n: int) -> int:
    return n * (n - 1) // 2


def evaluate(records_by_type: dict[str, list[Record]], result: ResolutionResult) -> dict:
    """Score resolution against ground truth, pairwise.

    Pairwise metrics measure the decisions the system actually made, and punish
    a bad merge in proportion to how many spurious pairs it created. Computed
    from a contingency table rather than by enumerating pairs, so it stays
    linear in the number of records.
    """
    report = {}
    grand = {"tp": 0, "predicted": 0, "actual": 0, "records": 0}

    for type_name, records in sorted(records_by_type.items()):
        labelled = [r for r in records if r.truth_id]
        if not labelled:
            continue

        cluster_sizes: dict[str, int] = defaultdict(int)
        truth_sizes: dict[str, int] = defaultdict(int)
        cells: dict[tuple[str, str], int] = defaultdict(int)

        for record in labelled:
            entity_id = result.assignment.get(record.key)
            if entity_id is None:
                continue
            cluster_sizes[entity_id] += 1
            truth_sizes[record.truth_id] += 1
            cells[(entity_id, record.truth_id)] += 1

        tp = sum(_pair_count(n) for n in cells.values())
        predicted = sum(_pair_count(n) for n in cluster_sizes.values())
        actual = sum(_pair_count(n) for n in truth_sizes.values())

        precision = tp / predicted if predicted else 1.0
        recall = tp / actual if actual else 1.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        # A cluster counts as exact when every record in it shares one truth
        # id and it holds every record carrying that id.
        exact = 0
        for entity_id, size in cluster_sizes.items():
            members = {t: c for (e, t), c in cells.items() if e == entity_id}
            if len(members) == 1:
                truth_id, count = next(iter(members.items()))
                if truth_sizes[truth_id] == count:
                    exact += 1

        report[type_name] = {
            "labelled_records": len(labelled),
            "true_entities": len(truth_sizes),
            "predicted_entities": len(cluster_sizes),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "exact_clusters": exact,
            "exact_cluster_rate": round(exact / len(truth_sizes), 4) if truth_sizes else 0.0,
            "pairs": {"tp": tp, "predicted": predicted, "actual": actual},
        }
        grand["tp"] += tp
        grand["predicted"] += predicted
        grand["actual"] += actual
        grand["records"] += len(labelled)

    precision = grand["tp"] / grand["predicted"] if grand["predicted"] else 1.0
    recall = grand["tp"] / grand["actual"] if grand["actual"] else 1.0
    report["overall"] = {
        "labelled_records": grand["records"],
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(2 * precision * recall / (precision + recall), 4) if (precision + recall) else 0.0,
    }
    return report


def find_errors(records_by_type: dict[str, list[Record]], result: ResolutionResult,
                limit: int = 10) -> dict:
    """Surface concrete false merges and misses.

    An aggregate F1 says the system is imperfect without saying how. These are
    the examples worth putting in front of someone.
    """
    false_merges, missed = [], []
    for type_name, records in sorted(records_by_type.items()):
        labelled = [r for r in records if r.truth_id]
        by_entity: dict[str, list[Record]] = defaultdict(list)
        by_truth: dict[str, list[Record]] = defaultdict(list)
        for record in labelled:
            entity_id = result.assignment.get(record.key)
            if entity_id:
                by_entity[entity_id].append(record)
            by_truth[record.truth_id].append(record)

        for entity_id, members in by_entity.items():
            truths = {r.truth_id for r in members}
            if len(truths) > 1 and len(false_merges) < limit:
                false_merges.append({
                    "entity": entity_id,
                    "type": type_name,
                    "truths": sorted(truths),
                    "records": [
                        {"key": r.key, "truth": r.truth_id,
                         "name": r.props.get("full_name") or r.props.get("legal_name") or r.key}
                        for r in members
                    ],
                })

        for truth_id, members in by_truth.items():
            entities = {result.assignment.get(r.key) for r in members}
            entities.discard(None)
            if len(entities) > 1 and len(missed) < limit:
                missed.append({
                    "truth": truth_id,
                    "type": type_name,
                    "fragments": len(entities),
                    "records": [
                        {"key": r.key, "entity": result.assignment.get(r.key),
                         "name": r.props.get("full_name") or r.props.get("legal_name") or r.key}
                        for r in members
                    ],
                })
    return {"false_merges": false_merges, "fragmented": missed}
