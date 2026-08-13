"""End-to-end pipeline and the investigation surface built on top of it.

``Pipeline.run`` takes a directory of raw source files to a scored knowledge
graph. ``Investigation`` is what an analyst actually touches: dossiers,
connection discovery, and the ranked case list.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import analytics
from .analytics import Finding
from .graph import KnowledgeGraph
from .ingest import Ingestor
from .ontology import Ontology, load_default
from .resolve import Resolver, ResolutionResult, evaluate, find_errors


@dataclass
class PipelineResult:
    graph: KnowledgeGraph
    resolution: ResolutionResult
    ingestor: Ingestor
    findings: list[Finding] = field(default_factory=list)
    risk: dict = field(default_factory=dict)
    communities: dict = field(default_factory=dict)
    cases: list = field(default_factory=list)
    evaluation: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)
    seeds: list = field(default_factory=list)


class Pipeline:
    def __init__(self, ontology: Ontology | None = None, verbose: bool = True):
        self.ontology = ontology or load_default()
        self.verbose = verbose

    def _say(self, message: str):
        if self.verbose:
            print(message, flush=True)

    def run(self, data_dir: str | Path, evaluate_against_truth: bool = True) -> PipelineResult:
        data_dir = Path(data_dir)
        timings = {}

        start = time.time()
        ingestor = Ingestor(self.ontology)
        report = ingestor.ingest_dir(data_dir)
        by_type = ingestor.by_type()
        timings["ingest"] = time.time() - start
        self._say(
            f"  ingest    {sum(len(v) for v in by_type.values()):>7,} records  "
            f"{len(ingestor.links):>7,} link observations  "
            f"{len(report)} sources  [{timings['ingest']:.1f}s]"
        )

        truth_attached = 0
        if evaluate_against_truth:
            truth_attached = ingestor.attach_truth(data_dir / "truth.json")

        start = time.time()
        resolver = Resolver(self.ontology)
        resolution = resolver.resolve(by_type)
        timings["resolve"] = time.time() - start
        total = resolution.stats["total"]
        self._say(
            f"  resolve   {total['entities']:>7,} entities  from {total['records']:,} records  "
            f"({total['scored_pairs']:,} pairs scored, {total['conflicts_blocked']:,} merges vetoed)  "
            f"[{timings['resolve']:.1f}s]"
        )

        start = time.time()
        graph = KnowledgeGraph.build(self.ontology, resolution, ingestor.links)
        timings["graph"] = time.time() - start
        stats = graph.stats()
        self._say(
            f"  graph     {stats['entities']:>7,} nodes  {stats['edges']:,} edges  "
            f"{stats['observations']:,} observations  [{timings['graph']:.1f}s]"
        )

        seeds = self._mark_watchlist(graph)
        self._say(f"  watchlist {len(seeds):>7,} entities flagged from listings")

        start = time.time()
        results = analytics.run_all(graph, seeds)
        timings["analytics"] = time.time() - start
        findings = results["findings"]
        cases = analytics.correlate(graph, findings, results["risk"])
        timings["analytics"] = time.time() - start
        self._say(
            f"  analytics {len(findings):>7,} findings  {len(cases)} correlated cases  "
            f"modularity {results['modularity']:.3f}  [{timings['analytics']:.1f}s]"
        )

        for entity_id, score in results["risk"].items():
            if score >= 0.5:
                graph.entities[entity_id].flags["risk"] = round(score, 3)
        for finding in findings:
            for entity_id in finding.entities:
                entity = graph.entities.get(entity_id)
                if entity is not None:
                    entity.flags.setdefault("typologies", [])
                    if finding.kind not in entity.flags["typologies"]:
                        entity.flags["typologies"].append(finding.kind)

        evaluation = {}
        if truth_attached:
            evaluation = evaluate(by_type, resolution)
            overall = evaluation["overall"]
            self._say(
                f"  evaluate  precision {overall['precision']:.4f}  "
                f"recall {overall['recall']:.4f}  f1 {overall['f1']:.4f}  "
                f"({overall['labelled_records']:,} labelled records)"
            )

        return PipelineResult(
            graph=graph,
            resolution=resolution,
            ingestor=ingestor,
            findings=findings,
            risk=results["risk"],
            communities=results["communities"],
            cases=cases,
            evaluation=evaluation,
            timings=timings,
            seeds=seeds,
        )

    def _mark_watchlist(self, graph: KnowledgeGraph) -> list[str]:
        """Any entity carrying a watchlist record becomes a risk seed.

        This is the payoff of resolution in one line: the watchlist holds only
        a name and a date of birth, yet the flag lands on a fully-formed entity
        with accounts, employers, and counterparties attached.
        """
        seeds = []
        for entity in graph.entities.values():
            listings = [m for m in entity.members if m.startswith("watch:")]
            if listings:
                entity.flags["watchlisted"] = True
                entity.flags["listings"] = listings
                seeds.append(entity.id)
        return seeds


class Investigation:
    """Analyst-facing views over a completed pipeline run."""

    def __init__(self, result: PipelineResult):
        self.result = result
        self.graph = result.graph

    # -- lookup ------------------------------------------------------------

    def resolve_ref(self, reference: str):
        """Accept an entity id, a record key, or a name fragment."""
        if reference in self.graph.entities:
            return self.graph.entities[reference]
        entity = self.result.resolution.entity_for(reference)
        if entity is not None:
            return self.graph.entities.get(entity.id)
        hits = self.graph.search(reference, limit=1)
        return hits[0] if hits else None

    # -- dossier -----------------------------------------------------------

    def dossier(self, reference: str) -> str:
        entity = self.resolve_ref(reference)
        if entity is None:
            return f"No entity matches '{reference}'."

        lines = ["=" * 78, self.graph.describe(entity.id), "=" * 78]

        risk = self.result.risk.get(entity.id, 0.0)
        lines.append(f"\nRISK  {risk:.3f}  {_bar(risk)}")
        if entity.flags.get("watchlisted"):
            lines.append("      ON WATCHLIST")

        lines.append(f"\nPROVENANCE  {len(entity.members)} source records resolved into this entity")
        by_source: dict[str, int] = {}
        for member in entity.members:
            by_source[member.split(":", 1)[0]] = by_source.get(member.split(":", 1)[0], 0) + 1
        for source, count in sorted(by_source.items(), key=lambda kv: -kv[1]):
            examples = [m for m in entity.members if m.startswith(source + ":")][:3]
            lines.append(f"    {source:<10} {count:>4}   e.g. {', '.join(examples)}")

        groups: dict[str, list] = {}
        for neighbour in self.graph.neighbours(entity.id):
            groups.setdefault(neighbour.edge.link_type, []).append(neighbour)
        if groups:
            lines.append("\nRELATIONSHIPS")
            for link_type, neighbours in sorted(groups.items()):
                link = self.graph.ontology.links[link_type]
                lines.append(f"    {link.label} ({len(neighbours)})")
                ranked = sorted(neighbours, key=lambda n: -self.result.risk.get(n.other, 0))
                for neighbour in ranked[:6]:
                    other_risk = self.result.risk.get(neighbour.other, 0.0)
                    marker = "!" if other_risk >= 0.5 else " "
                    detail = f" x{neighbour.edge.count}" if neighbour.edge.count > 1 else ""
                    amount = neighbour.edge.total("amount")
                    money = f"  {amount:,.0f}" if amount else ""
                    lines.append(
                        f"      {marker} {self.graph.label(neighbour.other):<34} "
                        f"[{neighbour.other}]{detail}{money}"
                    )
                if len(ranked) > 6:
                    lines.append(f"        ... and {len(ranked) - 6} more")

        hits = [f for f in self.result.findings if entity.id in f.entities]
        if hits:
            lines.append("\nFINDINGS")
            for finding in hits:
                lines.append(f"    [{finding.severity:.2f}] {finding.title}")
                lines.append(f"           {finding.detail}")

        seeds = [s for s in self.result.seeds if s != entity.id]
        if seeds and not entity.flags.get("watchlisted"):
            nearest = None
            for seed in seeds[:12]:
                path = self.graph.shortest_path(entity.id, seed, max_hops=4)
                if path and (nearest is None or len(path) < len(nearest[1])):
                    nearest = (seed, path)
            if nearest:
                lines.append(f"\nNEAREST WATCHLISTED ENTITY  ({len(nearest[1])} hops)")
                lines.append(_indent(self.graph.format_path(nearest[1], entity.id), 4))

        return "\n".join(lines)

    # -- connections -------------------------------------------------------

    def connect(self, left: str, right: str, max_hops: int = 4, limit: int = 5) -> str:
        source = self.resolve_ref(left)
        target = self.resolve_ref(right)
        if source is None:
            return f"No entity matches '{left}'."
        if target is None:
            return f"No entity matches '{right}'."

        paths = self.graph.all_paths(source.id, target.id, max_hops=max_hops, limit=limit)
        if not paths:
            return (f"No connection within {max_hops} hops between "
                    f"{source.label()} and {target.label()}.")

        lines = [f"{len(paths)} connection(s) between {source.label()} and {target.label()}:"]
        for index, path in enumerate(paths, start=1):
            lines.append(f"\n  path {index}  ({len(path)} hops)")
            lines.append(_indent(self.graph.format_path(path, source.id), 4))
        return "\n".join(lines)

    # -- case list ---------------------------------------------------------

    def case_report(self, limit: int = 3) -> str:
        if not self.result.cases:
            return "No correlated cases."
        lines = []
        for index, case in enumerate(self.result.cases[:limit], start=1):
            lines.append("=" * 78)
            lines.append(f"CASE {index}   score {case['score']:.2f}   "
                         f"signals: {', '.join(case['kinds'])}")
            lines.append("=" * 78)
            if case["people"]:
                lines.append(f"  Subjects ({len(case['people'])})")
                for entity_id in case["people"][:10]:
                    entity = self.graph.entities[entity_id]
                    lines.append(
                        f"    {'!' if entity.flags.get('watchlisted') else ' '} "
                        f"{entity.label():<32} risk {self.result.risk.get(entity_id, 0):.2f}  "
                        f"[{entity_id}]"
                    )
            if case["organizations"]:
                lines.append(f"  Entities ({len(case['organizations'])})")
                for entity_id in case["organizations"][:10]:
                    lines.append(f"      {self.graph.label(entity_id):<32} [{entity_id}]")
            lines.append("  Findings")
            for finding in case["findings"]:
                lines.append(f"    [{finding['severity']:.2f}] {finding['title']}")
                lines.append(f"           {finding['detail']}")
        return "\n".join(lines)

    def top_risk(self, type_name: str = "person", limit: int = 15) -> str:
        ranked = sorted(
            (e for e in self.graph.entities.values() if e.entity_type == type_name),
            key=lambda e: -self.result.risk.get(e.id, 0.0),
        )[:limit]
        lines = [f"{'entity':<34} {'risk':>6}  {'deg':>4}  flags"]
        lines.append("-" * 78)
        for entity in ranked:
            flags = []
            if entity.flags.get("watchlisted"):
                flags.append("watchlist")
            flags.extend(entity.flags.get("typologies", []))
            lines.append(
                f"{entity.label()[:33]:<34} {self.result.risk.get(entity.id, 0):>6.3f}  "
                f"{self.graph.degree(entity.id):>4}  {', '.join(flags)}"
            )
        return "\n".join(lines)


def _bar(value: float, width: int = 24) -> str:
    filled = int(round(max(0.0, min(1.0, value)) * width))
    return "#" * filled + "." * (width - filled)


def _indent(text: str, spaces: int) -> str:
    pad = " " * spaces
    return "\n".join(pad + line for line in text.splitlines())


def save_artifacts(result: PipelineResult, out_dir: str | Path) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result.graph.save(out / "graph.json")
    (out / "findings.json").write_text(
        json.dumps([f.to_dict() for f in result.findings], indent=1, default=str), encoding="utf-8"
    )
    (out / "cases.json").write_text(
        json.dumps(result.cases, indent=1, default=str), encoding="utf-8"
    )
    (out / "resolution_stats.json").write_text(
        json.dumps(result.resolution.stats, indent=1), encoding="utf-8"
    )
    if result.evaluation:
        (out / "evaluation.json").write_text(
            json.dumps(result.evaluation, indent=1), encoding="utf-8"
        )
    return {"out_dir": str(out)}
