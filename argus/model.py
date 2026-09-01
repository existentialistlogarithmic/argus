"""Core data model.

Three representations, kept separate so provenance survives the pipeline:

    Record   one row as a source system asserted it. Never mutated or merged.
    Entity   a cluster of Records the resolver decided are the same object.
    Edge     a relationship between two Entities, plus every observation of it.

"Why does the system think this?" is always answerable as a set of Records.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime


def _jsonable(value):
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    return value


@dataclass
class Record:
    """A single assertion from a single source about a single object."""

    source: str
    local_id: str
    entity_type: str
    raw: dict = field(default_factory=dict)
    props: dict = field(default_factory=dict)
    truth_id: str | None = None  # evaluation only; the resolver never reads it

    @property
    def key(self) -> str:
        return f"{self.source}:{self.local_id}"

    def get(self, name):
        return self.props.get(name)

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "source": self.source,
            "local_id": self.local_id,
            "entity_type": self.entity_type,
            "props": _jsonable(self.props),
            "raw": _jsonable(self.raw),
        }


@dataclass
class LinkRecord:
    """An observed relationship, expressed in record keys.

    Links are asserted between source-system identifiers rather than entities,
    because no entities exist at ingest time. graph.build rewires them once
    resolution has decided which records are which entity.
    """

    source: str
    local_id: str
    link_type: str
    source_key: str
    target_key: str
    props: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.source}:{self.local_id}"


@dataclass
class Entity:
    """A resolved real-world object."""

    id: str
    entity_type: str
    props: dict = field(default_factory=dict)
    members: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    confidence: float = 1.0
    flags: dict = field(default_factory=dict)

    def get(self, name, default=None):
        return self.props.get(name, default)

    def label(self, ontology=None) -> str:
        """Best human-readable name for this entity."""
        for candidate in ("full_name", "legal_name", "account_number", "address", "label"):
            value = self.props.get(candidate)
            if value:
                if isinstance(value, list):
                    value = value[0]
                text = str(value)
                return text.title() if candidate in ("full_name", "legal_name") else text
        return self.id

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.entity_type,
            "label": self.label(),
            "props": _jsonable(self.props),
            "members": list(self.members),
            "sources": list(self.sources),
            "confidence": round(self.confidence, 4),
            "flags": _jsonable(self.flags),
        }


@dataclass
class Edge:
    """A relationship between two entities, with every supporting observation."""

    id: str
    link_type: str
    src: str
    dst: str
    observations: list[dict] = field(default_factory=list)
    provenance: list[str] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.observations) or 1

    def total(self, prop: str = "amount") -> float:
        return sum(float(o.get(prop) or 0) for o in self.observations)

    def dates(self) -> list:
        return sorted(o["date"] for o in self.observations if o.get("date"))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.link_type,
            "src": self.src,
            "dst": self.dst,
            "count": self.count,
            "observations": _jsonable(self.observations),
            "provenance": list(self.provenance),
        }


@dataclass
class MatchDecision:
    """Why the resolver linked (or refused to link) two records."""

    left: str
    right: str
    score: float
    verdict: str  # match | review | no-match
    evidence: list[tuple[str, float, float]] = field(default_factory=list)

    def explain(self) -> str:
        lines = [f"{self.left}  <->  {self.right}   score={self.score:+.2f}  [{self.verdict}]"]
        for prop, similarity, weight in sorted(self.evidence, key=lambda e: -abs(e[2])):
            mark = "+" if weight >= 0 else "-"
            lines.append(f"    {mark} {prop:<14} sim={similarity:.3f}  weight={weight:+.2f}")
        return "\n".join(lines)


def merge_props(entity_type, records: list[Record]) -> dict:
    """Merge clustered records into one property bag.

    Single-valued properties take the most frequently asserted value, ties
    broken by length, since a fuller value usually carries more information
    than a truncated one. Multi-valued properties take the union: a person
    really can have three email addresses, and dropping two loses linkage.
    """
    merged: dict = {}
    for prop_name, prop in entity_type.properties.items():
        values = []
        for record in records:
            value = record.props.get(prop_name)
            if value in (None, "", []):
                continue
            values.extend(value if isinstance(value, list) else [value])
        if not values:
            continue
        if prop.multi:
            seen, unique = set(), []
            for value in values:
                if value not in seen:
                    seen.add(value)
                    unique.append(value)
            merged[prop_name] = unique
        else:
            counts = Counter(values)
            best = max(counts.items(), key=lambda kv: (kv[1], len(str(kv[0]))))
            merged[prop_name] = best[0]
    return merged
