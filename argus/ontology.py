"""The ontology: typed entities, links, and the resolution policy per type.

Nothing in the platform hardcodes "person" or "transaction". Object types,
their properties, how each property is normalized and compared, and how much
evidence a property carries toward an identity decision are all declared in an
ontology document and loaded here. Swapping the domain means swapping the JSON,
not editing the engine.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .normalize import normalize
from .similarity import compare


@dataclass(frozen=True)
class PropertyDef:
    """One property on an entity type.

    ``m_prob`` / ``u_prob`` are the Fellegi-Sunter parameters:

    - ``m_prob``: P(this property agrees | the two records are the same entity).
      Below 1.0 because real source systems disagree even about matches --
      typos, stale values, nicknames.
    - ``u_prob``: P(this property agrees | the records are different entities).
      Essentially the collision rate of the value space. A national ID has a
      tiny u_prob and so carries enormous weight; a first name has a large one.

    The ratio between them, not either alone, sets how much a property counts.
    """

    name: str
    kind: str = "text"
    comparator: str | None = None
    m_prob: float = 0.9
    u_prob: float = 0.1
    threshold: float = 0.85
    disagree_below: float | None = None
    multi: bool = False
    display: bool = False

    @property
    def partial_floor(self) -> float:
        """Similarity at or below which a property counts as full disagreement.

        Between this and ``threshold`` the evidence is interpolated. The band is
        narrow by default: partial credit exists for near-misses like a
        transposed digit, not for values that are merely somewhat alike.
        """
        if self.disagree_below is not None:
            return self.disagree_below
        return self.threshold * 0.9

    @property
    def compare_kind(self) -> str:
        return self.comparator or self.kind

    def normalize(self, value):
        if self.multi and isinstance(value, (list, tuple, set)):
            return [normalize(self.kind, v) for v in value if v not in (None, "")]
        return normalize(self.kind, value)

    def similarity(self, left, right) -> float:
        """Best-matching similarity, handling multi-valued properties."""
        if self.multi:
            left_values = left if isinstance(left, list) else [left]
            right_values = right if isinstance(right, list) else [right]
            best = 0.0
            for a in left_values:
                for b in right_values:
                    if a in (None, "") or b in (None, ""):
                        continue
                    best = max(best, compare(self.compare_kind, a, b))
            return best
        if left in (None, "") or right in (None, ""):
            return 0.0
        return compare(self.compare_kind, left, right)


@dataclass(frozen=True)
class BlockingRule:
    """A cheap key that groups records worth comparing.

    Comparing every record against every other is quadratic and unaffordable.
    Blocking trades a little recall for a large constant factor: only records
    sharing at least one blocking key are ever scored. Multiple independent
    rules cover for each other -- a record with a mangled name still blocks on
    its phone number.
    """

    name: str
    properties: tuple[str, ...]
    strategy: str = "exact"  # exact | prefix | soundex | sorted_tokens
    length: int = 4

    def keys(self, props: dict) -> list[str]:
        """Produce the blocking keys a record falls into (possibly none)."""
        values = []
        for prop_name in self.properties:
            value = props.get(prop_name)
            if isinstance(value, list):
                value = value[0] if value else None
            if value in (None, ""):
                return []
            values.append(value)

        from .normalize import soundex

        parts = []
        for value in values:
            text = str(value)
            if self.strategy == "prefix":
                parts.append(text[: self.length])
            elif self.strategy == "soundex":
                # First and last significant token, skipping initials. Keying
                # on the first two tokens instead means a middle initial shifts
                # the whole key, so "Robert J Whitlock" and "Robert Whitlock"
                # land in different blocks and are never even compared.
                tokens = [t for t in text.split() if len(t) > 1] or text.split()
                if not tokens:
                    return []
                parts.append(soundex(tokens[0]) + soundex(tokens[-1]))
            elif self.strategy == "sorted_tokens":
                parts.append("".join(sorted(text.replace(" ", ""))[: self.length]))
            else:
                parts.append(text)
        return [f"{self.name}:" + "|".join(parts)]


@dataclass
class EntityType:
    """An object type in the ontology."""

    name: str
    label: str = ""
    properties: dict[str, PropertyDef] = field(default_factory=dict)
    blocking: tuple[BlockingRule, ...] = ()
    match_threshold: float = 8.0
    review_threshold: float = 4.0
    resolvable: bool = True
    icon: str = "*"

    def normalize_props(self, raw: dict) -> dict:
        """Normalize a raw property bag, dropping unknown and empty fields."""
        out = {}
        for key, value in raw.items():
            prop = self.properties.get(key)
            if prop is None or value in (None, ""):
                continue
            normalized = prop.normalize(value)
            if normalized in (None, "", []):
                continue
            out[key] = normalized
        return out

    def blocking_keys(self, props: dict) -> list[str]:
        keys = []
        for rule in self.blocking:
            keys.extend(rule.keys(props))
        return keys

    def display_props(self) -> list[str]:
        return [name for name, prop in self.properties.items() if prop.display]


@dataclass
class LinkType:
    """A typed, directed relationship between two entity types."""

    name: str
    source: str
    target: str
    label: str = ""
    symmetric: bool = False
    weight: float = 1.0
    properties: dict[str, PropertyDef] = field(default_factory=dict)


class Ontology:
    """Registry of entity and link types, loaded from a JSON document."""

    def __init__(self, entities: dict[str, EntityType], links: dict[str, LinkType], name: str = "ontology"):
        self.name = name
        self.entities = entities
        self.links = links

    @classmethod
    def load(cls, path: str | Path) -> "Ontology":
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(document)

    @classmethod
    def from_dict(cls, document: dict) -> "Ontology":
        entities: dict[str, EntityType] = {}
        for name, spec in document.get("entities", {}).items():
            properties = {
                prop_name: PropertyDef(name=prop_name, **prop_spec)
                for prop_name, prop_spec in spec.get("properties", {}).items()
            }
            blocking = tuple(
                BlockingRule(
                    name=rule["name"],
                    properties=tuple(rule["properties"]),
                    strategy=rule.get("strategy", "exact"),
                    length=rule.get("length", 4),
                )
                for rule in spec.get("blocking", [])
            )
            entities[name] = EntityType(
                name=name,
                label=spec.get("label", name.title()),
                properties=properties,
                blocking=blocking,
                match_threshold=spec.get("match_threshold", 8.0),
                review_threshold=spec.get("review_threshold", 4.0),
                resolvable=spec.get("resolvable", True),
                icon=spec.get("icon", "*"),
            )

        links: dict[str, LinkType] = {}
        for name, spec in document.get("links", {}).items():
            links[name] = LinkType(
                name=name,
                source=spec["source"],
                target=spec["target"],
                label=spec.get("label", name.replace("_", " ")),
                symmetric=spec.get("symmetric", False),
                weight=spec.get("weight", 1.0),
                properties={
                    prop_name: PropertyDef(name=prop_name, **prop_spec)
                    for prop_name, prop_spec in spec.get("properties", {}).items()
                },
            )

        ontology = cls(entities, links, name=document.get("name", "ontology"))
        ontology.validate()
        return ontology

    def validate(self) -> None:
        """Fail loudly on a malformed ontology rather than at query time."""
        for link in self.links.values():
            for endpoint in (link.source, link.target):
                if endpoint not in self.entities:
                    raise ValueError(
                        f"link '{link.name}' references unknown entity type '{endpoint}'"
                    )
        for entity in self.entities.values():
            for rule in entity.blocking:
                for prop_name in rule.properties:
                    if prop_name not in entity.properties:
                        raise ValueError(
                            f"blocking rule '{rule.name}' on '{entity.name}' "
                            f"references unknown property '{prop_name}'"
                        )
            if entity.review_threshold > entity.match_threshold:
                raise ValueError(
                    f"entity '{entity.name}' has review_threshold above match_threshold"
                )

    def entity(self, name: str) -> EntityType:
        if name not in self.entities:
            raise KeyError(f"unknown entity type '{name}'")
        return self.entities[name]

    def link(self, name: str) -> LinkType:
        if name not in self.links:
            raise KeyError(f"unknown link type '{name}'")
        return self.links[name]

    def links_between(self, source: str, target: str) -> list[LinkType]:
        return [
            link for link in self.links.values()
            if (link.source == source and link.target == target)
            or (link.symmetric and link.source == target and link.target == source)
        ]

    def summary(self) -> str:
        lines = [f"ontology '{self.name}': {len(self.entities)} entity types, {len(self.links)} link types"]
        for entity in self.entities.values():
            lines.append(
                f"  {entity.icon} {entity.name:<14} {len(entity.properties):>2} props, "
                f"{len(entity.blocking)} blocking rules"
                + ("" if entity.resolvable else "  [not resolved]")
            )
        for link in self.links.values():
            arrow = "<->" if link.symmetric else "->"
            lines.append(f"  - {link.name:<20} {link.source} {arrow} {link.target}")
        return "\n".join(lines)


DEFAULT_ONTOLOGY_PATH = Path(__file__).resolve().parent.parent / "ontology" / "intel.json"


def load_default() -> Ontology:
    return Ontology.load(DEFAULT_ONTOLOGY_PATH)
