"""The knowledge graph.

Built by rewiring ingest-time links -- which point at source records -- onto
resolved entities. This step is where the value of resolution actually lands:
a transaction observed between two bare account numbers becomes an edge
between two accounts whose owners are known, and a chain of relationships
appears that no single source contained.

Parallel observations collapse into one edge that remembers all of them, so
"these two accounts transacted" and "here are the 14 transfers that prove it"
are the same object.
"""

from __future__ import annotations

import json
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

from .model import Edge, Entity, LinkRecord, _jsonable
from .ontology import Ontology
from .resolve import ResolutionResult


@dataclass
class Neighbour:
    edge: Edge
    other: str
    outgoing: bool


class KnowledgeGraph:
    def __init__(self, ontology: Ontology):
        self.ontology = ontology
        self.entities: dict[str, Entity] = {}
        self.edges: dict[str, Edge] = {}
        self._adjacency: dict[str, list[Neighbour]] = defaultdict(list)
        self._by_type: dict[str, list[str]] = defaultdict(list)
        self.dangling_links = 0

    # -- construction ------------------------------------------------------

    @classmethod
    def build(cls, ontology: Ontology, resolution: ResolutionResult,
              links: list[LinkRecord]) -> "KnowledgeGraph":
        graph = cls(ontology)
        for entity in resolution.entities.values():
            graph.add_entity(entity)

        aggregated: dict[tuple, Edge] = {}
        for link in links:
            src = resolution.assignment.get(link.source_key)
            dst = resolution.assignment.get(link.target_key)
            if not src or not dst:
                graph.dangling_links += 1
                continue
            # Resolution can collapse both endpoints into one entity -- two
            # accounts of the same person, say. A self-loop is real but adds
            # nothing traversable, so it is recorded as an observation on the
            # entity rather than an edge.
            if src == dst:
                continue

            link_type = ontology.links.get(link.link_type)
            if link_type is None:
                continue
            if link_type.symmetric and src > dst:
                src, dst = dst, src

            key = (link.link_type, src, dst)
            edge = aggregated.get(key)
            if edge is None:
                edge = Edge(
                    id=f"E-{len(aggregated):07d}",
                    link_type=link.link_type,
                    src=src,
                    dst=dst,
                )
                aggregated[key] = edge
            edge.observations.append(dict(link.props))
            if len(edge.provenance) < 25:
                edge.provenance.append(link.key)

        for edge in aggregated.values():
            graph.add_edge(edge)
        return graph

    def add_entity(self, entity: Entity):
        self.entities[entity.id] = entity
        self._by_type[entity.entity_type].append(entity.id)

    def add_edge(self, edge: Edge):
        self.edges[edge.id] = edge
        self._adjacency[edge.src].append(Neighbour(edge, edge.dst, True))
        self._adjacency[edge.dst].append(Neighbour(edge, edge.src, False))

    # -- accessors ---------------------------------------------------------

    def entity(self, entity_id: str) -> Entity | None:
        return self.entities.get(entity_id)

    def of_type(self, type_name: str) -> list[Entity]:
        return [self.entities[e] for e in self._by_type.get(type_name, [])]

    def neighbours(self, entity_id: str, link_types=None,
                   direction: str = "both") -> list[Neighbour]:
        results = []
        for neighbour in self._adjacency.get(entity_id, ()):
            if link_types and neighbour.edge.link_type not in link_types:
                continue
            if direction == "out" and not neighbour.outgoing:
                continue
            if direction == "in" and neighbour.outgoing:
                continue
            results.append(neighbour)
        return results

    def degree(self, entity_id: str) -> int:
        return len(self._adjacency.get(entity_id, ()))

    def label(self, entity_id: str) -> str:
        entity = self.entities.get(entity_id)
        return entity.label() if entity else entity_id

    def describe(self, entity_id: str) -> str:
        entity = self.entities.get(entity_id)
        if not entity:
            return f"<unknown entity {entity_id}>"
        etype = self.ontology.entity(entity.entity_type)
        parts = [f"{entity.id}  [{etype.label}]  {entity.label()}"]
        for name in etype.display_props():
            value = entity.props.get(name)
            if value:
                if isinstance(value, list):
                    value = ", ".join(str(v) for v in value[:3])
                parts.append(f"    {name:<14} {value}")
        parts.append(f"    {'sources':<14} {', '.join(entity.sources)} "
                     f"({len(entity.members)} records)")
        parts.append(f"    {'degree':<14} {self.degree(entity_id)}")
        if entity.flags:
            for name, value in entity.flags.items():
                parts.append(f"    ! {name:<12} {value}")
        return "\n".join(parts)

    def find(self, type_name: str | None = None, **predicates) -> list[Entity]:
        """Simple property-equality search over entities."""
        candidates = self.of_type(type_name) if type_name else list(self.entities.values())
        results = []
        for entity in candidates:
            ok = True
            for key, wanted in predicates.items():
                value = entity.props.get(key)
                if isinstance(value, list):
                    if wanted not in value:
                        ok = False
                        break
                elif value != wanted:
                    ok = False
                    break
            if ok:
                results.append(entity)
        return results

    def search(self, text: str, limit: int = 20) -> list[Entity]:
        """Substring search across an entity's displayable properties."""
        needle = text.casefold().strip()
        hits = []
        for entity in self.entities.values():
            haystack = " ".join(
                str(v) for v in entity.props.values() if not isinstance(v, (dict,))
            ).casefold()
            if needle in haystack:
                hits.append(entity)
                if len(hits) >= limit:
                    break
        return hits

    # -- traversal ---------------------------------------------------------

    def shortest_path(self, source: str, target: str, max_hops: int = 6,
                      link_types=None) -> list[tuple[str, Edge]] | None:
        """Breadth-first shortest path, returned as (entity, edge-taken) steps."""
        if source == target or source not in self.entities or target not in self.entities:
            return None
        previous: dict[str, tuple[str, Edge]] = {}
        seen = {source}
        frontier = deque([(source, 0)])
        while frontier:
            current, depth = frontier.popleft()
            if depth >= max_hops:
                continue
            for neighbour in self.neighbours(current, link_types):
                if neighbour.other in seen:
                    continue
                seen.add(neighbour.other)
                previous[neighbour.other] = (current, neighbour.edge)
                if neighbour.other == target:
                    return self._reconstruct(previous, source, target)
                frontier.append((neighbour.other, depth + 1))
        return None

    def _reconstruct(self, previous, source, target):
        path = []
        node = target
        while node != source:
            parent, edge = previous[node]
            path.append((node, edge))
            node = parent
        path.reverse()
        return path

    def all_paths(self, source: str, target: str, max_hops: int = 4,
                  limit: int = 25, link_types=None) -> list[list[tuple[str, Edge]]]:
        """Enumerate distinct simple paths up to ``max_hops``.

        Analysts rarely want *the* shortest connection -- they want to see the
        several independent ways two entities are tied together, because one
        path is a coincidence and four are a relationship.
        """
        results: list[list[tuple[str, Edge]]] = []
        stack = [(source, [], {source})]
        while stack and len(results) < limit:
            node, path, visited = stack.pop()
            if len(path) >= max_hops:
                continue
            for neighbour in self.neighbours(node, link_types):
                other = neighbour.other
                if other in visited:
                    continue
                step = path + [(other, neighbour.edge)]
                if other == target:
                    results.append(step)
                    if len(results) >= limit:
                        break
                else:
                    stack.append((other, step, visited | {other}))
        results.sort(key=len)
        return results

    def ego(self, entity_id: str, hops: int = 1, link_types=None) -> set[str]:
        """Entity ids within ``hops`` of a starting entity, inclusive."""
        seen = {entity_id}
        frontier = [entity_id]
        for _ in range(hops):
            nxt = []
            for node in frontier:
                for neighbour in self.neighbours(node, link_types):
                    if neighbour.other not in seen:
                        seen.add(neighbour.other)
                        nxt.append(neighbour.other)
            frontier = nxt
            if not frontier:
                break
        return seen

    def subgraph_edges(self, entity_ids: set[str]) -> list[Edge]:
        return [e for e in self.edges.values() if e.src in entity_ids and e.dst in entity_ids]

    def format_path(self, path: list[tuple[str, Edge]], source: str) -> str:
        parts = [self.label(source)]
        node = source
        for target, edge in path:
            link = self.ontology.links[edge.link_type]
            arrow = "--" if link.symmetric else ("->" if edge.src == node else "<-")
            parts.append(f"  {arrow}[{link.label} x{edge.count}]{arrow}  {self.label(target)}")
            node = target
        return "\n".join(parts)

    # -- persistence -------------------------------------------------------

    def stats(self) -> dict:
        by_entity = {t: len(ids) for t, ids in sorted(self._by_type.items())}
        by_link: dict[str, int] = defaultdict(int)
        observations = 0
        for edge in self.edges.values():
            by_link[edge.link_type] += 1
            observations += edge.count
        return {
            "entities": len(self.entities),
            "edges": len(self.edges),
            "observations": observations,
            "dangling_links": self.dangling_links,
            "by_entity_type": by_entity,
            "by_link_type": dict(sorted(by_link.items())),
        }

    def save(self, path: str | Path):
        payload = {
            "ontology": self.ontology.name,
            "entities": [e.to_dict() for e in self.entities.values()],
            "edges": [e.to_dict() for e in self.edges.values()],
            "dangling_links": self.dangling_links,
        }
        Path(path).write_text(json.dumps(payload, default=str), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, ontology: Ontology) -> "KnowledgeGraph":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        graph = cls(ontology)
        for row in payload["entities"]:
            graph.add_entity(Entity(
                id=row["id"],
                entity_type=row["type"],
                props=row["props"],
                members=row["members"],
                sources=row["sources"],
                confidence=row.get("confidence", 1.0),
                flags=row.get("flags", {}),
            ))
        for row in payload["edges"]:
            graph.add_edge(Edge(
                id=row["id"],
                link_type=row["type"],
                src=row["src"],
                dst=row["dst"],
                observations=row.get("observations", []),
                provenance=row.get("provenance", []),
            ))
        graph.dangling_links = payload.get("dangling_links", 0)
        return graph
