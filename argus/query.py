"""Structured query language over the graph.

A question has to end up as something exact and re-runnable. The
natural-language layer compiles to this rather than querying directly, and the
compiled query is shown to the analyst, so what runs is always something they
can read and correct.

    find person where risk > 0.8 and nationality = "cy" limit 10
    find organization where sector ~ "commodities" linked to PER-000023 within 2
    path PER-000023 to ORG-000052 within 4
    show PER-000023
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .graph import KnowledgeGraph

TOKEN_RE = re.compile(r"""
    \s*(?:
        (?P<string>"[^"]*"|'[^']*')
      | (?P<number>-?\d+(?:\.\d+)?)
      | (?P<op><=|>=|!=|=|<|>|~)
      | (?P<word>[A-Za-z_][A-Za-z0-9_.\-]*)
      | (?P<punct>[(),])
    )
""", re.VERBOSE)

KEYWORDS = {
    "find", "where", "and", "or", "not", "linked", "to", "via", "within",
    "order", "by", "limit", "path", "show", "asc", "desc", "count", "in",
}


class QueryError(ValueError):
    pass


@dataclass
class Token:
    kind: str
    value: str

    def __repr__(self):
        return f"{self.kind}:{self.value}"


def tokenize(text: str) -> list[Token]:
    tokens, position = [], 0
    while position < len(text):
        match = TOKEN_RE.match(text, position)
        if not match or match.end() == match.start():
            remainder = text[position:].strip()
            if not remainder:
                break
            raise QueryError(f"cannot parse near: {remainder[:20]!r}")
        position = match.end()
        kind = match.lastgroup
        value = match.group(kind)
        if kind == "string":
            tokens.append(Token("value", value[1:-1]))
        elif kind == "number":
            tokens.append(Token("number", value))
        elif kind == "word":
            lowered = value.lower()
            tokens.append(Token("keyword" if lowered in KEYWORDS else "word",
                                lowered if lowered in KEYWORDS else value))
        else:
            tokens.append(Token(kind, value))
    return tokens


# --------------------------------------------------------------------------
# predicates
# --------------------------------------------------------------------------

@dataclass
class Comparison:
    field: str
    op: str
    value: object

    def evaluate(self, graph: KnowledgeGraph, entity, context: dict) -> bool:
        actual = resolve_field(graph, entity, self.field, context)
        return apply_op(self.op, actual, self.value)


@dataclass
class BoolOp:
    op: str  # and | or
    terms: list

    def evaluate(self, graph, entity, context) -> bool:
        if self.op == "and":
            return all(t.evaluate(graph, entity, context) for t in self.terms)
        return any(t.evaluate(graph, entity, context) for t in self.terms)


@dataclass
class NotOp:
    term: object

    def evaluate(self, graph, entity, context) -> bool:
        return not self.term.evaluate(graph, entity, context)


def resolve_field(graph: KnowledgeGraph, entity, name: str, context: dict):
    """Look up a field, including computed ones that are not stored."""
    if name == "risk":
        return context.get("risk", {}).get(entity.id, 0.0)
    if name == "degree":
        return graph.degree(entity.id)
    if name == "type":
        return entity.entity_type
    if name == "id":
        return entity.id
    if name == "label":
        return entity.label()
    if name == "sources":
        return entity.sources
    if name == "records":
        return len(entity.members)
    if name.startswith("flags."):
        return entity.flags.get(name.split(".", 1)[1])
    if name == "community":
        return context.get("communities", {}).get(entity.id)
    return entity.props.get(name)


def apply_op(op: str, actual, expected) -> bool:
    if actual is None:
        return op == "!="
    if isinstance(actual, list):
        # A multi-valued property matches if any single value matches.
        return any(apply_op(op, item, expected) for item in actual)

    if op == "~":
        return str(expected).casefold() in str(actual).casefold()
    if op in ("=", "!="):
        equal = str(actual).casefold() == str(expected).casefold()
        return equal if op == "=" else not equal

    try:
        left, right = float(actual), float(expected)
    except (TypeError, ValueError):
        left, right = str(actual), str(expected)
    return {
        ">": left > right, "<": left < right,
        ">=": left >= right, "<=": left <= right,
    }[op]


# --------------------------------------------------------------------------
# query objects
# --------------------------------------------------------------------------

@dataclass
class FindQuery:
    entity_type: str | None = None
    predicate: object | None = None
    linked_to: str | None = None
    via: tuple[str, ...] | None = None
    within: int = 1
    order_by: str | None = None
    descending: bool = True
    limit: int = 25
    source: str = ""

    def run(self, graph: KnowledgeGraph, context: dict) -> list:
        if self.entity_type and self.entity_type not in graph.ontology.entities:
            raise QueryError(f"unknown entity type '{self.entity_type}'")
        candidates = graph.of_type(self.entity_type) if self.entity_type else list(graph.entities.values())

        if self.linked_to:
            anchor = _resolve_reference(graph, self.linked_to)
            if anchor is None:
                raise QueryError(f"no entity matches '{self.linked_to}'")
            reachable = graph.ego(anchor.id, hops=self.within, link_types=self.via)
            candidates = [e for e in candidates if e.id in reachable and e.id != anchor.id]

        if self.predicate is not None:
            candidates = [e for e in candidates if self.predicate.evaluate(graph, e, context)]

        if self.order_by:
            candidates.sort(
                key=lambda e: _sort_key(resolve_field(graph, e, self.order_by, context)),
                reverse=self.descending,
            )
        return candidates[: self.limit]


@dataclass
class PathQuery:
    left: str
    right: str
    within: int = 4
    limit: int = 5
    source: str = ""

    def run(self, graph: KnowledgeGraph, context: dict):
        source = _resolve_reference(graph, self.left)
        target = _resolve_reference(graph, self.right)
        if source is None:
            raise QueryError(f"no entity matches '{self.left}'")
        if target is None:
            raise QueryError(f"no entity matches '{self.right}'")
        paths = graph.all_paths(source.id, target.id, max_hops=self.within, limit=self.limit)
        return {"source": source, "target": target, "paths": paths}


@dataclass
class ShowQuery:
    reference: str
    source: str = ""

    def run(self, graph: KnowledgeGraph, context: dict):
        entity = _resolve_reference(graph, self.reference)
        if entity is None:
            raise QueryError(f"no entity matches '{self.reference}'")
        return entity


def _sort_key(value):
    if value is None:
        return (0, 0.0, "")
    if isinstance(value, (int, float)):
        return (1, float(value), "")
    return (1, 0.0, str(value))


def _resolve_reference(graph: KnowledgeGraph, reference: str):
    if reference in graph.entities:
        return graph.entities[reference]
    hits = graph.search(reference, limit=1)
    return hits[0] if hits else None


# --------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------

class Parser:
    def __init__(self, text: str):
        self.text = text
        self.tokens = tokenize(text)
        self.position = 0

    def peek(self) -> Token | None:
        return self.tokens[self.position] if self.position < len(self.tokens) else None

    def next(self) -> Token:
        if self.position >= len(self.tokens):
            raise QueryError("unexpected end of query")
        token = self.tokens[self.position]
        self.position += 1
        return token

    def accept(self, kind: str, value: str | None = None) -> Token | None:
        token = self.peek()
        if token and token.kind == kind and (value is None or token.value == value):
            return self.next()
        return None

    def expect(self, kind: str, value: str | None = None) -> Token:
        token = self.accept(kind, value)
        if token is None:
            found = self.peek()
            raise QueryError(f"expected {value or kind}, found {found.value if found else 'end of query'}")
        return token

    def parse(self):
        token = self.peek()
        if token is None:
            raise QueryError("empty query")
        if token.kind == "keyword" and token.value == "find":
            return self._parse_find()
        if token.kind == "keyword" and token.value == "path":
            return self._parse_path()
        if token.kind == "keyword" and token.value == "show":
            self.next()
            return ShowQuery(self._parse_reference(), source=self.text)
        raise QueryError(f"query must start with find, path, or show (found '{token.value}')")

    def _parse_find(self) -> FindQuery:
        self.expect("keyword", "find")
        query = FindQuery(source=self.text)
        token = self.peek()
        if token and token.kind == "word":
            query.entity_type = self.next().value.lower()

        while True:
            if self.accept("keyword", "where"):
                query.predicate = self._parse_expression()
            elif self.accept("keyword", "linked"):
                self.accept("keyword", "to")
                query.linked_to = self._parse_reference()
                if self.accept("keyword", "via"):
                    query.via = self._parse_link_types()
                if self.accept("keyword", "within"):
                    query.within = int(self.expect("number").value)
            elif self.accept("keyword", "order"):
                self.expect("keyword", "by")
                query.order_by = self._parse_field()
                if self.accept("keyword", "asc"):
                    query.descending = False
                else:
                    self.accept("keyword", "desc")
            elif self.accept("keyword", "limit"):
                query.limit = int(self.expect("number").value)
            else:
                break
        if self.peek() is not None:
            raise QueryError(f"unexpected trailing input: '{self.peek().value}'")
        return query

    def _parse_path(self) -> PathQuery:
        self.expect("keyword", "path")
        left = self._parse_reference()
        self.expect("keyword", "to")
        right = self._parse_reference()
        query = PathQuery(left=left, right=right, source=self.text)
        if self.accept("keyword", "within"):
            query.within = int(self.expect("number").value)
        if self.accept("keyword", "limit"):
            query.limit = int(self.expect("number").value)
        return query

    def _parse_reference(self) -> str:
        token = self.next()
        if token.kind in ("word", "value"):
            return token.value
        raise QueryError(f"expected an entity id or name, found '{token.value}'")

    def _parse_link_types(self) -> tuple[str, ...]:
        types = [self.expect("word").value]
        while self.accept("punct", ","):
            types.append(self.expect("word").value)
        return tuple(types)

    def _parse_field(self) -> str:
        token = self.next()
        if token.kind not in ("word", "keyword"):
            raise QueryError(f"expected a field name, found '{token.value}'")
        return token.value

    # -- expressions ---------------------------------------------------

    def _parse_expression(self):
        terms = [self._parse_and()]
        while self.accept("keyword", "or"):
            terms.append(self._parse_and())
        return terms[0] if len(terms) == 1 else BoolOp("or", terms)

    def _parse_and(self):
        terms = [self._parse_term()]
        while self.accept("keyword", "and"):
            terms.append(self._parse_term())
        return terms[0] if len(terms) == 1 else BoolOp("and", terms)

    def _parse_term(self):
        if self.accept("keyword", "not"):
            return NotOp(self._parse_term())
        if self.accept("punct", "("):
            inner = self._parse_expression()
            self.expect("punct", ")")
            return inner
        name = self._parse_field()
        op = self.expect("op").value
        token = self.next()
        if token.kind == "number":
            value = float(token.value)
        elif token.kind in ("value", "word", "keyword"):
            value = token.value
        else:
            raise QueryError(f"expected a value, found '{token.value}'")
        return Comparison(name, op, value)


def parse(text: str):
    return Parser(text.strip()).parse()


def execute(graph: KnowledgeGraph, text: str, context: dict | None = None):
    """Parse and run a query, returning its result."""
    return parse(text).run(graph, context or {})


GRAMMAR_HELP = """\
find <type> [where <expr>] [linked to <ref> [via <link,...>] [within <n>]]
            [order by <field> [asc|desc]] [limit <n>]
path <ref> to <ref> [within <n>] [limit <n>]
show <ref>

  <expr>   comparisons joined by and / or / not, grouped with parentheses
  <op>     =  !=  >  <  >=  <=  ~   (~ is "contains")
  <field>  any ontology property, or: risk, degree, type, id, label,
           sources, records, community, flags.<name>
  <ref>    an entity id (PER-000023) or a quoted name fragment
"""
