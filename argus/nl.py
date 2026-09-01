"""Natural-language layer.

The model never touches the graph. It has one job: turn a question into a
query, which is then printed and executed deterministically. That boundary is
the whole design. A model answering from the graph can hallucinate an entity,
an edge or a number with nothing to check it against. A model that only writes
a query can still be wrong, but wrong in one visible line you can read,
correct and re-run.

The provider is pluggable. The fallback is not a stub: it handles the common
question shapes with no network and no API key, so the system is usable before
anyone configures a model.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

from .graph import KnowledgeGraph
from .query import GRAMMAR_HELP, QueryError, parse

# Overridable so a deployment can pin whatever model it has access to.
DEFAULT_MODEL = os.environ.get("ARGUS_MODEL", "claude-opus-5")

QUERY_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "A single query in the Argus query language.",
        },
        "reasoning": {
            "type": "string",
            "description": "One sentence explaining the translation choice.",
        },
    },
    "required": ["query", "reasoning"],
    "additionalProperties": False,
}


@dataclass
class Translation:
    query: str
    reasoning: str
    provider: str

    def __str__(self) -> str:
        return self.query


class LLMProvider:
    """Interface every backend implements."""

    name = "abstract"

    def available(self) -> bool:
        raise NotImplementedError

    def translate(self, question: str, schema_prompt: str) -> Translation:
        raise NotImplementedError


class AnthropicProvider(LLMProvider):
    """Translation through the Anthropic API.

    The SDK is imported lazily on first use, so the rest of the platform keeps
    its zero-dependency guarantee. The engine runs and every test passes on a
    machine that has never installed it.
    """

    name = "anthropic"

    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = None

    def available(self) -> bool:
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return False
        return bool(self.api_key or os.environ.get("ANTHROPIC_AUTH_TOKEN"))

    def client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(api_key=self.api_key) if self.api_key \
                else anthropic.Anthropic()
        return self._client

    def translate(self, question: str, schema_prompt: str) -> Translation:
        response = self.client().messages.create(
            model=self.model,
            max_tokens=2000,
            # Short, scoped task. Low effort keeps it fast and cheap without
            # costing accuracy on a grammar this small.
            output_config={
                "effort": "low",
                "format": {"type": "json_schema", "schema": QUERY_SCHEMA},
            },
            system=schema_prompt,
            messages=[{"role": "user", "content": question}],
        )
        if response.stop_reason == "refusal":
            raise RuntimeError("the model declined to translate this question")
        text = next((b.text for b in response.content if b.type == "text"), "")
        payload = json.loads(text)
        return Translation(payload["query"], payload.get("reasoning", ""), self.name)


class RuleBasedProvider(LLMProvider):
    """Pattern-based translation. No network, no key, deterministic.

    Covers the question shapes that actually get repeated. Anything it cannot
    translate confidently raises instead of guessing; a wrong query returned
    with confidence is worse than asking for a rephrase.
    """

    name = "rule-based"

    PATTERNS = [
        # connections between two things
        (r"(?:how (?:is|are)|what connects?|connection|link|path)\b.*?\bbetween\s+(?P<a>.+?)\s+and\s+(?P<b>.+?)[\?\.]?$",
         lambda m, c: f'path "{_clean(m["a"])}" to "{_clean(m["b"])}" within 4'),
        (r"(?:how (?:is|are))\s+(?P<a>.+?)\s+(?:connected|linked|related)\s+to\s+(?P<b>.+?)[\?\.]?$",
         lambda m, c: f'path "{_clean(m["a"])}" to "{_clean(m["b"])}" within 4'),
        # ranked risk
        (r"(?:most |highest[- ])?(?:risk|risky|suspicious|dangerous)\w*\s+(?P<type>\w+?)s?\b",
         lambda m, c: f"find {_singular(m['type'], c)} order by risk desc limit 15"),
        (r"\btop\s+(?P<n>\d+)\s+(?:most\s+)?(?:risky|risk|suspicious)\s+(?P<type>\w+?)s?\b",
         lambda m, c: f"find {_singular(m['type'], c)} order by risk desc limit {m['n']}"),
        # neighbourhood
        (r"(?:who|what)(?:'s| is| are)?\s+(?:connected|linked|related)\s+to\s+(?P<a>.+?)[\?\.]?$",
         lambda m, c: f'find where linked to "{_clean(m["a"])}" within 1 order by risk desc limit 25'),
        (r"(?:who|which people)\s+.*\b(?:around|near)\s+(?P<a>.+?)[\?\.]?$",
         lambda m, c: f'find person linked to "{_clean(m["a"])}" within 2 order by risk desc limit 25'),
        # watchlist
        (r"\bwatch\s?list(?:ed)?\b",
         lambda m, c: 'find person where flags.watchlisted = "True" order by risk desc limit 25'),
        # typology mentions
        (r"\b(structuring|circular|shell|smurf\w*)\b",
         lambda m, c: 'find where flags.typologies != "" order by risk desc limit 25'),
        # lookup
        (r"^(?:who|what)\s+is\s+(?P<a>.+?)[\?\.]?$",
         lambda m, c: f'show "{_clean(m["a"])}"'),
        (r"^(?:show|tell me about|look up|profile)\s+(?:me\s+)?(?P<a>.+?)[\?\.]?$",
         lambda m, c: f'show "{_clean(m["a"])}"'),
        # counting / listing by a property
        (r"(?P<type>\w+?)s?\s+(?:in|from|registered in)\s+(?P<place>[A-Za-z ]+?)[\?\.]?$",
         lambda m, c: f'find {_singular(m["type"], c)} where country = "{_clean(m["place"])}" limit 25'),
    ]

    def available(self) -> bool:
        return True

    def translate(self, question: str, schema_prompt: str) -> Translation:
        text = question.strip().casefold()
        context = {"types": _TYPES_HINT}
        for pattern, build in self.PATTERNS:
            match = re.search(pattern, text)
            if match:
                query = build(match, context)
                try:
                    parse(query)
                except QueryError:
                    continue
                return Translation(query, f"matched pattern /{pattern[:40]}.../", self.name)
        raise QueryError(
            "Could not translate that question without a language model.\n"
            "Try a structured query instead:\n\n" + GRAMMAR_HELP
        )


_TYPES_HINT = {"person", "people", "organization", "organisation", "company",
               "account", "location", "address"}

_SINGULARS = {
    "people": "person", "persons": "person", "person": "person",
    "individual": "person", "individuals": "person", "subject": "person",
    "company": "organization", "companies": "organization", "firm": "organization",
    "org": "organization", "orgs": "organization", "organisation": "organization",
    "organization": "organization", "business": "organization",
    "account": "account", "accounts": "account",
    "address": "location", "addresses": "location", "location": "location",
}


def _singular(word: str, context: dict) -> str:
    return _SINGULARS.get(word.casefold(), word.casefold())


def _clean(text: str) -> str:
    return re.sub(r"[\"']", "", text).strip().strip("?.").strip()


# --------------------------------------------------------------------------

def schema_prompt(graph: KnowledgeGraph) -> str:
    """Describe the ontology and grammar to the model.

    Built from the live ontology rather than hardcoded, so a new entity type or
    property becomes queryable in natural language as soon as it is declared.
    No prompt edit needed.
    """
    lines = [
        "You translate an intelligence analyst's question into a single query in the",
        "Argus query language. Reply with the query only, never prose or an answer.",
        "",
        "GRAMMAR",
        GRAMMAR_HELP,
        "ENTITY TYPES AND PROPERTIES",
    ]
    for entity in graph.ontology.entities.values():
        properties = ", ".join(sorted(entity.properties))
        lines.append(f"  {entity.name}: {properties}")
    lines.append("")
    lines.append("LINK TYPES")
    for link in graph.ontology.links.values():
        arrow = "<->" if link.symmetric else "->"
        lines.append(f"  {link.name}: {link.source} {arrow} {link.target}")
    lines.append("")
    lines.extend([
        "COMPUTED FIELDS",
        "  risk (0-1), degree, type, id, label, sources, records, community,",
        "  flags.watchlisted, flags.typologies",
        "",
        "RULES",
        "  - Property values are normalized: lowercase, no punctuation, countries",
        "    are two-letter codes (gb, cy, ae).",
        "  - Prefer `~` (contains) over `=` for names and free text.",
        "  - When the question is about a named individual or company, quote the",
        "    name and let the engine resolve it.",
        "  - Never invent a property that is not listed above.",
    ])
    return "\n".join(lines)


class NaturalLanguage:
    """Translate questions to queries, preferring a model when one is configured."""

    def __init__(self, graph: KnowledgeGraph, provider: LLMProvider | None = None):
        self.graph = graph
        self.fallback = RuleBasedProvider()
        if provider is not None:
            self.provider = provider
        else:
            candidate = AnthropicProvider()
            self.provider = candidate if candidate.available() else self.fallback

    @property
    def using_model(self) -> bool:
        return self.provider is not self.fallback

    def translate(self, question: str) -> Translation:
        prompt = schema_prompt(self.graph)
        try:
            translation = self.provider.translate(question, prompt)
        except QueryError:
            raise
        except Exception as error:  # network, auth, malformed output
            if self.provider is self.fallback:
                raise
            translation = self.fallback.translate(question, prompt)
            translation.reasoning = f"model unavailable ({error}); {translation.reasoning}"
            return translation

        # Never execute model output directly. Parse it first and report a
        # failure rather than running something unchecked.
        parse(translation.query)
        return translation

    def ask(self, question: str, context: dict | None = None):
        translation = self.translate(question)
        result = parse(translation.query).run(self.graph, context or {})
        return translation, result
