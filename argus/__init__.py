"""Argus -- an ontology-driven intelligence platform.

Raw records from many source systems become resolved entities, resolved
entities become a knowledge graph, and the graph becomes findings an analyst
can act on -- with provenance preserved end to end.

Pure standard library. No dependency is required to ingest, resolve, build the
graph, run the analytics, or query it; the optional Claude integration is
imported lazily and only when configured.
"""

from .analytics import Finding
from .graph import KnowledgeGraph
from .ingest import Ingestor
from .model import Edge, Entity, MatchDecision, Record
from .ontology import Ontology, load_default
from .pipeline import Investigation, Pipeline, PipelineResult
from .resolve import Resolver, evaluate

__version__ = "0.1.0"

__all__ = [
    "Edge",
    "Entity",
    "Finding",
    "Ingestor",
    "Investigation",
    "KnowledgeGraph",
    "MatchDecision",
    "Ontology",
    "Pipeline",
    "PipelineResult",
    "Record",
    "Resolver",
    "evaluate",
    "load_default",
]
