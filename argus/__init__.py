"""Argus: entity resolution and link analysis.

Records from many source systems become resolved entities, entities become a
knowledge graph, and the graph produces findings. Provenance is preserved
throughout.

Standard library only. The optional LLM integration is imported lazily.
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
