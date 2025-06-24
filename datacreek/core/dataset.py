from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from copy import deepcopy

from .knowledge_graph import KnowledgeGraph
from ..pipelines import DatasetType


@dataclass
class DatasetBuilder:
    """Manage a dataset under construction with its own knowledge graph."""

    dataset_type: DatasetType
    name: Optional[str] = None
    graph: KnowledgeGraph = field(default_factory=KnowledgeGraph)

    def add_document(self, doc_id: str, source: str) -> None:
        """Insert a document node in the dataset graph."""
        self.graph.add_document(doc_id, source)

    def add_chunk(
        self, doc_id: str, chunk_id: str, text: str, source: Optional[str] = None
    ) -> None:
        """Insert a chunk node in the dataset graph."""
        self.graph.add_chunk(doc_id, chunk_id, text, source)

    def search_chunks(self, query: str) -> list[str]:
        return self.graph.search_chunks(query)

    def search(self, query: str, node_type: str = "chunk") -> list[str]:
        return self.graph.search(query, node_type=node_type)

    def search_documents(self, query: str) -> list[str]:
        return self.graph.search_documents(query)

    def get_chunks_for_document(self, doc_id: str) -> list[str]:
        return self.graph.get_chunks_for_document(doc_id)

    def clone(self, name: Optional[str] = None) -> "DatasetBuilder":
        """Return a deep copy of this dataset with a new optional name."""
        return DatasetBuilder(self.dataset_type, name, deepcopy(self.graph))
