from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import networkx as nx


@dataclass
class KnowledgeGraph:
    """Simple wrapper storing documents and chunks with source info."""

    graph: nx.DiGraph = field(default_factory=nx.DiGraph)

    def add_document(self, doc_id: str, source: str) -> None:
        self.graph.add_node(doc_id, type="document", source=source)

    def add_chunk(
        self, doc_id: str, chunk_id: str, text: str, source: Optional[str] = None
    ) -> None:
        if source is None:
            source = self.graph.nodes[doc_id].get("source")
        self.graph.add_node(chunk_id, type="chunk", text=text, source=source)
        self.graph.add_edge(doc_id, chunk_id, relation="has_chunk")

    def search(self, query: str, node_type: str = "chunk") -> list[str]:
        """Return node IDs of the given type matching the query.

        For chunks we search in the ``text`` attribute while documents are
        matched against their id or ``source``.
        """

        query_lower = query.lower()
        results: list[str] = []
        for node, data in self.graph.nodes(data=True):
            if data.get("type") != node_type:
                continue
            if node_type == "document":
                if query_lower in node.lower() or query_lower in str(data.get("source", "")).lower():
                    results.append(node)
            else:
                if query_lower in str(data.get("text", "")).lower():
                    results.append(node)
        return results

    def search_chunks(self, query: str) -> list[str]:
        """Return chunk IDs containing the query string."""

        return self.search(query, node_type="chunk")

    def search_documents(self, query: str) -> list[str]:
        """Return document IDs whose id or source matches the query."""

        return self.search(query, node_type="document")

    def get_chunks_for_document(self, doc_id: str) -> list[str]:
        """Return all chunk IDs that belong to the given document."""

        return [
            tgt for src, tgt, data in self.graph.edges(doc_id, data=True) if data.get("relation") == "has_chunk"
        ]
