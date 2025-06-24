from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List
import networkx as nx

from ..utils.retrieval import EmbeddingIndex


@dataclass
class KnowledgeGraph:
    """Simple wrapper storing documents and chunks with source info."""

    graph: nx.DiGraph = field(default_factory=nx.DiGraph)
    index: EmbeddingIndex = field(default_factory=EmbeddingIndex)

    def add_document(self, doc_id: str, source: str) -> None:
        self.graph.add_node(doc_id, type="document", source=source)

    def add_chunk(
        self, doc_id: str, chunk_id: str, text: str, source: Optional[str] = None
    ) -> None:
        if source is None:
            source = self.graph.nodes[doc_id].get("source")
        self.graph.add_node(chunk_id, type="chunk", text=text, source=source)
        self.graph.add_edge(doc_id, chunk_id, relation="has_chunk")
        self.index.add(chunk_id, text)

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

    def search_embeddings(self, query: str, k: int = 3, fetch_neighbors: bool = True) -> list[str]:
        """Return chunk IDs most relevant to the query using embeddings."""
        indices = self.index.search(query, k)
        chunk_ids: List[str] = []
        for idx in indices:
            cid = self.index.get_id(idx)
            chunk_ids.append(cid)
            if fetch_neighbors:
                # add previous and next chunks from same document if available
                preds = list(self.graph.predecessors(cid))
                if preds:
                    doc = preds[0]
                    doc_chunks = self.get_chunks_for_document(doc)
                    pos = doc_chunks.index(cid)
                    if pos > 0:
                        chunk_ids.append(doc_chunks[pos - 1])
                    if pos < len(doc_chunks) - 1:
                        chunk_ids.append(doc_chunks[pos + 1])
        # remove duplicates while preserving order
        seen = set()
        result = []
        for c in chunk_ids:
            if c not in seen:
                seen.add(c)
                result.append(c)
        return result

    def get_chunks_for_document(self, doc_id: str) -> list[str]:
        """Return all chunk IDs that belong to the given document."""

        return [
            tgt for src, tgt, data in self.graph.edges(doc_id, data=True) if data.get("relation") == "has_chunk"
        ]
