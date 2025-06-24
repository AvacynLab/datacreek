from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import networkx as nx
from neo4j import Driver, GraphDatabase

from ..utils.retrieval import EmbeddingIndex


@dataclass
class KnowledgeGraph:
    """Simple wrapper storing documents and chunks with source info."""

    graph: nx.DiGraph = field(default_factory=nx.DiGraph)
    index: EmbeddingIndex = field(default_factory=EmbeddingIndex)

    def add_document(self, doc_id: str, source: str) -> None:
        if self.graph.has_node(doc_id):
            raise ValueError(f"Document already exists: {doc_id}")
        self.graph.add_node(doc_id, type="document", source=source)

    def add_chunk(
        self, doc_id: str, chunk_id: str, text: str, source: Optional[str] = None
    ) -> None:
        if source is None:
            source = self.graph.nodes[doc_id].get("source")
        if self.graph.has_node(chunk_id):
            raise ValueError(f"Chunk already exists: {chunk_id}")
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
                if (
                    query_lower in node.lower()
                    or query_lower in str(data.get("source", "")).lower()
                ):
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
            tgt
            for src, tgt, data in self.graph.edges(doc_id, data=True)
            if data.get("relation") == "has_chunk"
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the graph to a dictionary."""

        return {
            "nodes": [{"id": n, **data} for n, data in self.graph.nodes(data=True)],
            "edges": [
                {"source": u, "target": v, **data} for u, v, data in self.graph.edges(data=True)
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeGraph":
        kg = cls()
        for node in data.get("nodes", []):
            node_id = node.pop("id")
            kg.graph.add_node(node_id, **node)
            if node.get("type") == "chunk" and "text" in node:
                kg.index.add(node_id, node["text"])
        for edge in data.get("edges", []):
            src = edge.pop("source")
            tgt = edge.pop("target")
            kg.graph.add_edge(src, tgt, **edge)
        kg.index.build()
        return kg

    def to_json(self, path: str) -> str:
        """Save the graph to a JSON file."""

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @classmethod
    def from_json(cls, path: str) -> "KnowledgeGraph":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    # ------------------------------------------------------------------
    # Neo4j helpers
    # ------------------------------------------------------------------

    def to_neo4j(self, driver: Driver, *, clear: bool = True) -> None:
        """Persist the graph to a Neo4j database.

        Parameters
        ----------
        driver: Driver
            Neo4j driver instance.
        clear: bool, optional
            Whether to remove existing nodes before writing.
        """

        def _write(tx):
            if clear:
                tx.run("MATCH (n) DETACH DELETE n")
            for n, data in self.graph.nodes(data=True):
                label = data.get("type", "Node").capitalize()
                tx.run(
                    f"MERGE (m:{label} {{id:$id}}) SET m += $props",
                    id=n,
                    props={k: v for k, v in data.items() if k != "type"},
                )
            for u, v, edata in self.graph.edges(data=True):
                rel = edata.get("relation", "RELATED_TO").upper()
                tx.run(
                    f"MATCH (a {{id:$u}}), (b {{id:$v}}) MERGE (a)-[r:{rel}]->(b) SET r += $props",
                    u=u,
                    v=v,
                    props={k: v for k, v in edata.items() if k != "relation"},
                )

        with driver.session() as session:
            session.execute_write(_write)

    @classmethod
    def from_neo4j(cls, driver: Driver) -> "KnowledgeGraph":
        kg = cls()

        def _read(tx):
            nodes = tx.run("MATCH (n) RETURN n, labels(n)[0] AS label")
            for record in nodes:
                props = record["n"]
                node_id = props.pop("id")
                node_type = props.pop("type", record["label"]).lower()
                kg.graph.add_node(node_id, type=node_type, **props)
                if node_type == "chunk" and "text" in props:
                    kg.index.add(node_id, props["text"])
            edges = tx.run("MATCH (a)-[r]->(b) RETURN a.id AS src, type(r) AS rel, b.id AS tgt")
            for record in edges:
                kg.graph.add_edge(record["src"], record["tgt"], relation=record["rel"].lower())

        with driver.session() as session:
            session.execute_read(_read)
        kg.index.build()
        return kg
