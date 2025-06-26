from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import networkx as nx
import numpy as np
from neo4j import Driver, GraphDatabase
from sklearn.cluster import KMeans

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

    def add_entity(self, entity_id: str, text: str, source: str | None = None) -> None:
        """Insert an entity node."""

        if self.graph.has_node(entity_id):
            raise ValueError(f"Entity already exists: {entity_id}")
        self.graph.add_node(entity_id, type="entity", text=text, source=source)
        if text:
            self.index.add(entity_id, text)

    def link_entity(
        self,
        node_id: str,
        entity_id: str,
        relation: str = "mentions",
        *,
        provenance: str | None = None,
    ) -> None:
        """Create a relation between ``node_id`` and ``entity_id``.

        Parameters
        ----------
        node_id: str
            ID of the source node (chunk or document).
        entity_id: str
            ID of the entity node.
        relation: str, optional
            Label for the relation. Defaults to ``"mentions"``.
        provenance: str, optional
            Provenance for the relation. If omitted, ``node_id``'s ``source`` is
            used when available.
        """

        if not self.graph.has_node(node_id) or not self.graph.has_node(entity_id):
            raise ValueError("Unknown node")
        if provenance is None:
            provenance = self.graph.nodes[node_id].get("source")
        self.graph.add_edge(node_id, entity_id, relation=relation, provenance=provenance)

    def add_chunk(
        self, doc_id: str, chunk_id: str, text: str, source: Optional[str] = None
    ) -> None:
        if source is None:
            source = self.graph.nodes[doc_id].get("source")
        if self.graph.has_node(chunk_id):
            raise ValueError(f"Chunk already exists: {chunk_id}")

        # Determine the sequence index based on existing chunks
        existing_chunks = self.get_chunks_for_document(doc_id)
        sequence = len(existing_chunks)

        # Add the chunk node and relation to the document
        self.graph.add_node(chunk_id, type="chunk", text=text, source=source)
        self.graph.add_edge(
            doc_id,
            chunk_id,
            relation="has_chunk",
            sequence=sequence,
            provenance=source,
        )

        # Connect to the previous chunk to keep the original order
        if existing_chunks:
            prev_chunk = existing_chunks[-1]
            self.graph.add_edge(prev_chunk, chunk_id, relation="next_chunk")

        self.index.add(chunk_id, text)

    def _renumber_chunks(self, doc_id: str) -> None:
        """Update sequence numbers and next_chunk links for ``doc_id``."""
        chunks = self.get_chunks_for_document(doc_id)
        # Update sequence numbers
        for i, cid in enumerate(chunks):
            if (doc_id, cid) in self.graph.edges:
                self.graph.edges[doc_id, cid]["sequence"] = i

        # Remove existing next_chunk edges for this document
        for cid in chunks:
            for succ in list(self.graph.successors(cid)):
                if self.graph.edges[cid, succ].get("relation") == "next_chunk":
                    self.graph.remove_edge(cid, succ)

        # Recreate next_chunk edges
        for a, b in zip(chunks, chunks[1:]):
            self.graph.add_edge(a, b, relation="next_chunk")

    def remove_chunk(self, chunk_id: str) -> None:
        """Delete ``chunk_id`` from the graph and index."""
        if not self.graph.has_node(chunk_id):
            return
        preds = [
            p
            for p in self.graph.predecessors(chunk_id)
            if self.graph.edges[p, chunk_id].get("relation") == "has_chunk"
        ]
        doc_id = preds[0] if preds else None
        self.graph.remove_node(chunk_id)
        self.index.remove(chunk_id)
        if doc_id:
            self._renumber_chunks(doc_id)

    def remove_document(self, doc_id: str) -> None:
        """Remove a document and all its chunks."""
        if not self.graph.has_node(doc_id):
            return
        chunks = self.get_chunks_for_document(doc_id)
        for cid in chunks:
            self.index.remove(cid)
        self.graph.remove_nodes_from(chunks)
        if self.graph.has_node(doc_id):
            self.graph.remove_node(doc_id)

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

    def search_hybrid(self, query: str, k: int = 5) -> list[str]:
        """Return chunk IDs by combining lexical and embedding search.

        Results from plain text search are returned first followed by
        semantic matches from the embedding index. Duplicate IDs are
        removed while preserving order. Only the top ``k`` items are
        returned.
        """

        lexical_matches = self.search_chunks(query)
        embedding_ids = [self.index.get_id(i) for i in self.index.search(query, k)]

        seen = set()
        results: List[str] = []
        for cid in lexical_matches + embedding_ids:
            if cid not in seen:
                seen.add(cid)
                results.append(cid)
            if len(results) >= k:
                break
        return results

    def search_with_links(self, query: str, k: int = 5, hops: int = 1) -> list[str]:
        """Return chunk IDs related to a query and expand via graph links.

        Parameters
        ----------
        query:
            Text to search for.
        k:
            Number of initial matches to retrieve via :meth:`search_hybrid`.
        hops:
            How many hops to traverse through ``next_chunk`` or ``similar_to``
            relations starting from the initial results.
        """

        seeds = self.search_hybrid(query, k)
        seen = set(seeds)
        results = list(seeds)
        queue = [(cid, 0) for cid in seeds]

        while queue:
            node, depth = queue.pop(0)
            if depth >= hops:
                continue
            # traverse both successors and predecessors so that similar chunks
            # are discovered regardless of edge direction
            for neighbor in list(self.graph.successors(node)) + list(self.graph.predecessors(node)):
                rel = self.graph.edges.get((node, neighbor)) or self.graph.edges.get(
                    (neighbor, node)
                )
                if not rel:
                    continue
                if rel.get("relation") not in {"next_chunk", "similar_to"}:
                    continue
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                results.append(neighbor)
                queue.append((neighbor, depth + 1))

        return results

    def search_with_links_data(self, query: str, k: int = 5, hops: int = 1) -> List[Dict[str, Any]]:
        """Return enriched search results expanding through graph links.

        Each item contains the chunk ``id``, its ``text``, owning ``document``
        and ``source`` plus the hop ``depth`` and the ``path`` from the initial
        result used to reach it.
        """

        seeds = self.search_hybrid(query, k)
        seen = set(seeds)
        queue: List[tuple[str, int, List[str]]] = [(cid, 0, [cid]) for cid in seeds]
        results: List[tuple[str, int, List[str]]] = queue.copy()

        while queue:
            node, depth, path = queue.pop(0)
            if depth >= hops:
                continue
            for nb in list(self.graph.successors(node)) + list(self.graph.predecessors(node)):
                rel = self.graph.edges.get((node, nb)) or self.graph.edges.get((nb, node))
                if not rel or rel.get("relation") not in {"next_chunk", "similar_to"}:
                    continue
                if nb in seen:
                    continue
                seen.add(nb)
                new_path = path + [nb]
                results.append((nb, depth + 1, new_path))
                queue.append((nb, depth + 1, new_path))

        out: List[Dict[str, Any]] = []
        for cid, depth, path in results:
            node = self.graph.nodes[cid]
            doc_id = None
            for pred in self.graph.predecessors(cid):
                if self.graph.edges[pred, cid].get("relation") == "has_chunk":
                    doc_id = pred
                    break
            out.append(
                {
                    "id": cid,
                    "text": node.get("text"),
                    "document": doc_id,
                    "source": node.get("source"),
                    "depth": depth,
                    "path": path,
                }
            )
        return out

    def link_similar_chunks(self, k: int = 3) -> None:
        """Add ``similar_to`` edges between semantically close chunks."""

        neighbors = self.index.nearest_neighbors(k, return_distances=True)
        for src, nb_list in neighbors.items():
            for tgt, score in nb_list:
                if src == tgt:
                    continue
                if (
                    self.graph.has_edge(src, tgt)
                    and self.graph.edges[src, tgt].get("relation") == "similar_to"
                ):
                    continue
                self.graph.add_edge(src, tgt, relation="similar_to", similarity=score)

    # ------------------------------------------------------------------
    # Advanced operations
    # ------------------------------------------------------------------

    def consolidate_schema(self) -> None:
        """Normalize node types and relation labels to lowercase."""

        for n, data in list(self.graph.nodes(data=True)):
            if "type" in data:
                data["type"] = str(data["type"]).lower()
        for u, v, data in list(self.graph.edges(data=True)):
            if "relation" in data:
                data["relation"] = str(data["relation"]).lower()

    def _node_embedding(self, node: str) -> Optional[np.ndarray]:
        data = self.graph.nodes[node]
        text = data.get("text")
        if not text:
            return None
        vec = self.index.embed(text)
        if vec.size == 0:
            return None
        return vec

    def cluster_chunks(self, n_clusters: int = 3) -> None:
        """Cluster chunk nodes and attach ``community`` nodes."""

        chunks = [n for n, d in self.graph.nodes(data=True) if d.get("type") == "chunk"]
        embeddings = [self._node_embedding(n) for n in chunks]
        embeddings = [e for e in embeddings if e is not None]
        if not embeddings:
            return
        X = np.vstack(embeddings)
        n_clusters = min(n_clusters, len(X))
        km = KMeans(n_clusters=n_clusters, n_init=10)
        labels = km.fit_predict(X)
        for cid in {f"community_{i}" for i in labels}:
            if not self.graph.has_node(cid):
                self.graph.add_node(cid, type="community")
        for node, label in zip(chunks, labels):
            cid = f"community_{label}"
            self.graph.add_edge(node, cid, relation="in_community")

    def cluster_entities(self, n_clusters: int = 3) -> None:
        """Cluster entity nodes into groups using embeddings."""

        entities = [n for n, d in self.graph.nodes(data=True) if d.get("type") == "entity"]
        embeddings = [self._node_embedding(n) for n in entities]
        embeddings = [e for e in embeddings if e is not None]
        if not embeddings:
            return
        X = np.vstack(embeddings)
        n_clusters = min(n_clusters, len(X))
        km = KMeans(n_clusters=n_clusters, n_init=10)
        labels = km.fit_predict(X)
        for gid in {f"entity_group_{i}" for i in labels}:
            if not self.graph.has_node(gid):
                self.graph.add_node(gid, type="entity_group")
        for node, label in zip(entities, labels):
            gid = f"entity_group_{label}"
            self.graph.add_edge(node, gid, relation="in_group")

    def summarize_communities(self) -> None:
        """Create a simple summary text for each community node."""

        for c in [n for n, d in self.graph.nodes(data=True) if d.get("type") == "community"]:
            members = [
                u
                for u, v in self.graph.in_edges(c)
                if self.graph.edges[u, c].get("relation") == "in_community"
            ]
            texts = [self.graph.nodes[m].get("text", "") for m in members]
            joined = " ".join(texts)
            words = joined.split()
            summary = " ".join(words[:20])
            self.graph.nodes[c]["summary"] = summary

    def summarize_entity_groups(self) -> None:
        """Assign a naive summary to each entity group."""

        for g in [n for n, d in self.graph.nodes(data=True) if d.get("type") == "entity_group"]:
            members = [
                u
                for u, v in self.graph.in_edges(g)
                if self.graph.edges[u, g].get("relation") == "in_group"
            ]
            texts = [self.graph.nodes[m].get("text", "") for m in members]
            joined = " ".join(texts)
            words = joined.split()
            self.graph.nodes[g]["summary"] = " ".join(words[:20])

    def score_trust(self) -> None:
        """Assign a naive trust score based on source frequency."""

        src_counts: Dict[str, int] = {}
        for n, d in self.graph.nodes(data=True):
            src = d.get("source")
            if src:
                src_counts[src] = src_counts.get(src, 0) + 1
        for u, v, d in self.graph.edges(data=True):
            src = d.get("provenance")
            if src:
                src_counts[src] = src_counts.get(src, 0) + 1
        for n, d in self.graph.nodes(data=True):
            src = d.get("source")
            if not src:
                continue
            count = src_counts.get(src, 1)
            d["trust"] = min(1.0, count / 3)
        for u, v, d in self.graph.edges(data=True):
            src = d.get("provenance")
            if not src:
                continue
            count = src_counts.get(src, 1)
            d["trust"] = min(1.0, count / 3)

    def get_chunks_for_document(self, doc_id: str) -> list[str]:
        """Return all chunk IDs that belong to the given document."""

        edges = [
            (data.get("sequence", i), tgt)
            for i, (src, tgt, data) in enumerate(self.graph.edges(doc_id, data=True))
            if data.get("relation") == "has_chunk"
        ]
        edges.sort(key=lambda x: x[0])
        return [t for _, t in edges]

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
            edges = tx.run(
                "MATCH (a)-[r]->(b) RETURN a.id AS src, type(r) AS rel, b.id AS tgt, r as rel_props"
            )
            for record in edges:
                props = dict(record["rel_props"])
                kg.graph.add_edge(
                    record["src"],
                    record["tgt"],
                    relation=record["rel"].lower(),
                    **props,
                )

        with driver.session() as session:
            session.execute_read(_read)
        kg.index.build()
        return kg
