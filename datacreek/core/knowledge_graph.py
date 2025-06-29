from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import networkx as nx
import numpy as np
import requests
from dateutil import parser
try:
    from neo4j import Driver, GraphDatabase
except Exception:  # pragma: no cover - optional dependency for tests
    Driver = object  # type: ignore
    GraphDatabase = None  # type: ignore

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:  # pragma: no cover - optional dependency for tests
    KMeans = None
    cosine_similarity = None

from ..utils.retrieval import EmbeddingIndex


@dataclass
class KnowledgeGraph:
    """Simple wrapper storing documents and chunks with source info."""

    graph: nx.DiGraph = field(default_factory=nx.DiGraph)
    index: EmbeddingIndex = field(default_factory=EmbeddingIndex)

    def add_document(
        self,
        doc_id: str,
        source: str,
        *,
        text: str | None = None,
        author: str | None = None,
        organization: str | None = None,
    ) -> None:
        if self.graph.has_node(doc_id):
            raise ValueError(f"Document already exists: {doc_id}")
        self.graph.add_node(
            doc_id,
            type="document",
            source=source,
            author=author,
            organization=organization,
        )
        if text:
            self.graph.nodes[doc_id]["text"] = text
            self.index.add(doc_id, text)

    def add_entity(self, entity_id: str, text: str, source: str | None = None) -> None:
        """Insert an entity node."""

        if self.graph.has_node(entity_id):
            raise ValueError(f"Entity already exists: {entity_id}")
        self.graph.add_node(entity_id, type="entity", text=text, source=source)
        if text:
            self.index.add(entity_id, text)

    def add_fact(
        self,
        subject: str,
        predicate: str,
        obj: str,
        fact_id: Optional[str] = None,
        *,
        source: Optional[str] = None,
    ) -> str:
        """Insert a standalone fact and link corresponding entities."""

        fact_id = fact_id or f"fact_{len(self.graph.nodes)}"
        if self.graph.has_node(fact_id):
            raise ValueError(f"Fact already exists: {fact_id}")

        # ensure entity nodes exist
        if not self.graph.has_node(subject):
            self.add_entity(subject, subject, source)
        if not self.graph.has_node(obj):
            self.add_entity(obj, obj, source)

        self.graph.add_node(
            fact_id,
            type="fact",
            subject=subject,
            predicate=predicate,
            object=obj,
            source=source,
        )
        self.graph.add_edge(fact_id, subject, relation="subject", provenance=source)
        self.graph.add_edge(fact_id, obj, relation="object", provenance=source)
        self.graph.add_edge(subject, obj, relation=predicate, provenance=source)
        self.index.add(fact_id, f"{subject} {predicate} {obj}")
        return fact_id

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

    def add_section(
        self,
        doc_id: str,
        section_id: str,
        title: str | None = None,
        source: Optional[str] = None,
        *,
        page: int | None = None,
    ) -> None:
        """Insert a section node and attach it to ``doc_id``."""

        if source is None:
            source = self.graph.nodes[doc_id].get("source")
        if self.graph.has_node(section_id):
            raise ValueError(f"Section already exists: {section_id}")

        existing = self.get_sections_for_document(doc_id)
        sequence = len(existing)

        self.graph.add_node(
            section_id,
            type="section",
            title=title,
            source=source,
            page=page,
        )
        if title:
            self.index.add(section_id, title)
        self.graph.add_edge(
            doc_id,
            section_id,
            relation="has_section",
            sequence=sequence,
            provenance=source,
        )

        if existing:
            prev = existing[-1]
            self.graph.add_edge(prev, section_id, relation="next_section")

    def add_chunk(
        self,
        doc_id: str,
        chunk_id: str,
        text: str,
        source: Optional[str] = None,
        *,
        section_id: str | None = None,
        page: int | None = None,
    ) -> None:
        if source is None:
            source = self.graph.nodes[doc_id].get("source")
        if self.graph.has_node(chunk_id):
            raise ValueError(f"Chunk already exists: {chunk_id}")

        # Determine the sequence index within the document and optional section
        doc_chunks = self.get_chunks_for_document(doc_id)
        doc_sequence = len(doc_chunks)
        if section_id and self.graph.has_node(section_id):
            existing_chunks = self.get_chunks_for_section(section_id)
            section_sequence = len(existing_chunks)
        else:
            existing_chunks = doc_chunks
            section_sequence = None

        # Add the chunk node and relation to the document
        if page is None:
            page = 1

        self.graph.add_node(chunk_id, type="chunk", text=text, source=source, page=page)
        self.graph.add_edge(
            doc_id,
            chunk_id,
            relation="has_chunk",
            sequence=doc_sequence,
            provenance=source,
        )
        if section_id and self.graph.has_node(section_id):
            self.graph.add_edge(
                section_id,
                chunk_id,
                relation="under_section",
                sequence=section_sequence,
                provenance=source,
            )
            if self.graph.nodes[section_id].get("page") is None and page is not None:
                self.graph.nodes[section_id]["page"] = page

        # Connect to the previous chunk to keep the original order
        if doc_chunks:
            prev_chunk = doc_chunks[-1]
            self.graph.add_edge(prev_chunk, chunk_id, relation="next_chunk")

        self.index.add(chunk_id, text)

    def add_image(
        self,
        doc_id: str,
        image_id: str,
        path: str,
        source: Optional[str] = None,
        *,
        page: int | None = None,
    ) -> None:
        """Insert an image node linked to ``doc_id``."""

        if source is None:
            source = self.graph.nodes[doc_id].get("source")
        if self.graph.has_node(image_id):
            raise ValueError(f"Image already exists: {image_id}")

        doc_images = self.get_images_for_document(doc_id)
        sequence = len(doc_images)
        if page is None:
            page = 1

        self.graph.add_node(image_id, type="image", path=path, source=source, page=page)
        self.graph.add_edge(
            doc_id,
            image_id,
            relation="has_image",
            sequence=sequence,
            provenance=source,
        )

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
        if chunks:
            self.index.build()

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
            elif node_type == "fact":
                if (
                    query_lower in str(data.get("subject", "")).lower()
                    or query_lower in str(data.get("predicate", "")).lower()
                    or query_lower in str(data.get("object", "")).lower()
                ):
                    results.append(node)
            elif node_type == "section":
                if query_lower in node.lower() or query_lower in str(data.get("title", "")).lower():
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

    def search_embeddings(
        self,
        query: str,
        k: int = 3,
        fetch_neighbors: bool = True,
        *,
        node_type: str = "chunk",
    ) -> list[str]:
        """Return node IDs of ``node_type`` relevant to ``query`` using embeddings."""

        # retrieve a larger candidate set and then filter by node type
        indices = self.index.search(query, max(k * 4, k))
        ids: List[str] = []
        for idx in indices:
            nid = self.index.get_id(idx)
            if self.graph.nodes[nid].get("type") != node_type:
                continue
            ids.append(nid)
            if fetch_neighbors and node_type == "chunk":
                # add previous and next chunks from same document if available
                preds = list(self.graph.predecessors(nid))
                if preds:
                    doc = preds[0]
                    doc_chunks = self.get_chunks_for_document(doc)
                    pos = doc_chunks.index(nid)
                    if pos > 0:
                        ids.append(doc_chunks[pos - 1])
                    if pos < len(doc_chunks) - 1:
                        ids.append(doc_chunks[pos + 1])
            if len(ids) >= k:
                break

        # remove duplicates while preserving order
        seen = set()
        result = []
        for n in ids:
            if n not in seen:
                seen.add(n)
                result.append(n)
        return result

    def search_hybrid(self, query: str, k: int = 5, *, node_type: str = "chunk") -> list[str]:
        """Return node IDs by combining lexical and embedding search.

        Results from plain text search are returned first followed by
        semantic matches from the embedding index. Duplicate IDs are
        removed while preserving order. Only the top ``k`` items are
        returned. The ``node_type`` parameter restricts results to nodes
        of the given type.
        """

        lexical_matches = self.search(query, node_type=node_type)
        embedding_ids = self.search_embeddings(
            query, k=k, fetch_neighbors=False, node_type=node_type
        )

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
            if self.graph.nodes[src].get("type") != "chunk":
                continue
            for tgt, score in nb_list:
                if src == tgt or self.graph.nodes[tgt].get("type") != "chunk":
                    continue
                if (
                    self.graph.has_edge(src, tgt)
                    and self.graph.edges[src, tgt].get("relation") == "similar_to"
                ):
                    continue
                self.graph.add_edge(src, tgt, relation="similar_to", similarity=score)

    def link_similar_sections(self, k: int = 3) -> None:
        """Add ``similar_to`` edges between section titles."""

        neighbors = self.index.nearest_neighbors(k, return_distances=True)
        for src, nb_list in neighbors.items():
            if self.graph.nodes[src].get("type") != "section":
                continue
            for tgt, score in nb_list:
                if src == tgt or self.graph.nodes[tgt].get("type") != "section":
                    continue
                if (
                    self.graph.has_edge(src, tgt)
                    and self.graph.edges[src, tgt].get("relation") == "similar_to"
                ):
                    continue
                self.graph.add_edge(src, tgt, relation="similar_to", similarity=score)

    def link_similar_documents(self, k: int = 3) -> None:
        """Add ``similar_to`` edges between document texts."""

        neighbors = self.index.nearest_neighbors(k, return_distances=True)
        for src, nb_list in neighbors.items():
            if self.graph.nodes[src].get("type") != "document":
                continue
            for tgt, score in nb_list:
                if src == tgt or self.graph.nodes[tgt].get("type") != "document":
                    continue
                if (
                    self.graph.has_edge(src, tgt)
                    and self.graph.edges[src, tgt].get("relation") == "similar_to"
                ):
                    continue
                self.graph.add_edge(src, tgt, relation="similar_to", similarity=score)

    def deduplicate_chunks(self) -> int:
        """Remove duplicate chunk nodes based on their text."""

        seen: Dict[str, str] = {}
        removed = 0
        for node, data in list(self.graph.nodes(data=True)):
            if data.get("type") != "chunk":
                continue
            text = data.get("text", "")
            if text in seen:
                self.remove_chunk(node)
                removed += 1
            else:
                seen[text] = node
        if removed:
            self.index.build()
        return removed

    def prune_sources(self, sources: list[str]) -> int:
        """Remove nodes and edges originating from the given sources.

        Parameters
        ----------
        sources:
            List of source identifiers to remove from the graph. Nodes with a
            ``source`` attribute matching any of these values are deleted along
            with their associated edges. Edges whose ``provenance`` attribute
            matches are removed as well.

        Returns
        -------
        int
            Number of nodes removed from the graph.
        """

        removed = 0

        for node, data in list(self.graph.nodes(data=True)):
            if data.get("source") in sources:
                self.graph.remove_node(node)
                self.index.remove(node)
                removed += 1

        for u, v, edata in list(self.graph.edges(data=True)):
            if edata.get("provenance") in sources:
                if self.graph.has_edge(u, v):
                    self.graph.remove_edge(u, v)

        if removed:
            self.index.build()

        return removed

    def mark_conflicting_facts(self) -> int:
        """Flag edges that express conflicting facts.

        A conflict occurs when the same subject and predicate are linked to
        multiple distinct objects. All edges in such groups receive a
        ``conflict`` attribute set to ``True``.

        Returns
        -------
        int
            Number of edges marked as conflicting.
        """

        groups: Dict[tuple[str, str], set[str]] = {}
        for u, v, data in self.graph.edges(data=True):
            rel = data.get("relation")
            if not rel or rel in {
                "has_chunk",
                "next_chunk",
                "subject",
                "object",
                "has_fact",
                "mentions",
            }:
                continue
            groups.setdefault((u, rel), set()).add(v)

        marked = 0
        for (subj, rel), objs in groups.items():
            if len(objs) <= 1:
                continue
            for obj in objs:
                edge = self.graph.edges[subj, obj]
                if not edge.get("conflict"):
                    edge["conflict"] = True
                    marked += 1
        return marked

    def clean_chunk_texts(self) -> int:
        """Normalize text of chunk nodes by stripping HTML and collapsing whitespace.

        Returns
        -------
        int
            Number of chunks modified.
        """

        changed = 0
        for cid, data in list(self.graph.nodes(data=True)):
            if data.get("type") != "chunk":
                continue
            text = data.get("text", "")
            cleaned = re.sub(r"<[^>]+>", " ", text)
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            if cleaned != text:
                data["text"] = cleaned
                self.index.remove(cid)
                self.index.add(cid, cleaned)
                changed += 1
        if changed:
            self.index.build()
        return changed

    def normalize_date_fields(self) -> int:
        """Normalize date-like attributes on nodes to ISO format.

        Fields with names containing ``date`` or ``time`` are parsed using
        :func:`dateutil.parser.parse` and converted to ``YYYY-MM-DD`` strings.

        Returns
        -------
        int
            Number of fields updated.
        """

        changed = 0
        for _, data in list(self.graph.nodes(data=True)):
            for key, value in list(data.items()):
                if not isinstance(value, str):
                    continue
                if "date" not in key and "time" not in key:
                    continue
                try:
                    dt = parser.parse(value)
                except Exception:  # pragma: no cover - parse failure
                    continue
                iso = dt.date().isoformat()
                if iso != value:
                    data[key] = iso
                    changed += 1
        return changed

    def validate_coherence(self) -> int:
        """Flag logically inconsistent edges such as impossible birth dates.

        The current implementation checks ``parent_of`` relations. If both the
        parent node and child node have a ``birth_date`` attribute and the
        parent's date is later than the child's, the edge receives an
        ``inconsistent`` attribute.

        Returns
        -------
        int
            Number of edges marked as inconsistent.
        """

        marked = 0
        for u, v, data in self.graph.edges(data=True):
            if data.get("relation") != "parent_of":
                continue
            p_date = self.graph.nodes[u].get("birth_date")
            c_date = self.graph.nodes[v].get("birth_date")
            if not p_date or not c_date:
                continue
            try:
                p_dt = parser.parse(p_date)
                c_dt = parser.parse(c_date)
            except Exception:  # pragma: no cover - parse failure
                continue
            if p_dt > c_dt and not data.get("inconsistent"):
                data["inconsistent"] = True
                marked += 1
        return marked

    def link_chunks_by_entity(self) -> int:
        """Connect chunks mentioning the same entity with ``co_mentions`` edges.

        Returns
        -------
        int
            Number of edges added.
        """

        ent_to_chunks: Dict[str, list[str]] = {}
        for u, v, data in self.graph.edges(data=True):
            if data.get("relation") != "mentions":
                continue
            if self.graph.nodes[u].get("type") != "chunk":
                continue
            ent_to_chunks.setdefault(v, []).append(u)

        added = 0
        for chunks in ent_to_chunks.values():
            for i, c1 in enumerate(chunks):
                for c2 in chunks[i + 1 :]:
                    if (
                        self.graph.has_edge(c1, c2)
                        and self.graph.edges[c1, c2].get("relation") == "co_mentions"
                    ):
                        continue
                    self.graph.add_edge(c1, c2, relation="co_mentions")
                    added += 1
        return added

    def link_documents_by_entity(self) -> int:
        """Connect documents that mention the same entity with ``co_mentions`` edges.

        Returns
        -------
        int
            Number of edges added.
        """

        ent_to_docs: Dict[str, list[str]] = {}
        for u, v, data in self.graph.edges(data=True):
            if data.get("relation") != "mentions":
                continue
            src_type = self.graph.nodes[u].get("type")
            doc_id: str | None = None
            if src_type == "chunk":
                doc_id = self.get_document_for_chunk(u)
            elif src_type == "document":
                doc_id = u
            if not doc_id:
                continue
            ent_to_docs.setdefault(v, []).append(doc_id)

        added = 0
        for docs in ent_to_docs.values():
            uniq_docs = list(dict.fromkeys(docs))
            for i, d1 in enumerate(uniq_docs):
                for d2 in uniq_docs[i + 1 :]:
                    if (
                        self.graph.has_edge(d1, d2)
                        and self.graph.edges[d1, d2].get("relation") == "co_mentions"
                    ):
                        continue
                    self.graph.add_edge(d1, d2, relation="co_mentions")
                    added += 1
        return added

    def link_sections_by_entity(self) -> int:
        """Connect sections that mention the same entity with ``co_mentions`` edges.

        Returns
        -------
        int
            Number of edges added.
        """

        ent_to_sections: Dict[str, list[str]] = {}
        for u, v, data in self.graph.edges(data=True):
            if data.get("relation") != "mentions":
                continue
            src_type = self.graph.nodes[u].get("type")
            sec_id: str | None = None
            if src_type == "chunk":
                sec_id = self.get_section_for_chunk(u)
            elif src_type == "section":
                sec_id = u
            if not sec_id:
                continue
            ent_to_sections.setdefault(v, []).append(sec_id)

        added = 0
        for secs in ent_to_sections.values():
            uniq_secs = list(dict.fromkeys(secs))
            for i, s1 in enumerate(uniq_secs):
                for s2 in uniq_secs[i + 1 :]:
                    if (
                        self.graph.has_edge(s1, s2)
                        and self.graph.edges[s1, s2].get("relation") == "co_mentions"
                    ):
                        continue
                    self.graph.add_edge(s1, s2, relation="co_mentions")
                    added += 1
        return added

    def link_authors_organizations(self) -> int:
        """Connect author nodes to their organizations using ``affiliated_with``.

        Documents may store ``author`` and ``organization`` attributes. This
        helper ensures corresponding entity nodes exist and links them.

        Returns
        -------
        int
            Number of edges created.
        """

        added = 0
        for doc_id, data in list(self.graph.nodes(data=True)):
            if data.get("type") != "document":
                continue
            author = data.get("author")
            org = data.get("organization")
            if not author or not org:
                continue
            if not self.graph.has_node(author):
                self.add_entity(author, author, data.get("source"))
            if not self.graph.has_node(org):
                self.add_entity(org, org, data.get("source"))
            if not self.graph.has_edge(author, org):
                self.graph.add_edge(author, org, relation="affiliated_with")
                added += 1
        return added

    def _merge_entity_nodes(self, target: str, source: str) -> None:
        """Merge ``source`` entity into ``target``."""

        for pred, _, edata in list(self.graph.in_edges(source, data=True)):
            self.graph.add_edge(pred, target, **edata)
        for _, succ, edata in list(self.graph.out_edges(source, data=True)):
            self.graph.add_edge(target, succ, **edata)
        if "text" not in self.graph.nodes[target]:
            self.graph.nodes[target]["text"] = self.graph.nodes[source].get("text")
        self.graph.remove_node(source)

    def resolve_entities(
        self,
        threshold: float = 0.8,
        aliases: dict[str, list[str]] | None = None,
    ) -> int:
        """Merge entity nodes that refer to the same concept.

        Parameters
        ----------
        threshold:
            Minimum cosine similarity for merging using text embeddings.
        aliases:
            Optional mapping of canonical labels to lists of aliases. Nodes whose
            ``text`` matches an alias will be merged into the node matching the
            canonical label before similarity-based merging occurs.
        """

        entities = [n for n, d in self.graph.nodes(data=True) if d.get("type") == "entity"]
        texts = [self.graph.nodes[e].get("text", "") for e in entities]
        if len(entities) < 2:
            return 0

        merged = 0

        used_alias_indices: set[int] = set()
        if aliases:

            def _norm(t: str) -> str:
                return re.sub(r"\W+", "", t).lower()

            label_to_id = {_norm(self.graph.nodes[e].get("text", "")): e for e in entities}
            for canon, alist in aliases.items():
                target = label_to_id.get(_norm(canon))
                if not target:
                    continue
                for alias in alist:
                    source = label_to_id.get(_norm(alias))
                    if source and source != target:
                        self._merge_entity_nodes(target, source)
                        merged += 1
                        idx = entities.index(source)
                        texts[idx] = canon
                        used_alias_indices.add(idx)
                        label_to_id[_norm(canon)] = target
                        label_to_id.pop(_norm(alias), None)

        embeddings = self.index.transform(texts)
        used: set[int] = set()
        for i, eid1 in enumerate(entities):
            if i in used or i in used_alias_indices:
                continue
            for j in range(i + 1, len(entities)):
                if j in used or j in used_alias_indices:
                    continue
                sim = float(
                    cosine_similarity(embeddings[i].reshape(1, -1), embeddings[j].reshape(1, -1))[
                        0, 0
                    ]
                )
                t1 = re.sub(r"\W+", "", texts[i]).lower()
                t2 = re.sub(r"\W+", "", texts[j]).lower()
                if sim >= threshold or t1 == t2 or t1 in t2 or t2 in t1:
                    self._merge_entity_nodes(eid1, entities[j])
                    used.add(j)
                    merged += 1
        if merged:
            self.index = EmbeddingIndex()
            for n, d in self.graph.nodes(data=True):
                if d.get("type") in {"chunk", "entity"} and "text" in d:
                    self.index.add(n, d["text"])
            self.index.build()
        return merged

    def extract_entities(self, model: str | None = "en_core_web_sm") -> None:
        """Run named entity recognition on all chunks and link entities."""

        from datacreek.utils.entity_extraction import extract_entities

        for cid, data in list(self.graph.nodes(data=True)):
            if data.get("type") != "chunk":
                continue
            ents = extract_entities(data.get("text", ""), model=model)
            for ent in ents:
                if not self.graph.has_node(ent):
                    self.add_entity(ent, ent, data.get("source"))
                if not self.graph.has_edge(cid, ent):
                    self.link_entity(cid, ent, provenance=data.get("source"))

    def enrich_entity_wikidata(self, entity_id: str) -> None:
        """Fetch description from Wikidata and store it on the entity node."""

        node = self.graph.nodes.get(entity_id)
        if not node or node.get("type") != "entity":
            raise ValueError("Unknown entity")
        label = node.get("text")
        if not label:
            return
        search = requests.get(
            "https://www.wikidata.org/w/api.php",
            params={
                "action": "wbsearchentities",
                "search": label,
                "format": "json",
                "language": "en",
                "limit": 1,
            },
            timeout=10,
        )
        data = search.json()
        if not data.get("search"):
            return
        item = data["search"][0]
        node["wikidata_id"] = item.get("id")
        if desc := item.get("description"):
            node["description"] = desc

    def enrich_entity_dbpedia(self, entity_id: str) -> None:
        """Fetch description from DBpedia and store it on the entity node."""

        node = self.graph.nodes.get(entity_id)
        if not node or node.get("type") != "entity":
            raise ValueError("Unknown entity")
        label = node.get("text")
        if not label:
            return
        search = requests.get(
            "https://lookup.dbpedia.org/api/search",
            params={"query": label, "maxResults": 1},
            headers={"Accept": "application/json"},
            timeout=10,
        )
        data = search.json()
        results = data.get("docs") or data.get("results")
        if not results:
            return
        item = results[0]
        uri = item.get("uri") or item.get("id")
        if uri:
            node["dbpedia_uri"] = uri
        desc = item.get("description") or item.get("overview")
        if desc:
            node["description_dbpedia"] = desc

    def predict_links(self, threshold: float = 0.8, *, use_graph_embeddings: bool = False) -> None:
        """Create ``related_to`` edges between similar entities.

        Parameters
        ----------
        threshold:
            Minimum cosine similarity for creating a link.
        use_graph_embeddings:
            If ``True`` use Node2Vec embeddings stored on the nodes instead of
            text embeddings for similarity computation. If embeddings are not
            present they will be generated automatically.
        """

        entities = [n for n, d in self.graph.nodes(data=True) if d.get("type") == "entity"]
        if len(entities) < 2:
            return

        if use_graph_embeddings:
            if not all("embedding" in self.graph.nodes[e] for e in entities):
                self.compute_node2vec_embeddings()
            emb_matrix = np.array([self.graph.nodes[e]["embedding"] for e in entities])
        else:
            texts = [self.graph.nodes[e].get("text", "") for e in entities]
            emb_matrix = self.index.transform(texts)

        for i, eid1 in enumerate(entities):
            for j in range(i + 1, len(entities)):
                eid2 = entities[j]
                sim = float(
                    cosine_similarity(emb_matrix[i].reshape(1, -1), emb_matrix[j].reshape(1, -1))[
                        0, 0
                    ]
                )
                if sim >= threshold and not self.graph.has_edge(eid1, eid2):
                    self.graph.add_edge(eid1, eid2, relation="related_to", similarity=sim)

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
        if "embedding" in data:
            emb = np.array(data["embedding"], dtype=float)
            return emb
        text = data.get("text")
        if not text:
            return None
        vec = self.index.embed(text)
        if vec.size == 0:
            return None
        data["embedding"] = vec.tolist()
        return vec

    def update_embeddings(self, node_type: str = "chunk") -> None:
        """Materialize embeddings for nodes of ``node_type``."""

        for n, d in self.graph.nodes(data=True):
            if d.get("type") != node_type:
                continue
            text = d.get("text")
            if not text:
                continue
            vec = self.index.embed(text)
            if vec.size:
                self.graph.nodes[n]["embedding"] = vec.tolist()

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

    def compute_centrality(self, node_type: str = "entity", metric: str = "degree") -> None:
        """Compute centrality scores for nodes of ``node_type``."""

        if metric == "degree":
            values = nx.degree_centrality(self.graph)
        elif metric == "betweenness":
            values = nx.betweenness_centrality(self.graph)
        else:
            raise ValueError("Unknown metric")

        for node, score in values.items():
            if self.graph.nodes[node].get("type") == node_type:
                self.graph.nodes[node]["centrality"] = float(score)

    def compute_node2vec_embeddings(
        self,
        dimensions: int = 64,
        walk_length: int = 10,
        num_walks: int = 50,
        workers: int = 1,
        seed: int = 0,
    ) -> None:
        """Compute Node2Vec embeddings for all nodes and store them on the nodes."""

        try:
            from node2vec import Node2Vec
        except Exception as e:  # pragma: no cover - dependency missing
            raise RuntimeError("node2vec package is required") from e

        n2v = Node2Vec(
            self.graph,
            dimensions=dimensions,
            walk_length=walk_length,
            num_walks=num_walks,
            workers=workers,
            seed=seed,
        )
        model = n2v.fit()
        for node in self.graph.nodes:
            vec = model.wv[str(node)]
            self.graph.nodes[node]["embedding"] = vec.tolist()

    # ------------------------------------------------------------------
    # Structure helpers
    # ------------------------------------------------------------------

    def get_sections_for_document(self, doc_id: str) -> list[str]:
        """Return IDs of sections belonging to ``doc_id`` ordered by sequence."""

        edges = [
            (data.get("sequence", i), tgt)
            for i, (src, tgt, data) in enumerate(self.graph.edges(doc_id, data=True))
            if data.get("relation") == "has_section"
        ]
        edges.sort(key=lambda x: x[0])
        return [t for _, t in edges]

    def get_chunks_for_section(self, section_id: str) -> list[str]:
        """Return chunk IDs under ``section_id`` ordered by sequence."""

        edges = [
            (data.get("sequence", i), tgt)
            for i, (src, tgt, data) in enumerate(self.graph.edges(section_id, data=True))
            if data.get("relation") == "under_section"
        ]
        edges.sort(key=lambda x: x[0])
        return [t for _, t in edges]

    def get_section_for_chunk(self, chunk_id: str) -> str | None:
        """Return the section containing ``chunk_id`` if any."""

        for pred in self.graph.predecessors(chunk_id):
            if self.graph.edges[pred, chunk_id].get("relation") == "under_section":
                return pred
        return None

    def get_next_section(self, section_id: str) -> str | None:
        """Return the section that follows ``section_id`` if any."""

        for _, succ, data in self.graph.out_edges(section_id, data=True):
            if data.get("relation") == "next_section":
                return succ
        # fallback using sequence metadata
        for doc_id, _, data in self.graph.in_edges(section_id, data=True):
            if data.get("relation") == "has_section":
                seq = data.get("sequence")
                if seq is None:
                    return None
                sections = self.get_sections_for_document(doc_id)
                if seq + 1 < len(sections):
                    return sections[seq + 1]
        return None

    def get_previous_section(self, section_id: str) -> str | None:
        """Return the section preceding ``section_id`` if any."""

        for pred, _, data in self.graph.in_edges(section_id, data=True):
            if data.get("relation") == "next_section":
                return pred
        # fallback using sequence metadata
        for doc_id, _, data in self.graph.in_edges(section_id, data=True):
            if data.get("relation") == "has_section":
                seq = data.get("sequence")
                if seq is None:
                    return None
                sections = self.get_sections_for_document(doc_id)
                if seq > 0:
                    return sections[seq - 1]
        return None

    def get_next_chunk(self, chunk_id: str) -> str | None:
        """Return the chunk that follows ``chunk_id`` if any."""

        for _, succ, data in self.graph.out_edges(chunk_id, data=True):
            if data.get("relation") == "next_chunk":
                return succ
        # fallback using sequence metadata
        for doc_id, _, data in self.graph.in_edges(chunk_id, data=True):
            if data.get("relation") == "has_chunk":
                seq = data.get("sequence")
                if seq is None:
                    return None
                chunks = self.get_chunks_for_document(doc_id)
                if seq + 1 < len(chunks):
                    return chunks[seq + 1]
        return None

    def get_previous_chunk(self, chunk_id: str) -> str | None:
        """Return the chunk preceding ``chunk_id`` if any."""

        for pred, _, data in self.graph.in_edges(chunk_id, data=True):
            if data.get("relation") == "next_chunk":
                return pred
        # fallback using sequence metadata
        for doc_id, _, data in self.graph.in_edges(chunk_id, data=True):
            if data.get("relation") == "has_chunk":
                seq = data.get("sequence")
                if seq is None:
                    return None
                chunks = self.get_chunks_for_document(doc_id)
                if seq > 0:
                    return chunks[seq - 1]
        return None

    def get_page_for_chunk(self, chunk_id: str) -> int | None:
        """Return the page number associated with ``chunk_id``."""

        node = self.graph.nodes.get(chunk_id)
        if node and node.get("type") == "chunk":
            return node.get("page")
        return None

    def get_page_for_section(self, section_id: str) -> int | None:
        """Return the starting page recorded for ``section_id``."""

        node = self.graph.nodes.get(section_id)
        if node and node.get("type") == "section":
            return node.get("page")
        return None

    def get_chunks_for_document(self, doc_id: str) -> list[str]:
        """Return all chunk IDs that belong to the given document."""

        edges = [
            (data.get("sequence", i), tgt)
            for i, (src, tgt, data) in enumerate(self.graph.edges(doc_id, data=True))
            if data.get("relation") == "has_chunk"
        ]
        edges.sort(key=lambda x: x[0])
        return [t for _, t in edges]

    def get_images_for_document(self, doc_id: str) -> list[str]:
        """Return image IDs associated with ``doc_id`` ordered by sequence."""

        edges = [
            (data.get("sequence", i), tgt)
            for i, (src, tgt, data) in enumerate(self.graph.edges(doc_id, data=True))
            if data.get("relation") == "has_image"
        ]
        edges.sort(key=lambda x: x[0])
        return [t for _, t in edges]

    def get_facts_for_entity(self, entity_id: str) -> list[str]:
        """Return IDs of facts linked to ``entity_id``."""

        facts = [
            src
            for src, _ in self.graph.in_edges(entity_id)
            if self.graph.edges[src, entity_id].get("relation") in {"subject", "object"}
            and self.graph.nodes[src].get("type") == "fact"
        ]
        return facts

    def get_chunks_for_entity(self, entity_id: str) -> list[str]:
        """Return chunk IDs that mention ``entity_id``."""

        chunks = [
            src
            for src, _ in self.graph.in_edges(entity_id)
            if self.graph.edges[src, entity_id].get("relation") == "mentions"
            and self.graph.nodes[src].get("type") == "chunk"
        ]
        return chunks

    def get_facts_for_chunk(self, chunk_id: str) -> list[str]:
        """Return fact IDs attached to ``chunk_id``."""

        facts = [
            tgt
            for _, tgt, data in self.graph.out_edges(chunk_id, data=True)
            if data.get("relation") == "has_fact" and self.graph.nodes[tgt].get("type") == "fact"
        ]
        return facts

    def get_facts_for_document(self, doc_id: str) -> list[str]:
        """Return fact IDs related to any chunk of ``doc_id``."""

        fact_ids: list[str] = []
        for cid in self.get_chunks_for_document(doc_id):
            fact_ids.extend(self.get_facts_for_chunk(cid))
        return fact_ids

    def get_chunks_for_fact(self, fact_id: str) -> list[str]:
        """Return chunk IDs referencing ``fact_id``."""

        chunks = [
            src
            for src, _ in self.graph.in_edges(fact_id)
            if self.graph.edges[src, fact_id].get("relation") == "has_fact"
            and self.graph.nodes[src].get("type") == "chunk"
        ]
        return chunks

    def get_sections_for_fact(self, fact_id: str) -> list[str]:
        """Return section IDs referencing ``fact_id`` via a chunk."""

        sections: list[str] = []
        for cid in self.get_chunks_for_fact(fact_id):
            sec = self.get_section_for_chunk(cid)
            if sec and sec not in sections:
                sections.append(sec)
        return sections

    def get_documents_for_fact(self, fact_id: str) -> list[str]:
        """Return document IDs referencing ``fact_id`` via a chunk."""

        docs: list[str] = []
        for cid in self.get_chunks_for_fact(fact_id):
            doc = self.get_document_for_chunk(cid)
            if doc and doc not in docs:
                docs.append(doc)
        return docs

    def get_pages_for_fact(self, fact_id: str) -> list[int]:
        """Return page numbers referencing ``fact_id`` via chunks."""

        pages: list[int] = []
        for cid in self.get_chunks_for_fact(fact_id):
            page = self.get_page_for_chunk(cid)
            if page is not None and page not in pages:
                pages.append(page)
        return pages

    def get_entities_for_fact(self, fact_id: str) -> list[str]:
        """Return entity IDs linked as subject or object of ``fact_id``."""

        entities = [
            tgt
            for _, tgt, data in self.graph.out_edges(fact_id, data=True)
            if data.get("relation") in {"subject", "object"}
            and self.graph.nodes[tgt].get("type") == "entity"
        ]
        return entities

    def get_entities_for_chunk(self, chunk_id: str) -> list[str]:
        """Return entity IDs mentioned in ``chunk_id``."""

        entities = [
            tgt
            for _, tgt, data in self.graph.out_edges(chunk_id, data=True)
            if data.get("relation") == "mentions" and self.graph.nodes[tgt].get("type") == "entity"
        ]
        return entities

    def get_entities_for_document(self, doc_id: str) -> list[str]:
        """Return entity IDs mentioned anywhere in ``doc_id``."""

        entities: list[str] = [
            tgt
            for _, tgt, data in self.graph.out_edges(doc_id, data=True)
            if data.get("relation") == "mentions" and self.graph.nodes[tgt].get("type") == "entity"
        ]
        for cid in self.get_chunks_for_document(doc_id):
            entities.extend(self.get_entities_for_chunk(cid))
        # remove duplicates while preserving order
        seen = set()
        out = []
        for e in entities:
            if e not in seen:
                seen.add(e)
                out.append(e)
        return out

    def get_document_for_section(self, section_id: str) -> str | None:
        """Return the document containing ``section_id`` if any."""

        for pred in self.graph.predecessors(section_id):
            if self.graph.edges[pred, section_id].get("relation") == "has_section":
                return pred
        return None

    def get_document_for_chunk(self, chunk_id: str) -> str | None:
        """Return the document containing ``chunk_id`` if any."""

        for pred in self.graph.predecessors(chunk_id):
            if self.graph.edges[pred, chunk_id].get("relation") == "has_chunk":
                return pred
        section_id = self.get_section_for_chunk(chunk_id)
        if section_id is not None:
            return self.get_document_for_section(section_id)
        return None

    def get_similar_chunks(self, chunk_id: str, k: int = 3) -> list[str]:
        """Return up to ``k`` chunk IDs most similar to ``chunk_id``."""

        if chunk_id not in self.graph.nodes or self.graph.nodes[chunk_id].get("type") != "chunk":
            return []

        text = self.graph.nodes[chunk_id].get("text")
        if not text:
            return []

        indices = self.index.search(text, k=k + 1)
        neighbors: list[str] = []
        for idx in indices:
            nid = self.index.get_id(idx)
            if nid == chunk_id:
                continue
            if self.graph.nodes[nid].get("type") != "chunk":
                continue
            neighbors.append(nid)
            if len(neighbors) >= k:
                break
        return neighbors

    def get_similar_chunks_data(self, chunk_id: str, k: int = 3) -> list[dict[str, Any]]:
        """Return up to ``k`` similar chunk infos for ``chunk_id``."""

        if chunk_id not in self.graph.nodes or self.graph.nodes[chunk_id].get("type") != "chunk":
            return []

        data = []
        neighbors = self.index.nearest_neighbors(k=k, return_distances=True).get(chunk_id, [])
        for nid, score in neighbors:
            if self.graph.nodes[nid].get("type") != "chunk":
                continue
            data.append(
                {
                    "id": nid,
                    "similarity": score,
                    "text": self.graph.nodes[nid].get("text"),
                    "document": self.get_document_for_chunk(nid),
                }
            )
        return data

    def get_chunk_neighbors(self, k: int = 3) -> dict[str, list[str]]:
        """Return the ``k`` nearest chunk neighbors for each chunk."""

        raw = self.index.nearest_neighbors(k)
        neighbors: dict[str, list[str]] = {}
        for cid, neigh in raw.items():
            if self.graph.nodes.get(cid, {}).get("type") != "chunk":
                continue
            filtered = [n for n in neigh if self.graph.nodes.get(n, {}).get("type") == "chunk"]
            neighbors[cid] = filtered
        return neighbors

    def get_chunk_neighbors_data(self, k: int = 3) -> dict[str, list[dict[str, Any]]]:
        """Return neighbor information for each chunk."""

        raw = self.index.nearest_neighbors(k, return_distances=True)
        out: dict[str, list[dict[str, Any]]] = {}
        for cid, neigh in raw.items():
            if self.graph.nodes.get(cid, {}).get("type") != "chunk":
                continue
            data: list[dict[str, Any]] = []
            for nid, score in neigh:
                if self.graph.nodes.get(nid, {}).get("type") != "chunk":
                    continue
                data.append(
                    {
                        "id": nid,
                        "similarity": score,
                        "text": self.graph.nodes[nid].get("text"),
                        "document": self.get_document_for_chunk(nid),
                    }
                )
            out[cid] = data
        return out

    def get_similar_sections(self, section_id: str, k: int = 3) -> list[str]:
        """Return up to ``k`` section IDs similar to ``section_id``."""

        if (
            section_id not in self.graph.nodes
            or self.graph.nodes[section_id].get("type") != "section"
        ):
            return []

        title = self.graph.nodes[section_id].get("title")
        if not title:
            return []

        indices = self.index.search(title, k=k + 1)
        neighbors: list[str] = []
        for idx in indices:
            sid = self.index.get_id(idx)
            if sid == section_id:
                continue
            if self.graph.nodes[sid].get("type") != "section":
                continue
            neighbors.append(sid)
            if len(neighbors) >= k:
                break
        return neighbors

    def get_similar_documents(self, doc_id: str, k: int = 3) -> list[str]:
        """Return up to ``k`` document IDs similar to ``doc_id``."""

        if doc_id not in self.graph.nodes or self.graph.nodes[doc_id].get("type") != "document":
            return []

        text = self.graph.nodes[doc_id].get("text")
        if not text:
            return []

        indices = self.index.search(text, k=k + 1)
        neighbors: list[str] = []
        for idx in indices:
            did = self.index.get_id(idx)
            if did == doc_id:
                continue
            if self.graph.nodes[did].get("type") != "document":
                continue
            neighbors.append(did)
            if len(neighbors) >= k:
                break
        return neighbors

    def get_chunk_context(self, chunk_id: str, before: int = 1, after: int = 1) -> list[str]:
        """Return chunk IDs surrounding ``chunk_id`` including itself."""

        if chunk_id not in self.graph.nodes or self.graph.nodes[chunk_id].get("type") != "chunk":
            return []

        context: list[str] = [chunk_id]

        current = chunk_id
        for _ in range(before):
            prev = self.get_previous_chunk(current)
            if prev is None:
                break
            context.insert(0, prev)
            current = prev

        current = chunk_id
        for _ in range(after):
            nxt = self.get_next_chunk(current)
            if nxt is None:
                break
            context.append(nxt)
            current = nxt

        return context

    def get_documents_for_entity(self, entity_id: str) -> list[str]:
        """Return document IDs where ``entity_id`` is mentioned."""

        docs: set[str] = set()
        # direct links from document to entity
        for src, _, data in self.graph.in_edges(entity_id, data=True):
            if (
                data.get("relation") == "mentions"
                and self.graph.nodes[src].get("type") == "document"
            ):
                docs.add(src)
        # links via chunks
        for chunk_id in self.get_chunks_for_entity(entity_id):
            for pred in self.graph.predecessors(chunk_id):
                if self.graph.edges[pred, chunk_id].get("relation") == "has_chunk":
                    docs.add(pred)
        return list(docs)

    def get_pages_for_entity(self, entity_id: str) -> list[int]:
        """Return page numbers mentioning ``entity_id`` via chunks."""

        pages: list[int] = []
        for cid in self.get_chunks_for_entity(entity_id):
            page = self.get_page_for_chunk(cid)
            if page is not None and page not in pages:
                pages.append(page)
        return pages

    # ------------------------------------------------------------------
    # Fact helpers
    # ------------------------------------------------------------------

    def find_facts(
        self,
        *,
        subject: str | None = None,
        predicate: str | None = None,
        object: str | None = None,
    ) -> list[str]:
        """Return fact IDs matching the provided components."""

        matches: list[str] = []
        for node, data in self.graph.nodes(data=True):
            if data.get("type") != "fact":
                continue
            if subject is not None and data.get("subject") != subject:
                continue
            if predicate is not None and data.get("predicate") != predicate:
                continue
            if object is not None and data.get("object") != object:
                continue
            matches.append(node)
        return matches

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

    def to_text(self) -> str:
        """Return all chunk texts concatenated in document order."""

        parts: list[str] = []
        docs = [n for n, d in self.graph.nodes(data=True) if d.get("type") == "document"]
        docs.sort()
        for doc_id in docs:
            chunks = self.get_chunks_for_document(doc_id)
            if chunks:
                for cid in chunks:
                    text = self.graph.nodes[cid].get("text")
                    if text:
                        parts.append(text)
            else:
                text = self.graph.nodes[doc_id].get("text")
                if text:
                    parts.append(text)
        return "\n\n".join(parts)

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
