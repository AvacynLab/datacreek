from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import redis

from ..pipelines import DatasetType, PipelineStep
from .knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)


@dataclass
class HistoryEvent:
    """Record a dataset modification for auditing."""

    operation: str
    message: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    params: Optional[Dict[str, Any]] | None = None


@dataclass
class DatasetBuilder:
    """Manage a dataset under construction with its own knowledge graph."""

    dataset_type: DatasetType
    id: str = field(default_factory=lambda: secrets.token_hex(8))
    name: Optional[str] = None
    graph: KnowledgeGraph = field(default_factory=KnowledgeGraph)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    history: List[str] = field(default_factory=list)
    events: List[HistoryEvent] = field(default_factory=list)
    versions: List[Dict[str, Any]] = field(default_factory=list)
    stage: int = 0  # 0=created, 1=ingest, 2=generation, 3=curation, 4=exported

    def _record_event(self, operation: str, message: str, **params: Any) -> None:
        """Store a history event and log it."""

        event = HistoryEvent(operation, message, params=params or None)
        self.events.append(event)
        self.history.append(message)
        logger.info(message)

    def add_document(
        self,
        doc_id: str,
        source: str,
        *,
        text: str | None = None,
        author: str | None = None,
        organization: str | None = None,
    ) -> None:
        """Insert a document node in the dataset graph."""

        self.graph.add_document(
            doc_id,
            source,
            text=text,
            author=author,
            organization=organization,
        )
        self._record_event(
            "add_document",
            f"Added document {doc_id}",
            source=source,
        )

    def add_section(
        self,
        doc_id: str,
        section_id: str,
        title: str | None = None,
        source: Optional[str] = None,
        *,
        page: int | None = None,
    ) -> None:
        """Insert a section node for ``doc_id``."""

        self.graph.add_section(doc_id, section_id, title=title, source=source, page=page)
        self._record_event(
            "add_section",
            f"Added section {section_id} to {doc_id}",
            source=source,
        )

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
        """Insert a chunk node in the dataset graph."""

        self.graph.add_chunk(doc_id, chunk_id, text, source, section_id=section_id, page=page)
        self._record_event(
            "add_chunk",
            f"Added chunk {chunk_id} to {doc_id}",
            source=source,
        )

    def add_image(
        self,
        doc_id: str,
        image_id: str,
        path: str,
        *,
        page: int | None = None,
    ) -> None:
        """Insert an image node in the dataset graph."""

        self.graph.add_image(doc_id, image_id, path, page=page)
        self._record_event("add_image", f"Added image {image_id} to {doc_id}")

    def add_entity(self, entity_id: str, text: str, source: Optional[str] = None) -> None:
        """Insert an entity node."""

        self.graph.add_entity(entity_id, text, source)
        self._record_event(
            "add_entity",
            f"Added entity {entity_id}",
            source=source,
        )

    def link_entity(
        self,
        node_id: str,
        entity_id: str,
        relation: str = "mentions",
        *,
        provenance: Optional[str] = None,
    ) -> None:
        """Link an entity to another node."""

        self.graph.link_entity(node_id, entity_id, relation, provenance=provenance)
        self._record_event(
            "link_entity",
            f"Linked {node_id} to {entity_id}",
            relation=relation,
            provenance=provenance,
        )

    def search_chunks(self, query: str) -> list[str]:
        return self.graph.search_chunks(query)

    def search(self, query: str, node_type: str = "chunk") -> list[str]:
        return self.graph.search(query, node_type=node_type)

    def search_documents(self, query: str) -> list[str]:
        return self.graph.search_documents(query)

    def search_entities(self, query: str) -> list[str]:
        """Return entity IDs matching ``query``."""

        return self.graph.search(query, node_type="entity")

    def search_sections(self, query: str) -> list[str]:
        """Return section IDs whose title or ID matches ``query``."""

        return self.graph.search(query, node_type="section")

    def search_facts(self, query: str) -> list[str]:
        """Return fact IDs whose subject, predicate or object matches the query."""

        return self.graph.search(query, node_type="fact")

    def search_embeddings(
        self,
        query: str,
        k: int = 3,
        fetch_neighbors: bool = True,
        *,
        node_type: str = "chunk",
    ) -> list[str]:
        """Wrapper for :meth:`KnowledgeGraph.search_embeddings`."""

        return self.graph.search_embeddings(
            query,
            k=k,
            fetch_neighbors=fetch_neighbors,
            node_type=node_type,
        )

    def search_hybrid(self, query: str, k: int = 5, *, node_type: str = "chunk") -> list[str]:
        """Wrapper for :meth:`KnowledgeGraph.search_hybrid`."""

        return self.graph.search_hybrid(query, k=k, node_type=node_type)

    def search_with_links(self, query: str, k: int = 5, hops: int = 1) -> list[str]:
        """Wrapper for :meth:`KnowledgeGraph.search_with_links`."""

        return self.graph.search_with_links(query, k=k, hops=hops)

    def search_with_links_data(self, query: str, k: int = 5, hops: int = 1) -> List[Dict[str, Any]]:
        """Wrapper for :meth:`KnowledgeGraph.search_with_links_data`.

        Returns detailed chunk information, hop depth and traversal path.
        """

        return self.graph.search_with_links_data(query, k=k, hops=hops)

    def link_similar_chunks(self, k: int = 3) -> None:
        """Create similarity edges between chunks using embeddings."""

        self.graph.link_similar_chunks(k)

    def link_similar_sections(self, k: int = 3) -> None:
        """Create similarity edges between section titles."""

        self.graph.link_similar_sections(k)

    def link_similar_documents(self, k: int = 3) -> None:
        """Create similarity edges between document texts."""

        self.graph.link_similar_documents(k)

    def link_chunks_by_entity(self) -> int:
        """Connect chunks that mention the same entity."""

        added = self.graph.link_chunks_by_entity()
        if added:
            msg = f"Added {added} co-mention links"
            self._record_event(
                "link_chunks_by_entity",
                msg,
                added=added,
            )
        return added

    def link_documents_by_entity(self) -> int:
        """Connect documents that mention the same entity."""

        added = self.graph.link_documents_by_entity()
        if added:
            msg = f"Linked {added} co-mentioned documents"
            self._record_event(
                "link_documents_by_entity",
                msg,
                added=added,
            )
        return added

    def link_sections_by_entity(self) -> int:
        """Connect sections that mention the same entity."""

        added = self.graph.link_sections_by_entity()
        if added:
            msg = f"Linked {added} co-mentioned sections"
            self._record_event(
                "link_sections_by_entity",
                msg,
                added=added,
            )
        return added

    def link_authors_organizations(self) -> int:
        """Create affiliation links between authors and organizations."""

        added = self.graph.link_authors_organizations()
        if added:
            msg = f"Linked {added} authors to organizations"
            self._record_event(
                "link_authors_organizations",
                msg,
                added=added,
            )
        return added

    def get_similar_chunks(self, chunk_id: str, k: int = 3) -> list[str]:
        """Return up to ``k`` chunk IDs most similar to ``chunk_id``."""

        return self.graph.get_similar_chunks(chunk_id, k=k)

    def get_similar_chunks_data(self, chunk_id: str, k: int = 3) -> List[Dict[str, Any]]:
        """Return up to ``k`` similar chunk infos for ``chunk_id``."""

        return self.graph.get_similar_chunks_data(chunk_id, k=k)

    def get_chunk_neighbors(self, k: int = 3) -> Dict[str, List[str]]:
        """Return the ``k`` nearest neighbors for each chunk."""

        return self.graph.get_chunk_neighbors(k=k)

    def get_chunk_neighbors_data(self, k: int = 3) -> Dict[str, List[Dict[str, Any]]]:
        """Return neighbor information for every chunk."""

        return self.graph.get_chunk_neighbors_data(k=k)

    def get_similar_sections(self, section_id: str, k: int = 3) -> list[str]:
        """Return up to ``k`` section IDs most similar to ``section_id``."""

        return self.graph.get_similar_sections(section_id, k=k)

    def get_similar_documents(self, doc_id: str, k: int = 3) -> list[str]:
        """Return up to ``k`` document IDs most similar to ``doc_id``."""

        return self.graph.get_similar_documents(doc_id, k=k)

    def get_chunk_context(self, chunk_id: str, before: int = 1, after: int = 1) -> list[str]:
        """Return chunk IDs surrounding ``chunk_id`` including itself."""

        return self.graph.get_chunk_context(chunk_id, before=before, after=after)

    def deduplicate_chunks(self, similarity: float = 1.0) -> int:
        """Remove duplicate chunk nodes from the graph.

        Parameters
        ----------
        similarity:
            Similarity threshold between 0 and 1 for detecting duplicates. A
            value of 1.0 removes only exact matches.
        """

        removed = self.graph.deduplicate_chunks(similarity)
        if removed:
            msg = f"Removed {removed} duplicate chunks"
            self._record_event("deduplicate_chunks", msg)
        return removed

    def clean_chunks(self) -> int:
        """Normalize chunk text to remove markup and extra whitespace."""

        cleaned = self.graph.clean_chunk_texts()
        if cleaned:
            msg = f"Cleaned {cleaned} chunks"
            self._record_event("clean_chunks", msg)
        return cleaned

    def cleanup_graph(
        self,
        *,
        resolve_threshold: float = 0.8,
        resolve_aliases: dict[str, list[str]] | None = None,
        dedup_similarity: float = 1.0,
    ) -> tuple[int, int]:
        """Run standard deduplication and cleaning steps on the graph.

        This removes duplicate chunks, normalizes their text and resolves
        entities so later pipeline stages work with a clean knowledge graph.

        Parameters
        ----------
        resolve_threshold:
            Minimum similarity score when merging entities.
        resolve_aliases:
            Optional mapping of canonical labels to lists of aliases used during
            entity resolution.
        dedup_similarity:
            Threshold used when removing duplicate chunks. Values below 1.0
            allow near-duplicate detection.

        Returns
        -------
        tuple[int, int]
            Number of chunks removed and cleaned respectively.
        """

        removed = self.deduplicate_chunks(similarity=dedup_similarity)
        cleaned = self.clean_chunks()
        self.resolve_entities(threshold=resolve_threshold, aliases=resolve_aliases)
        return removed, cleaned

    def normalize_dates(self) -> int:
        """Standardize date attributes on nodes to ISO format."""

        changed = self.graph.normalize_date_fields()
        if changed:
            msg = f"Normalized {changed} date fields"
            self._record_event("normalize_dates", msg)
        return changed

    def prune_sources(self, sources: List[str]) -> int:
        """Remove nodes and edges associated with ``sources`` from the graph."""

        removed = self.graph.prune_sources(sources)
        if removed:
            joined = ", ".join(sources)
            msg = f"Pruned {removed} nodes from {joined}"
            self._record_event("prune_sources", msg, sources=sources)
        return removed

    def resolve_entities(
        self,
        threshold: float = 0.8,
        aliases: dict[str, list[str]] | None = None,
    ) -> int:
        """Merge entity nodes that likely refer to the same real world entity."""

        merged = self.graph.resolve_entities(threshold=threshold, aliases=aliases)
        if merged:
            msg = f"Merged {merged} entities"
            self._record_event(
                "resolve_entities",
                msg,
                threshold=threshold,
                aliases=list(aliases.keys()) if aliases else None,
            )
        return merged

    def predict_links(self, threshold: float = 0.8, *, use_graph_embeddings: bool = False) -> None:
        """Infer missing relations between entities based on similarity."""

        self.graph.predict_links(threshold=threshold, use_graph_embeddings=use_graph_embeddings)
        self._record_event(
            "predict_links",
            "Predicted entity links",
            threshold=threshold,
            use_graph_embeddings=use_graph_embeddings,
        )

    def enrich_entity(self, entity_id: str) -> None:
        """Enrich an entity node using external sources like Wikidata."""

        self.graph.enrich_entity_wikidata(entity_id)
        self._record_event("enrich_entity", f"Entity {entity_id} enriched")

    def enrich_entity_dbpedia(self, entity_id: str) -> None:
        """Enrich an entity node using DBpedia."""

        self.graph.enrich_entity_dbpedia(entity_id)
        self._record_event(
            "enrich_entity_dbpedia",
            f"Entity {entity_id} enriched from DBpedia",
        )

    def consolidate_schema(self) -> None:
        """Normalize labels in the underlying knowledge graph."""

        self.graph.consolidate_schema()

    def detect_communities(self, n_clusters: int = 3) -> None:
        """Cluster chunks into communities."""

        self.graph.cluster_chunks(n_clusters=n_clusters)

    def detect_entity_groups(self, n_clusters: int = 3) -> None:
        """Cluster entity nodes into groups."""

        self.graph.cluster_entities(n_clusters=n_clusters)

    def summarize_entity_groups(self) -> None:
        self.graph.summarize_entity_groups()

    def summarize_communities(self) -> None:
        self.graph.summarize_communities()

    def score_trust(self) -> None:
        self.graph.score_trust()
        self._record_event("score_trust", "Computed trust scores")

    def compute_centrality(self, node_type: str = "entity", metric: str = "degree") -> None:
        """Compute centrality metrics for graph nodes."""
        self.graph.compute_centrality(node_type=node_type, metric=metric)
        self._record_event(
            "compute_centrality",
            f"Centrality ({metric}) computed for {node_type} nodes",
            node_type=node_type,
            metric=metric,
        )

    def compute_graph_embeddings(
        self,
        dimensions: int = 64,
        walk_length: int = 10,
        num_walks: int = 50,
        seed: int = 0,
        workers: int = 1,
    ) -> None:
        """Generate Node2Vec embeddings for all nodes."""

        self.graph.compute_node2vec_embeddings(
            dimensions=dimensions,
            walk_length=walk_length,
            num_walks=num_walks,
            workers=workers,
            seed=seed,
        )
        self._record_event(
            "compute_graph_embeddings",
            "Graph embeddings computed",
            dimensions=dimensions,
            walk_length=walk_length,
            num_walks=num_walks,
            workers=workers,
            seed=seed,
        )

    def update_embeddings(self, node_type: str = "chunk") -> None:
        """Materialize embeddings for nodes of ``node_type``."""

        self.graph.update_embeddings(node_type=node_type)

    def extract_facts(self, client: Optional["LLMClient"] = None) -> None:
        """Run fact extraction on all chunk nodes."""

        from datacreek.utils.fact_extraction import extract_facts

        for cid, data in list(self.graph.graph.nodes(data=True)):
            if data.get("type") != "chunk":
                continue
            facts = extract_facts(data.get("text", ""), client)
            for i, fact in enumerate(facts):
                fid = f"{cid}_fact_{i}"
                self.graph.add_fact(
                    fact["subject"],
                    fact["predicate"],
                    fact["object"],
                    fact_id=fid,
                    source=data.get("source"),
                )
                self.graph.graph.add_edge(
                    cid, fid, relation="has_fact", provenance=data.get("source")
                )

    def extract_entities(self, model: str | None = "en_core_web_sm") -> None:
        """Run named entity recognition on all chunks."""

        self.graph.extract_entities(model=model)
        self._record_event("extract_entities", "Entities extracted", model=model)

    def find_conflicting_facts(self) -> List[tuple[str, str, Dict[str, List[str]]]]:
        """Return edges with the same subject/predicate but different objects."""

        conflicts: Dict[tuple[str, str], Dict[str, List[str]]] = {}
        for u, v, edata in self.graph.graph.edges(data=True):
            rel = edata.get("relation")
            if not rel or rel in {
                "has_chunk",
                "next_chunk",
                "subject",
                "object",
                "has_fact",
                "mentions",
            }:
                continue
            prov = edata.get("provenance")
            key = (u, rel)
            conflicts.setdefault(key, {})
            conflicts[key].setdefault(v, [])
            if prov:
                conflicts[key][v].append(prov)

        result = []
        for key, obj_map in conflicts.items():
            if len(obj_map) > 1:
                result.append((key[0], key[1], obj_map))
        return result

    def mark_conflicting_facts(self) -> int:
        """Annotate edges that belong to conflicting fact groups."""

        marked = self.graph.mark_conflicting_facts()
        if marked:
            msg = f"Marked {marked} conflicting facts"
            self._record_event("mark_conflicts", msg)
        return marked

    def validate_coherence(self) -> int:
        """Flag logically inconsistent edges in the underlying graph."""

        marked = self.graph.validate_coherence()
        if marked:
            msg = f"Marked {marked} inconsistent relations"
            self._record_event("validate_coherence", msg)
        return marked

    def get_chunks_for_document(self, doc_id: str) -> list[str]:
        return self.graph.get_chunks_for_document(doc_id)

    def get_images_for_document(self, doc_id: str) -> list[str]:
        return self.graph.get_images_for_document(doc_id)

    def get_sections_for_document(self, doc_id: str) -> list[str]:
        return self.graph.get_sections_for_document(doc_id)

    def get_document_for_section(self, section_id: str) -> str | None:
        return self.graph.get_document_for_section(section_id)

    def get_document_for_chunk(self, chunk_id: str) -> str | None:
        return self.graph.get_document_for_chunk(chunk_id)

    def get_chunks_for_section(self, section_id: str) -> list[str]:
        return self.graph.get_chunks_for_section(section_id)

    def get_section_for_chunk(self, chunk_id: str) -> str | None:
        return self.graph.get_section_for_chunk(chunk_id)

    def get_next_chunk(self, chunk_id: str) -> str | None:
        """Return the chunk following ``chunk_id`` if any."""

        return self.graph.get_next_chunk(chunk_id)

    def get_previous_chunk(self, chunk_id: str) -> str | None:
        """Return the chunk preceding ``chunk_id`` if any."""

        return self.graph.get_previous_chunk(chunk_id)

    def get_page_for_chunk(self, chunk_id: str) -> int | None:
        """Return the page number stored on ``chunk_id``."""

        return self.graph.get_page_for_chunk(chunk_id)

    def get_page_for_section(self, section_id: str) -> int | None:
        """Return the page number recorded for ``section_id``."""

        return self.graph.get_page_for_section(section_id)

    def get_next_section(self, section_id: str) -> str | None:
        """Return the section following ``section_id`` if any."""

        return self.graph.get_next_section(section_id)

    def get_previous_section(self, section_id: str) -> str | None:
        """Return the section preceding ``section_id`` if any."""

        return self.graph.get_previous_section(section_id)

    def get_facts_for_chunk(self, chunk_id: str) -> list[str]:
        """Return fact IDs attached to ``chunk_id``."""

        return self.graph.get_facts_for_chunk(chunk_id)

    def get_facts_for_document(self, doc_id: str) -> list[str]:
        """Return fact IDs related to any chunk of ``doc_id``."""

        return self.graph.get_facts_for_document(doc_id)

    def get_chunks_for_fact(self, fact_id: str) -> list[str]:
        """Return chunk IDs referencing ``fact_id``."""

        return self.graph.get_chunks_for_fact(fact_id)

    def get_entities_for_fact(self, fact_id: str) -> list[str]:
        """Return entity IDs linked as subject or object of ``fact_id``."""

        return self.graph.get_entities_for_fact(fact_id)

    def get_sections_for_fact(self, fact_id: str) -> list[str]:
        """Return section IDs referencing ``fact_id`` via a chunk."""

        return self.graph.get_sections_for_fact(fact_id)

    def get_documents_for_fact(self, fact_id: str) -> list[str]:
        """Return document IDs referencing ``fact_id`` via a chunk."""

        return self.graph.get_documents_for_fact(fact_id)

    def get_pages_for_fact(self, fact_id: str) -> list[int]:
        """Return page numbers referencing ``fact_id`` via chunks."""

        return self.graph.get_pages_for_fact(fact_id)

    def get_facts_for_entity(self, entity_id: str) -> list[str]:
        """Return fact IDs connected to ``entity_id``."""

        return self.graph.get_facts_for_entity(entity_id)

    def get_chunks_for_entity(self, entity_id: str) -> list[str]:
        """Return chunk IDs mentioning ``entity_id``."""

        return self.graph.get_chunks_for_entity(entity_id)

    def get_entities_for_chunk(self, chunk_id: str) -> list[str]:
        """Return entity IDs mentioned in ``chunk_id``."""

        return self.graph.get_entities_for_chunk(chunk_id)

    def get_entities_for_document(self, doc_id: str) -> list[str]:
        """Return entity IDs mentioned anywhere in ``doc_id``."""

        return self.graph.get_entities_for_document(doc_id)

    def find_facts(
        self,
        *,
        subject: str | None = None,
        predicate: str | None = None,
        object: str | None = None,
    ) -> list[str]:
        """Wrapper for :meth:`KnowledgeGraph.find_facts`."""

        return self.graph.find_facts(subject=subject, predicate=predicate, object=object)

    def get_documents_for_entity(self, entity_id: str) -> list[str]:
        """Return document IDs where ``entity_id`` is mentioned."""

        return self.graph.get_documents_for_entity(entity_id)

    def get_pages_for_entity(self, entity_id: str) -> list[int]:
        """Return page numbers where ``entity_id`` is mentioned."""

        return self.graph.get_pages_for_entity(entity_id)

    def remove_chunk(self, chunk_id: str) -> None:
        """Remove a chunk node from the dataset graph."""
        if self.graph.graph.has_node(chunk_id):
            preds = list(self.graph.graph.predecessors(chunk_id))
            self.graph.remove_chunk(chunk_id)
            if preds:
                msg = f"Removed chunk {chunk_id} from {preds[0]}"
                self._record_event("remove_chunk", msg)

    def remove_document(self, doc_id: str) -> None:
        """Remove a document and all its chunks."""
        if not self.graph.graph.has_node(doc_id):
            return
        self.graph.remove_document(doc_id)
        self._record_event("remove_document", f"Removed document {doc_id}")

    def clone(self, name: Optional[str] = None) -> "DatasetBuilder":
        """Return a deep copy of this dataset with a new optional name."""
        clone = DatasetBuilder(self.dataset_type, name=name, graph=deepcopy(self.graph))
        clone.history = self.history.copy()
        clone.versions = deepcopy(self.versions)
        clone.stage = self.stage
        return clone

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this dataset to a Python dictionary."""

        return {
            "dataset_type": self.dataset_type.value,
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "history": self.history,
            "versions": self.versions,
            "graph": self.graph.to_dict(),
            "stage": self.stage,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetBuilder":
        ds = cls(DatasetType(data["dataset_type"]), name=data.get("name"))
        if "id" in data:
            ds.id = data["id"]
        if ts := data.get("created_at"):
            ds.created_at = datetime.fromisoformat(ts)
        ds.history = list(data.get("history", []))
        ds.versions = list(data.get("versions", []))
        ds.graph = KnowledgeGraph.from_dict(data.get("graph", {}))
        ds.stage = int(data.get("stage", 0))
        return ds

    def to_json(self, path: str) -> str:
        """Save this dataset to ``path`` in JSON format."""

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @classmethod
    def from_json(cls, path: str) -> "DatasetBuilder":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    # ------------------------------------------------------------------
    # Redis helpers
    # ------------------------------------------------------------------

    def to_redis(self, client: redis.Redis, key: str | None = None) -> str:
        """Persist the dataset in Redis under ``key``."""

        key = key or (self.name or "dataset")
        client.set(key, json.dumps(self.to_dict()))
        return key

    @classmethod
    def from_redis(cls, client: redis.Redis, key: str) -> "DatasetBuilder":
        data = client.get(key)
        if data is None:
            raise KeyError(key)
        return cls.from_dict(json.loads(data))

    # ------------------------------------------------------------------
    # Generation helpers
    # ------------------------------------------------------------------

    def get_raw_text(self) -> str:
        """Return text representation of the underlying graph."""

        return self.graph.to_text()

    def run_post_kg_pipeline(
        self,
        *,
        config_path: Path | None = None,
        provider: str | None = None,
        profile: str | None = None,
        api_base: str | None = None,
        model: str | None = None,
        num_pairs: int | None = None,
        threshold: float | None = None,
        fmt: str | None = None,
        overrides: Dict[str, Any] | None = None,
        verbose: bool = False,
        async_mode: bool = False,
        multi_answer: bool = False,
        batch_size: int | None = None,
        inference_batch: int | None = None,
        start_step: PipelineStep | None = None,
        checkpoint_dir: Path | None = None,
        pipeline_config_path: Path | None = None,
        dedup_similarity: float = 1.0,
        keep_ratings: bool = False,
    ) -> Any:
        """Run generation steps after the knowledge graph stage.

        When ``async_mode`` is ``True`` the underlying LLM calls may be
        executed concurrently. ``multi_answer`` forwards to the knowledge graph
        generator to produce several answers per fact. ``pipeline_config_path``
        can override the default pipeline definition. ``dedup_similarity``
        adjusts near-duplicate detection during cleanup and ``keep_ratings``
        controls whether ratings are preserved in the curation result. The
        dataset builder is passed to the pipeline so cleanup events are
        recorded.
        """

        from datacreek.pipelines import run_generation_pipeline, run_generation_pipeline_async

        if async_mode:
            return asyncio.run(
                run_generation_pipeline_async(
                    self.dataset_type,
                    self.graph,
                    dataset_builder=self,
                    config_path=config_path,
                    provider=provider,
                    profile=profile,
                    api_base=api_base,
                    model=model,
                    num_pairs=num_pairs,
                    threshold=threshold,
                    fmt=fmt,
                    overrides=overrides,
                    verbose=verbose,
                    async_mode=True,
                    multi_answer=multi_answer,
                    batch_size=batch_size,
                    inference_batch=inference_batch,
                    start_step=start_step,
                    checkpoint_dir=checkpoint_dir,
                    pipeline_config_path=pipeline_config_path,
                    dedup_similarity=dedup_similarity,
                    keep_ratings=keep_ratings,
                )
            )

        return run_generation_pipeline(
            self.dataset_type,
            self.graph,
            dataset_builder=self,
            config_path=config_path,
            provider=provider,
            profile=profile,
            api_base=api_base,
            model=model,
            num_pairs=num_pairs,
            threshold=threshold,
            fmt=fmt,
            overrides=overrides,
            verbose=verbose,
            async_mode=False,
            multi_answer=multi_answer,
            batch_size=batch_size,
            inference_batch=inference_batch,
            start_step=start_step,
            checkpoint_dir=checkpoint_dir,
            pipeline_config_path=pipeline_config_path,
            dedup_similarity=dedup_similarity,
            keep_ratings=keep_ratings,
        )
