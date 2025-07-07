from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import secrets
from copy import deepcopy
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional

import networkx as nx
import numpy as np

if TYPE_CHECKING:
    from .ingest import IngestOptions

import redis
from neo4j import Driver

from ..backends import get_redis_graph

try:  # optional redisgraph dependency
    from redisgraph import Edge as RGEdge
    from redisgraph import Graph as RGGraph
    from redisgraph import Node as RGNode
except Exception:  # pragma: no cover - optional
    RGGraph = None
    RGNode = None
    RGEdge = None

from ..models.stage import DatasetStage
from ..pipelines import DatasetType, PipelineStep
from .knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)

MAX_NAME_LENGTH = 64
NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")


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
    owner_id: int | None = None
    history: List[str] = field(default_factory=list)
    events: List[HistoryEvent] = field(default_factory=list)
    versions: List[Dict[str, Any]] = field(default_factory=list)
    ingested_docs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    accessed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    stage: DatasetStage = DatasetStage.CREATED
    redis_client: redis.Redis | None = field(default=None, repr=False)
    neo4j_driver: Driver | None = field(default=None, repr=False, compare=False)
    redis_key: str | None = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Name validation
    # ------------------------------------------------------------------

    @staticmethod
    def validate_name(name: str) -> str:
        """Validate that ``name`` is a safe identifier."""
        if len(name) > MAX_NAME_LENGTH or not NAME_PATTERN.match(name):
            raise ValueError(f"Invalid dataset name: {name}")
        return name

    def __post_init__(self) -> None:
        """Ensure persistence backends are configured when required."""
        if self.name:
            self.validate_name(self.name)
        require = os.getenv("DATACREEK_REQUIRE_PERSISTENCE", "1") != "0"
        if require and (self.redis_client is None or self.neo4j_driver is None):
            raise ValueError("Redis and Neo4j must be configured")

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _persist(self) -> None:
        """Persist the dataset state to configured backends."""

        if self.redis_client and self.name:
            key = self.redis_key or f"dataset:{self.name}"
            try:
                pipe = self.redis_client.pipeline()
                self.to_redis(pipe, key)
                pipe.sadd("datasets", self.name)
                if self.owner_id is not None:
                    pipe.sadd(f"user:{self.owner_id}:datasets", self.name)
                pipe.execute()
            except Exception:
                logger.exception("Failed to persist dataset %s", self.name)
        if self.neo4j_driver and self.name:
            try:
                self.graph.to_neo4j(
                    self.neo4j_driver,
                    dataset=self.name,
                )
            except Exception:
                logger.exception("Failed to persist graph %s to Neo4j", self.name)
        graph = get_redis_graph(self.name)
        if graph is not None:
            try:
                DatasetBuilder.save_redis_graph.__wrapped__(self, graph)
            except Exception:
                logger.exception("Failed to persist graph %s to RedisGraph", self.name)

    def _touch(self) -> None:
        """Update the ``accessed_at`` timestamp in Redis."""

        if self.redis_client and self.redis_key:
            self.accessed_at = datetime.now(timezone.utc)
            try:
                self.redis_client.set(self.redis_key, json.dumps(self.to_dict()))
            except Exception:
                logger.exception("Failed to update access time for %s", self.name)

    # ------------------------------------------------------------------
    # Decorators
    # ------------------------------------------------------------------

    @staticmethod
    def persist_after(func):
        """Decorator ensuring dataset state is persisted after ``func``."""

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            self._persist()
            return result

        return wrapper

    def _record_event(self, operation: str, message: str, **params: Any) -> None:
        """Store a history event and log it."""

        event = HistoryEvent(operation, message, params=params or None)
        self.events.append(event)
        self.history.append(message)
        logger.info(message)
        if self.redis_client and self.name:
            key = f"dataset:{self.name}"
            data = {
                "operation": event.operation,
                "message": event.message,
                "timestamp": event.timestamp.isoformat(),
                "params": event.params,
            }
            try:
                self.redis_client.rpush(f"{key}:events", json.dumps(data))
                log = data | {"dataset": self.name}
                self.redis_client.rpush("dataset:events", json.dumps(log))
            except Exception:
                logger.exception("Failed to persist event %s", self.name)

        self._persist()

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
        emotion: str | None = None,
        modality: str | None = None,
    ) -> None:
        """Insert a chunk node in the dataset graph."""

        self.graph.add_chunk(
            doc_id,
            chunk_id,
            text,
            source,
            section_id=section_id,
            page=page,
            emotion=emotion,
            modality=modality,
        )
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
        alt_text: str | None = None,
    ) -> None:
        """Insert an image node in the dataset graph."""

        self.graph.add_image(doc_id, image_id, path, page=page, alt_text=alt_text)
        self._record_event("add_image", f"Added image {image_id} to {doc_id}")

    def add_atom(
        self,
        doc_id: str,
        atom_id: str,
        text: str,
        element_type: str,
        source: Optional[str] = None,
        *,
        page: int | None = None,
        emotion: str | None = None,
        modality: str | None = None,
    ) -> None:
        """Insert an atom node."""

        self.graph.add_atom(
            doc_id,
            atom_id,
            text,
            element_type,
            source,
            page=page,
            emotion=emotion,
            modality=modality,
        )
        self._record_event(
            "add_atom",
            f"Added atom {atom_id} to {doc_id}",
            source=source,
        )

    def add_molecule(
        self,
        doc_id: str,
        molecule_id: str,
        atom_ids: Iterable[str],
        source: Optional[str] = None,
    ) -> None:
        """Insert a molecule node made of existing atoms."""

        self.graph.add_molecule(doc_id, molecule_id, atom_ids, source=source)
        self._record_event(
            "add_molecule",
            f"Added molecule {molecule_id} to {doc_id}",
            source=source,
        )

    def add_hyperedge(
        self,
        edge_id: str,
        node_ids: Iterable[str],
        *,
        relation: str = "member",
        source: str | None = None,
    ) -> None:
        """Insert a hyperedge node connecting ``node_ids``."""

        self.graph.add_hyperedge(
            edge_id,
            node_ids,
            relation=relation,
            source=source,
        )
        self._record_event(
            "add_hyperedge",
            f"Added hyperedge {edge_id}",
            relation=relation,
            source=source,
        )

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
        self._record_event(
            "link_similar_chunks",
            f"Linked similar chunks (k={k})",
            k=k,
        )

    def link_similar_sections(self, k: int = 3) -> None:
        """Create similarity edges between section titles."""

        self.graph.link_similar_sections(k)
        self._record_event(
            "link_similar_sections",
            f"Linked similar sections (k={k})",
            k=k,
        )

    def link_similar_documents(self, k: int = 3) -> None:
        """Create similarity edges between document texts."""

        self.graph.link_similar_documents(k)
        self._record_event(
            "link_similar_documents",
            f"Linked similar documents (k={k})",
            k=k,
        )

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

    @persist_after
    def cleanup_graph(
        self,
        *,
        resolve_threshold: float = 0.8,
        resolve_aliases: dict[str, list[str]] | None = None,
        dedup_similarity: float = 1.0,
        normalize_dates: bool = True,
        mark_conflicts: bool = False,
        validate: bool = False,
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
        if normalize_dates:
            self.normalize_dates()
        self.resolve_entities(threshold=resolve_threshold, aliases=resolve_aliases)
        if mark_conflicts:
            self.mark_conflicting_facts()
        if validate:
            self.validate_coherence()
        self._record_event(
            "cleanup_graph",
            f"Graph cleaned (removed={removed}, cleaned={cleaned})",
            resolve_threshold=resolve_threshold,
            dedup_similarity=dedup_similarity,
            removed=removed,
            cleaned=cleaned,
        )
        self.stage = max(self.stage, DatasetStage.CURATED)
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
        self._record_event("consolidate_schema", "Schema consolidated")

    def detect_communities(self, n_clusters: int = 3) -> None:
        """Cluster chunks into communities."""

        self.graph.cluster_chunks(n_clusters=n_clusters)
        self._record_event(
            "detect_communities",
            "Communities detected",
            n_clusters=n_clusters,
        )

    def detect_entity_groups(self, n_clusters: int = 3) -> None:
        """Cluster entity nodes into groups."""

        self.graph.cluster_entities(n_clusters=n_clusters)
        self._record_event(
            "detect_entity_groups",
            "Entity groups detected",
            n_clusters=n_clusters,
        )

    def summarize_entity_groups(self) -> None:
        self.graph.summarize_entity_groups()
        self._record_event("summarize_entity_groups", "Entity groups summarized")

    def summarize_communities(self) -> None:
        self.graph.summarize_communities()
        self._record_event("summarize_communities", "Communities summarized")

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

    def compute_graphwave_embeddings(self, scales: Iterable[float], num_points: int = 10) -> None:
        """Wrapper for :meth:`KnowledgeGraph.compute_graphwave_embeddings`."""

        self.graph.compute_graphwave_embeddings(scales=scales, num_points=num_points)
        self._record_event(
            "compute_graphwave_embeddings",
            "GraphWave embeddings computed",
            scales=list(scales),
            num_points=num_points,
        )

    def compute_poincare_embeddings(
        self,
        dim: int = 2,
        negative: int = 5,
        epochs: int = 50,
        learning_rate: float = 0.1,
        burn_in: int = 10,
    ) -> None:
        """Wrapper for :meth:`KnowledgeGraph.compute_poincare_embeddings`."""

        self.graph.compute_poincare_embeddings(
            dim=dim,
            negative=negative,
            epochs=epochs,
            learning_rate=learning_rate,
            burn_in=burn_in,
        )
        self._record_event(
            "compute_poincare_embeddings",
            "Poincaré embeddings computed",
            dim=dim,
            negative=negative,
            epochs=epochs,
            learning_rate=learning_rate,
            burn_in=burn_in,
        )

    def compute_graphsage_embeddings(
        self,
        *,
        dimensions: int = 64,
        num_layers: int = 2,
    ) -> None:
        """Wrapper for :meth:`KnowledgeGraph.compute_graphsage_embeddings`."""

        self.graph.compute_graphsage_embeddings(dimensions=dimensions, num_layers=num_layers)
        self._record_event(
            "compute_graphsage_embeddings",
            "GraphSAGE embeddings computed",
            dimensions=dimensions,
            num_layers=num_layers,
        )

    def compute_transe_embeddings(
        self,
        *,
        dimensions: int = 64,
    ) -> None:
        """Wrapper for :meth:`KnowledgeGraph.compute_transe_embeddings`."""

        self.graph.compute_transe_embeddings(dimensions=dimensions)
        self._record_event(
            "compute_transe_embeddings",
            "TransE relation embeddings computed",
            dimensions=dimensions,
        )

    def compute_multigeometric_embeddings(
        self,
        *,
        node2vec_dim: int = 64,
        graphwave_scales: Iterable[float] | None = None,
        graphwave_points: int = 10,
        poincare_dim: int = 2,
        negative: int = 5,
        epochs: int = 50,
        learning_rate: float = 0.1,
        burn_in: int = 10,
    ) -> None:
        """Compute Node2Vec, GraphWave, Poincar\u00e9 and GraphSAGE embeddings."""

        self.compute_graph_embeddings(
            dimensions=node2vec_dim,
            walk_length=10,
            num_walks=50,
            workers=1,
            seed=0,
        )
        self.compute_graphwave_embeddings(
            scales=graphwave_scales or [0.5, 1.0],
            num_points=graphwave_points,
        )
        self.compute_poincare_embeddings(
            dim=poincare_dim,
            negative=min(negative, max(1, len(self.graph.graph.nodes) - 2)),
            epochs=epochs,
            learning_rate=learning_rate,
            burn_in=burn_in,
        )
        self.compute_graphsage_embeddings(dimensions=node2vec_dim, num_layers=2)
        self._record_event(
            "compute_multigeometric_embeddings",
            "Multi-geometry embeddings computed",
            node2vec_dim=node2vec_dim,
            graphwave_scales=list(graphwave_scales or [0.5, 1.0]),
            graphwave_points=graphwave_points,
            poincare_dim=poincare_dim,
            negative=negative,
            epochs=epochs,
            learning_rate=learning_rate,
            burn_in=burn_in,
        )

    def fractal_dimension(self, radii: Iterable[int]) -> tuple[float, list[tuple[int, int]]]:
        """Wrapper for :meth:`KnowledgeGraph.box_counting_dimension`."""

        dim, counts = self.graph.box_counting_dimension(radii)
        self._record_event(
            "fractal_dimension",
            "Fractal dimension computed",
            radii=list(radii),
        )
        return dim, counts

    def compute_fractal_features(self, radii: Iterable[int], *, max_dim: int = 1) -> Dict[str, Any]:
        """Compute fractal metrics and record the event."""

        features = self.graph.compute_fractal_features(radii, max_dim=max_dim)
        self._record_event(
            "compute_fractal_features",
            "Fractal features computed",
            radii=list(radii),
            dimension=features["dimension"],
            radius=features["radius"],
        )
        return features

    def spectral_dimension(self, times: Iterable[float]) -> tuple[float, list[tuple[float, float]]]:
        """Wrapper for :meth:`KnowledgeGraph.spectral_dimension`."""

        dim, traces = self.graph.spectral_dimension(times)
        self._record_event(
            "spectral_dimension",
            "Spectral dimension computed",
            times=list(times),
        )
        return dim, traces

    def spectral_entropy(self, normed: bool = True) -> float:
        """Wrapper for :meth:`KnowledgeGraph.spectral_entropy`."""

        ent = self.graph.spectral_entropy(normed=normed)
        self._record_event(
            "spectral_entropy",
            "Spectral entropy computed",
            normed=normed,
        )
        return ent

    def spectral_gap(self, normed: bool = True) -> float:
        """Wrapper for :meth:`KnowledgeGraph.spectral_gap`."""

        gap = self.graph.spectral_gap(normed=normed)
        self._record_event(
            "spectral_gap",
            "Spectral gap computed",
            normed=normed,
        )
        return gap

    def laplacian_energy(self, normed: bool = True) -> float:
        """Wrapper for :meth:`KnowledgeGraph.laplacian_energy`."""

        energy = self.graph.laplacian_energy(normed=normed)
        self._record_event(
            "laplacian_energy",
            "Laplacian energy computed",
            normed=normed,
        )
        return energy

    def sheaf_laplacian(self, edge_attr: str = "sheaf_sign") -> np.ndarray:
        """Wrapper for :meth:`KnowledgeGraph.sheaf_laplacian`."""

        L = self.graph.sheaf_laplacian(edge_attr=edge_attr)
        self._record_event(
            "sheaf_laplacian",
            "Sheaf Laplacian computed",
            edge_attr=edge_attr,
        )
        return L

    def graph_information_bottleneck(
        self,
        labels: Dict[str, int],
        *,
        beta: float = 1.0,
    ) -> float:
        """Wrapper for :meth:`KnowledgeGraph.graph_information_bottleneck`."""

        loss = self.graph.graph_information_bottleneck(labels, beta=beta)
        self._record_event(
            "graph_information_bottleneck",
            "Information bottleneck computed",
            beta=beta,
        )
        return loss

    def prototype_subgraph(
        self,
        labels: Dict[str, int],
        class_id: int,
        *,
        radius: int = 1,
    ) -> nx.Graph:
        """Wrapper for :meth:`KnowledgeGraph.prototype_subgraph`."""

        sub = self.graph.prototype_subgraph(labels, class_id, radius=radius)
        self._record_event(
            "prototype_subgraph",
            "Prototype subgraph extracted",
            class_id=class_id,
            radius=radius,
        )
        return sub

    def laplacian_spectrum(self, normed: bool = True) -> np.ndarray:
        """Wrapper for :meth:`KnowledgeGraph.laplacian_spectrum`."""

        evals = self.graph.laplacian_spectrum(normed=normed)
        self._record_event(
            "laplacian_spectrum",
            "Laplacian spectrum computed",
            normed=normed,
        )
        return evals

    def spectral_density(
        self, bins: int = 50, *, normed: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Wrapper for :meth:`KnowledgeGraph.spectral_density`."""

        hist, edges = self.graph.spectral_density(bins=bins, normed=normed)
        self._record_event(
            "spectral_density",
            "Spectral density computed",
            bins=bins,
            normed=normed,
        )
        return hist, edges

    def graph_fourier_transform(
        self, signal: Dict[str, float] | np.ndarray, *, normed: bool = True
    ) -> np.ndarray:
        """Wrapper for :meth:`KnowledgeGraph.graph_fourier_transform`."""

        coeffs = self.graph.graph_fourier_transform(signal, normed=normed)
        self._record_event(
            "graph_fourier_transform",
            "Graph Fourier transform computed",
            normed=normed,
        )
        return coeffs

    def inverse_graph_fourier_transform(
        self, coeffs: np.ndarray, *, normed: bool = True
    ) -> np.ndarray:
        """Wrapper for :meth:`KnowledgeGraph.inverse_graph_fourier_transform`."""

        signal = self.graph.inverse_graph_fourier_transform(coeffs, normed=normed)
        self._record_event(
            "inverse_graph_fourier_transform",
            "Inverse graph Fourier transform computed",
            normed=normed,
        )
        return signal

    def persistence_entropy(self, dimension: int = 0) -> float:
        """Wrapper for :meth:`KnowledgeGraph.persistence_entropy`."""

        ent = self.graph.persistence_entropy(dimension)
        self._record_event(
            "persistence_entropy",
            "Persistence entropy computed",
            dimension=dimension,
        )
        return ent

    def persistence_diagrams(self, max_dim: int = 2) -> Dict[int, np.ndarray]:
        """Wrapper for :meth:`KnowledgeGraph.persistence_diagrams`."""

        diags = self.graph.persistence_diagrams(max_dim)
        self._record_event(
            "persistence_diagrams",
            "Persistence diagrams computed",
            max_dim=max_dim,
        )
        return diags

    def topological_signature(self, max_dim: int = 1) -> Dict[str, Any]:
        """Wrapper for :meth:`KnowledgeGraph.topological_signature`."""

        sig = self.graph.topological_signature(max_dim=max_dim)
        self._record_event(
            "topological_signature",
            "Topological signature computed",
            max_dim=max_dim,
        )
        return sig

    def fractalize_level(self, radius: int) -> tuple[nx.Graph, Dict[str, int]]:
        """Wrapper for :meth:`KnowledgeGraph.fractalize_level`."""

        coarse, mapping = self.graph.fractalize_level(radius)
        self._record_event(
            "fractalize_level",
            "Graph coarse-grained via box covering",
            radius=radius,
        )
        return coarse, mapping

    def fractalize_optimal(self, radii: Iterable[int]) -> tuple[nx.Graph, Dict[str, int], int]:
        """Wrapper for :meth:`KnowledgeGraph.fractalize_optimal`."""

        coarse, mapping, radius = self.graph.fractalize_optimal(radii)
        self._record_event(
            "fractalize_optimal",
            "Graph coarse-grained at optimal radius",
            radii=list(radii),
            chosen_radius=radius,
        )
        return coarse, mapping, radius

    def build_fractal_hierarchy(
        self, radii: Iterable[int], *, max_levels: int = 5
    ) -> list[tuple[nx.Graph, Dict[str, int], int]]:
        """Wrapper for :meth:`KnowledgeGraph.build_fractal_hierarchy`."""

        hierarchy = self.graph.build_fractal_hierarchy(radii, max_levels=max_levels)
        self._record_event(
            "build_fractal_hierarchy",
            "Fractal hierarchy constructed",
            radii=list(radii),
            max_levels=max_levels,
        )
        return hierarchy

    def optimize_topology(
        self,
        target: nx.Graph,
        *,
        dimension: int = 1,
        epsilon: float = 0.0,
        max_iter: int = 100,
        seed: int | None = None,
        use_generator: bool = False,
    ) -> float:
        """Wrapper for :meth:`KnowledgeGraph.optimize_topology`."""

        dist = self.graph.optimize_topology(
            target,
            dimension=dimension,
            epsilon=epsilon,
            max_iter=max_iter,
            seed=seed,
            use_generator=use_generator,
        )
        self._record_event(
            "optimize_topology",
            "Topology adjusted via bottleneck minimization",
            dimension=dimension,
            epsilon=epsilon,
            max_iter=max_iter,
            use_generator=use_generator,
        )
        return dist

    def apply_perception(
        self,
        node_id: str,
        new_text: str,
        *,
        perception_id: str | None = None,
        strength: float | None = None,
    ) -> None:
        """Wrapper for :meth:`KnowledgeGraph.apply_perception`."""

        self.graph.apply_perception(
            node_id,
            new_text,
            perception_id=perception_id,
            strength=strength,
        )
        self._record_event(
            "apply_perception",
            "Node perception applied",
            node_id=node_id,
            perception_id=perception_id,
            strength=strength,
        )

    def gds_quality_check(
        self,
        *,
        min_component_size: int = 2,
        similarity_threshold: float = 0.95,
        driver: Driver | None = None,
    ) -> Dict[str, Any]:
        """Run Neo4j GDS quality checks via :class:`KnowledgeGraph`."""

        driver = driver or self.neo4j_driver
        if not driver:
            raise ValueError("Neo4j driver required")

        result = self.graph.gds_quality_check(
            driver,
            dataset=self.name,
            min_component_size=min_component_size,
            similarity_threshold=similarity_threshold,
        )
        self._record_event(
            "gds_quality_check",
            "Neo4j GDS quality check performed",
            min_component_size=min_component_size,
            similarity_threshold=similarity_threshold,
        )
        return result

    def update_embeddings(self, node_type: str = "chunk") -> None:
        """Materialize embeddings for nodes of ``node_type``."""

        self.graph.update_embeddings(node_type=node_type)
        self._record_event("update_embeddings", "Embeddings materialized", node_type=node_type)

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
        self._record_event("extract_facts", "Facts extracted")

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

    def get_atoms_for_document(self, doc_id: str) -> list[str]:
        return self.graph.get_atoms_for_document(doc_id)

    def get_molecules_for_document(self, doc_id: str) -> list[str]:
        return self.graph.get_molecules_for_document(doc_id)

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
        if name is not None:
            self.validate_name(name)
        clone = DatasetBuilder(self.dataset_type, name=name, graph=deepcopy(self.graph))
        clone.owner_id = self.owner_id
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
            "owner_id": self.owner_id,
            "created_at": self.created_at.isoformat(),
            "accessed_at": self.accessed_at.isoformat(),
            "history": self.history,
            "versions": self.versions,
            "ingested_docs": self.ingested_docs,
            "events": [{**asdict(e), "timestamp": e.timestamp.isoformat()} for e in self.events],
            "graph": self.graph.to_dict(),
            "stage": int(self.stage),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetBuilder":
        name = data.get("name")
        if name is not None:
            cls.validate_name(name)
        ds = cls(DatasetType(data["dataset_type"]), name=name)
        if "id" in data:
            ds.id = data["id"]
        if ts := data.get("created_at"):
            ds.created_at = datetime.fromisoformat(ts)
        if ts := data.get("accessed_at"):
            ds.accessed_at = datetime.fromisoformat(ts)
        ds.owner_id = data.get("owner_id")
        ds.history = list(data.get("history", []))
        ds.versions = list(data.get("versions", []))
        ds.ingested_docs = dict(data.get("ingested_docs", {}))
        ds.events = [
            HistoryEvent(
                e.get("operation"),
                e.get("message"),
                timestamp=(
                    datetime.fromisoformat(e["timestamp"])
                    if "timestamp" in e
                    else datetime.now(timezone.utc)
                ),
                params=e.get("params"),
            )
            for e in data.get("events", [])
        ]
        ds.graph = KnowledgeGraph.from_dict(data.get("graph", {}))
        ds.stage = DatasetStage(int(data.get("stage", 0)))
        return ds

    # ------------------------------------------------------------------
    # Redis helpers
    # ------------------------------------------------------------------

    def to_redis(self, client: redis.Redis | redis.client.Pipeline, key: str | None = None) -> str:
        """Persist the dataset in Redis under ``key``."""

        key = key or (self.name or "dataset")
        self.redis_key = key
        if self.name:
            self.validate_name(self.name)
        pipe = client.pipeline() if not isinstance(client, redis.client.Pipeline) else client
        pipe.set(key, json.dumps(self.to_dict()))
        events_key = f"{key}:events"
        pipe.delete(events_key)
        if self.events:
            pipe.rpush(
                events_key,
                *[
                    json.dumps(
                        {
                            "operation": ev.operation,
                            "message": ev.message,
                            "timestamp": ev.timestamp.isoformat(),
                            "params": ev.params,
                        }
                    )
                    for ev in self.events
                ],
            )
        if pipe is not client:
            pipe.execute()
        return key

    @classmethod
    def from_redis(
        cls, client: redis.Redis | None, key: str, driver: Driver | None = None
    ) -> "DatasetBuilder":
        if client is None:
            raise ValueError("Redis client required")
        data = client.get(key)
        if data is None:
            raise KeyError(key)
        ds = cls.from_dict(json.loads(data))
        if ds.name:
            cls.validate_name(ds.name)
        ds.redis_client = client
        ds.neo4j_driver = driver
        ds.redis_key = key
        require = os.getenv("DATACREEK_REQUIRE_PERSISTENCE", "1") != "0"
        if require and driver is None:
            raise ValueError("Neo4j driver required")
        graph = get_redis_graph(ds.name)
        if graph is not None:
            try:
                DatasetBuilder.load_redis_graph.__wrapped__(ds, graph)
            except Exception:
                logger.exception("Failed to load graph %s from RedisGraph", ds.name)
        events_key = f"{key}:events"
        raw_events = client.lrange(events_key, 0, -1)
        if raw_events:
            ds.events = []
            for e in raw_events:
                ev = json.loads(e)
                ts = ev.get("timestamp")
                if ts:
                    ev["timestamp"] = datetime.fromisoformat(ts)
                ds.events.append(HistoryEvent(**ev))
            ds.history = [ev.message for ev in ds.events]
        ds._touch()
        return ds

    @persist_after
    def ingest_file(
        self,
        path: str,
        doc_id: str | None = None,
        *,
        config: dict[str, Any] | None = None,
        high_res: bool = False,
        ocr: bool = False,
        use_unstructured: bool | None = None,
        extract_entities: bool = False,
        extract_facts: bool = False,
        client: "LLMClient" | None = None,
        options: "IngestOptions" | None = None,
        progress_callback: Callable[[int], None] | None = None,
    ) -> str:
        """Parse ``path`` and ingest its content into the dataset."""

        from .ingest import ingest_into_dataset

        if options is not None:
            config = options.config
            high_res = options.high_res
            ocr = options.ocr
            use_unstructured = options.use_unstructured
            extract_entities = options.extract_entities
            extract_facts = options.extract_facts

        doc = ingest_into_dataset(
            path,
            self,
            doc_id=doc_id,
            config=config,
            high_res=high_res,
            ocr=ocr,
            use_unstructured=use_unstructured,
            extract_entities=extract_entities,
            extract_facts=extract_facts,
            client=client,
            options=None,
            progress_callback=progress_callback,
        )
        self.ingested_docs[doc] = {
            "path": path,
            "config": config,
            "high_res": high_res,
            "ocr": ocr,
            "use_unstructured": use_unstructured,
            "extract_entities": extract_entities,
            "extract_facts": extract_facts,
        }
        self.stage = max(self.stage, DatasetStage.INGESTED)
        self._record_event(
            "ingest_document",
            f"Ingested {doc}",
            path=path,
            high_res=high_res,
            ocr=ocr,
            use_unstructured=use_unstructured,
            extract_entities=extract_entities,
            extract_facts=extract_facts,
        )
        return doc

    async def ingest_file_async(
        self,
        path: str,
        doc_id: str | None = None,
        *,
        config: dict[str, Any] | None = None,
        high_res: bool = False,
        ocr: bool = False,
        use_unstructured: bool | None = None,
        extract_entities: bool = False,
        extract_facts: bool = False,
        client: "LLMClient" | None = None,
        options: "IngestOptions" | None = None,
        progress_callback: Callable[[int], None] | None = None,
    ) -> str:
        """Asynchronous counterpart to :meth:`ingest_file`."""

        from .ingest import ingest_into_dataset_async

        if options is not None:
            config = options.config
            high_res = options.high_res
            ocr = options.ocr
            use_unstructured = options.use_unstructured
            extract_entities = options.extract_entities
            extract_facts = options.extract_facts

        doc = await ingest_into_dataset_async(
            path,
            self,
            doc_id=doc_id,
            config=config,
            high_res=high_res,
            ocr=ocr,
            use_unstructured=use_unstructured,
            extract_entities=extract_entities,
            extract_facts=extract_facts,
            client=client,
            options=None,
            progress_callback=progress_callback,
        )
        self.ingested_docs[doc] = {
            "path": path,
            "config": config,
            "high_res": high_res,
            "ocr": ocr,
            "use_unstructured": use_unstructured,
            "extract_entities": extract_entities,
            "extract_facts": extract_facts,
        }
        self.stage = max(self.stage, DatasetStage.INGESTED)
        self._record_event(
            "ingest_document",
            f"Ingested {doc}",
            path=path,
            high_res=high_res,
            ocr=ocr,
            use_unstructured=use_unstructured,
            extract_entities=extract_entities,
            extract_facts=extract_facts,
        )
        self._persist()
        return doc

    # ------------------------------------------------------------------
    # Generation helpers
    # ------------------------------------------------------------------

    def get_raw_text(self) -> str:
        """Return text representation of the underlying graph."""

        return self.graph.to_text()

    @persist_after
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
        pipeline_config_path: Path | None = None,
        dedup_similarity: float = 1.0,
        keep_ratings: bool = False,
        redis_client: redis.Redis | None = None,
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

        if redis_client is None:
            redis_client = self.redis_client

        if async_mode:
            result = asyncio.run(
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
                    curation_threshold=threshold,
                    fmt=fmt,
                    overrides=overrides,
                    verbose=verbose,
                    async_mode=True,
                    multi_answer=multi_answer,
                    batch_size=batch_size,
                    inference_batch=inference_batch,
                    start_step=start_step,
                    pipeline_config_path=pipeline_config_path,
                    dedup_similarity=dedup_similarity,
                    keep_ratings=keep_ratings,
                    redis_client=redis_client,
                )
            )
        else:
            result = run_generation_pipeline(
                self.dataset_type,
                self.graph,
                dataset_builder=self,
                config_path=config_path,
                provider=provider,
                profile=profile,
                api_base=api_base,
                model=model,
                num_pairs=num_pairs,
                curation_threshold=threshold,
                fmt=fmt,
                overrides=overrides,
                verbose=verbose,
                async_mode=False,
                multi_answer=multi_answer,
                batch_size=batch_size,
                inference_batch=inference_batch,
                start_step=start_step,
                pipeline_config_path=pipeline_config_path,
                dedup_similarity=dedup_similarity,
                keep_ratings=keep_ratings,
                redis_client=redis_client,
            )

        params = {
            "provider": provider,
            "profile": profile,
            "api_base": api_base,
            "model": model,
            "num_pairs": num_pairs,
            "threshold": threshold,
            "fmt": fmt,
            "multi_answer": multi_answer,
            "batch_size": batch_size,
            "inference_batch": inference_batch,
            "start_step": start_step.value if start_step else None,
            "pipeline_config_path": str(pipeline_config_path) if pipeline_config_path else None,
            "dedup_similarity": dedup_similarity,
            "keep_ratings": keep_ratings,
        }

        res_data = (
            asdict(result) if is_dataclass(result) else result if isinstance(result, dict) else None
        )
        self.versions.append(
            {"params": params, "time": datetime.now(timezone.utc).isoformat(), "result": res_data}
        )
        self.stage = max(self.stage, DatasetStage.GENERATED)
        self._record_event("generate", f"Post-KG pipeline run (v{len(self.versions)})")

        env_limit = os.getenv("DATASET_MAX_VERSIONS")
        if env_limit:
            try:
                limit_val = int(env_limit)
            except Exception:
                limit_val = None
            if limit_val and limit_val > 0:
                try:
                    self.prune_versions(limit_val)
                except Exception:
                    logger.exception("Failed to prune versions for %s", self.name)

        return result

    @persist_after
    def mark_exported(self) -> None:
        """Update the dataset stage and history after export."""

        self.stage = max(self.stage, DatasetStage.EXPORTED)
        self._record_event("export_dataset", "Dataset exported")

    def export_prompts(self) -> List[Dict[str, Any]]:
        """Return prompt records with fractal and perception metadata."""

        signature = self.graph.topological_signature(max_dim=1)
        data: List[Dict[str, Any]] = []
        for node, attrs in self.graph.graph.nodes(data=True):
            if attrs.get("type") != "chunk":
                continue
            record = {
                "prompt": attrs.get("text", ""),
                "fractal_level": attrs.get("fractal_level"),
                "perception_id": attrs.get("perception_id"),
                "topo_signature": signature,
            }
            data.append(record)
        self._record_event("export_prompts", "Prompt data exported")
        return data

    @persist_after
    def delete_version(self, index: int) -> None:
        """Remove a stored generation version."""

        if index < 1 or index > len(self.versions):
            raise IndexError("version index out of range")
        removed = self.versions.pop(index - 1)
        self._record_event(
            "delete_version",
            f"Removed version {index}",
            params={"version": removed.get("time")},
        )

    @persist_after
    def restore_version(self, index: int) -> None:
        """Restore ``index`` as the latest generation result."""

        if index < 1 or index > len(self.versions):
            raise IndexError("version index out of range")
        restored = deepcopy(self.versions[index - 1])
        self.versions.append(restored)
        self.stage = max(self.stage, DatasetStage.GENERATED)
        self._record_event(
            "restore_version",
            f"Restored version {index}",
            version=restored.get("time"),
        )

    @persist_after
    def prune_versions(self, limit: int | None = None) -> int:
        """Remove oldest versions beyond ``limit``.

        The default ``limit`` is read from the ``DATASET_MAX_VERSIONS``
        environment variable if not provided. Returns the number of removed
        versions.
        """

        if limit is None:
            env = os.getenv("DATASET_MAX_VERSIONS")
            try:
                limit = int(env) if env else None
            except Exception:
                limit = None
        if not limit or limit < 1:
            return 0
        removed = 0
        while len(self.versions) > limit:
            self.versions.pop(0)
            removed += 1
        if removed:
            self._record_event(
                "prune_versions",
                f"Pruned to {limit} versions",
                removed=removed,
            )
        return removed

    # ------------------------------------------------------------------
    # Neo4j helpers
    # ------------------------------------------------------------------

    @persist_after
    def save_neo4j(self, driver: Driver | None = None) -> None:
        """Persist the knowledge graph to Neo4j."""

        driver = driver or self.neo4j_driver
        if not driver:
            raise ValueError("Neo4j driver required")
        self.graph.to_neo4j(driver, dataset=self.name)
        self._record_event("save_neo4j", "Graph saved to Neo4j")

    @persist_after
    def load_neo4j(self, driver: Driver | None = None) -> None:
        """Load the knowledge graph from Neo4j."""

        driver = driver or self.neo4j_driver
        if not driver:
            raise ValueError("Neo4j driver required")
        self.graph = self.graph.__class__.from_neo4j(driver, dataset=self.name)
        self._record_event("load_neo4j", "Graph loaded from Neo4j")

    # ------------------------------------------------------------------
    # RedisGraph helpers
    # ------------------------------------------------------------------

    @persist_after
    def save_redis_graph(self, graph: RGGraph | None = None) -> None:
        """Persist the knowledge graph to RedisGraph."""

        graph = graph or get_redis_graph(self.name)
        if graph is None:
            raise ValueError("RedisGraph not configured")
        try:
            graph.query("MATCH (n {dataset:$ds}) DETACH DELETE n", {"ds": self.name})
        except Exception:
            pass
        for node_id, attrs in self.graph.graph.nodes(data=True):
            label = attrs.get("type", "Node")
            props = {k: v for k, v in attrs.items() if k != "type"}
            props_str = ", ".join(f"{k}: ${k}_{node_id}" for k in props)
            params = {f"{k}_{node_id}": v for k, v in props.items()}
            params.update({"id": node_id, "ds": self.name})
            q = f"CREATE (:{label} {{id:$id, dataset:$ds{', ' + props_str if props_str else ''}}})"
            graph.query(q, params)
        for src, dst, attrs in self.graph.graph.edges(data=True):
            relation = attrs.get("relation", "REL")
            props = {k: v for k, v in attrs.items() if k != "relation"}
            props_str = ", ".join(f"{k}: ${k}_{src}_{dst}" for k in props)
            params = {f"{k}_{src}_{dst}": v for k, v in props.items()}
            params.update({"src": src, "dst": dst, "ds": self.name})
            q = (
                f"MATCH (a {{id:$src, dataset:$ds}}), (b {{id:$dst, dataset:$ds}}) "
                f"CREATE (a)-[:{relation} {{dataset:$ds{', ' + props_str if props_str else ''}}}]->(b)"
            )
            graph.query(q, params)
        self._record_event("save_redis_graph", "Graph saved to RedisGraph")

    @persist_after
    def load_redis_graph(self, graph: RGGraph | None = None) -> None:
        """Load the knowledge graph from RedisGraph."""

        graph = graph or get_redis_graph(self.name)
        if graph is None:
            raise ValueError("RedisGraph not configured")
        self.graph.graph.clear()
        res_nodes = graph.query(
            "MATCH (n {dataset:$ds}) RETURN n.id, labels(n)[0], properties(n)",
            {"ds": self.name},
        )
        for nid, label, props in res_nodes.result_set:
            if isinstance(props, dict):
                props.pop("dataset", None)
            self.graph.graph.add_node(nid, type=label, **props)
        res_edges = graph.query(
            "MATCH (a {dataset:$ds})-[r]->(b {dataset:$ds}) RETURN a.id, b.id, type(r), properties(r)",
            {"ds": self.name},
        )
        for src, dst, rel, props in res_edges.result_set:
            if isinstance(props, dict):
                props.pop("dataset", None)
            self.graph.graph.add_edge(src, dst, relation=rel, **props)
        self._record_event("load_redis_graph", "Graph loaded from RedisGraph")

    def delete_redis_graph(self, graph: RGGraph | None = None) -> None:
        """Remove the dataset's graph from RedisGraph if configured."""

        graph = graph or get_redis_graph(self.name)
        if graph is None:
            return
        try:
            graph.query("MATCH (n {dataset:$ds}) DETACH DELETE n", {"ds": self.name})
        except Exception:
            logger.exception("Failed to delete graph %s from RedisGraph", self.name)
