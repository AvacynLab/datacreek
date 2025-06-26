from __future__ import annotations

import json
import os
import secrets
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import redis

from ..pipelines import DatasetType
from .knowledge_graph import KnowledgeGraph


@dataclass
class DatasetBuilder:
    """Manage a dataset under construction with its own knowledge graph."""

    dataset_type: DatasetType
    id: str = field(default_factory=lambda: secrets.token_hex(8))
    name: Optional[str] = None
    graph: KnowledgeGraph = field(default_factory=KnowledgeGraph)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    history: List[str] = field(default_factory=list)
    versions: List[Dict[str, Any]] = field(default_factory=list)
    stage: int = 0  # 0=created, 1=ingest, 2=generation, 3=curation, 4=exported

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

    def search_embeddings(self, query: str, k: int = 3, fetch_neighbors: bool = True) -> list[str]:
        """Wrapper for :meth:`KnowledgeGraph.search_embeddings`."""

        return self.graph.search_embeddings(query, k=k, fetch_neighbors=fetch_neighbors)

    def search_hybrid(self, query: str, k: int = 5) -> list[str]:
        """Wrapper for :meth:`KnowledgeGraph.search_hybrid`."""

        return self.graph.search_hybrid(query, k=k)

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

    def get_chunks_for_document(self, doc_id: str) -> list[str]:
        return self.graph.get_chunks_for_document(doc_id)

    def remove_chunk(self, chunk_id: str) -> None:
        """Remove a chunk node from the dataset graph."""
        if self.graph.graph.has_node(chunk_id):
            preds = list(self.graph.graph.predecessors(chunk_id))
            self.graph.remove_chunk(chunk_id)
            if preds:
                self.history.append(f"Removed chunk {chunk_id} from {preds[0]}")

    def remove_document(self, doc_id: str) -> None:
        """Remove a document and all its chunks."""
        if not self.graph.graph.has_node(doc_id):
            return
        self.graph.remove_document(doc_id)
        self.history.append(f"Removed document {doc_id}")

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
