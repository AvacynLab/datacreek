"""Embedding-based retrieval utilities using TF-IDF."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


class EmbeddingIndex:
    """Maintain a simple in-memory retrieval index."""

    def __init__(self) -> None:
        self.texts: List[str] = []
        self.ids: List[str] = []
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._matrix: Optional[np.ndarray] = None
        self._nn: Optional[NearestNeighbors] = None

    def add(self, chunk_id: str, text: str) -> None:
        self.ids.append(chunk_id)
        self.texts.append(text)

    def remove(self, chunk_id: str) -> None:
        """Remove a chunk from the index."""
        if chunk_id not in self.ids:
            return
        idx = self.ids.index(chunk_id)
        del self.ids[idx]
        del self.texts[idx]
        self._vectorizer = None
        self._matrix = None
        self._nn = None

    def build(self) -> None:
        if not self.texts:
            return
        self._vectorizer = TfidfVectorizer().fit(self.texts)
        self._matrix = self._vectorizer.transform(self.texts)
        self._nn = NearestNeighbors(metric="cosine")
        self._nn.fit(self._matrix)

    def _ensure_index(self) -> None:
        """Build the index if it has not been constructed yet."""
        if self._vectorizer is None or self._matrix is None or self._nn is None:
            self.build()

    def nearest_neighbors(
        self, k: int = 3, return_distances: bool = False
    ) -> Dict[str, List[tuple[str, float]] | List[str]]:
        """Return the ``k`` nearest neighbors for each text ID.

        Parameters
        ----------
        k:
            Number of neighbors to return.
        return_distances:
            If ``True``, return a list of ``(id, similarity)`` tuples for each
            chunk. Similarity is ``1 - cosine_distance``.
        """

        self._ensure_index()
        if not self.texts or self._matrix is None:
            return {}

        distances, indices = self._nn.kneighbors(
            self._matrix, n_neighbors=min(k + 1, len(self.ids))
        )
        neighbors: Dict[str, List[tuple[str, float]] | List[str]] = {}
        for idx, (dist_row, neigh_row) in enumerate(zip(distances, indices)):
            items: List[tuple[str, float]] | List[str] = []
            for dist, i in zip(dist_row, neigh_row):
                if i == idx:
                    continue
                if len(items) >= k:
                    break
                if return_distances:
                    items.append((self.ids[i], 1 - float(dist)))
                else:
                    items.append(self.ids[i])
            neighbors[self.ids[idx]] = items
        return neighbors

    def search(self, query: str, k: int = 5) -> List[int]:
        if not self.texts:
            return []
        if self._vectorizer is None:
            self.build()
        query_vec = self._vectorizer.transform([query])
        distances, indices = self._nn.kneighbors(query_vec, n_neighbors=min(k, len(self.ids)))
        return indices[0].tolist()

    def get_text(self, idx: int) -> str:
        return self.texts[idx]

    def get_id(self, idx: int) -> str:
        return self.ids[idx]
