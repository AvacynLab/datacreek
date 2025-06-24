"""Embedding-based retrieval utilities using TF-IDF."""

from __future__ import annotations

from typing import List, Dict, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import numpy as np


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

    def build(self) -> None:
        if not self.texts:
            return
        self._vectorizer = TfidfVectorizer().fit(self.texts)
        self._matrix = self._vectorizer.transform(self.texts)
        self._nn = NearestNeighbors(metric="cosine")
        self._nn.fit(self._matrix)

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
