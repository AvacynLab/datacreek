"""ANN indexing utilities with HNSW fallback and recall tracking."""

from __future__ import annotations

import time
from typing import Iterable, Sequence, Tuple

try:
    import faiss  # type: ignore
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    np = None  # type: ignore
    faiss = None  # type: ignore

from .autotune import recall_at_k


def search_with_fallback(
    xb: "np.ndarray",
    xq: "np.ndarray",
    k: int = 10,
    *,
    index: object | None = None,
    latency_threshold: float = 0.1,
) -> Tuple[Sequence[int], float, object]:
    """Return nearest neighbors using FAISS with optional HNSW fallback."""
    if faiss is None or np is None:
        raise RuntimeError("faiss not installed")
    if index is None:
        index = faiss.IndexFlatIP(xb.shape[1])
        index.add(xb)
    start = time.monotonic()
    _, idx = index.search(xq, k)
    latency = time.monotonic() - start
    if latency > latency_threshold and not isinstance(index, faiss.IndexHNSWFlat):
        hnsw = faiss.IndexHNSWFlat(xb.shape[1], 32)
        hnsw.hnsw.efSearch = 200
        hnsw.add(xb)
        index = hnsw
        start = time.monotonic()
        _, idx = index.search(xq, k)
        latency = time.monotonic() - start
    return idx[0], latency, index


def recall10(
    graph,
    queries: Iterable[object],
    ground_truth: dict,
    *,
    gamma: float = 0.5,
    eta: float = 0.25,
) -> float:
    """Compute recall@10 and store it in ``graph.graph['recall10']``."""
    score = recall_at_k(graph, list(queries), ground_truth, k=10, gamma=gamma, eta=eta)
    if hasattr(graph, "graph"):
        graph.graph["recall10"] = score
    else:
        graph["recall10"] = score
    return score
