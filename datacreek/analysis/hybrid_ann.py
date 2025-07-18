"""Hybrid ANN retrieval combining HNSW and IVFPQ."""

from __future__ import annotations

from typing import Sequence

import numpy as np

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover - faiss optional
    faiss = None  # type: ignore

__all__ = ["search_hnsw_pq"]


def search_hnsw_pq(
    xb: np.ndarray,
    xq: np.ndarray,
    *,
    k: int = 10,
    prefetch: int = 50,
) -> Sequence[int]:
    """Return ``k`` neighbours using an HNSW stage then PQ reranking.

    Parameters
    ----------
    xb:
        Database vectors of shape ``(n, d)``.
    xq:
        Query vectors of shape ``(1, d)``.
    k:
        Number of neighbours to return.
    prefetch:
        Size of candidate set retrieved with HNSW before reranking.

    Returns
    -------
    Sequence[int]
        Indices of the nearest neighbours.
    """
    if faiss is None:
        raise RuntimeError("faiss not installed")

    d = xb.shape[1]
    hnsw = faiss.IndexHNSWFlat(d, 32)
    hnsw.hnsw.efSearch = max(prefetch * 2, 64)
    hnsw.add(xb)
    _, idx = hnsw.search(xq, prefetch)
    candidates = idx[0]

    quantizer = faiss.IndexFlatIP(d)
    pq = faiss.IndexIVFPQ(quantizer, d, 1, 8, 8)
    pq.train(xb[candidates])
    pq.add(xb[candidates])
    pq.nprobe = 1
    _, rerank = pq.search(xq, k)

    return [int(candidates[i]) for i in rerank[0]]
