"""Hybrid ANN retrieval combining HNSW and IVFPQ."""

from __future__ import annotations

from typing import Sequence

import numpy as np

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover - faiss optional
    faiss = None  # type: ignore

__all__ = ["rerank_pq", "search_hnsw_pq"]


def rerank_pq(
    xb: np.ndarray,
    xq: np.ndarray,
    *,
    k: int = 10,
    gpu: bool = True,
) -> np.ndarray:
    """Return ``k`` nearest indices using IVFPQ on CPU or GPU.

    Parameters
    ----------
    xb:
        Database vectors of shape ``(n, d)`` used for training and search.
    xq:
        Query vectors of shape ``(m, d)``.
    k:
        Number of neighbours to return.
    gpu:
        Whether to run the search on GPU when available.

    Returns
    -------
    ``np.ndarray``
        Indices of size ``(m, k)`` referring to ``xb``.
    """
    if faiss is None:
        raise RuntimeError("faiss not installed")

    d = xb.shape[1]
    quantizer = faiss.IndexFlatIP(d)

    if xb.shape[0] < 256:
        index = faiss.IndexFlatIP(d)
        index.add(xb)
    else:
        nlist = max(4, int(np.sqrt(xb.shape[0])))
        nbits = 8 if xb.shape[0] >= 1000 else 6
        index = faiss.IndexIVFPQ(quantizer, d, nlist, 8, int(nbits))

        if (
            gpu
            and getattr(faiss, "StandardGpuResources", None)
            and faiss.get_num_gpus() > 0
        ):
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)

        index.train(xb)
        index.add(xb)
        index.nprobe = max(1, nlist // 4)

    _, rerank = index.search(xq, k)
    return rerank


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

    rerank = rerank_pq(xb[candidates], xq, k=k, gpu=True)
    return [int(candidates[i]) for i in rerank[0]]
