"""Hybrid ANN retrieval combining HNSW and IVFPQ."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from datacreek.backend import get_xp

xp = get_xp()

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover - faiss optional
    faiss = None  # type: ignore

__all__ = ["rerank_pq", "search_hnsw_pq"]


def expected_recall(nprobe: int, n_cells: int, tables: int) -> float:
    """Return approximate recall with multi-probing.

    Parameters
    ----------
    nprobe:
        Number of coarse centroids probed per table.
    n_cells:
        Total number of centroids in the IVF index.
    tables:
        Number of product quantisation tables.

    Notes
    -----
    The estimate follows

    .. math::

       1 - \left(1 - \frac{n_{\text{probe}}}{N_{\text{cells}}}\right)^{L}

    where ``L`` is ``tables``.
    """

    if n_cells == 0:
        return 0.0
    return 1.0 - (1.0 - nprobe / n_cells) ** tables


def choose_nprobe_multi(n_cells: int, tables: int, *, target: float = 0.9) -> int:
    """Return ``nprobe_multi`` achieving at least ``target`` recall."""

    if n_cells <= 0 or tables <= 0:
        return 1
    value = n_cells * (1.0 - (1.0 - target) ** (1.0 / tables))
    return max(1, int(math.ceil(value)))


def rerank_pq(
    xb: np.ndarray,
    xq: np.ndarray,
    *,
    k: int = 10,
    gpu: bool = True,
    n_subprobe: int = 1,
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
    n_subprobe:
        Number of additional FAISS multi-probes. ``nprobe`` becomes
        ``base * n_subprobe`` where ``base`` is ``max(1, nlist // 4)``.

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
        nlist = max(4, int(xp.sqrt(xb.shape[0])))
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
        base = max(1, nlist // 4)
        index.nprobe = base * max(1, int(n_subprobe))
        if not gpu and hasattr(index, "nprobe_multi"):
            index.nprobe_multi = choose_nprobe_multi(nlist, index.pq.M)

    _, rerank = index.search(xq, k)
    return rerank


def search_hnsw_pq(
    xb: np.ndarray,
    xq: np.ndarray,
    *,
    k: int = 10,
    prefetch: int = 50,
    n_subprobe: int = 1,
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
    n_subprobe:
        Number of sub-probes used during PQ reranking.

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

    rerank = rerank_pq(
        xb[candidates],
        xq,
        k=k,
        gpu=True,
        n_subprobe=n_subprobe,
    )
    return [int(candidates[i]) for i in rerank[0]]
