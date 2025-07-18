"""GPU-accelerated GraphWave utilities."""

from __future__ import annotations

import networkx as nx
import numpy as np

try:  # pragma: no cover - optional dependency
    import cupy as cp
    import cupyx.scipy.sparse as csp
    from cupyx.scipy.sparse import csr_matrix as csr_gpu
except Exception:  # pragma: no cover - cupy not installed
    cp = None  # type: ignore
    csp = None  # type: ignore
    csr_gpu = None  # type: ignore

__all__ = ["chebyshev_heat_kernel_gpu", "graphwave_embedding_gpu"]


def chebyshev_heat_kernel_gpu(
    L: np.ndarray | "csp.spmatrix", t: float, m: int = 7
) -> np.ndarray:
    """Return GPU approximation of ``exp(-t L)`` using Chebyshev polynomials.

    Parameters
    ----------
    L:
        Laplacian matrix (CSR) on CPU or GPU.
    t:
        Diffusion scale.
    m:
        Order of the approximation.

    Returns
    -------
    np.ndarray
        Heat kernel matrix on the host.
    """
    if cp is None:
        raise RuntimeError("cupy is required for GPU GraphWave")

    arr = L
    if not isinstance(arr, csr_gpu):
        arr = csp.csr_matrix(L)

    n = arr.shape[0]
    # Estimate lmax with power iteration on GPU
    x = cp.random.rand(n, dtype=cp.float32)
    x /= cp.linalg.norm(x)
    for _ in range(5):
        x = arr @ x
        norm = cp.linalg.norm(x)
        if norm == 0:
            break
        x /= norm
    lmax = cp.float32(x.T @ (arr @ x))
    if float(lmax) == 0.0:
        lmax = cp.float32(1.0)

    L_norm = (2.0 / lmax) * arr - csp.identity(n, format="csr", dtype=cp.float32)

    from cupyx.scipy.special import iv

    ak = [2 * iv(k, t * lmax / 2.0) for k in range(m + 1)]
    a0, a1 = ak[0], ak[1]
    Tk_minus = csp.identity(n, format="csr", dtype=cp.float32)
    Tk = L_norm
    psi = a0 * Tk_minus + a1 * Tk
    for k in range(2, m + 1):
        Tk_plus = 2 * L_norm.dot(Tk) - Tk_minus
        psi = psi + ak[k] * Tk_plus
        Tk_minus, Tk = Tk, Tk_plus

    psi = cp.exp(-t * lmax / 2.0, dtype=cp.float32) * psi
    return cp.asnumpy(psi.toarray())


def graphwave_embedding_gpu(
    graph: nx.Graph,
    scales: Iterable[float],
    *,
    num_points: int = 10,
    order: int = 7,
) -> dict[int, np.ndarray]:
    """Return GraphWave embeddings using :mod:`cupy` for acceleration."""
    if cp is None:
        raise RuntimeError("cupy is required for GPU GraphWave")

    nodes = list(graph.nodes())
    lap = nx.to_scipy_sparse_array(graph, format="csr")
    ts = np.linspace(0, 2 * np.pi, num_points)
    emb: dict[int, list[float]] = {n: [] for n in nodes}

    for s in scales:
        heat = chebyshev_heat_kernel_gpu(lap, s, m=order)
        for idx, node in enumerate(nodes):
            coeffs = np.exp(1j * np.outer(ts, heat[idx, :])).sum(axis=1)
            emb[node].extend(coeffs.real)
            emb[node].extend(coeffs.imag)

    return {n: np.asarray(v, dtype=float) for n, v in emb.items()}
