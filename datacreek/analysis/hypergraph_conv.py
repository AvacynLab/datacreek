"""Spectral convolutions on hypergraphs."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def hypergraph_laplacian(B: np.ndarray, w: Iterable[float] | None = None) -> np.ndarray:
    """Return the normalized hypergraph Laplacian.

    Parameters
    ----------
    B:
        Incidence matrix of shape ``(num_nodes, num_edges)`` where ``B[v, e] = 1``
        if vertex ``v`` is incident to hyperedge ``e``.
    w:
        Optional iterable of edge weights. Defaults to ``1`` for all edges.

    Returns
    -------
    np.ndarray
        Normalized Laplacian matrix ``Delta``.
    """
    B = np.asarray(B, dtype=float)
    num_nodes, num_edges = B.shape
    if w is None:
        w = np.ones(num_edges, dtype=float)
    w = np.asarray(list(w), dtype=float)
    if w.shape[0] != num_edges:
        raise ValueError("weight length must match number of edges")

    dv = B @ w
    de = B.sum(axis=0)

    dv_inv_sqrt = np.diag(1.0 / np.sqrt(dv + 1e-12))
    de_inv = np.diag(1.0 / (de + 1e-12))
    W = np.diag(w)

    return (
        np.eye(num_nodes)
        - dv_inv_sqrt @ B @ W @ de_inv @ B.T @ dv_inv_sqrt
    )


def chebyshev_conv(X: np.ndarray, Delta: np.ndarray, K: int, theta: Iterable[float] | None = None) -> np.ndarray:
    """Return Chebyshev spectral convolution on hypergraph features.

    Parameters
    ----------
    X:
        Node feature matrix of shape ``(num_nodes, feat_dim)``.
    Delta:
        Normalized hypergraph Laplacian.
    K:
        Order of the Chebyshev approximation.
    theta:
        Iterable of ``K`` filter coefficients. Defaults to ones.

    Returns
    -------
    np.ndarray
        Filtered features with the same shape as ``X``.
    """
    X = np.asarray(X, dtype=float)
    Delta = np.asarray(Delta, dtype=float)
    if theta is None:
        theta = np.ones(K, dtype=float)
    theta = np.asarray(list(theta), dtype=float)
    if theta.shape[0] != K:
        raise ValueError("theta length must equal K")

    lamb_max = float(np.linalg.eigvalsh(Delta).max())
    if lamb_max == 0.0:
        lamb_max = 1.0
    Delta_tilde = (2.0 / lamb_max) * Delta - np.eye(Delta.shape[0])

    T_k_minus = X
    out = theta[0] * T_k_minus
    if K > 1:
        T_k = Delta_tilde @ X
        out = out + theta[1] * T_k
    else:
        return out
    for k in range(2, K):
        T_k_plus = 2 * Delta_tilde @ T_k - T_k_minus
        out = out + theta[k] * T_k_plus
        T_k_minus, T_k = T_k, T_k_plus
    return out
