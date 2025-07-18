"""Poincaré embedding recentering utilities."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping

import numpy as np

__all__ = ["recenter_embeddings"]


# --- Möbius geometry helpers -------------------------------------------------


def _mobius_add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return Möbius addition of ``x`` and ``y``."""
    x2 = np.dot(x, x)
    y2 = np.dot(y, y)
    xy = np.dot(x, y)
    num = (1 + 2 * xy + y2) * x + (1 - x2) * y
    denom = 1 + 2 * xy + x2 * y2
    return num / max(denom, 1e-15)


def _mobius_neg(x: np.ndarray) -> np.ndarray:
    """Return the Möbius inverse of ``x``."""
    return -x


def _exp_map_zero(v: np.ndarray) -> np.ndarray:
    """Exponential map at the origin."""
    norm = np.linalg.norm(v)
    if norm < 1e-15:
        return v
    return np.tanh(norm) * v / norm


def _log_map(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Logarithmic map of ``y`` at ``x``."""
    u = _mobius_add(_mobius_neg(x), y)
    norm_u = np.linalg.norm(u)
    if norm_u < 1e-15:
        return np.zeros_like(x)
    lam = 2.0 / (1.0 - np.dot(x, x))
    return (1.0 / lam) * np.arctanh(lam * norm_u) * (u / norm_u)


# --- Public API ----------------------------------------------------------------


def recenter_embeddings(
    embeddings: Mapping[object, Iterable[float]],
) -> Dict[object, np.ndarray]:
    """Recenter Poincaré embeddings using a moving hyperbolic center.

    Parameters
    ----------
    embeddings:
        Mapping of node identifier to embedding vectors lying in the
        open unit ball.

    Returns
    -------
    dict
        Embeddings translated so the hyperbolic center of mass sits at
        the origin.
    """

    vecs = {k: np.asarray(v, dtype=np.float64) for k, v in embeddings.items()}
    if not vecs:
        return {}

    # Compute approximate hyperbolic barycenter using Euclidean mean.
    center = np.mean(list(vecs.values()), axis=0)
    norm = np.linalg.norm(center)
    if norm >= 1.0:
        center = center / norm * (1.0 - 1e-6)

    recentered: Dict[object, np.ndarray] = {}
    for key, x in vecs.items():
        v = -_log_map(x, center)
        y = _exp_map_zero(v)
        n = np.linalg.norm(y)
        if n >= 1.0:
            y = y / n * (1.0 - 1e-6)
        recentered[key] = y.astype(np.float16)

    return recentered
