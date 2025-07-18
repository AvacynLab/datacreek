"""Poincaré embedding recentering utilities."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Sequence

import numpy as np

try:  # pragma: no cover - optional dependency
    import torch
except Exception:  # pragma: no cover - torch optional
    torch = None  # type: ignore

__all__ = ["recenter_embeddings", "hyperbolic_radius", "measure_overshoot"]


# --- Möbius geometry helpers -------------------------------------------------


def _mobius_add(
    x: np.ndarray,
    y: np.ndarray,
    *,
    c: float = 1.0,
    delta: float = 1e-6,
    clamp: bool = True,
) -> np.ndarray:
    """Return Möbius addition of ``x`` and ``y``.

    Parameters
    ----------
    x, y:
        Points in the open Poincaré ball.
    c:
        Curvature constant ``c>0`` for ``kappa=-c``.
    delta:
        Safety margin to avoid hitting the boundary when ``clamp`` is ``True``.
    clamp:
        Whether to enforce ``||result|| < 1 - delta``.
    """

    x2 = c * np.dot(x, x)
    y2 = c * np.dot(y, y)
    xy = c * np.dot(x, y)
    num = (1 + 2 * xy + y2) * x + (1 - x2) * y
    denom = 1 + 2 * xy + x2 * y2
    res = num / max(denom, 1e-15)
    if clamp:
        n = np.linalg.norm(res)
        if n >= 1.0 - delta:
            res = res / n * (1.0 - delta)
    return res


def _mobius_neg(x: np.ndarray) -> np.ndarray:
    """Return the Möbius inverse of ``x``."""
    return -x


def _exp_map_zero(v: np.ndarray, *, c: float = 1.0) -> np.ndarray:
    """Exponential map at the origin for curvature ``-c``."""
    norm = np.linalg.norm(v)
    if norm < 1e-15:
        return v
    sqc = float(c) ** 0.5
    return np.tanh(sqc * norm) * v / (sqc * norm)


def _log_map(x: np.ndarray, y: np.ndarray, *, c: float = 1.0) -> np.ndarray:
    """Logarithmic map of ``y`` at ``x`` for curvature ``-c``."""
    u = _mobius_add(_mobius_neg(x), y, c=c, clamp=False)
    norm_u = np.linalg.norm(u)
    if norm_u < 1e-15:
        return np.zeros_like(x)
    sqc = float(c) ** 0.5
    return (1.0 / sqc) * np.arctanh(sqc * norm_u) * (u / norm_u)


if torch is not None:  # pragma: no cover - optional autodiff helpers

    def _exp_map_zero_torch(v: "torch.Tensor", *, c: float = 1.0) -> "torch.Tensor":
        norm = torch.linalg.norm(v, dim=-1, keepdim=True)
        sqc = float(c) ** 0.5
        return torch.tanh(sqc * norm) * v / (sqc * norm.clamp_min(1e-8))

    def _log_map_zero_torch(y: "torch.Tensor", *, c: float = 1.0) -> "torch.Tensor":
        norm = torch.linalg.norm(y, dim=-1, keepdim=True)
        sqc = float(c) ** 0.5
        return torch.atanh(sqc * norm) * y / (sqc * norm.clamp_min(1e-8))


def hyperbolic_radius(x: np.ndarray, *, c: float = 1.0) -> float:
    """Return the hyperbolic radius of ``x`` in the Poincaré ball."""
    norm = np.linalg.norm(x)
    sqc = float(c) ** 0.5
    return 2.0 / sqc * np.arctanh(sqc * norm)


def measure_overshoot(
    radii: Sequence[float], *, kappa: float = -1.0, num_samples: int = 256
) -> dict[str, list[float]]:
    """Return overshoot curves with and without clamping.

    Parameters
    ----------
    radii:
        Hyperbolic radii to sample.
    kappa:
        Negative curvature value.
    num_samples:
        Number of random points per radius.

    Returns
    -------
    dict
        ``{"clamp": overshoot, "noclamp": overshoot}`` lists.
    """
    c = -kappa
    rng = np.random.default_rng(0)
    clamp_cur = []
    raw_cur = []
    for r in radii:
        eucl = np.tanh(0.5 * (c**0.5) * r)
        dirs = rng.standard_normal((num_samples, 2))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        pts = dirs * eucl
        o_clamp = 0.0
        o_raw = 0.0
        for x in pts:
            y_raw = _mobius_add(x, -x, c=c, clamp=False)
            y_clamp = _mobius_add(x, -x, c=c)
            o_raw += hyperbolic_radius(y_raw, c=c)
            o_clamp += hyperbolic_radius(y_clamp, c=c)
        clamp_cur.append(o_clamp / num_samples)
        raw_cur.append(o_raw / num_samples)
    return {"clamp": clamp_cur, "noclamp": raw_cur}


# --- Public API ----------------------------------------------------------------


def recenter_embeddings(
    embeddings: Mapping[object, Iterable[float]],
    *,
    curvature: float = 1.0,
    delta: float = 1e-6,
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
    if norm >= 1.0 - delta:
        center = center / norm * (1.0 - delta)

    recentered: Dict[object, np.ndarray] = {}
    for key, x in vecs.items():
        v = -_log_map(x, center, c=curvature)
        y = _exp_map_zero(v, c=curvature)
        n = np.linalg.norm(y)
        if n >= 1.0 - delta:
            y = y / n * (1.0 - delta)
        recentered[key] = y.astype(np.float16)

    return recentered
