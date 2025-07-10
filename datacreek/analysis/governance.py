"""Governance and surveillance utilities for embedding spaces."""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
from scipy.stats import wasserstein_distance


def alignment_correlation(x: Dict[str, Iterable[float]], y: Dict[str, Iterable[float]]) -> float:
    """Return Pearson correlation between two aligned embedding dictionaries."""
    common = [k for k in x if k in y]
    if not common:
        raise ValueError("no common keys between embeddings")
    a = np.stack([np.asarray(x[k], dtype=float) for k in common])
    b = np.stack([np.asarray(y[k], dtype=float) for k in common])
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    num = float(np.sum(a * b))
    den = float(np.sqrt(np.sum(a * a)) * np.sqrt(np.sum(b * b)))
    if den == 0.0:
        return 0.0
    return num / den


def average_hyperbolic_radius(x: Dict[str, Iterable[float]]) -> float:
    """Return mean hyperbolic radius of embeddings in the Poincare ball."""
    arr = np.stack([np.asarray(v, dtype=float) for v in x.values()])
    norms = np.linalg.norm(arr, axis=1)
    norms = np.clip(norms, 0.0, 1 - 1e-7)
    radii = np.arctanh(norms)
    return float(np.mean(radii))


def scale_bias_wasserstein(*embeddings: Dict[str, Iterable[float]]) -> float:
    """Return maximum Wasserstein distance between embedding norm distributions."""
    dists = []
    for emb in embeddings:
        arr = np.stack([np.asarray(v, dtype=float) for v in emb.values()])
        norms = np.linalg.norm(arr, axis=1)
        dists.append(norms)
    max_w = 0.0
    for i in range(len(dists)):
        for j in range(i + 1, len(dists)):
            w = wasserstein_distance(dists[i], dists[j])
            if w > max_w:
                max_w = float(w)
    return max_w


def governance_metrics(
    n2v: Dict[str, Iterable[float]],
    gw: Dict[str, Iterable[float]],
    hyp: Dict[str, Iterable[float]],
) -> Dict[str, float]:
    """Compute governance metrics for three embedding spaces."""
    return {
        "alignment_corr": alignment_correlation(n2v, gw),
        "hyperbolic_radius": average_hyperbolic_radius(hyp),
        "bias_wasserstein": scale_bias_wasserstein(n2v, gw, hyp),
    }
