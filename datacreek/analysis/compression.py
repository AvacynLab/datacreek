"""Model compression utilities."""

from __future__ import annotations

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    np = None  # type: ignore


def prune_fractalnet(weights: "np.ndarray | list[float]", ratio: float = 0.5):
    """Return weights pruned by magnitude preserving ``ratio`` of parameters.

    Parameters
    ----------
    weights:
        Weight matrix or vector to prune.
    ratio:
        Fraction of weights to keep based on absolute value.

    Returns
    -------
    numpy.ndarray
        Pruned weight array with the same shape.
    """

    if np is None:
        flat = [abs(x) for x in (weights if isinstance(weights, list) else list(weights))]
        k = int(len(flat) * ratio)
        if k <= 0:
            return [0.0 for _ in flat]
        thresh = sorted(flat)[-k]
        result = [w if abs(w) >= thresh else 0.0 for w in (weights if isinstance(weights, list) else list(weights))]
        return result if isinstance(weights, list) else np.array(result)

    flat = weights.flatten()
    k = int(len(flat) * ratio)
    if k <= 0:
        return np.zeros_like(weights)
    thresh = np.partition(np.abs(flat), -k)[-k]
    mask = np.abs(weights) >= thresh
    return weights * mask
