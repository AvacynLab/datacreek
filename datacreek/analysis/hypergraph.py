import math
from typing import Iterable, List, Sequence

import numpy as np


def hyper_sagnn_embeddings(
    hyperedges: List[Sequence[int]],
    node_features: np.ndarray,
    *,
    embed_dim: int | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Return Hyper-SAGNN style embeddings for each hyperedge.

    Parameters
    ----------
    hyperedges:
        List of hyperedges, each given as a sequence of node indices.
    node_features:
        Array of shape ``(num_nodes, feat_dim)`` containing node features.
    embed_dim:
        Dimension ``d`` of the output embedding. Defaults to ``feat_dim``.
    seed:
        Optional random seed controlling the internal attention weights.
    """
    rng = np.random.default_rng(seed)
    feat_dim = node_features.shape[1]
    d = embed_dim or feat_dim

    W_q = rng.normal(scale=1.0 / math.sqrt(feat_dim), size=(feat_dim, d))
    W_k = rng.normal(scale=1.0 / math.sqrt(feat_dim), size=(feat_dim, d))
    W_v = rng.normal(scale=1.0 / math.sqrt(feat_dim), size=(feat_dim, d))

    def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        x_max = x.max(axis=axis, keepdims=True)
        e = np.exp(x - x_max)
        return e / e.sum(axis=axis, keepdims=True)

    embeddings = []
    for edge in hyperedges:
        X = node_features[np.asarray(edge)]
        q = X @ W_q
        k = X @ W_k
        v = X @ W_v
        scores = q @ k.T / math.sqrt(d)
        weights = _softmax(scores, axis=1)
        context = weights @ v
        emb = context.mean(axis=0)
        embeddings.append(emb)
    return np.stack(embeddings)
