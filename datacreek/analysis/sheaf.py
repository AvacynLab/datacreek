from __future__ import annotations

import networkx as nx
import numpy as np


def sheaf_laplacian(graph: nx.Graph, *, edge_attr: str = "sheaf_sign") -> np.ndarray:
    """Return the sheaf Laplacian matrix of a signed graph.

    Each edge may carry a sign stored in ``edge_attr``. Missing attributes
    default to ``+1``. The graph is treated as undirected.
    """
    nodes = list(graph.nodes())
    index = {n: i for i, n in enumerate(nodes)}
    L = np.zeros((len(nodes), len(nodes)))
    for u, v, data in graph.to_undirected().edges(data=True):
        sign = data.get(edge_attr, 1.0)
        i, j = index[u], index[v]
        L[i, i] += 1
        L[j, j] += 1
        L[i, j] -= sign
        L[j, i] -= sign
    return L


def sheaf_convolution(
    graph: nx.Graph,
    features: dict[object, np.ndarray],
    *,
    edge_attr: str = "sheaf_sign",
    alpha: float = 0.1,
) -> dict[object, np.ndarray]:
    """Return one step of sheaf convolution on ``features``.

    The update rule is ``X - alpha * L @ X`` where ``L`` is the sheaf
    Laplacian obtained from ``edge_attr``. ``features`` must map each
    node to a numeric vector of equal length.
    """

    nodes = list(graph.nodes())
    if not nodes:
        return {}

    L = sheaf_laplacian(graph, edge_attr=edge_attr)
    dim = len(next(iter(features.values())))
    X = np.zeros((len(nodes), dim))
    for i, n in enumerate(nodes):
        if n in features:
            X[i] = np.asarray(features[n], dtype=float)
    Y = X - alpha * L @ X
    return {n: Y[i] for i, n in enumerate(nodes)}


def sheaf_neural_network(
    graph: nx.Graph,
    features: dict[object, np.ndarray],
    *,
    layers: int = 2,
    alpha: float = 0.1,
    edge_attr: str = "sheaf_sign",
) -> dict[object, np.ndarray]:
    """Return node features after a simple sheaf neural network.

    Parameters
    ----------
    graph:
        Input graph with optional sign information on edges.
    features:
        Initial node features as a mapping ``node -> vector``.
    layers:
        Number of sheaf convolution layers to apply.
    alpha:
        Step size used by :func:`sheaf_convolution`.
    edge_attr:
        Name of the edge attribute storing the sheaf sign.

    Returns
    -------
    dict
        Mapping of nodes to their updated feature vectors after ``layers``
        convolutions with a ReLU nonlinearity.
    """

    out = {n: np.asarray(v, dtype=float) for n, v in features.items()}
    for _ in range(max(1, layers)):
        out = sheaf_convolution(graph, out, edge_attr=edge_attr, alpha=alpha)
        for n in out:
            out[n] = np.maximum(out[n], 0.0)
    return out
