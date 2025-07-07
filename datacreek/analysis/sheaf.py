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
