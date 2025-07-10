"""Topological Perception Layer utilities."""

from __future__ import annotations

from typing import Dict, Iterable

import networkx as nx

from .fractal import persistence_wasserstein_distance
from .generation import generate_graph_rnn_like
from .sheaf import resolve_sheaf_obstruction


def tpl_correct_graph(
    graph: nx.Graph,
    target: nx.Graph,
    *,
    epsilon: float = 0.1,
    dimension: int = 1,
    order: int = 1,
    max_iter: int = 5,
) -> Dict[str, float | bool]:
    """Correct ``graph`` topology if Wasserstein distance is too high."""

    dist_before = persistence_wasserstein_distance(
        graph, target, dimension=dimension, order=order
    )

    corrected = False
    if dist_before > epsilon:
        motif = generate_graph_rnn_like(
            target.number_of_nodes(), target.number_of_edges()
        )
        mapping = {i: n for i, n in enumerate(graph.nodes())}
        for u, v in motif.edges():
            a = mapping.get(u)
            b = mapping.get(v)
            if a is None or b is None:
                continue
            if not graph.has_edge(a, b):
                graph.add_edge(a, b, relation="perception_link")
        resolve_sheaf_obstruction(graph, max_iter=max_iter)
        corrected = True

    dist_after = persistence_wasserstein_distance(
        graph, target, dimension=dimension, order=order
    )

    return {
        "distance_before": float(dist_before),
        "distance_after": float(dist_after),
        "corrected": corrected,
    }

