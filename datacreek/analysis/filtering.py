"""Graph filtering utilities."""

from __future__ import annotations

from typing import Iterable, Set

import networkx as nx


def filter_semantic_cycles(
    graph: nx.Graph,
    *,
    attr: str = "text",
    stopwords: Iterable[str] | None = None,
    max_len: int = 4,
) -> nx.Graph:
    """Return ``graph`` after removing trivial cycles.

    Parameters
    ----------
    graph:
        Input graph potentially containing meaningless cycles.
    attr:
        Node attribute containing textual labels. If missing, node IDs are used.
    stopwords:
        Words considered non-informative. Cycles whose node labels only contain
        these words are removed.
    max_len:
        Maximum cycle length to examine.

    Returns
    -------
    nx.Graph
        The modified graph with trivial cycles removed.
    """
    if stopwords is None:
        stopwords = {"the", "a", "an", "of", "and", "to", "in"}
    sw: Set[str] = {w.lower() for w in stopwords}

    g = graph.copy()
    cycles = nx.simple_cycles(g) if g.is_directed() else nx.cycle_basis(g)
    for cyc in cycles:
        if len(cyc) > max_len:
            continue
        labels = [str(g.nodes[n].get(attr, n)).lower() for n in cyc]
        if all(all(tok in sw for tok in lab.split()) for lab in labels):
            edges = list(zip(cyc, cyc[1:] + [cyc[0]]))
            g.remove_edges_from(edges)
    return g


import numpy as np


def entropy_triangle_threshold(
    graph: nx.Graph,
    *,
    weight: str = "weight",
    base: float = 2.0,
    scale: float = 10.0,
) -> int:
    """Return triangle threshold derived from edge weight entropy.

    Parameters
    ----------
    graph:
        Input graph with weighted edges.
    weight:
        Edge attribute storing weights.
    base:
        Logarithm base for entropy computation.
    scale:
        Scaling factor converting entropy to an integer threshold.

    Returns
    -------
    int
        Threshold on triangle counts; at least ``1``.
    """
    weights = [abs(data.get(weight, 1.0)) for _, _, data in graph.edges(data=True)]
    if not weights:
        return 1
    arr = np.asarray(weights, dtype=float)
    p = arr / arr.sum()
    H = -float(np.sum(p * np.log(p)) / np.log(base))
    return max(1, int(round(scale * H)))
