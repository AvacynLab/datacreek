from __future__ import annotations

"""Graph generation utilities."""

import networkx as nx


def generate_graph_rnn_like(num_nodes: int, num_edges: int) -> nx.Graph:
    """Return a random graph mimicking GraphRNN output.

    This simplified implementation generates an undirected graph with
    ``num_nodes`` nodes and roughly ``num_edges`` edges using a uniform
    random model. It serves as a lightweight stand-in for a true
    GraphRNN model.
    """
    g = nx.gnm_random_graph(num_nodes, num_edges, seed=0)
    return g
