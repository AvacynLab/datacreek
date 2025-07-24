import networkx as nx
import numpy as np
import pytest

from datacreek.analysis import tpl


class FakeSimplexTree:
    """Minimal simplex tree emulating Gudhi's API for testing."""

    def __init__(self):
        self.inserted = []

    def insert(self, simplex, filtration=0.0):
        self.inserted.append((tuple(simplex), filtration))

    def compute_persistence(self, persistence_dim_max=True):
        """No-op for fake persistence calculation."""

    def persistence_intervals_in_dimension(self, dim):
        if dim == 1 and any(len(s) == 2 for s, _ in self.inserted):
            return np.array([[0.0, 2.0]])
        return np.empty((0, 2))


class FakeGd:
    def SimplexTree(self):
        return FakeSimplexTree()


def test_sinkhorn_distance_properties(monkeypatch):
    """Validate that Sinkhorn W1 returns zero for identical diagrams."""

    monkeypatch.setattr(tpl, "gd", FakeGd())
    d1 = np.array([[0.0, 1.0], [0.5, 1.5]])
    d2 = np.array([[0.0, 1.0], [1.0, 2.0]])
    assert np.isclose(tpl.sinkhorn_w1(d1, d1), 0.0, atol=1e-2)
    assert tpl.sinkhorn_w1(d1, d2) > 0.0


def test_diagram_and_correction(monkeypatch):
    """Ensure graph correction reduces the diagram distance."""
    monkeypatch.setattr(tpl, "gd", FakeGd())
    def fake_diagram(graph, dimension=1):
        # return different diagrams based on edge count
        if graph.number_of_edges() < graph.number_of_nodes():
            return np.array([[0.0, 1.0]])
        return np.array([[0.0, 2.0]])

    monkeypatch.setattr(tpl, "_diagram", fake_diagram)
    monkeypatch.setattr(tpl, "generate_graph_rnn_like", lambda n, m: nx.complete_graph(n))
    monkeypatch.setattr(tpl, "resolve_sheaf_obstruction", lambda g, max_iter=5: None)

    g1 = nx.path_graph(4)
    g2 = nx.cycle_graph(4)
    res = tpl.tpl_correct_graph(g1, g2, epsilon=0.0, max_iter=1)
    assert res["corrected"] is True
    assert res["distance_after"] <= res["distance_before"]


def test_diagram_with_fake_gudhi(monkeypatch):
    """Ensure `_diagram` uses the provided Gudhi wrapper."""

    monkeypatch.setattr(tpl, "gd", FakeGd())
    diag = tpl._diagram(nx.path_graph(3), dimension=1)
    assert diag.shape == (1, 2)

