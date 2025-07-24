import networkx as nx
import numpy as np
import pytest

import datacreek.analysis.tpl_incremental as tpli


def test_local_hash_changes_on_update():
    g = nx.path_graph(3)
    h1 = tpli._local_hash(g, 0, 1)
    g.add_edge(0, 2)
    h2 = tpli._local_hash(g, 0, 1)
    assert h1 != h2


def test_incremental_diagrams_and_global(monkeypatch):
    g = nx.path_graph(4)
    calls = 0

    def fake_local_persistence(*args, **kwargs):
        nonlocal calls
        calls += 1
        return np.array([[0.0, 1.0]])

    monkeypatch.setattr(tpli, "_local_persistence", fake_local_persistence)

    diags = tpli.tpl_incremental(g, radius=1)
    assert set(diags) == set(g.nodes())
    assert "tpl_global" in g.graph
    first_calls = calls

    diags2 = tpli.tpl_incremental(g, radius=1)
    assert diags2
    assert calls == first_calls



def test_local_persistence_with_fake_gudhi(monkeypatch):
    g = nx.path_graph(3)

    class FakeSimplexTree:
        def __init__(self):
            self.inserted = []

        def insert(self, simplex, filtration=0.0):
            self.inserted.append((tuple(simplex), filtration))

        def compute_persistence(self, persistence_dim_max=True):
            pass

        def persistence_intervals_in_dimension(self, dim):
            return np.array([[0.0, 2.0]]) if dim == 1 else np.empty((0, 2))

    class FakeGd:
        def SimplexTree(self):
            return FakeSimplexTree()

    monkeypatch.setattr(tpli, "gd", FakeGd())
    diag = tpli._local_persistence(g, 1, radius=1, dimension=1)
    assert diag.shape == (1, 2)
    assert np.all(diag == np.array([[0.0, 2.0]]))

