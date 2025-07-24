import sys
import types
import numpy as np
import networkx as nx
import pytest

import datacreek.analysis.sheaf as sheaf

import datacreek.utils.config as config

def test_resolve_no_improvement(monkeypatch):
    g = nx.path_graph(3)
    for u, v in g.edges():
        g[u][v]["sheaf_sign"] = 1
    calls = [1, 2]

    def fake_h1(*_a, **_k):
        return calls.pop(0) if calls else 2

    monkeypatch.setattr(sheaf, "sheaf_first_cohomology", fake_h1)
    h1 = sheaf.resolve_sheaf_obstruction(g, max_iter=1)
    assert h1 == 1
    assert all(d["sheaf_sign"] == 1 for *_e, d in g.edges(data=True))


def test_first_cohomology_blocksmith_config(monkeypatch):
    monkeypatch.setattr(sheaf, "sheaf_incidence_matrix", lambda g, edge_attr="sheaf_sign": np.eye(2, dtype=int))
    monkeypatch.setattr(config, "load_config", lambda: {"sheaf": {"lam_thresh": 2}})
    g = nx.Graph([(0, 1), (1, 2)])
    assert sheaf.sheaf_first_cohomology_blocksmith(g, block_size=5, lam_thresh=None) == 0


def test_first_cohomology_blocksmith_fallback(monkeypatch):
    monkeypatch.setattr(sheaf, "sheaf_incidence_matrix", lambda g, edge_attr="sheaf_sign": np.zeros((3, 2), int))
    monkeypatch.setattr(sheaf, "block_smith", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(np.linalg, "eigvalsh", lambda m: np.zeros(min(m.shape)))
    g = nx.Graph([(0, 1), (1, 2), (2, 0)])
    assert sheaf.sheaf_first_cohomology_blocksmith(g, block_size=1, lam_thresh=1) == 2


def test_block_smith_config_threshold(monkeypatch):
    monkeypatch.setattr(config, "load_config", lambda: {"sheaf": {"lam_thresh": 0.5}})
    monkeypatch.setattr(np.linalg, "eigvalsh", lambda m: np.array([1.0, 2.0]))
    assert sheaf.block_smith(np.eye(2, dtype=int), block_size=2, lam_thresh=None) == 0


def test_block_smith_invariants_empty(monkeypatch):
    module = types.SimpleNamespace()

    class Matrix:
        def __init__(self, arr):
            self.arr = np.array(arr)

        @property
        def shape(self):
            return self.arr.shape

        def __getitem__(self, idx):
            return self.arr[idx]

    module.Matrix = Matrix
    module.matrices = types.SimpleNamespace(normalforms=types.SimpleNamespace(smith_normal_form=lambda m: (m, None, None)))
    sys.modules['sympy'] = module
    sys.modules['sympy.matrices.normalforms'] = module.matrices.normalforms

    inv = sheaf.block_smith_invariants(np.zeros((2, 0), int))
    assert inv == []
