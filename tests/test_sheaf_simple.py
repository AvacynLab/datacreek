import sys
import types
import numpy as np
import networkx as nx
import pytest

import datacreek.analysis.sheaf as sheaf


def _triangle_graph():
    g = nx.Graph()
    g.add_edge(0, 1, sheaf_sign=1)
    g.add_edge(1, 2, sheaf_sign=-1)
    g.add_edge(2, 0, sheaf_sign=1)
    return g


def test_laplacian_and_incidence():
    g = _triangle_graph()
    L = sheaf.sheaf_laplacian(g)
    expected_L = np.array([[2, -1, -1], [-1, 2, 1], [-1, 1, 2]], dtype=float)
    assert np.array_equal(L, expected_L)

    B = sheaf.sheaf_incidence_matrix(g)
    cols = sorted([tuple(B[:, j]) for j in range(B.shape[1])])
    expected = sorted([(1, -1, 0), (1, 0, -1), (0, 1, 1)])
    assert cols == expected


def test_convolution_and_network():
    g = _triangle_graph()
    feats = {0: np.array([1.0, 0.0]), 1: np.array([0.0, 1.0]), 2: np.array([1.0, 1.0])}
    conv = sheaf.sheaf_convolution(g, feats, alpha=0.5)
    assert pytest.approx(conv[0][0]) == 0.5
    nn = sheaf.sheaf_neural_network(g, feats, layers=2, alpha=0.1)
    assert all(np.all(vec >= 0) for vec in nn.values())


def test_convolution_empty_graph():
    g = nx.Graph()
    out = sheaf.sheaf_convolution(g, {}, alpha=0.5)
    assert out == {}


def test_cohomology_and_score():
    g = _triangle_graph()
    assert sheaf.sheaf_first_cohomology(g) == 0
    assert sheaf.sheaf_consistency_score(g) == 1.0


def test_obstruction_reduction():
    g = nx.cycle_graph(3)
    for u, v in g.edges():
        g[u][v]["sheaf_sign"] = 1
    assert sheaf.sheaf_first_cohomology(g) == 1
    assert sheaf.resolve_sheaf_obstruction(g, max_iter=3) == 0
    assert sheaf.sheaf_first_cohomology(g) == 0


def test_obstruction_zero_sign():
    g = nx.Graph()
    g.add_edge(0, 1, sheaf_sign=0)
    assert sheaf.resolve_sheaf_obstruction(g, max_iter=2) == 0


def test_blocksmith_helpers(monkeypatch):
    # Provide minimal sympy stub
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
    def snf(mat):
        return mat, None, None
    module.matrices = types.SimpleNamespace(normalforms=types.SimpleNamespace(smith_normal_form=snf))
    sys.modules['sympy'] = module
    sys.modules['sympy.matrices.normalforms'] = module.matrices.normalforms

    delta = np.eye(2, dtype=int)
    inv = sheaf.block_smith_invariants(delta, block_size=1)
    assert all(isinstance(i, int) for i in inv)
    assert sheaf.block_smith(delta, block_size=1, lam_thresh=2) == 0


def test_blocksmith_large(monkeypatch):
    monkeypatch.setattr(
        sheaf,
        "sheaf_incidence_matrix",
        lambda g, edge_attr="sheaf_sign": np.zeros((40001, 1), int),
    )
    monkeypatch.setattr(sheaf, "block_smith", lambda d, block_size=40000, lam_thresh=None: 0)
    g = _triangle_graph()
    assert sheaf.sheaf_first_cohomology_blocksmith(g, block_size=40000, lam_thresh=1) == 0


def test_remaining_utils():
    g = _triangle_graph()
    scores = sheaf.sheaf_consistency_score_batched(g, [[0, 1], [1, 2]])
    assert len(scores) == 2
    assert sheaf.spectral_bound_exceeded(g, 2, 0.5)
    assert not sheaf.spectral_bound_exceeded(nx.Graph(), 1, 0.1)
    assert 0.0 <= sheaf.validate_section(g, [0, 1]) <= 1.0
    assert not sheaf.spectral_bound_exceeded(g, 5, 10.0)


def test_spectral_bound_with_scipy(monkeypatch):
    class FakeEig:
        def __call__(self, A, k, which="LM", return_eigenvectors=False):
            M = np.asarray(A)
            return np.linalg.eigvalsh(M)[-k:]

    sp = types.SimpleNamespace(
        csr_matrix=lambda x: np.asarray(x),
        linalg=types.SimpleNamespace(eigsh=FakeEig()),
    )
    scipy = types.SimpleNamespace(sparse=sp)
    monkeypatch.setitem(sys.modules, "scipy", scipy)
    monkeypatch.setitem(sys.modules, "scipy.sparse", sp)
    monkeypatch.setitem(sys.modules, "scipy.sparse.linalg", sp.linalg)
    g = _triangle_graph()
    assert sheaf.spectral_bound_exceeded(g, 2, 0.5)
