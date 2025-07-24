import random
import sys
import types

import networkx as nx
import numpy as np

import datacreek.analysis.generation as gen


def test_generate_graph_rnn_like_and_netgan():
    g = gen.generate_graph_rnn_like(5, 4)
    assert g.number_of_nodes() == 5
    assert g.number_of_edges() <= 4

    base = nx.path_graph(5)
    h = gen.generate_netgan_like(base, num_walks=2, walk_length=3, p=1.0)
    assert h.number_of_nodes() == 5
    assert h.number_of_edges() > 0


def test_generate_graph_rnn_variants():
    random.seed(0)
    g = gen.generate_graph_rnn(5, 3, p=0.9)
    assert g.number_of_nodes() == 5
    assert g.number_of_edges() >= 3

    g_stateful = gen.generate_graph_rnn_stateful(5, 3, seed=0)
    assert g_stateful.number_of_nodes() == 5
    g_seq = gen.generate_graph_rnn_sequential(5, 3, seed=0)
    assert g_seq.number_of_nodes() == 5


def test_sheaf_score_helpers(monkeypatch):
    def fake_laplacian(graph, edge_attr="sheaf_sign"):
        return np.eye(2)

    def fake_cg(A, b_vec, atol=0.0, rtol=1e-5, maxiter=1000):
        return np.linalg.solve(np.asarray(A), b_vec), 0

    monkeypatch.setitem(
        sys.modules,
        "datacreek.analysis.sheaf",
        types.SimpleNamespace(sheaf_laplacian=fake_laplacian),
    )
    monkeypatch.setitem(
        sys.modules, "scipy.sparse", types.SimpleNamespace(csr_matrix=lambda a: a)
    )
    monkeypatch.setitem(
        sys.modules, "scipy.sparse.linalg", types.SimpleNamespace(cg=fake_cg)
    )

    g = nx.path_graph(2)
    b = [1.0, -1.0]
    score = gen.sheaf_consistency_real(g, b)
    assert 0.0 < score <= 1.0

    Delta = np.eye(2)
    score2 = gen.sheaf_score(b, Delta)
    assert 0.0 < score2 <= 1.0


def test_bias_reweighting_and_apply(monkeypatch):
    loc = {"A": 0, "B": 2}
    glob = {"A": 2, "B": 2}
    weights = {"A": 0.5, "B": 0.5}

    def fake_bw(loc_hist, glob_hist, logits):
        return np.array(logits) * 0.5, 0.2

    monkeypatch.setattr(gen, "bias_wasserstein", fake_bw)
    stats_mod = types.SimpleNamespace(wasserstein_distance=lambda a, b: 0.2)
    monkeypatch.setitem(sys.modules, "scipy", types.SimpleNamespace(stats=stats_mod))
    monkeypatch.setitem(sys.modules, "scipy.stats", stats_mod)
    biased = gen.bias_reweighting(loc, glob, weights, threshold=0.0)
    assert biased["A"] > weights["A"]
    payload = {"logits": [0.0, 1.0]}
    w = gen.apply_logit_bias(payload, loc, glob)
    assert w == 0.2
    assert payload["logits"] == [0.0, 0.5]


def test_bias_wasserstein(monkeypatch):
    class DummyLoss:
        def __call__(self, a, b):
            return float(np.abs(a - b).sum())

    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(
            as_tensor=lambda x, dtype=None: np.array(x, dtype=float),
            float32=np.float32,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "geomloss",
        types.SimpleNamespace(
            SamplesLoss=lambda *_a, **_k: DummyLoss(),
        ),
    )

    loc_hist = [1, 0]
    glob_hist = [0, 1]
    logits = [1.0, 2.0]
    scaled, w = gen.bias_wasserstein(loc_hist, glob_hist, logits)
    assert len(scaled) == 2 and w > 0
