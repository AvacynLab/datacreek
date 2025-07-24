import math
import concurrent.futures
import numpy as np
import networkx as nx
import pytest
import sys
import datacreek.analysis.fractal as fractal


def test_with_timeout_success(monkeypatch):
    calls = {"c": 0}

    def fast(x):
        calls["c"] += 1
        return x * 2

    class Gauge:
        def __init__(self):
            self.val = 0.0

        def set(self, v):
            self.val = v
            raise RuntimeError

    g = Gauge()
    c = type("C", (), {"inc": lambda self: None})()
    wrapped = fractal.with_timeout(0.1, counter=c, duration_gauge=g)(fast)
    assert wrapped(3) == 6
    assert calls["c"] == 1
    assert g.val > 0


def test_laplacian_fallback(monkeypatch):
    g = nx.path_graph(3)
    monkeypatch.setattr(fractal, "csgraph", None)
    L = fractal._laplacian(g, normed=True)
    expected = np.array(
        [[1.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 1.0]], dtype=float
    )
    inv_deg = np.array([1.0, 1.0 / np.sqrt(2.0), 1.0])
    norm = (inv_deg[:, None] * expected) * inv_deg
    assert np.allclose(L, norm)


def test_lanczos_lmax_fallback(monkeypatch):
    L = np.array([[2.0, 0.0], [0.0, 1.0]], dtype=float)
    monkeypatch.setattr(np.linalg, "eigvalsh", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError))
    monkeypatch.setattr(np.random, "rand", lambda n: np.ones(n))
    val = fractal.lanczos_lmax(L, iters=3)
    assert pytest.approx(2.0, rel=1e-2) == val


def test_box_dim_single_radius():
    g = nx.path_graph(3)
    dim1, counts1 = fractal.box_counting_dimension(g, [1])
    dim2, counts2 = fractal.colour_box_dimension(g, [1])
    assert counts1 and counts2
    assert dim1 == 0.0 and dim2 == 0.0


def test_embedding_box_dim_single_radius():
    coords = {0: [0.0, 0.0], 1: [2.0, 0.0]}
    dim, counts = fractal.embedding_box_counting_dimension(coords, [1.0])
    assert counts and dim == 0.0


def test_with_timeout_error_branches(monkeypatch):
    def slow(x):
        raise concurrent.futures.TimeoutError

    class Counter:
        def inc(self):
            raise RuntimeError

    class Gauge:
        def __init__(self):
            self.val = 0.0

        def set(self, v):
            self.val = v
            raise RuntimeError

    class DummyFuture:
        def result(self, timeout=None):
            raise concurrent.futures.TimeoutError

        def cancel(self):
            pass

    class DummyExecutor:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args, **kwargs):
            return DummyFuture()

    monkeypatch.setattr(fractal.concurrent.futures, "ThreadPoolExecutor", DummyExecutor)
    wrapped = fractal.with_timeout(0.01, counter=Counter(), duration_gauge=Gauge(), fallback=lambda x: -x)(slow)
    assert wrapped(2) == -2


def test_edge_cases_metrics(monkeypatch):
    g = nx.Graph()
    g.add_node(0)
    assert fractal.graph_lacunarity(g) == 1.0
    assert fractal.fractal_level_coverage(g) == 0.0
    monkeypatch.setattr(fractal, "eigh", lambda a, eigvals_only=False: np.linalg.eigvalsh(a) if eigvals_only else (np.linalg.eigvalsh(a), np.eye(a.shape[0])))
    assert fractal.spectral_entropy(g) == 0.0
    assert fractal.laplacian_energy(g) == 0.0
    monkeypatch.setattr(fractal, "bottleneck_distance", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError))
    g_a = nx.path_graph(2)
    g_b = nx.path_graph(2)
    score = fractal.diversification_score(g_a, g_b, [1])
    assert np.isnan(score) or isinstance(score, float)


def test_mdl_helpers_short_sequences():
    counts = [(1, 2)]
    assert fractal.mdl_optimal_radius(counts) == 0
    assert fractal.mdl_value(counts) == 0.0
    assert fractal._slope(counts) == 0.0
    counts2 = [(1, 3), (2, 1)]
    target = fractal._slope(counts2)
    idx = fractal.dichotomic_radius(counts2, target)
    assert idx == 1


def test_entropy_edge_cases(monkeypatch):
    assert fractal.graphwave_entropy({0: []}) == 0.0
    monkeypatch.setattr(np.linalg, "slogdet", lambda m: (-1, 0.0))
    vals = {0: [0.0, 0.0], 1: [0.0, 0.0]}
    assert math.isnan(fractal.embedding_entropy(vals))


def test_spectral_tools_and_fourier():
    g = nx.path_graph(3)
    dim, traces = fractal.spectral_dimension(g, [0.1, 0.2])
    assert traces and dim >= 0
    evals = fractal.laplacian_spectrum(g)
    gap = fractal.spectral_gap(g)
    energy = fractal.laplacian_energy(g)
    hist, edges = fractal.spectral_density(g, bins=3)
    coeffs = fractal.graph_fourier_transform(g, {n: float(n) for n in g})
    recon = fractal.inverse_graph_fourier_transform(g, coeffs)
    assert len(evals) == g.number_of_nodes()
    assert gap >= 0 and energy >= 0 and hist.size == 3
    assert np.allclose(recon, np.arange(g.number_of_nodes(), dtype=float))


def test_information_and_hierarchy(monkeypatch):
    g = nx.path_graph(4)
    monkeypatch.setattr(fractal, "gd", None)
    monkeypatch.setattr(fractal, "gr", None)
    metrics = fractal.fractal_information_metrics(g, [1])
    density = fractal.fractal_information_density(g, [1])
    coarse, mapping = fractal.fractalize_graph(g, 1)
    coarse2, mapping2, radius = fractal.fractalize_optimal(g, [1])
    levels = fractal.build_fractal_hierarchy(g, [1])
    mdl = fractal.build_mdl_hierarchy(g, [1])
    assert metrics["dimension"] >= 0 and isinstance(density, float)
    assert coarse.number_of_nodes() <= g.number_of_nodes()
    assert radius == 1 and levels and mdl


def test_hyperbolic_utils():
    embs = {0: [0.0, 0.0], 1: [0.1, 0.0], 2: [0.0, 0.2]}
    d01 = fractal.hyperbolic_distance(np.array(embs[0]), np.array(embs[1]))
    nn = fractal.hyperbolic_nearest_neighbors(embs, k=1)
    path = fractal.hyperbolic_reasoning(embs, 0, 2)
    hpath = fractal.hyperbolic_hypergraph_reasoning(embs, {1}, 0, 2)
    mpath = fractal.hyperbolic_multi_curvature_reasoning({0.1: embs}, 0, 2)
    assert d01 > 0 and nn[0]
    assert path[0] == 0 and path[-1] in {0,2}
    assert hpath and mpath


def test_prune_compress_and_bootstrap(monkeypatch):
    embs = {0: [0.0, 0.0], 1: [0.0, 0.0]}
    centers, mapping = fractal.fractal_net_prune(embs)
    levels = {0: 1, 1: 1}
    compressed = fractal.fractalnet_compress(embs, levels)
    g = nx.path_graph(3)
    sigma = fractal.bootstrap_sigma_db(g, [1])
    assert centers and mapping and compressed and sigma >= 0


def test_injection_helpers(monkeypatch):
    g = nx.path_graph(3)
    monkeypatch.setattr(fractal, "GraphRNN_Lite", None, raising=False)
    import types
    gen_mod = types.SimpleNamespace(generate_graph_rnn_like=lambda n, e: nx.path_graph(n))
    monkeypatch.setattr(fractal, "generation", gen_mod, raising=False)
    monkeypatch.setattr(fractal.generation, "generate_graph_rnn_like", lambda n, e: nx.path_graph(n))
    nodes = fractal.inject_graphrnn_subgraph(g, 2, 1)
    assert nodes and all(n in g for n in nodes)
    monkeypatch.setattr(fractal, "inject_graphrnn_subgraph", lambda *a, **k: nodes)
    import types
    sheaf_mod = types.SimpleNamespace(validate_section=lambda *_a, **_k: 0.9)
    monkeypatch.setitem(sys.modules, "datacreek.analysis.sheaf", sheaf_mod)
    score = fractal.inject_and_validate(g, 2, 1, rollback=True, driver=None)
    assert 0 <= score <= 1
    cfg = {"tpl": {"rnn_size": 2}}
    monkeypatch.setattr(fractal, "inject_and_validate", lambda *_a, **_k: 1.0)
    assert fractal.tpl_motif_injection(g, cfg) == 1.0

