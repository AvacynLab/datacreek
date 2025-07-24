import time
import concurrent.futures

import networkx as nx
import numpy as np
import pytest

import datacreek.analysis.fractal as fractal


def constant_polyfit(x, y, deg=1):
    """Return slope 0.5 and intercept 0.0 regardless of input."""
    return 0.5, 0.0


@pytest.fixture(autouse=True)
def _stub_polyfit(monkeypatch):
    """Stub ``np.polyfit`` to keep tests deterministic."""
    monkeypatch.setattr(np, "polyfit", constant_polyfit)
    yield


def test_with_timeout_fallback(monkeypatch):
    def slow(x):
        time.sleep(0.05)
        return x

    def fallback(x):
        return -x

    called = {"n": 0}

    class Counter:
        def inc(self):
            called["n"] += 1

    class Gauge:
        def __init__(self):
            self.val = 0.0

        def set(self, v):
            self.val = v

    c = Counter()
    g = Gauge()

    class DummyFuture:
        def result(self, timeout=None):
            raise concurrent.futures.TimeoutError

        def cancel(self):
            pass

    class DummyExecutor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args, **kwargs):
            return DummyFuture()

    monkeypatch.setattr(fractal.concurrent.futures, "ThreadPoolExecutor", DummyExecutor)

    wrapped = fractal.with_timeout(
        0.01, counter=c, duration_gauge=g, fallback=fallback
    )(slow)
    assert wrapped(2) == -2
    assert called["n"] == 1
    assert g.val >= 0.01


def test_laplacian_and_spectrum_metrics(monkeypatch):
    g = nx.path_graph(3)
    L = fractal._laplacian(g, normed=False)
    expected = np.array([[1, -1, 0], [-1, 2, -1], [0, -1, 1]], dtype=float)
    assert np.allclose(L, expected)

    def fake_eigh(a, eigvals_only=False):
        vals = np.linalg.eigvalsh(a)
        return vals if eigvals_only else (vals, np.eye(a.shape[0]))

    monkeypatch.setattr(fractal, "eigh", fake_eigh)

    evals = fractal.laplacian_spectrum(g)
    assert len(evals) == 3
    ent = fractal.spectral_entropy(g)
    gap = fractal.spectral_gap(g)
    energy = fractal.laplacian_energy(g)
    monkeypatch.setattr(
        fractal,
        "spectral_density",
        lambda *_a, **_k: (np.zeros(3), np.arange(4)),
    )
    hist, edges = fractal.spectral_density(g, bins=3)
    assert ent >= 0
    assert gap >= 0
    assert energy > 0
    assert hist.size == 3 and edges.size == 4


def test_box_and_colour_dimension():
    g = nx.path_graph(4)
    boxes = fractal.box_cover(g, 1)
    assert sum(len(b) for b in boxes) == 4
    dim1, counts1 = fractal.box_counting_dimension(g, [1, 2])
    dim2, counts2 = fractal.colour_box_dimension(g, [1, 2])
    assert counts1 and counts2
    assert isinstance(dim1, float) and isinstance(dim2, float)


def test_embedding_box_counting_and_lacunarity():
    coords = {0: [0.0, 0.0], 1: [1.0, 0.0], 2: [2.0, 0.0]}
    dim, counts = fractal.embedding_box_counting_dimension(coords, [0.5, 1.5])
    assert counts and dim >= 0
    g = nx.path_graph(3)
    lac = fractal.graph_lacunarity(g, radius=1)
    assert lac >= 1.0


def test_lanczos_and_mdl_helpers():
    L = np.array([[3.0, 0.0], [0.0, 1.0]], dtype=float)
    assert pytest.approx(3.0, rel=1e-3) == fractal.lanczos_lmax(L)
    assert fractal.lanczos_top_eigenvalue(L, k=20) > 0

    counts = [(1, 3), (2, 1)]
    idx = fractal.mdl_optimal_radius(counts)
    assert idx == 1
    val = fractal.mdl_value(counts)
    slope = fractal._slope(counts)
    best = fractal.dichotomic_radius(counts, slope)
    assert best == idx


def test_fractal_level_coverage_and_diversification(monkeypatch):
    g1 = nx.path_graph(3)
    g2 = nx.path_graph(3)
    for n in g1.nodes():
        g1.nodes[n]["fractal_level"] = 0
    cov = fractal.fractal_level_coverage(g1)
    assert cov == 1.0
    monkeypatch.setattr(fractal, "bottleneck_distance", lambda *a, **k: 0.0)
    score = fractal.diversification_score(g1, g2, [1])
    assert isinstance(score, float)
