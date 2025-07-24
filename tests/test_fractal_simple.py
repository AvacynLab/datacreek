import time
import concurrent.futures
import math
import numpy as np
import networkx as nx
import sys
import pytest

import datacreek.analysis.fractal as fractal


def test_with_timeout_fallback(monkeypatch):
    def slow(x):
        time.sleep(0.05)
        return x

    def fallback(x):
        return -x

    calls = {"n": 0}

    class Counter:
        def inc(self):
            calls["n"] += 1

    class Gauge:
        def __init__(self):
            self.val = 0.0
        def set(self, v):
            self.val = v

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

    wrapped = fractal.with_timeout(0.01, counter=Counter(), duration_gauge=Gauge(), fallback=fallback)(slow)
    assert wrapped(2) == -2
    assert calls["n"] == 1


def test_with_timeout_success(monkeypatch):
    def fast(x):
        return x * 2

    class Gauge:
        def __init__(self):
            self.val = 0
        def set(self, v):
            self.val = v
            raise RuntimeError

    g = Gauge()
    wrapped = fractal.with_timeout(0.1, duration_gauge=g)(fast)
    assert wrapped(3) == 6
    assert g.val > 0


def test_with_timeout_error_branches(monkeypatch):
    def slow(x):
        raise concurrent.futures.TimeoutError

    class Counter:
        def inc(self):
            raise RuntimeError

    class Gauge:
        def __init__(self):
            self.val = 0
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


def test_laplacian_csgraph(monkeypatch):
    g = nx.path_graph(3)

    class FakeCS:
        def laplacian(self, arr, normed=False):
            class A:
                def __init__(self, m):
                    self.m = m
                def toarray(self):
                    return self.m
            return A(np.array([[1, -1, 0], [-1, 2, -1], [0, -1, 1]], dtype=float))

    monkeypatch.setattr(fractal, "csgraph", FakeCS())
    L = fractal._laplacian(g, normed=True)
    assert L.shape == (3, 3)


def test_lanczos_lmax_dense(monkeypatch):
    L = np.array([[2.0, 0.0], [0.0, 1.0]], dtype=float)

    monkeypatch.setattr(np.linalg, "eigvalsh", lambda x: np.array([1.0, 2.0]))
    import types
    sp = types.SimpleNamespace(issparse=lambda x: False)
    monkeypatch.setitem(sys.modules, "scipy.sparse", sp)
    assert fractal.lanczos_lmax(L) == 2.0


def test_laplacian_and_spectral(monkeypatch):
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
    entropy = fractal.spectral_entropy(g)
    gap = fractal.spectral_gap(g)
    energy = fractal.laplacian_energy(g)
    hist, edges = fractal.spectral_density(g, bins=3)
    assert entropy >= 0
    assert gap >= 0
    assert energy >= 0
    assert hist.size == 3 and edges.size == 4


def test_box_dimension_and_helpers():
    g = nx.path_graph(4)
    boxes = fractal.box_cover(g, 1)
    assert sum(len(b) for b in boxes) == 4
    dim1, counts1 = fractal.box_counting_dimension(g, [1, 2])
    dim2, counts2 = fractal.colour_box_dimension(g, [1, 2])
    assert counts1 and counts2
    assert isinstance(dim1, float) and isinstance(dim2, float)

    coords = {0: [0.0, 0.0], 1: [1.0, 0.0], 2: [2.0, 0.0]}
    dim3, counts3 = fractal.embedding_box_counting_dimension(coords, [0.5, 1.5])
    assert counts3 and dim3 >= 0
    lac = fractal.graph_lacunarity(g, radius=1)
    assert lac >= 1.0


def test_lanczos_and_mdl(monkeypatch):
    L = np.array([[3.0, 0.0], [0.0, 1.0]], dtype=float)
    assert pytest.approx(3.0, rel=1e-3) == fractal.lanczos_lmax(L)
    assert fractal.lanczos_top_eigenvalue(L, k=10) > 0

    counts = [(1, 3), (2, 1)]
    assert fractal.mdl_optimal_radius(counts) == 1
    assert isinstance(fractal.mdl_value(counts), float)
    slope = fractal._slope(counts)
    best = fractal.dichotomic_radius(counts, slope)
    assert best == 1


def test_coverage_and_diversification(monkeypatch):
    g1 = nx.path_graph(3)
    g2 = nx.path_graph(3)
    for n in g1.nodes():
        g1.nodes[n]["fractal_level"] = 0
    cov = fractal.fractal_level_coverage(g1)
    assert cov == 1.0
    monkeypatch.setattr(fractal, "bottleneck_distance", lambda *a, **k: 0.0)
    score = fractal.diversification_score(g1, g2, [1])
    assert isinstance(score, float)
