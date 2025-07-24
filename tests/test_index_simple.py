import sys
import types
from types import SimpleNamespace

# stub sklearn to avoid heavy dependency
sk_mod = types.ModuleType("sklearn")
sk_cd = types.ModuleType("sklearn.cross_decomposition")
sk_cd.CCA = lambda *a, **k: None
sys.modules.setdefault("sklearn", sk_mod)
sys.modules.setdefault("sklearn.cross_decomposition", sk_cd)

import networkx as nx
import numpy as np

import datacreek.analysis.index as idx


def test_search_with_fallback(monkeypatch):
    class FakeIndex:
        def __init__(self, dim):
            self.xb = None
            self.dim = dim
            self.hnsw = SimpleNamespace(efSearch=0)

        def add(self, xb):
            self.xb = xb

        def search(self, xq, k):
            scores = xq @ self.xb.T
            order = np.argsort(-scores, axis=1)
            return None, order[:, :k]

    class FakeHNSW(FakeIndex):
        def __init__(self, dim, M=16):
            super().__init__(dim)

    fake_faiss = SimpleNamespace(IndexFlatIP=FakeIndex, IndexHNSWFlat=FakeHNSW)
    times = iter([0.0, 0.2, 0.2, 0.25])
    monkeypatch.setattr(idx, "faiss", fake_faiss)
    monkeypatch.setattr(idx, "np", np)
    monkeypatch.setattr(
        idx,
        "ann_latency",
        SimpleNamespace(time=lambda: __import__("contextlib").nullcontext()),
    )
    monkeypatch.setattr(idx.time, "monotonic", lambda: next(times))

    xb = np.eye(2, dtype=np.float32)
    xq = np.asarray([[1.0, 0.0]], dtype=np.float32)
    idxs, latency, used = idx.search_with_fallback(xb, xq, k=1, latency_threshold=0.1)
    assert idxs.tolist() == [0]
    assert isinstance(used, FakeHNSW)
    assert latency > 0.0


def test_recall10(monkeypatch):
    def fake_score(*a, **k):
        return float(a[1][0]) * float(a[0][0])

    metrics = {}

    def fake_update(name, val):
        metrics[name] = val

    class Gauge:
        def __init__(self, *a, **kw):
            self.val = None

        def set(self, value):
            self.val = value

    monkeypatch.setattr(idx, "hybrid_score", fake_score)
    monkeypatch.setattr(idx, "update_metric", fake_update)
    monkeypatch.setattr(idx, "recall_gauge", Gauge())

    G = nx.Graph()
    G.add_node(
        "a", embedding=[1.0], graphwave_embedding=[1.0], poincare_embedding=[1.0]
    )
    G.add_node(
        "b", embedding=[0.0], graphwave_embedding=[0.0], poincare_embedding=[0.0]
    )

    r = idx.recall10(G, ["a"], {"a": ["b"]})
    assert r == 0.1
    assert G.graph["recall10"] == r
    assert metrics["recall10"] == r
