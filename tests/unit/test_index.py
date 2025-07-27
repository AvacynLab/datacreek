# Tests for analysis.index utilities
import types
import numpy as np
import networkx as nx
import pytest

import datacreek.analysis.index as index


class FakeIndex:
    """Simple FAISS-like index using cosine similarity."""

    def __init__(self, dim, m=None):
        self.xb = None
        self.dim = dim
        self.hnsw = types.SimpleNamespace(efSearch=0)

    def add(self, xb):
        self.xb = np.asarray(xb)

    def search(self, xq, k):
        xb = self.xb
        xq = np.asarray(xq)
        sims = xb @ xq.T
        order = np.argsort(-sims[:, 0])[:k]
        return None, order.reshape(1, -1)


class FakeHNSW(FakeIndex):
    pass


@pytest.mark.parametrize("use_hnsw", [True, False])
def test_search_with_fallback(monkeypatch, use_hnsw):
    monkeypatch.setattr(
        index,
        "faiss",
        types.SimpleNamespace(IndexFlatIP=FakeIndex, IndexHNSWFlat=FakeHNSW),
    )
    monkeypatch.setattr(index, "np", np)

    xb = np.eye(2, dtype=float)
    xq = np.array([[1.0, 0.0]], dtype=float)

    def fake_monotonic():
        fake_monotonic.t += 0.2
        return fake_monotonic.t

    fake_monotonic.t = 0.0
    monkeypatch.setattr(index.time, "monotonic", fake_monotonic)

    idx, latency, out_index = index.search_with_fallback(xb, xq, latency_threshold=0.1)
    if use_hnsw:
        assert isinstance(out_index, FakeHNSW)
        assert latency > 0.1
    else:
        assert isinstance(out_index, FakeIndex)
        assert latency > 0
    assert idx[0] == 0


def test_recall10(monkeypatch):
    G = nx.Graph()
    G.add_node(
        "q", embedding=[1, 0], graphwave_embedding=[1, 0], poincare_embedding=[0, 0]
    )
    G.add_node(
        "t", embedding=[1, 0], graphwave_embedding=[1, 0], poincare_embedding=[0, 0]
    )
    ground_truth = {"q": ["t"]}

    called = {}
    monkeypatch.setattr(
        index, "update_metric", lambda name, val: called.setdefault(name, val)
    )
    monkeypatch.setattr(
        index,
        "recall_gauge",
        types.SimpleNamespace(set=lambda v: called.setdefault("gauge", v)),
    )

    r = index.recall10(G, ["q"], ground_truth)
    assert r == 0.1
    assert called.get("recall10") == 0.1
    assert called.get("gauge") == 0.1
