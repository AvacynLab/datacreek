import json
import networkx as nx
import types
import time

from datacreek.analysis import node2vec_tuning


class DummyKG:
    def __init__(self):
        self.graph = nx.path_graph(2)
        self.graph.graph = {}

    def compute_node2vec_embeddings(self, *, p: float = 1.0, q: float = 1.0):
        for n in self.graph.nodes:
            self.graph.nodes[n]["embedding"] = [p + n, q + n]


def test_autotune_node2vec_runs(monkeypatch, tmp_path):
    kg = DummyKG()

    monkeypatch.setattr(node2vec_tuning, "ARTIFACT_PATH", tmp_path / "art.json")

    monkeypatch.setattr(
        node2vec_tuning, "recall10", lambda g, q, gt, gamma=0.5, eta=0.25: 0.5
    )
    best = node2vec_tuning.autotune_node2vec(
        kg, [0], {0: [1]}, max_evals=3, max_minutes=0.1
    )
    assert 0.1 <= best[0] <= 4.0
    assert 0.1 <= best[1] <= 4.0
    data = json.loads((tmp_path / "art.json").read_text())
    assert data["p"] == best[0]
    assert data["q"] == best[1]


def test_autotune_node2vec_early_stop(monkeypatch, tmp_path):
    kg = DummyKG()
    monkeypatch.setattr(node2vec_tuning, "ARTIFACT_PATH", tmp_path / "art.json")
    calls = []

    def _compute(p: float = 1.0, q: float = 1.0):
        calls.append((p, q))
        for n in kg.graph.nodes:
            kg.graph.nodes[n]["embedding"] = [1.0, 1.0]

    monkeypatch.setattr(kg, "compute_node2vec_embeddings", _compute)
    monkeypatch.setattr(
        node2vec_tuning, "recall10", lambda g, q, gt, gamma=0.5, eta=0.25: 0.6
    )
    node2vec_tuning.autotune_node2vec(
        kg, [0], {0: [1]}, var_threshold=0.05, max_evals=5
    )
    assert len(calls) == 2  # one for trial, one for final selection
    assert (tmp_path / "art.json").exists()


def test_autotune_node2vec_time_budget(monkeypatch, tmp_path):
    kg = DummyKG()
    monkeypatch.setattr(node2vec_tuning, "ARTIFACT_PATH", tmp_path / "art.json")
    times = [0.0, 70.0]

    def fake_time():
        return times.pop(0)

    monkeypatch.setattr(node2vec_tuning.time, "monotonic", fake_time)
    monkeypatch.setattr(
        node2vec_tuning, "recall10", lambda g, q, gt, gamma=0.5, eta=0.25: 0.5
    )
    node2vec_tuning.autotune_node2vec(kg, [0], {0: [1]}, max_minutes=0.01)
    assert (tmp_path / "art.json").exists()
