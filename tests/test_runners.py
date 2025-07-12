import types

import networkx as nx

import datacreek.core.runners as runners
from datacreek.core.knowledge_graph import KnowledgeGraph
from datacreek.core.runners import GraphWaveRunner, Node2VecRunner


def test_node2vec_runner_uses_config(monkeypatch):
    kg = KnowledgeGraph()
    kg.graph.add_edge("a", "b")
    captured = {}

    def fake_compute(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(kg, "compute_node2vec_embeddings", fake_compute)
    monkeypatch.setattr(
        runners,
        "load_config",
        lambda: {"embeddings": {"node2vec": {"p": 3.0, "q": 4.0, "dimension": 16}}},
    )
    runner = Node2VecRunner(kg)
    runner.run()
    assert captured["p"] == 3.0
    assert captured["q"] == 4.0
    assert captured["dimensions"] == 16


def test_graphwave_runner_sets_entropy(monkeypatch):
    kg = KnowledgeGraph()
    kg.graph.add_edge("a", "b")
    monkeypatch.setattr(kg, "compute_graphwave_embeddings", lambda *a, **k: None)
    monkeypatch.setattr(kg, "graphwave_entropy", lambda: 0.5)
    runner = GraphWaveRunner(kg)
    runner.run([1.0])
    assert kg.graph["gw_entropy"] == 0.5
