import networkx as nx
import numpy as np
import types
import sys
import datacreek.core.runners as runners

class FakeKG:
    def __init__(self):
        self.graph = nx.Graph()
        self.graph.add_edge('a', 'b')
    def compute_node2vec_embeddings(self, **kwargs):
        for n in self.graph.nodes:
            self.graph.nodes[n]['embedding'] = [1.0, 0.0]
    def compute_graphwave_embeddings(self, *a, **k):
        for n in self.graph.nodes:
            self.graph.nodes[n]['graphwave_embedding'] = [0.1, 0.2]
    def graphwave_entropy(self):
        return 0.3


def test_node2vec_runner(monkeypatch):
    kg = FakeKG()
    calls = {}
    monkeypatch.setattr(runners, 'load_config', lambda: {'embeddings': {'node2vec': {'p': 2, 'q': 3, 'd': 4}}})
    monkeypatch.setattr(sys.modules['datacreek.analysis.monitoring'], 'update_metric', lambda n, v: calls.setdefault(n, v))
    runner = runners.Node2VecRunner(kg)
    runner.run()
    assert kg.graph.graph['var_norm'] == 0.0
    assert calls['n2v_var_norm'] == 0.0


def test_graphwave_runner(monkeypatch):
    kg = FakeKG()
    fake_mod = types.SimpleNamespace(update_graphwave_bandwidth=lambda g: 1.0)
    monkeypatch.setitem(sys.modules, 'datacreek.analysis.graphwave_bandwidth', fake_mod)
    monkeypatch.setattr(sys.modules['datacreek.analysis.fractal'], 'graphwave_embedding_chebyshev', lambda g, s, num_points=1, order=1: {n: np.zeros(1) for n in g.nodes()})
    metrics = {}
    monkeypatch.setattr(sys.modules['datacreek.analysis.monitoring'], 'update_metric', lambda n, v: metrics.setdefault(n, v))
    runner = runners.GraphWaveRunner(kg)
    runner.run([1.0])
    assert kg.graph.graph['gw_entropy'] == 0.3
    assert metrics['gw_entropy'] == 0.3


def test_poincare_runner_rescale(monkeypatch):
    kg = FakeKG()
    monkeypatch.setattr(sys.modules['datacreek.analysis.fractal'], 'poincare_embedding', lambda *a, **k: {n: np.array([1.0, 0.0]) for n in kg.graph.nodes})
    runner = runners.PoincareRunner(kg)
    runner.run()
    for v in kg.graph.nodes:
        assert np.isclose(np.linalg.norm(kg.graph.nodes[v]['poincare_embedding']), 0.8)
