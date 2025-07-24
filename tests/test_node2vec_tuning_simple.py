import json
from types import ModuleType
import sys
from pathlib import Path

# Provide a lightweight skopt stub so the module loads without the real package
class DummyOpt:
    def __init__(self, *a, **k):
        self.i = 0
        self.logged = []

    def ask(self):
        self.i += 1
        return [0.1 * self.i, 0.2 * self.i]

    def tell(self, cand, loss):
        self.logged.append((tuple(cand), loss))

skopt = ModuleType("skopt")
skopt.Optimizer = DummyOpt
sys.modules.setdefault("skopt", skopt)

# Ensure the repository root is on ``sys.path`` so ``datacreek`` is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import datacreek.analysis.node2vec_tuning as n2v

class SimpleGraph:
    def __init__(self):
        self.nodes_dict = {}
        self.edges_list = []
        self.graph = self  # mimic networkx attribute

    def add_node(self, n, **data):
        self.nodes_dict[n] = data

    def add_edge(self, u, v):
        self.edges_list.append((u, v))

    def nodes(self, data=False):
        if data:
            for n, d in self.nodes_dict.items():
                yield n, d
        else:
            return list(self.nodes_dict.keys())

    def edges(self):
        return list(self.edges_list)


class DummyKG:
    def __init__(self):
        self.graph = SimpleGraph()
        self.graph.add_node(0, embedding=[0.0])
        self.graph.add_node(1, embedding=[1.0])
        self.graph.add_edge(0, 1)

    def compute_node2vec_embeddings(self, *, p: float = 1.0, q: float = 1.0):
        for n in self.graph.nodes_dict:
            self.graph.nodes_dict[n]["embedding"] = [p + n, q + n]


def test_dataset_hash_stable():
    kg = DummyKG()
    h1 = n2v._dataset_hash(kg)
    # reorder nodes and edges; hash should remain the same
    kg.graph.nodes_dict = {1: kg.graph.nodes_dict[1], 0: kg.graph.nodes_dict[0]}
    kg.graph.edges_list = [(0,1)]
    h2 = n2v._dataset_hash(kg)
    assert h1 == h2


def test_var_norm_and_save(tmp_path):
    kg = DummyKG()
    v = n2v._var_norm(kg)
    assert v > 0
    path = tmp_path / "art.json"
    n2v._save_artifact(path, "d", 0.1, 0.2)
    data = json.loads(path.read_text())
    assert data["p"] == 0.1 and data["q"] == 0.2


def test_autotune_node2vec_basic(monkeypatch, tmp_path):
    kg = DummyKG()
    monkeypatch.setattr(n2v, "BEST_PQ_PATH", tmp_path / "best.json")
    monkeypatch.setattr(n2v, "Optimizer", DummyOpt)
    monkeypatch.setattr(n2v, "recall10", lambda *a, **k: 0.5)
    monkeypatch.setattr(n2v, "_var_norm", lambda g: 1.0)
    best = n2v.autotune_node2vec(kg, [0], {0: [1]}, max_evals=2, max_minutes=0.01)
    assert isinstance(best, tuple) and len(best) == 2
    assert (tmp_path / "best.json").exists()
