import sys
from pathlib import Path
from types import ModuleType

import networkx as nx
import numpy as np
import pytest
sys.modules.pop("sklearn", None)
sys.modules.pop("sklearn.linear_model", None)
pytest.importorskip("sklearn")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datacreek.analysis import information as info


class DummyLR:
    def __init__(self, *a, **k):
        pass

    def fit(self, X, y):
        self.X_ = np.asarray(X)
        self.y_ = np.asarray(y)

    def predict_proba(self, X):
        X = np.asarray(X)
        p1 = 1 / (1 + np.exp(-X[:, 0]))
        return np.column_stack([1 - p1, p1])


sklearn = ModuleType("sklearn")
sklearn.linear_model = ModuleType("linear_model")
sklearn.linear_model.LogisticRegression = DummyLR
sys.modules.setdefault("sklearn", sklearn)
sys.modules.setdefault("sklearn.linear_model", sklearn.linear_model)


def test_graph_information_bottleneck_basic():
    features = {0: np.array([1.0, 0.0]), 1: np.array([0.0, 1.0])}
    labels = {0: 0, 1: 1}
    loss = info.graph_information_bottleneck(features, labels, beta=0.5)
    assert loss > 0


def test_graph_information_bottleneck_no_features():
    with pytest.raises(ValueError):
        info.graph_information_bottleneck({}, {0: 0})


def test_prototype_subgraph_and_errors():
    g = nx.path_graph(3)
    features = {i: np.array([i, i]) for i in g.nodes}
    labels = {0: 0, 1: 0, 2: 1}
    sub = info.prototype_subgraph(g, features, labels, class_id=0, radius=1)
    assert set(sub.nodes) == {0, 1}
    with pytest.raises(ValueError):
        info.prototype_subgraph(g, features, labels, class_id=3)


def test_mdl_description_and_selection():
    g = nx.cycle_graph(4)
    motifs = [nx.path_graph(2), nx.cycle_graph(3)]
    length = info.mdl_description_length(g, motifs)
    assert length > 0
    selected = info.select_mdl_motifs(g, motifs)
    assert selected


def test_entropy_helpers():
    g = nx.path_graph(4)
    assert info.graph_entropy(nx.Graph()) == 0.0
    e = info.graph_entropy(g)
    assert e > 0
    sub_e = info.subgraph_entropy(g, [0, 1, 2])
    assert sub_e <= e + 1e-6
    struct_e = info.structural_entropy(nx.cycle_graph(4), tau=2)
    assert struct_e >= 0
