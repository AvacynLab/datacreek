import sys
from pathlib import Path
from types import ModuleType

import networkx as nx
import numpy as np

# stub sklearn modules used in analysis utilities
sklearn = ModuleType("sklearn")
sklearn.cross_decomposition = ModuleType("cross_decomposition")
sklearn.linear_model = ModuleType("linear_model")
sklearn.cross_decomposition.CCA = lambda *a, **k: None
sklearn.gaussian_process = ModuleType("gaussian_process")


class DummyGPR:
    def __init__(self, *a, **k):
        pass

    def fit(self, X, y):
        self.X_ = np.asarray(X)
        self.y_ = np.asarray(y)

    def predict(self, X, return_std=False):
        mu = np.mean(self.y_)
        if return_std:
            return np.array([mu]), np.array([0.1])
        return np.array([mu])


sklearn.gaussian_process.GaussianProcessRegressor = DummyGPR


class DummyLR:
    def __init__(self, *a, **k):
        pass

    def fit(self, X, y):
        self._n = len(y)

    def predict_proba(self, X):
        return np.tile([0.5, 0.5], (len(X), 1))


sklearn.linear_model.LogisticRegression = DummyLR
sys.modules.setdefault("sklearn", sklearn)
sys.modules.setdefault("sklearn.cross_decomposition", sklearn.cross_decomposition)
sys.modules.setdefault("sklearn.linear_model", sklearn.linear_model)
sys.modules.setdefault("sklearn.gaussian_process", sklearn.gaussian_process)

# stub scipy for expected improvement helpers
scipy = ModuleType("scipy")
scipy.stats = ModuleType("stats")
scipy.stats.norm = type(
    "N",
    (),
    {"cdf": staticmethod(lambda x: 0.5), "pdf": staticmethod(lambda x: 0.1)},
)()
sys.modules.setdefault("scipy", scipy)
sys.modules.setdefault("scipy.stats", scipy.stats)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datacreek.analysis import autotune as auto
from datacreek.analysis.autotune import AutoTuneState, update_theta

auto.hybrid_score = lambda *a, **k: 1.0
recall_at_k = auto.recall_at_k


def test_recall_at_k_basic():
    g = nx.Graph()
    for i in range(3):
        g.add_node(
            i,
            embedding=[float(i)],
            graphwave_embedding=[float(i)],
            poincare_embedding=[float(i)],
        )
    g.add_edge(0, 1)
    r = recall_at_k(g, [0], {0: [1]}, k=1)
    assert r == 1.0


def test_update_theta_changes_state():
    state = AutoTuneState()
    metrics = {
        "cost": 0.3,
        "tau": 2,
        "eps": 0.1,
        "beta": 0.2,
        "delta": 0.05,
        "jitter": 0.1,
    }
    update_theta(state, metrics)
    assert state.tau == 2
    assert state.eps == 0.1
    assert state.beta == 0.2
    assert state.delta == 0.05
