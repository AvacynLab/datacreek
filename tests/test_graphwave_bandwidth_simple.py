import networkx as nx
import numpy as np
import pytest

import datacreek.analysis.graphwave_bandwidth as gwb


def test_estimate_lambda_max_numpy(monkeypatch):
    """The numpy path should approximate the true spectral radius."""
    g = nx.path_graph(2)
    monkeypatch.setattr(gwb, "sp", None)
    val = gwb.estimate_lambda_max(g, iters=5)
    assert 1.9 < val < 2.1


class DummySp:
    @staticmethod
    def diags(arr):
        return np.diag(arr)

    @staticmethod
    def eye(n, format="csr"):
        return np.eye(n)


def test_estimate_lambda_max_scipy(monkeypatch):
    """Using the scipy branch should give the same result."""
    g = nx.path_graph(2)
    monkeypatch.setattr(gwb, "sp", DummySp)
    monkeypatch.setattr(
        nx,
        "to_scipy_sparse_array",
        lambda graph, format="csr": nx.to_numpy_array(graph),
    )
    val = gwb.estimate_lambda_max(g, iters=3)
    assert 1.9 < val < 2.1


def test_update_graphwave_bandwidth(monkeypatch):
    """Bandwidth updates when the spectral radius changes by threshold."""
    g = nx.path_graph(2)
    monkeypatch.setattr(gwb, "sp", None)
    t = gwb.update_graphwave_bandwidth(g, threshold=0.0, iters=3)
    assert g.graph["gw_lambda_max"] == pytest.approx(2.0, rel=0.1)
    assert t == pytest.approx(3.0 / 2.0, rel=0.1)
    g.graph["gw_lambda_max"] = 4.0
    t2 = gwb.update_graphwave_bandwidth(g, threshold=0.4, iters=3)
    assert g.graph["gw_lambda_max"] == pytest.approx(2.0, rel=0.1)
    assert t2 == pytest.approx(3.0 / 2.0, rel=0.1)
