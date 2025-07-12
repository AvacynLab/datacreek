import networkx as nx

from datacreek import AutoTuneState
from datacreek.analysis.autotune import autotune_step, recall_at_k


def test_autotune_state_defaults():
    state = AutoTuneState()
    assert hasattr(state, "alpha")
    assert hasattr(state, "gamma")
    assert hasattr(state, "eta")


def test_recall_and_autotune_step():
    g = nx.Graph()
    g.add_node(
        "a",
        embedding=[1.0, 0.0],
        graphwave_embedding=[1.0, 0.0],
        poincare_embedding=[0.0, 0.5],
    )
    g.add_node(
        "b",
        embedding=[0.0, 1.0],
        graphwave_embedding=[0.0, 1.0],
        poincare_embedding=[0.0, 0.6],
    )
    g.add_edge("a", "b")

    rec = recall_at_k(g, ["a"], {"a": ["b"]}, k=1)
    assert rec == 1.0

    state = AutoTuneState()
    emb = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
    labels = {"a": 0, "b": 1}
    res = autotune_step(
        g,
        emb,
        labels,
        [],
        state,
        recall_data=(["a"], {"a": ["b"]}),
        k=1,
        penalty_cfg={"lambda_sigma": 1.0, "lambda_cov": 1.0, "w_rec": 1.0},
    )
    assert res["recall"] == 1.0
