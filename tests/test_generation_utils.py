import networkx as nx

from datacreek.analysis.generation import bias_reweighting, sheaf_consistency_real


def test_sheaf_consistency_real_simple():
    g = nx.path_graph(3)
    for u, v in g.edges():
        g.edges[u, v]["sheaf_sign"] = 1
    score = sheaf_consistency_real(g, [0.0, 0.0, 0.0])
    assert score == 1.0


def test_bias_reweighting_adjusts(tmp_path):
    neigh = {"A": 1, "B": 0}
    global_d = {"A": 5, "B": 5}
    w = {"A": 1.0, "B": 1.0}
    out = bias_reweighting(neigh, global_d, w, threshold=0.0)
    assert out["B"] > w["B"]
