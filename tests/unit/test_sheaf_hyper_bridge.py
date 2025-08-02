import networkx as nx

from datacreek.analysis.sheaf_hyper_bridge import sheaf_hyper_bridge_score


def test_sheaf_hyper_bridge_score_dense():
    g = nx.complete_graph(5)
    for u, v in g.edges():
        g[u][v]["sheaf_sign"] = 1
    score = sheaf_hyper_bridge_score(g)
    assert 0.0 <= score <= 1.0
    assert score > 0.7
