import networkx as nx
from datacreek.analysis.fractal import bootstrap_sigma_db


def test_bootstrap_sigma_basic():
    g = nx.cycle_graph(6)
    sigma = bootstrap_sigma_db(g, [1])
    assert sigma >= 0.0
    assert 'fractal_sigma' in g.graph
