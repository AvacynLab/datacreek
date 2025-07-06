import networkx as nx
from datacreek.analysis.generation import generate_graph_rnn_like


def test_generate_graph_rnn_like():
    g = generate_graph_rnn_like(5, 4)
    assert g.number_of_nodes() == 5
    assert g.number_of_edges() >= 0
