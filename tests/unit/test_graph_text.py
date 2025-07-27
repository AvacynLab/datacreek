import networkx as nx
from datacreek.utils import graph_text


def _simple_graph():
    g = nx.Graph()
    g.add_node('A', text='Alpha')
    g.add_node('B', text='Bravo')
    g.add_node('C')
    g.add_edge('A', 'B', relation='r1')
    g.add_edge('B', 'C', relation='r2')
    return g


def test_neighborhood_to_sentence():
    g = _simple_graph()
    sent = graph_text.neighborhood_to_sentence(g, ['A', 'B'], depth=1)
    assert "Alpha" in sent and "r1" in sent and "r2" in sent
    assert sent.endswith('.')


def test_subgraph_and_full_text():
    g = _simple_graph()
    sub = graph_text.subgraph_to_text(g, ['A', 'B'])
    assert sub == 'Alpha r1 Bravo.'
    full = graph_text.graph_to_text(g)
    assert 'Alpha r1 Bravo.' in full
    assert 'Bravo r2 C.' in full
