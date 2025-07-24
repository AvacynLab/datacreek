import networkx as nx

from datacreek.utils.graph_text import (
    graph_to_text,
    neighborhood_to_sentence,
    subgraph_to_text,
)


def test_neighborhood_to_sentence_basic():
    g = nx.Graph()
    g.add_node("a", text="A")
    g.add_node("b", text="B")
    g.add_edge("a", "b", relation="rel")
    assert neighborhood_to_sentence(g, ["a", "b"]) == "A rel B."


def test_neighborhood_to_sentence_depth():
    g = nx.Graph()
    g.add_node("a", text="A")
    g.add_node("b", text="B")
    g.add_node("c", text="C")
    g.add_edge("a", "b", relation="rel")
    g.add_edge("b", "c", relation="to")
    sent = neighborhood_to_sentence(g, ["b"], depth=1)
    assert "B" in sent and "A" in sent and "C" in sent and "(" in sent


def test_neighborhood_to_sentence_empty():
    g = nx.Graph()
    assert neighborhood_to_sentence(g, []) == ""


def test_subgraph_to_text_and_graph_to_text():
    g = nx.Graph()
    g.add_node("a", text="A")
    g.add_node("b", text="B")
    g.add_edge("a", "b", relation="rel")
    assert subgraph_to_text(g, ["a", "b"]) == "A rel B."
    assert graph_to_text(g) == "A rel B."


def test_graph_to_text_default_relation():
    g = nx.Graph()
    g.add_edge(0, 1)
    assert graph_to_text(g) == "0 -> 1."
