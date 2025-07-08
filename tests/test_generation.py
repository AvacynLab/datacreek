import networkx as nx

from datacreek.analysis.generation import (
    generate_graph_rnn_like,
    generate_graph_rnn,
    generate_graph_rnn_stateful,
    generate_graph_rnn_sequential,
)
from datacreek.utils.graph_text import (
    neighborhood_to_sentence,
    subgraph_to_text,
    graph_to_text,
)
from datacreek.utils.toolformer import insert_tool_calls


def test_generate_graph_rnn_like():
    g = generate_graph_rnn_like(5, 4)
    assert g.number_of_nodes() == 5
    assert g.number_of_edges() >= 0


def test_generate_graph_rnn():
    g = generate_graph_rnn(6, 5, p=0.8)
    assert g.number_of_nodes() == 6
    assert g.number_of_edges() <= 5
    assert g.number_of_edges() > 0


def test_generate_graph_rnn_directed():
    g = generate_graph_rnn(4, 3, p=1.0, directed=True)
    assert g.is_directed()
    assert g.number_of_nodes() == 4


def test_generate_graph_rnn_stateful():
    g = generate_graph_rnn_stateful(5, 4, hidden_dim=4, seed=42)
    assert g.is_directed()
    assert g.number_of_nodes() == 5
    assert g.number_of_edges() <= 4


def test_generate_graph_rnn_sequential():
    g = generate_graph_rnn_sequential(5, 4, hidden_dim=4, seed=42)
    assert g.is_directed()
    assert g.number_of_nodes() == 5
    assert g.number_of_edges() <= 4


def test_generate_graph_rnn_sequential_undirected():
    g = generate_graph_rnn_sequential(5, 4, hidden_dim=4, seed=42, directed=False)
    assert not g.is_directed()
    assert g.number_of_nodes() == 5
    assert g.number_of_edges() <= 4


def test_neighborhood_to_sentence():
    g = nx.Graph()
    g.add_node(0, text="A")
    g.add_node(1, text="B")
    g.add_edge(0, 1, relation="rel")
    sent = neighborhood_to_sentence(g, [0, 1])
    assert "A" in sent and "B" in sent and "rel" in sent


def test_insert_tool_calls():
    text = "Search cats and then search dogs"
    out = insert_tool_calls(
        text,
        [
            ("search", r"search\s+\w+"),
            ("filter", r"dogs"),
        ],
    )
    assert out.count("[TOOL:search") == 1
    assert "[TOOL:filter" in out


def test_subgraph_to_text():
    g = nx.Graph()
    g.add_node("a", text="A")
    g.add_node("b", text="B")
    g.add_edge("a", "b", relation="rel")
    txt = subgraph_to_text(g, ["a", "b"])
    assert "A" in txt and "B" in txt and "rel" in txt


def test_graph_to_text():
    g = nx.Graph()
    g.add_node("a", text="A")
    g.add_node("b", text="B")
    g.add_edge("a", "b", relation="rel")
    txt = graph_to_text(g)
    assert "A" in txt and "B" in txt and "rel" in txt

