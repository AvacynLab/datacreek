import networkx as nx
import numpy as np

from datacreek.analysis.filtering import (
    entropy_triangle_threshold,
    filter_semantic_cycles,
)


def test_filter_semantic_cycles_removes_trivial():
    g = nx.Graph()
    g.add_node("a", text="the")
    g.add_node("b", text="and")
    g.add_node("c", text="of")
    g.add_edges_from([("a", "b"), ("b", "c"), ("c", "a")])
    filtered = filter_semantic_cycles(g, max_len=3)
    assert filtered.number_of_edges() == 0


def test_filter_semantic_cycles_keeps_meaningful():
    g = nx.Graph()
    g.add_node("x", text="apple")
    g.add_node("y", text="banana")
    g.add_node("z", text="carrot")
    g.add_edges_from([("x", "y"), ("y", "z"), ("z", "x")])
    filtered = filter_semantic_cycles(g, max_len=3)
    assert filtered.number_of_edges() == 3


def test_entropy_triangle_threshold():
    g = nx.Graph()
    g.add_edge(1, 2, weight=1.0)
    g.add_edge(2, 3, weight=2.0)
    arr = np.array([1.0, 2.0])
    p = arr / arr.sum()
    H = -float(np.sum(p * np.log(p)) / np.log(2.0))
    expected = max(1, int(round(10.0 * H)))
    th = entropy_triangle_threshold(g, base=2.0, scale=10.0)
    assert th == expected
