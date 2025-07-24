import sys
import types
import networkx as nx

# Provide stub for faiss so datacreek package imports succeed without the actual dependency
sys.modules.setdefault("faiss", types.ModuleType("faiss"))

from datacreek.analysis import symmetry


def test_cycle_automorphisms_and_group():
    g = nx.cycle_graph(4)
    autos = symmetry.automorphisms(g, max_count=10)
    assert len(autos) == 8
    orbits = symmetry.automorphism_orbits(g, max_count=10)
    assert set().union(*orbits) == set(g.nodes())
    order = symmetry.automorphism_group_order(g, max_count=10)
    assert order >= 8


def test_quotient_graph_orbits():
    g = nx.path_graph(4)
    orbits = symmetry.automorphism_orbits(g)
    expected = {frozenset({0, 3}), frozenset({1, 2})}
    assert {frozenset(o) for o in orbits} == expected
    q, mapping = symmetry.quotient_graph(g, orbits)
    assert q.number_of_nodes() == 2
    assert set(mapping.keys()) == set(g.nodes())
    assert set(mapping.values()) == {0, 1}


def test_quotient_graph_empty():
    g = nx.path_graph(3)
    q, mapping = symmetry.quotient_graph(g, [])
    assert q.number_of_nodes() == g.number_of_nodes()
    assert mapping == {n: i for i, n in enumerate(g.nodes())}
