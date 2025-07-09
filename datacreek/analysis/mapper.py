import networkx as nx
from typing import Iterable, List, Set, Tuple


def mapper_cover(graph: nx.Graph, radius: int = 1) -> List[Set[object]]:
    """Return a greedy cover of ``graph`` using BFS balls of ``radius``.

    The function repeatedly selects an uncovered node and forms a ball of
    radius ``radius`` around it until all nodes are covered.
    """
    remaining = set(graph.nodes())
    cover: List[Set[object]] = []
    while remaining:
        seed = remaining.pop()
        ball = set(nx.single_source_shortest_path_length(graph, seed, cutoff=radius).keys())
        cover.append(ball | {seed})
        remaining.difference_update(ball)
    return cover


def mapper_nerve(graph: nx.Graph, radius: int = 1) -> Tuple[nx.Graph, List[Set[object]]]:
    """Return the Mapper nerve of ``graph`` and the covering used."""
    cover = mapper_cover(graph, radius)
    nerve = nx.Graph()
    nerve.add_nodes_from(range(len(cover)))
    for i, A in enumerate(cover):
        for j in range(i + 1, len(cover)):
            if A & cover[j]:
                nerve.add_edge(i, j)
    return nerve, cover


def inverse_mapper(nerve: nx.Graph, cover: Iterable[Set[object]]) -> nx.Graph:
    """Reconstruct a graph from its Mapper ``nerve`` and ``cover`` sets."""
    cover_list = [set(c) for c in cover]
    g = nx.Graph()
    for cluster in cover_list:
        for node in cluster:
            g.add_node(node)
        for u in cluster:
            for v in cluster:
                if u != v:
                    g.add_edge(u, v)
    for u, v in nerve.edges():
        for a in cover_list[int(u)]:
            for b in cover_list[int(v)]:
                if a != b:
                    g.add_edge(a, b)
    return g
