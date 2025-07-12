from typing import Iterable, List, Set, Tuple

import networkx as nx


def mapper_cover(graph: nx.Graph, radius: int = 1) -> List[Set[object]]:
    """Return a greedy cover of ``graph`` using BFS balls of ``radius``.

    The function repeatedly selects an uncovered node and forms a ball of
    radius ``radius`` around it until all nodes are covered.
    """
    remaining = set(graph.nodes())
    cover: List[Set[object]] = []
    while remaining:
        seed = remaining.pop()
        ball = set(
            nx.single_source_shortest_path_length(graph, seed, cutoff=radius).keys()
        )
        cover.append(ball | {seed})
        remaining.difference_update(ball)
    return cover


def mapper_nerve(
    graph: nx.Graph, radius: int = 1
) -> Tuple[nx.Graph, List[Set[object]]]:
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


import pickle
from pathlib import Path
from typing import Iterable, Optional

try:  # optional dependencies
    import redis
except Exception:  # pragma: no cover - optional
    redis = None  # type: ignore

try:
    import lmdb  # type: ignore
except Exception:  # pragma: no cover - optional
    lmdb = None  # type: ignore


def _cache_put(
    key: str,
    nerve: nx.Graph,
    cover: Iterable[set[object]],
    *,
    redis_client: Optional["redis.Redis"] = None,
    lmdb_path: str = "hot_subgraph",
    ssd_dir: str = "nerve_cache",
    ttl: int = 3600,
) -> None:
    """Store ``nerve`` and ``cover`` in hierarchical caches.

    Parameters
    ----------
    key:
        Identifier for the cached subgraph.
    nerve, cover:
        Mapper nerve representation to persist.
    ttl:
        Time-to-live in seconds for the Redis entry. Defaults to one hour.
    """
    data = pickle.dumps((nx.node_link_data(nerve), [list(c) for c in cover]))
    if redis_client is None and redis is not None:  # pragma: no cover - fallback
        try:
            redis_client = redis.Redis()
        except Exception:
            redis_client = None
    if redis_client is not None:
        try:
            redis_client.hset("recent_nerve", key, data)
            redis_client.expire("recent_nerve", int(ttl))
        except Exception:  # pragma: no cover - network errors
            pass
    if lmdb is not None:
        try:
            env = lmdb.open(lmdb_path, map_size=10_485_760)
            with env.begin(write=True) as txn:
                txn.put(key.encode(), data)
            env.close()
        except Exception:  # pragma: no cover - disk errors
            pass
    try:
        Path(ssd_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(ssd_dir) / f"{key}.pkl", "wb") as fh:
            fh.write(data)
    except Exception:  # pragma: no cover - disk errors
        pass


def _cache_get(
    key: str,
    *,
    redis_client: Optional["redis.Redis"] = None,
    lmdb_path: str = "hot_subgraph",
    ssd_dir: str = "nerve_cache",
) -> Optional[tuple[nx.Graph, list[set[object]]]]:
    """Retrieve cached Mapper nerve from hierarchical caches."""
    blob = None
    if redis_client is None and redis is not None:  # pragma: no cover
        try:
            redis_client = redis.Redis()
        except Exception:
            redis_client = None
    if redis_client is not None:
        try:
            blob = redis_client.hget("recent_nerve", key)
        except Exception:
            blob = None
    if blob is None and lmdb is not None:
        try:
            env = lmdb.open(lmdb_path, readonly=True, lock=False)
            with env.begin() as txn:
                blob = txn.get(key.encode())
            env.close()
        except Exception:
            blob = None
    if blob is None:
        try:
            with open(Path(ssd_dir) / f"{key}.pkl", "rb") as fh:
                blob = fh.read()
        except Exception:
            blob = None
    if blob is None:
        return None
    nerve_data, cover_data = pickle.loads(blob)
    nerve = nx.node_link_graph(nerve_data)
    cover = [set(c) for c in cover_data]
    return nerve, cover


def _hash_graph(graph: nx.Graph) -> str:
    """Return SHA1 hash of ``graph`` structure."""

    import hashlib

    edges = sorted((sorted((str(u), str(v))) for u, v in graph.edges()))
    nodes = sorted(str(n) for n in graph.nodes())
    h = hashlib.sha1()
    h.update(pickle.dumps((nodes, edges)))
    return h.hexdigest()


def cache_mapper_nerve(
    graph: nx.Graph,
    radius: int,
    *,
    redis_client: Optional["redis.Redis"] = None,
    lmdb_path: str = "hot_subgraph",
    ssd_dir: str = "nerve_cache",
    ttl: int = 3600,
) -> tuple[nx.Graph, list[set[object]]]:
    """Return cached Mapper nerve for ``graph`` or compute it."""

    key = f"{radius}_{_hash_graph(graph)}"
    res = _cache_get(
        key, redis_client=redis_client, lmdb_path=lmdb_path, ssd_dir=ssd_dir
    )
    if res is not None:
        return res

    nerve, cover = mapper_nerve(graph, radius)
    _cache_put(
        key,
        nerve,
        cover,
        redis_client=redis_client,
        lmdb_path=lmdb_path,
        ssd_dir=ssd_dir,
        ttl=ttl,
    )
    return nerve, cover
