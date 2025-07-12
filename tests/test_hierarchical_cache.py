import tempfile
import networkx as nx
import pytest

from datacreek.analysis.mapper import _cache_put, _cache_get


def test_hierarchical_cache_roundtrip(tmp_path):
    fakeredis = pytest.importorskip("fakeredis")
    client = fakeredis.FakeRedis()
    g = nx.path_graph(3)
    cover = [{0, 1}, {1, 2}]
    lmdb_dir = tmp_path / "lmdb"
    ssd_dir = tmp_path / "ssd"
    _cache_put("g1", g, cover, redis_client=client, lmdb_path=str(lmdb_dir), ssd_dir=str(ssd_dir))
    res = _cache_get("g1", redis_client=client, lmdb_path=str(lmdb_dir), ssd_dir=str(ssd_dir))
    assert res is not None
    nerve, c = res
    assert nerve.number_of_edges() == g.number_of_edges()
    assert c == [set(x) for x in cover]
