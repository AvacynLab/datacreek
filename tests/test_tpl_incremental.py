import networkx as nx
import pytest

from datacreek.analysis.tpl_incremental import tpl_incremental
from datacreek.core.dataset import DatasetBuilder, DatasetType


def _gudhi_available():
    try:
        import gudhi  # noqa: F401
    except Exception:
        return False
    return True


@pytest.mark.skipif(not _gudhi_available(), reason="gudhi required")
def test_tpl_incremental_updates(monkeypatch):
    g = nx.path_graph(4)
    diags = tpl_incremental(g, radius=1)
    assert set(diags) == set(g.nodes())
    hashes = {n: g.nodes[n]["tpl_hash"] for n in g.nodes()}
    diags2 = tpl_incremental(g, radius=1)
    assert diags2 and {n: g.nodes[n]["tpl_hash"] for n in g.nodes()} == hashes
    g.add_edge(0, 3)
    diags3 = tpl_incremental(g, radius=1)
    assert any(g.nodes[n]["tpl_hash"] != hashes.get(n) for n in (0, 3))
    assert diags3


@pytest.mark.skipif(not _gudhi_available(), reason="gudhi required")
def test_tpl_incremental_wrapper(monkeypatch):
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_document("d", source="s")
    ds.add_chunk("d", "c1", "a")
    ds.add_chunk("d", "c2", "b")
    diags = ds.tpl_incremental(radius=1)
    assert diags
    assert any(e.operation == "tpl_incremental" for e in ds.events)
