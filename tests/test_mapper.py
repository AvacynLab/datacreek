import networkx as nx

from datacreek.analysis.mapper import inverse_mapper, mapper_nerve
from datacreek.core.dataset import DatasetBuilder
from datacreek.pipelines import DatasetType


def test_mapper_functions():
    g = nx.path_graph(5)
    nerve, cover = mapper_nerve(g, radius=1)
    assert nerve.number_of_nodes() == len(cover)
    recon = inverse_mapper(nerve, cover)
    for u, v in g.edges():
        assert recon.has_edge(u, v)


def test_dataset_mapper_wrappers():
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_document("d", source="s")
    for i in range(5):
        ds.add_chunk("d", f"c{i}", str(i))
    for i in range(4):
        ds.graph.graph.add_edge(f"c{i}", f"c{i+1}")
    nerve, cover = ds.mapper_nerve(radius=1)
    assert len(cover) == nerve.number_of_nodes()
    g = ds.inverse_mapper(nerve, cover)
    for u, v in ds.graph.graph.edges():
        assert g.has_edge(u, v)
    assert any(e.operation == "mapper_nerve" for e in ds.events)
    assert any(e.operation == "inverse_mapper" for e in ds.events)
