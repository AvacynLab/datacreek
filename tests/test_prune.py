from datacreek.core.dataset import DatasetBuilder
from datacreek.core.knowledge_graph import KnowledgeGraph
from datacreek.pipelines import DatasetType


def test_prune_sources_kg():
    kg = KnowledgeGraph()
    kg.add_document("doc1", source="bad")
    kg.add_chunk("doc1", "c1", "text", source="bad")
    kg.add_entity("e1", "ent", source="good")
    kg.link_entity("c1", "e1", provenance="bad")

    removed = kg.prune_sources(["bad"])
    assert removed == 2
    assert "doc1" not in kg.graph.nodes
    assert "c1" not in kg.graph.nodes
    # entity should remain
    assert "e1" in kg.graph.nodes


def test_prune_sources_wrapper():
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_document("doc1", source="bad")
    ds.add_chunk("doc1", "c1", "text", source="bad")
    ds.add_document("doc2", source="good")

    removed = ds.prune_sources(["bad"])
    assert removed == 2
    assert ds.search_documents("doc1") == []
    assert ds.search_documents("doc2") == ["doc2"]
