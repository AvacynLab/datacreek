import pytest

from datacreek.core.knowledge_graph import KnowledgeGraph


def test_add_document_and_chunk():
    kg = KnowledgeGraph()
    kg.add_document("doc1", source="paper.pdf")
    kg.add_chunk("doc1", "chunk1", "hello")

    assert kg.graph.nodes["doc1"]["source"] == "paper.pdf"
    assert kg.graph.nodes["chunk1"]["source"] == "paper.pdf"
    assert ("doc1", "chunk1") in kg.graph.edges


def test_search_chunks():
    kg = KnowledgeGraph()
    kg.add_document("doc1", source="paper.pdf")
    kg.add_chunk("doc1", "c1", "hello world")
    kg.add_chunk("doc1", "c2", "another line")

    matches = kg.search_chunks("world")
    assert matches == ["c1"]


def test_generic_search():
    kg = KnowledgeGraph()
    kg.add_document("doc1", source="paper.pdf")
    kg.add_chunk("doc1", "c1", "hello world")
    assert kg.search("hello") == ["c1"]


def test_document_helpers():
    kg = KnowledgeGraph()
    kg.add_document("doc1", source="paper.pdf")
    kg.add_document("guide", source="guide.txt")
    kg.add_chunk("doc1", "c1", "text1")
    kg.add_chunk("doc1", "c2", "text2")

    assert set(kg.search_documents("doc")) == {"doc1"}
    assert kg.get_chunks_for_document("doc1") == ["c1", "c2"]


def test_embedding_search():
    kg = KnowledgeGraph()
    kg.add_document("d", source="s")
    kg.add_chunk("d", "c1", "hello world")
    kg.add_chunk("d", "c2", "another world")
    kg.index.build()
    results = kg.search_embeddings("hello", k=1)
    assert results[0] == "c1"


def test_duplicate_checks():
    kg = KnowledgeGraph()
    kg.add_document("d", source="s")
    with pytest.raises(ValueError):
        kg.add_document("d", source="s")
    kg.add_chunk("d", "c1", "text")
    with pytest.raises(ValueError):
        kg.add_chunk("d", "c1", "text")


def test_hybrid_search():
    kg = KnowledgeGraph()
    kg.add_document("doc", source="s")
    kg.add_chunk("doc", "c1", "hello world")
    kg.add_chunk("doc", "c2", "bonjour le monde")
    kg.add_chunk("doc", "c3", "greetings planet")
    kg.index.build()

    results = kg.search_hybrid("hello", k=2)
    assert results[0] == "c1"
    assert len(results) == 2


def test_chunk_order_and_next_relations():
    kg = KnowledgeGraph()
    kg.add_document("doc", source="s")
    kg.add_chunk("doc", "c1", "t1")
    kg.add_chunk("doc", "c2", "t2")
    kg.add_chunk("doc", "c3", "t3")

    # edges should have sequence numbers
    assert kg.graph.edges["doc", "c1"]["sequence"] == 0
    assert kg.graph.edges["doc", "c2"]["sequence"] == 1
    assert kg.graph.edges["doc", "c3"]["sequence"] == 2

    # next_chunk relations preserve order
    assert ("c1", "c2") in kg.graph.edges
    assert kg.graph.edges["c1", "c2"]["relation"] == "next_chunk"
    assert kg.get_chunks_for_document("doc") == ["c1", "c2", "c3"]


def test_serialization_preserves_order():
    kg = KnowledgeGraph()
    kg.add_document("d", source="s")
    kg.add_chunk("d", "c1", "first")
    kg.add_chunk("d", "c2", "second")

    data = kg.to_dict()
    loaded = KnowledgeGraph.from_dict(data)
    assert loaded.get_chunks_for_document("d") == ["c1", "c2"]


def test_link_similar_chunks():
    kg = KnowledgeGraph()
    kg.add_document("doc", source="s")
    kg.add_chunk("doc", "c1", "hello world")
    kg.add_chunk("doc", "c2", "hello planet")
    kg.add_chunk("doc", "c3", "unrelated text")

    kg.index.build()
    kg.link_similar_chunks(k=1)

    assert ("c1", "c2") in kg.graph.edges
    edge = kg.graph.edges["c1", "c2"]
    assert edge["relation"] == "similar_to"
    assert 0 < edge["similarity"] <= 1


def test_search_with_links():
    kg = KnowledgeGraph()
    kg.add_document("doc", source="s")
    kg.add_chunk("doc", "c1", "hello world")
    kg.add_chunk("doc", "c2", "hello planet")
    kg.add_chunk("doc", "c3", "different text")

    kg.index.build()
    kg.link_similar_chunks(k=1)

    # search should return c1 and also c2 via the similarity edge
    results = kg.search_with_links("hello", k=1, hops=1)
    assert "c1" in results
    assert "c2" in results


def test_search_with_links_data():
    kg = KnowledgeGraph()
    kg.add_document("doc", source="s")
    kg.add_chunk("doc", "c1", "hello world")
    kg.add_chunk("doc", "c2", "hello planet")
    kg.add_chunk("doc", "c3", "different text")

    kg.index.build()
    kg.link_similar_chunks(k=1)

    results = kg.search_with_links_data("hello", k=1, hops=1)
    ids = [r["id"] for r in results]
    assert "c1" in ids
    assert "c2" in ids
    first = next(r for r in results if r["id"] == "c1")
    assert first["text"] == "hello world"
    assert first["document"] == "doc"
    assert first["depth"] == 0
    assert first["path"] == ["c1"]
    second = next(r for r in results if r["id"] == "c2")
    assert second["depth"] == 1
    assert second["path"] == ["c1", "c2"]

