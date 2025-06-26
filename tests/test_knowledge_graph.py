import pytest
import requests

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
    assert kg.get_next_chunk("c1") == "c2"
    assert kg.get_previous_chunk("c3") == "c2"


def test_serialization_preserves_order():
    kg = KnowledgeGraph()
    kg.add_document("d", source="s")
    kg.add_chunk("d", "c1", "first")
    kg.add_chunk("d", "c2", "second")

    data = kg.to_dict()
    loaded = KnowledgeGraph.from_dict(data)
    assert loaded.get_chunks_for_document("d") == ["c1", "c2"]


def test_section_helpers():
    kg = KnowledgeGraph()
    kg.add_document("doc", source="s")
    kg.add_section("doc", "s1", title="Intro")
    kg.add_section("doc", "s2", title="Body")
    kg.add_chunk("doc", "c1", "t1", section_id="s1")
    kg.add_chunk("doc", "c2", "t2", section_id="s2")

    assert kg.get_sections_for_document("doc") == ["s1", "s2"]
    assert kg.get_chunks_for_section("s1") == ["c1"]
    assert kg.get_section_for_chunk("c2") == "s2"
    assert kg.get_next_section("s1") == "s2"
    assert kg.get_previous_section("s2") == "s1"


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


def test_embeddings_filter_by_type():
    kg = KnowledgeGraph()
    kg.add_document("doc", source="s")
    kg.add_chunk("doc", "c1", "hello world")
    kg.add_entity("e1", "hello world")
    kg.index.build()

    # Should return only chunk IDs when searching embeddings
    results = kg.search_embeddings("hello", k=1, node_type="chunk")
    assert results == ["c1"]


def test_hybrid_filter_by_type():
    kg = KnowledgeGraph()
    kg.add_document("doc", source="s")
    kg.add_chunk("doc", "c1", "hello world")
    kg.add_entity("e1", "hello world")
    kg.index.build()

    # Should not return entity nodes when filtering for chunks
    results = kg.search_hybrid("hello", k=2, node_type="chunk")
    assert results == ["c1"]

    # Should return the entity when requested
    results = kg.search_hybrid("hello", k=1, node_type="entity")
    assert results == ["e1"]


def test_link_similar_chunks_ignores_entities():
    kg = KnowledgeGraph()
    kg.add_document("doc", source="s")
    kg.add_chunk("doc", "c1", "hello world")
    kg.add_chunk("doc", "c2", "hello planet")
    kg.add_entity("e1", "hello universe")
    kg.index.build()
    kg.link_similar_chunks(k=1)

    # no similar_to edge should involve entity nodes
    for u, v, d in kg.graph.edges(data=True):
        if d.get("relation") == "similar_to":
            assert kg.graph.nodes[u]["type"] == "chunk"
            assert kg.graph.nodes[v]["type"] == "chunk"


def test_fact_search():
    kg = KnowledgeGraph()
    fid = kg.add_fact("Paris", "capital_of", "France")
    kg.index.build()

    assert kg.search("capital_of", node_type="fact") == [fid]
    assert kg.search_embeddings("capital_of", k=1, node_type="fact") == [fid]
    assert kg.search_hybrid("capital_of", k=1, node_type="fact") == [fid]


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


def test_community_and_trust():
    kg = KnowledgeGraph()
    kg.add_document("doc", source="s")
    for i in range(3):
        kg.add_chunk("doc", f"c{i}", f"text {i}")
    kg.index.build()
    kg.cluster_chunks(n_clusters=1)
    kg.summarize_communities()
    kg.score_trust()

    comms = [n for n, d in kg.graph.nodes(data=True) if d.get("type") == "community"]
    assert len(comms) == 1
    cid = comms[0]
    assert "summary" in kg.graph.nodes[cid]
    for n, d in kg.graph.nodes(data=True):
        if d.get("type") == "chunk":
            assert "trust" in d


def test_entity_groups():
    kg = KnowledgeGraph()
    kg.add_entity("e1", "first entity")
    kg.add_entity("e2", "second entity")
    kg.add_entity("e3", "other text")
    kg.index.build()
    kg.cluster_entities(n_clusters=1)
    kg.summarize_entity_groups()

    groups = [n for n, d in kg.graph.nodes(data=True) if d.get("type") == "entity_group"]
    assert len(groups) == 1
    gid = groups[0]
    assert "summary" in kg.graph.nodes[gid]
    members = [u for u, _ in kg.graph.in_edges(gid)]
    assert set(members) == {"e1", "e2", "e3"}


def test_edge_provenance_and_trust():
    kg = KnowledgeGraph()
    kg.add_document("d", source="src")
    kg.add_chunk("d", "c1", "text", source="src")
    kg.add_entity("e", "ent", source="src")
    kg.link_entity("c1", "e", provenance="src")
    kg.index.build()
    kg.score_trust()

    assert kg.graph.edges["d", "c1"]["provenance"] == "src"
    assert "trust" in kg.graph.edges["d", "c1"]
    assert kg.graph.edges["c1", "e"]["provenance"] == "src"
    assert "trust" in kg.graph.edges["c1", "e"]


def test_update_embeddings():
    kg = KnowledgeGraph()
    kg.add_document("d", source="s")
    kg.add_chunk("d", "c1", "hello world")
    kg.index.build()
    kg.update_embeddings()
    emb = kg.graph.nodes["c1"].get("embedding")
    assert isinstance(emb, list)
    assert len(emb) > 0


def test_deduplicate_and_resolve_entities(monkeypatch):
    kg = KnowledgeGraph()
    kg.add_document("d", source="s")
    kg.add_chunk("d", "c1", "hello")
    kg.add_chunk("d", "c2", "hello")
    kg.add_entity("e1", "Beethoven")
    kg.add_entity("e2", "Ludwig van Beethoven")
    removed = kg.deduplicate_chunks()
    assert removed == 1
    merged = kg.resolve_entities(threshold=0.5)
    assert merged >= 1
    assert "e2" not in kg.graph.nodes


def test_enrich_entity_wikidata(monkeypatch):
    kg = KnowledgeGraph()
    kg.add_entity("e1", "Beethoven")

    class FakeResponse:
        def __init__(self, data):
            self._data = data

        def json(self):
            return self._data

    def fake_get(url, params=None, timeout=10):
        return FakeResponse({"search": [{"id": "Q1", "description": "composer"}]})

    monkeypatch.setattr(requests, "get", fake_get)
    kg.enrich_entity_wikidata("e1")
    node = kg.graph.nodes["e1"]
    assert node.get("wikidata_id") == "Q1"
    assert node.get("description") == "composer"


def test_compute_centrality():
    kg = KnowledgeGraph()
    kg.add_document("d", source="s")
    kg.add_chunk("d", "c1", "hello")
    kg.add_chunk("d", "c2", "world")
    kg.add_entity("e1", "A")
    kg.add_entity("e2", "B")
    kg.link_entity("c1", "e1")
    kg.link_entity("c2", "e2")
    kg.compute_centrality()
    assert "centrality" in kg.graph.nodes["e1"]


def test_predict_links():
    kg = KnowledgeGraph()
    kg.add_entity("e1", "Beethoven")
    kg.add_entity("e2", "Ludwig van Beethoven")
    kg.predict_links(threshold=0.4)
    assert kg.graph.has_edge("e1", "e2")


def test_consolidate_schema():
    kg = KnowledgeGraph()
    kg.add_entity("e1", "A")
    kg.add_entity("e2", "B")
    kg.graph.nodes["e1"]["type"] = "ENTITY"
    kg.graph.add_edge("e1", "e2", relation="RELATED")

    kg.consolidate_schema()

    assert kg.graph.nodes["e1"]["type"] == "entity"
    assert kg.graph.edges["e1", "e2"]["relation"] == "related"


def test_entity_helpers():
    kg = KnowledgeGraph()
    kg.add_document("d", source="s")
    kg.add_chunk("d", "c1", "Paris is nice")
    kg.add_chunk("d", "c2", "France is big")
    kg.add_entity("Paris", "Paris")
    kg.add_entity("France", "France")
    kg.link_entity("c1", "Paris")
    kg.link_entity("c2", "France")
    fid = kg.add_fact("Paris", "capital_of", "France")

    assert kg.get_chunks_for_entity("Paris") == ["c1"]
    assert kg.get_facts_for_entity("France") == [fid]


def test_fact_helpers():
    kg = KnowledgeGraph()
    kg.add_document("d", source="s")
    kg.add_chunk("d", "c1", "text")
    fid = kg.add_fact("A", "related", "B")
    kg.graph.add_edge("c1", fid, relation="has_fact")

    assert kg.get_facts_for_chunk("c1") == [fid]
    assert kg.get_facts_for_document("d") == [fid]
    assert kg.get_chunks_for_fact(fid) == ["c1"]
    assert set(kg.get_entities_for_fact(fid)) == {"A", "B"}


def test_find_facts():
    kg = KnowledgeGraph()
    kg.add_fact("A", "likes", "B", fact_id="f1")
    kg.add_fact("A", "likes", "C", fact_id="f2")

    assert set(kg.find_facts(subject="A", predicate="likes")) == {"f1", "f2"}


def test_entity_lookup_helpers():
    kg = KnowledgeGraph()
    kg.add_document("doc", source="s")
    kg.add_chunk("doc", "c1", "Paris is nice")
    kg.add_chunk("doc", "c2", "Berlin is big")
    kg.add_entity("Paris", "Paris")
    kg.add_entity("Berlin", "Berlin")
    kg.link_entity("c1", "Paris")
    kg.link_entity("c2", "Berlin")

    assert kg.get_entities_for_chunk("c1") == ["Paris"]
    assert set(kg.get_entities_for_document("doc")) == {"Paris", "Berlin"}
    assert kg.get_documents_for_entity("Berlin") == ["doc"]
