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
    assert 0 <= edge["similarity"] <= 1


def test_link_similar_sections():
    kg = KnowledgeGraph()
    kg.add_document("doc", source="s")
    kg.add_section("doc", "s1", title="Intro")
    kg.add_section("doc", "s2", title="Introduction")
    kg.add_section("doc", "s3", title="Other")

    kg.index.build()
    kg.link_similar_sections(k=1)

    assert ("s1", "s2") in kg.graph.edges
    edge = kg.graph.edges["s1", "s2"]
    assert edge["relation"] == "similar_to"
    assert 0 <= edge["similarity"] <= 1


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


def test_get_similar_chunks():
    kg = KnowledgeGraph()
    kg.add_document("doc", source="s")
    kg.add_chunk("doc", "c1", "hello world")
    kg.add_chunk("doc", "c2", "hello planet")
    kg.add_chunk("doc", "c3", "different text")

    kg.index.build()
    sims = kg.get_similar_chunks("c1", k=2)
    assert "c1" not in sims
    assert "c2" in sims


def test_get_similar_chunks_unknown():
    kg = KnowledgeGraph()
    assert kg.get_similar_chunks("missing") == []


def test_get_similar_chunks_data():
    kg = KnowledgeGraph()
    kg.add_document("doc", source="s")
    kg.add_chunk("doc", "c1", "hello world")
    kg.add_chunk("doc", "c2", "hello planet")
    kg.add_chunk("doc", "c3", "other")
    kg.index.build()

    data = kg.get_similar_chunks_data("c1", k=2)
    ids = [d["id"] for d in data]
    assert "c1" not in ids
    assert "c2" in ids


def test_get_chunk_neighbors_data():
    kg = KnowledgeGraph()
    kg.add_document("doc", source="s")
    kg.add_chunk("doc", "c1", "hello world")
    kg.add_chunk("doc", "c2", "hello planet")
    kg.add_chunk("doc", "c3", "different words")
    kg.index.build()

    data = kg.get_chunk_neighbors_data(k=1)
    assert set(data.keys()) == {"c1", "c2", "c3"}
    assert data["c1"][0]["id"] in {"c2", "c3"}


def test_get_chunk_context():
    kg = KnowledgeGraph()
    kg.add_document("doc", source="s")
    kg.add_chunk("doc", "c1", "t1")
    kg.add_chunk("doc", "c2", "t2")
    kg.add_chunk("doc", "c3", "t3")
    ctx = kg.get_chunk_context("c2", before=1, after=1)
    assert ctx == ["c1", "c2", "c3"]


def test_get_similar_sections():
    kg = KnowledgeGraph()
    kg.add_document("doc", source="s")
    kg.add_section("doc", "s1", title="Introduction")
    kg.add_section("doc", "s2", title="Introductory remarks")
    kg.add_section("doc", "s3", title="Other section")

    kg.index.build()
    sims = kg.get_similar_sections("s1", k=2)
    assert "s1" not in sims
    assert "s2" in sims


def test_get_similar_sections_unknown():
    kg = KnowledgeGraph()
    assert kg.get_similar_sections("missing") == []


def test_link_similar_documents():
    kg = KnowledgeGraph()
    kg.add_document("d1", source="s", text="hello world")
    kg.add_document("d2", source="s", text="hello planet")
    kg.add_document("d3", source="s", text="other text")

    kg.index.build()
    kg.link_similar_documents(k=1)

    assert ("d1", "d2") in kg.graph.edges
    edge = kg.graph.edges["d1", "d2"]
    assert edge["relation"] == "similar_to"
    assert 0 <= edge["similarity"] <= 1


def test_get_similar_documents():
    kg = KnowledgeGraph()
    kg.add_document("d1", source="s", text="hello world")
    kg.add_document("d2", source="s", text="hello planet")
    kg.add_document("d3", source="s", text="unrelated")

    kg.index.build()
    sims = kg.get_similar_documents("d1", k=2)
    assert "d1" not in sims
    assert "d2" in sims


def test_get_similar_documents_unknown():
    kg = KnowledgeGraph()
    assert kg.get_similar_documents("missing") == []


def test_get_chunk_context_unknown():
    kg = KnowledgeGraph()
    assert kg.get_chunk_context("missing") == []


def test_page_for_chunk():
    kg = KnowledgeGraph()
    kg.add_document("d", source="s")
    kg.add_chunk("d", "c1", "text", page=5)

    assert kg.get_page_for_chunk("c1") == 5


def test_page_for_section():
    kg = KnowledgeGraph()
    kg.add_document("doc", source="s")
    kg.add_section("doc", "s1", page=3)
    kg.add_chunk("doc", "c1", "t", section_id="s1", page=3)

    assert kg.get_page_for_section("s1") == 3


def test_next_section_fallback():
    kg = KnowledgeGraph()
    kg.add_document("doc", source="s")
    kg.add_section("doc", "s1")
    kg.add_section("doc", "s2")
    kg.add_section("doc", "s3")

    # remove explicit next_section edges
    for u, v, d in list(kg.graph.edges(data=True)):
        if d.get("relation") == "next_section":
            kg.graph.remove_edge(u, v)

    assert kg.get_next_section("s1") == "s2"
    assert kg.get_previous_section("s3") == "s2"


def test_document_lookup_helpers():
    kg = KnowledgeGraph()
    kg.add_document("doc", source="s")
    kg.add_section("doc", "sec1")
    kg.add_chunk("doc", "c1", "text", section_id="sec1")

    assert kg.get_document_for_section("sec1") == "doc"
    assert kg.get_document_for_chunk("c1") == "doc"

    kg.graph.remove_edge("doc", "c1")
    assert kg.get_document_for_chunk("c1") == "doc"


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


def test_resolve_entities_with_aliases():
    kg = KnowledgeGraph()
    kg.add_entity("e1", "IBM")
    kg.add_entity("e2", "International Business Machines")

    merged = kg.resolve_entities(
        threshold=1.0, aliases={"IBM": ["international business machines"]}
    )

    assert merged == 1
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


def test_enrich_entity_dbpedia(monkeypatch):
    kg = KnowledgeGraph()
    kg.add_entity("e1", "Beethoven")

    class FakeResponse:
        def __init__(self, data):
            self._data = data

        def json(self):
            return self._data

    def fake_get(url, params=None, headers=None, timeout=10):
        return FakeResponse(
            {"results": [{"id": "http://dbpedia.org/resource/Beethoven", "description": "desc"}]}
        )

    monkeypatch.setattr(requests, "get", fake_get)
    kg.enrich_entity_dbpedia("e1")
    node = kg.graph.nodes["e1"]
    assert node.get("dbpedia_uri") == "http://dbpedia.org/resource/Beethoven"
    assert node.get("description_dbpedia") == "desc"


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


def test_compute_node2vec_embeddings():
    kg = KnowledgeGraph()
    kg.add_document("d", source="s")
    kg.add_chunk("d", "c1", "hello")
    kg.add_chunk("d", "c2", "world")
    kg.add_entity("e1", "A")
    kg.add_entity("e2", "B")
    kg.link_entity("c1", "e1")
    kg.link_entity("c2", "e2")
    kg.compute_node2vec_embeddings(dimensions=8, walk_length=4, num_walks=5, seed=42)
    assert isinstance(kg.graph.nodes["e1"].get("embedding"), list)
    assert len(kg.graph.nodes["e1"]["embedding"]) == 8


def test_predict_links():
    kg = KnowledgeGraph()
    kg.add_entity("e1", "Beethoven")
    kg.add_entity("e2", "Ludwig van Beethoven")
    kg.predict_links(threshold=0.4)
    assert kg.graph.has_edge("e1", "e2")
    # Node2Vec based prediction
    kg2 = KnowledgeGraph()
    kg2.add_entity("a1", "X")
    kg2.add_entity("a2", "X")
    kg2.add_document("d", source="s")
    kg2.add_chunk("d", "c1", "x")
    kg2.link_entity("c1", "a1")
    kg2.link_entity("c1", "a2")
    kg2.compute_node2vec_embeddings(dimensions=8, walk_length=4, num_walks=5, seed=42)
    kg2.predict_links(threshold=0.1, use_graph_embeddings=True)
    assert "embedding" in kg2.graph.nodes["a1"]


def test_mark_conflicts():
    kg = KnowledgeGraph()
    kg.add_entity("A", "A")
    kg.add_entity("B", "B")
    kg.add_entity("C", "C")
    kg.graph.add_edge("A", "B", relation="related")
    kg.graph.add_edge("A", "C", relation="related")
    marked = kg.mark_conflicting_facts()
    assert marked == 2
    assert kg.graph.edges["A", "B"].get("conflict") is True
    assert kg.graph.edges["A", "C"].get("conflict") is True


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


def test_entity_pages_helper():
    kg = KnowledgeGraph()
    kg.add_document("doc", source="s")
    kg.add_chunk("doc", "c1", "Paris is nice", page=2)
    kg.add_entity("Paris", "Paris")
    kg.link_entity("c1", "Paris")

    assert kg.get_pages_for_entity("Paris") == [2]


def test_fact_lookup_helpers():
    kg = KnowledgeGraph()
    kg.add_document("doc", source="s")
    kg.add_section("doc", "sec1")
    kg.add_chunk("doc", "c1", "A is B", section_id="sec1")
    fid = kg.add_fact("A", "is", "B")
    kg.graph.add_edge("c1", fid, relation="has_fact")

    assert kg.get_sections_for_fact(fid) == ["sec1"]
    assert kg.get_documents_for_fact(fid) == ["doc"]
    assert kg.get_pages_for_fact(fid) == [1]


def test_extract_entities():
    kg = KnowledgeGraph()
    kg.add_document("doc", source="s")
    kg.add_chunk("doc", "c1", "Albert Einstein was born in Ulm.")
    kg.extract_entities(model=None)
    ents = set(kg.get_entities_for_chunk("c1"))
    assert "Albert Einstein" in ents
    assert "Ulm" in ents


def test_link_chunks_by_entity():
    kg = KnowledgeGraph()
    kg.add_document("doc1", source="s")
    kg.add_document("doc2", source="s")
    kg.add_chunk("doc1", "c1", "Paris is big")
    kg.add_chunk("doc2", "c2", "I love Paris")
    kg.add_entity("Paris", "Paris")
    kg.link_entity("c1", "Paris")
    kg.link_entity("c2", "Paris")

    added = kg.link_chunks_by_entity()
    assert added == 1
    assert kg.graph.has_edge("c1", "c2")
    assert kg.graph.edges["c1", "c2"]["relation"] == "co_mentions"


def test_link_documents_by_entity():
    kg = KnowledgeGraph()
    kg.add_document("doc1", source="s")
    kg.add_document("doc2", source="s")
    kg.add_chunk("doc1", "c1", "Paris is big")
    kg.add_chunk("doc2", "c2", "I love Paris")
    kg.add_entity("Paris", "Paris")
    kg.link_entity("c1", "Paris")
    kg.link_entity("c2", "Paris")

    added = kg.link_documents_by_entity()
    assert added == 1
    assert kg.graph.has_edge("doc1", "doc2")
    assert kg.graph.edges["doc1", "doc2"]["relation"] == "co_mentions"


def test_link_sections_by_entity():
    kg = KnowledgeGraph()
    kg.add_document("doc", source="s")
    kg.add_section("doc", "s1")
    kg.add_section("doc", "s2")
    kg.add_chunk("doc", "c1", "Paris", section_id="s1")
    kg.add_chunk("doc", "c2", "Paris", section_id="s2")
    kg.add_entity("Paris", "Paris")
    kg.link_entity("c1", "Paris")
    kg.link_entity("c2", "Paris")

    added = kg.link_sections_by_entity()
    assert added == 1
    assert kg.graph.has_edge("s1", "s2")
    assert kg.graph.edges["s1", "s2"]["relation"] == "co_mentions"


def test_clean_chunk_texts():
    kg = KnowledgeGraph()
    kg.add_document("doc", source="s")
    kg.add_chunk("doc", "c1", "<p>Hello\n world</p>")

    changed = kg.clean_chunk_texts()
    assert changed == 1
    assert kg.graph.nodes["c1"]["text"] == "Hello world"


def test_normalize_date_fields():
    kg = KnowledgeGraph()
    kg.graph.add_node("e1", type="entity", birth_date="Jan 2, 2024")

    changed = kg.normalize_date_fields()

    assert changed == 1
    assert kg.graph.nodes["e1"]["birth_date"] == "2024-01-02"


def test_validate_coherence():
    kg = KnowledgeGraph()
    kg.graph.add_node("p", type="entity", birth_date="2024-01-01")
    kg.graph.add_node("c", type="entity", birth_date="2023-01-01")
    kg.graph.add_edge("p", "c", relation="parent_of")

    marked = kg.validate_coherence()

    assert marked == 1
    assert kg.graph.edges["p", "c"].get("inconsistent") is True


def test_link_authors_organizations():
    kg = KnowledgeGraph()
    kg.add_document("doc", source="s", author="alice", organization="acme")

    added = kg.link_authors_organizations()

    assert added == 1
    assert kg.graph.has_edge("alice", "acme")
    assert kg.graph.edges["alice", "acme"]["relation"] == "affiliated_with"


def test_remove_document_rebuilds_index():
    kg = KnowledgeGraph()
    kg.add_document("d", source="s")
    kg.add_chunk("d", "c1", "hello world")
    kg.add_chunk("d", "c2", "more text")
    assert kg.search_embeddings("hello", k=1, fetch_neighbors=False) == ["c1"]
    kg.remove_document("d")
    assert kg.search_embeddings("hello", k=1, fetch_neighbors=False) == []


def test_deduplicate_chunks_similarity():
    kg = KnowledgeGraph()
    kg.add_document("d", source="s")
    kg.add_chunk("d", "c1", "Hello world!")
    kg.add_chunk("d", "c2", "Hello world")

    removed = kg.deduplicate_chunks(similarity=0.9)
    assert removed == 1
    assert "c2" not in kg.graph.nodes or "c1" not in kg.graph.nodes
