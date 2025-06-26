import fakeredis

from datacreek.core.dataset import DatasetBuilder
from datacreek.pipelines import DatasetType


def test_dataset_has_its_own_graph():
    ds1 = DatasetBuilder(DatasetType.QA, name="ds1")
    ds2 = DatasetBuilder(DatasetType.QA, name="ds2")
    ds1.add_document("doc1", source="a")
    ds1.add_chunk("doc1", "c1", "hello")

    assert ds1.search("hello") == ["c1"]
    # second dataset should be empty
    assert ds2.search("hello") == []


def test_dataset_search_wrappers():
    ds = DatasetBuilder(DatasetType.QA)
    ds.add_document("d", source="s")
    ds.add_chunk("d", "c", "text")
    assert ds.search_chunks("text") == ["c"]
    assert ds.search_documents("d") == ["d"]
    assert ds.get_chunks_for_document("d") == ["c"]
    ds.graph.index.build()
    assert ds.search_embeddings("text", k=1) == ["c"]
    assert ds.search_hybrid("text", k=1) == ["c"]


def test_dataset_clone():
    ds = DatasetBuilder(DatasetType.QA, name="orig")
    ds.add_document("d", source="s")
    ds.add_chunk("d", "c1", "text")
    clone = ds.clone(name="copy")
    clone.add_chunk("d", "c2", "more")

    # original should not have the new chunk
    assert ds.search("more") == []
    assert clone.search("more") == ["c2"]


def test_dataset_persistence_redis():
    ds = DatasetBuilder(DatasetType.TEXT, name="test")
    ds.add_document("d", source="source.txt")
    ds.add_chunk("d", "c1", "hello world")

    client = fakeredis.FakeStrictRedis()
    ds.to_redis(client, "ds:test")
    loaded = DatasetBuilder.from_redis(client, "ds:test")

    assert loaded.name == "test"
    assert loaded.search_chunks("hello") == ["c1"]
    # id should round-trip through persistence
    assert loaded.id == ds.id


def test_remove_chunk_updates_order():
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_document("d", source="s")
    ds.add_chunk("d", "c1", "t1")
    ds.add_chunk("d", "c2", "t2")
    ds.add_chunk("d", "c3", "t3")

    ds.remove_chunk("c2")
    assert ds.get_chunks_for_document("d") == ["c1", "c3"]
    assert ("c1", "c3") in ds.graph.graph.edges
    assert ds.graph.graph.edges["d", "c1"]["sequence"] == 0
    assert ds.graph.graph.edges["d", "c3"]["sequence"] == 1


def test_remove_document_cascades():
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_document("d", source="s")
    ds.add_chunk("d", "c1", "t1")
    ds.add_chunk("d", "c2", "t2")

    ds.remove_document("d")
    assert ds.search("t1") == []
    assert ds.search_documents("d") == []


def test_dataset_id_in_serialization():
    ds = DatasetBuilder(DatasetType.QA, name="demo")
    data = ds.to_dict()
    assert "id" in data
    loaded = DatasetBuilder.from_dict(data)
    assert loaded.id == ds.id


def test_link_similar_chunks_wrapper():
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_document("d", source="s")
    ds.add_chunk("d", "c1", "hello world")
    ds.add_chunk("d", "c2", "hello planet")
    ds.add_chunk("d", "c3", "other text")

    ds.graph.index.build()
    ds.link_similar_chunks(k=1)

    assert ("c1", "c2") in ds.graph.graph.edges
    edge = ds.graph.graph.edges["c1", "c2"]
    assert edge["relation"] == "similar_to"
    assert 0 < edge["similarity"] <= 1


def test_search_with_links_wrapper():
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_document("d", source="s")
    ds.add_chunk("d", "c1", "hello world")
    ds.add_chunk("d", "c2", "hello planet")
    ds.add_chunk("d", "c3", "unrelated text")

    ds.graph.index.build()
    ds.link_similar_chunks(k=1)

    results = ds.search_with_links("hello", k=1, hops=1)
    assert "c1" in results
    assert "c2" in results


def test_search_with_links_data_wrapper():
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_document("d", source="s")
    ds.add_chunk("d", "c1", "hello world")
    ds.add_chunk("d", "c2", "hello planet")
    ds.add_chunk("d", "c3", "unrelated text")

    ds.graph.index.build()
    ds.link_similar_chunks(k=1)

    results = ds.search_with_links_data("hello", k=1, hops=1)
    ids = [r["id"] for r in results]
    assert "c1" in ids
    assert "c2" in ids
    for r in results:
        if r["id"] == "c1":
            assert r["text"] == "hello world"
            assert r["document"] == "d"
            assert r["depth"] == 0
            assert r["path"] == ["c1"]
        if r["id"] == "c2":
            assert r["depth"] == 1
            assert r["path"] == ["c1", "c2"]


def test_link_entity_source_and_trust():
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_document("d", source="src")
    ds.add_chunk("d", "c1", "hello", source="src")
    ds.add_entity("e1", "entity", source="src")
    ds.link_entity("c1", "e1", provenance="src")
    ds.graph.index.build()
    ds.score_trust()

    edge = ds.graph.graph.edges["c1", "e1"]
    assert edge["provenance"] == "src"
    assert "trust" in edge
