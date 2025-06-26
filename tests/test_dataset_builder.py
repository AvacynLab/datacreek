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


def test_dataset_id_in_serialization():
    ds = DatasetBuilder(DatasetType.QA, name="demo")
    data = ds.to_dict()
    assert "id" in data
    loaded = DatasetBuilder.from_dict(data)
    assert loaded.id == ds.id
