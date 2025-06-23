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

