from datacreek import DatasetBuilder, DatasetType, ingest_file, to_kg
import pytest

def test_ingest_to_kg(tmp_path):
    text_file = tmp_path / "sample.txt"
    text_file.write_text("Hello world. This is a test document.")

    ds = DatasetBuilder(DatasetType.TEXT)
    text = ingest_file(str(text_file))
    to_kg(text, ds, "doc1")

    assert ds.search_chunks("Hello")
    assert ds.get_chunks_for_document("doc1")


def test_to_kg_no_index_build(tmp_path):
    text_file = tmp_path / "sample.txt"
    text_file.write_text("Hello world")

    ds = DatasetBuilder(DatasetType.TEXT)
    text = ingest_file(str(text_file))
    to_kg(text, ds, "doc1", build_index=False)

    assert ds.graph.index._vectorizer is None
    # calling search should auto-build the index
    assert ds.search_chunks("Hello") == ["doc1_chunk_0"]


def test_determine_parser_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        ingest_file(str(tmp_path / "missing.txt"))
    bad_file = tmp_path / "file.badext"
    bad_file.write_text("x")
    with pytest.raises(ValueError):
        ingest_file(str(bad_file))

