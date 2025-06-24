from datacreek import DatasetBuilder, DatasetType, ingest_file, to_kg

def test_ingest_to_kg(tmp_path):
    text_file = tmp_path / "sample.txt"
    text_file.write_text("Hello world. This is a test document.")

    ds = DatasetBuilder(DatasetType.TEXT)
    text = ingest_file(str(text_file))
    to_kg(text, ds, "doc1")

    assert ds.search_chunks("Hello")
    assert ds.get_chunks_for_document("doc1")

