import pytest

import datacreek.parsers
from datacreek import DatasetBuilder, DatasetType, ingest_file, to_kg
from datacreek.core.ingest import ingest_into_dataset, process_file
from datacreek.parsers import AudioParser, ImageParser
from datacreek.parsers.pdf_parser import PDFParser


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


def test_to_kg_with_pages(tmp_path):
    text_file = tmp_path / "sample.txt"
    text_file.write_text("Page1\fPage2")

    ds = DatasetBuilder(DatasetType.TEXT)
    to_kg("Page1\fPage2", ds, "doc1", pages=["Page1", "Page2"])

    assert ds.get_page_for_chunk("doc1_chunk_0") == 1
    assert ds.get_page_for_chunk("doc1_chunk_1") == 2


def test_determine_parser_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        ingest_file(str(tmp_path / "missing.txt"))
    bad_file = tmp_path / "file.badext"
    bad_file.write_text("x")
    with pytest.raises(ValueError):
        ingest_file(str(bad_file))


def test_ingest_resolves_input_path(tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("hi")

    cfg = {"paths": {"input": {"txt": str(tmp_path), "default": str(tmp_path)}}}
    text = ingest_file("doc.txt", config=cfg)
    assert text == "hi"


def test_ingest_into_dataset_with_facts(tmp_path):
    text_file = tmp_path / "doc.txt"
    text_file.write_text("Paris is the capital of France.")
    ds = DatasetBuilder(DatasetType.TEXT)
    ingest_into_dataset(str(text_file), ds, extract_facts=True)
    facts = ds.search_facts("Paris")
    assert facts


def test_ingest_into_dataset_with_entities(tmp_path):
    text_file = tmp_path / "doc.txt"
    text_file.write_text("Albert Einstein was born in Ulm.")
    ds = DatasetBuilder(DatasetType.TEXT)
    ingest_into_dataset(str(text_file), ds, extract_entities=True)
    ents = ds.search_entities("Einstein")
    assert ents


def test_ingest_into_dataset_records_source(tmp_path):
    f = tmp_path / "sample.txt"
    f.write_text("hello")

    ds = DatasetBuilder(DatasetType.TEXT)
    ingest_into_dataset(str(f), ds)

    assert ds.graph.graph.nodes["sample"]["source"] == str(f)


def test_ingest_image_and_audio(monkeypatch, tmp_path):
    img = tmp_path / "img.png"
    img.write_bytes(b"x")
    aud = tmp_path / "clip.wav"
    aud.write_bytes(b"y")

    monkeypatch.setattr(ImageParser, "parse", lambda self, p: "img text")
    monkeypatch.setattr(ImageParser, "save", lambda self, c, o: None)
    monkeypatch.setattr(AudioParser, "parse", lambda self, p: "audio text")
    monkeypatch.setattr(AudioParser, "save", lambda self, c, o: None)

    ds = DatasetBuilder(DatasetType.TEXT)
    ingest_into_dataset(str(img), ds)
    ingest_into_dataset(str(aud), ds)

    assert ds.search_chunks("img text")
    assert ds.search_chunks("audio text")


def test_determine_parser_remote_image(monkeypatch):
    called = {}

    class DummyParser:
        def parse(self, file_path):
            called["path"] = file_path
            return "ok"

        def save(self, content, output_path):
            pass

    monkeypatch.setitem(datacreek.parsers._PARSER_REGISTRY, ".png", DummyParser)

    text = ingest_file("https://example.com/test.png")
    assert text == "ok"
    assert called["path"] == "https://example.com/test.png"


def test_determine_parser_remote_audio(monkeypatch):
    called = {}

    class DummyParser:
        def parse(self, file_path):
            called["path"] = file_path
            return "ok"

        def save(self, content, output_path):
            pass

    monkeypatch.setitem(datacreek.parsers._PARSER_REGISTRY, ".mp3", DummyParser)

    text = ingest_file("https://example.com/test.mp3")
    assert text == "ok"
    assert called["path"] == "https://example.com/test.mp3"


def test_process_file_unstructured(monkeypatch, tmp_path):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"x")

    class Dummy(PDFParser):
        def parse(
            self,
            file_path,
            *,
            high_res=False,
            ocr=False,
            return_pages=False,
            use_unstructured=False,
        ):
            assert use_unstructured is True
            return "ok"

        def save(self, content, output_path):
            pass

    monkeypatch.setattr(datacreek.core.ingest, "determine_parser", lambda f, c: Dummy())
    text = process_file(str(pdf), use_unstructured=True)
    assert text == "ok"
