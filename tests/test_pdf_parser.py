import builtins
import sys
import types

import pytest

from datacreek.parsers.pdf_parser import PDFParser


def test_pdf_parser_basic(monkeypatch, tmp_path):
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF-1.4 test")
    dummy = types.SimpleNamespace(extract_text=lambda p: "hello")
    monkeypatch.setitem(sys.modules, "pdfminer.high_level", dummy)
    parser = PDFParser()
    assert parser.parse(str(pdf)) == "hello"


def test_pdf_parser_high_res_missing(monkeypatch, tmp_path):
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF-1.4 test")
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "llamaparse":
            raise ImportError("not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    parser = PDFParser()
    with pytest.raises(ImportError):
        parser.parse(str(pdf), high_res=True)


def test_pdf_parser_ocr(monkeypatch, tmp_path):
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF-1.4 test")
    dummy_pdfminer = types.SimpleNamespace(extract_text=lambda p: "")
    monkeypatch.setitem(sys.modules, "pdfminer.high_level", dummy_pdfminer)
    mod_pdf2image = types.ModuleType("pdf2image")
    mod_pytesseract = types.ModuleType("pytesseract")
    mod_pdf2image.convert_from_path = lambda p: ["img"]
    mod_pytesseract.image_to_string = lambda img: "ocr"
    monkeypatch.setitem(sys.modules, "pdf2image", mod_pdf2image)
    monkeypatch.setitem(sys.modules, "pytesseract", mod_pytesseract)
    parser = PDFParser()
    assert "ocr" in parser.parse(str(pdf), ocr=True)
