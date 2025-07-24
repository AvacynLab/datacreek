import tempfile
import os
from pathlib import Path
import datacreek.analysis.ingestion as ingestion


def test_partition_files_to_atoms_simple(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("a\n\n b \n")
    atoms = ingestion.partition_files_to_atoms(str(p))
    assert atoms == ["a", "b"]


def test_parse_code_to_atoms_simple(tmp_path):
    code = """\
# comment

def f():
    pass

class A:
    pass
"""
    p = tmp_path / "a.py"
    p.write_text(code)
    atoms = ingestion.parse_code_to_atoms(str(p))
    assert any("def f" in a for a in atoms)
    assert any("class A" in a for a in atoms)


def test_blip_caption_image_fallback(tmp_path):
    img = tmp_path / "x.jpg"
    img.write_bytes(b"\x00\x01")
    assert ingestion.blip_caption_image(str(img)) == ""


def test_transcribe_audio_fallback(tmp_path):
    wav = tmp_path / "x.wav"
    wav.write_bytes(b"00")
    assert ingestion.transcribe_audio(str(wav)) == ""
