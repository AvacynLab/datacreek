import os
from datacreek.analysis.ingestion import partition_files_to_atoms, transcribe_audio, blip_caption_image


def test_partition_files_to_atoms(tmp_path):
    p = tmp_path / "sample.txt"
    p.write_text("a\nb\nc")
    atoms = partition_files_to_atoms(str(p))
    assert atoms == ["a", "b", "c"]


def test_transcribe_audio_fallback(tmp_path):
    p = tmp_path / "audio.wav"
    p.write_bytes(b"fake")
    assert transcribe_audio(str(p)) == ""


def test_blip_caption_image_fallback(tmp_path):
    p = tmp_path / "img.jpg"
    p.write_bytes(b"fake")
    assert blip_caption_image(str(p)) == ""

def test_parse_code_to_atoms(tmp_path):
    code = """\
 def foo(x):
     return x+1
 
 class Bar:
     def baz(self):
         pass
 """
    p = tmp_path / "sample.py"
    p.write_text(code)
    from datacreek.analysis.ingestion import parse_code_to_atoms
    atoms = parse_code_to_atoms(str(p))
    assert len(atoms) == 2
    assert "def foo" in atoms[0]
    assert "class Bar" in atoms[1]
