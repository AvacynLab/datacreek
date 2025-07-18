import os

from datacreek.analysis.ingestion import (
    blip_caption_image,
    partition_files_to_atoms,
    transcribe_audio,
)
from datacreek.utils import image_captioning, whisper_batch


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


def test_caption_images_parallel(monkeypatch):
    paths = [f"img{i}.png" for i in range(300)]
    calls: list[str] = []

    def _fake(path: str) -> str:
        calls.append(path)
        return f"cap-{path}"

    monkeypatch.setattr(image_captioning, "caption_image", _fake)

    captions = image_captioning.caption_images_parallel(
        paths, max_workers=4, chunk_size=128
    )
    assert captions == [f"cap-{p}" for p in paths]
    assert calls == paths


def test_transcribe_audio_batch(monkeypatch):
    paths = [f"a{i}.wav" for i in range(5)]
    calls: list[str] = []

    class DummyModel:
        def transcribe(self, path: str, max_length: int = 30) -> str:
            calls.append(path)
            return f"txt-{path}"

    monkeypatch.setattr(whisper_batch, "_get_model", lambda *a, **k: DummyModel())

    result = whisper_batch.transcribe_audio_batch(paths, batch_size=2)
    assert result == [f"txt-{p}" for p in paths]
    assert calls == paths


def test_whisper_parser_uses_batch(monkeypatch):
    captured: list[str] = []

    def fake_batch(paths, **kw):
        captured.extend(paths)
        return ["hello"]

    monkeypatch.setattr(whisper_batch, "transcribe_audio_batch", fake_batch)
    from datacreek.parsers.whisper_audio_parser import WhisperAudioParser

    parser = WhisperAudioParser()
    assert parser.parse("x.wav") == "hello"
    assert captured == ["x.wav"]
