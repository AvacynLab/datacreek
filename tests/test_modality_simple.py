import os
from pathlib import Path
import importlib

import datacreek.utils.modality as modality


def test_detect_modality_files(tmp_path):
    img = tmp_path / "img.png"
    audio = tmp_path / "clip.mp3"
    code = tmp_path / "prog.py"
    txt = tmp_path / "doc.txt"
    for p in (img, audio, code, txt):
        p.write_text("data")

    assert modality.detect_modality(str(img)) == "IMAGE"
    assert modality.detect_modality(str(audio)) == "AUDIO"
    assert modality.detect_modality(str(code)) == "CODE"
    assert modality.detect_modality(str(txt)) == "TEXT"


def test_detect_modality_strings(monkeypatch):
    modality.detect_modality.cache_clear()
    assert modality.detect_modality("um yeah") == "spoken"
    assert modality.detect_modality("Hello world") == "spoken"
    assert modality.detect_modality("Hello world!") == "written"
    # cached value reused
    assert modality.detect_modality("um yeah") == "spoken"
