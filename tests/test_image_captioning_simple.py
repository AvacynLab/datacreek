import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

import datacreek.utils.image_captioning as image_captioning


class DummyModel:
    def __init__(self, responses):
        self.responses = responses

    def __call__(self, path):
        if isinstance(self.responses, Exception):
            raise self.responses
        return [{"generated_text": self.responses[path]}]


def test_caption_image(monkeypatch, tmp_path):
    responses = {"img1": "a cat"}
    importlib.reload(image_captioning)  # reset cache and pipeline
    monkeypatch.setattr(
        image_captioning, "pipeline", lambda *a, **k: DummyModel(responses)
    )
    assert image_captioning.caption_image("img1") == "a cat"


def test_caption_image_error(monkeypatch):
    importlib.reload(image_captioning)
    monkeypatch.setattr(
        image_captioning, "pipeline", lambda *a, **k: DummyModel(RuntimeError())
    )
    with pytest.raises(RuntimeError):
        image_captioning.caption_image("x")


def test_caption_images_parallel(monkeypatch):
    monkeypatch.setattr(image_captioning, "caption_image", lambda p: f"cap:{p}")
    caps = image_captioning.caption_images_parallel(["a", "b"], chunk_size=1)
    assert caps == ["cap:a", "cap:b"]
