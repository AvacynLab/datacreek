import json
import sys
from types import SimpleNamespace
from unittest.mock import patch
import pytest
from pathlib import Path

from PIL import Image

import datacreek.generators.vqa_generator as vg

class FakeClient:
    def __init__(self):
        self.config = {"prompt": "answer", "generation": {}}
        self.calls = []

    def batch_completion(self, message_batches, temperature, max_tokens, batch_size):
        self.calls.append((message_batches, temperature, max_tokens, batch_size))
        return [f"resp{i}" for i in range(len(message_batches))]

class FakeDataset:
    def __init__(self, data):
        self.data = data
    @classmethod
    def from_dict(cls, data):
        # replace placeholder image string with real PIL image
        img = make_image()
        return cls({"image": [img], "query": data["query"], "label": data["label"]})
    def __getitem__(self, key):
        return self
    def __len__(self):
        return len(self.data["image"])
    def select(self, indices):
        self.data = {k: [v[i] for i in indices] for k, v in self.data.items()}
        return self
    def map(self, func, batch_size=None, batched=False):
        self.data = func(self.data)
        return self
    def to_dict(self):
        out = {"image": ["img"] * len(self.data["image"])}
        out.update({k: v for k, v in self.data.items() if k != "image"})
        return out

class FakeBackend:
    def __init__(self):
        self.saved = {}
    def save(self, key, data):
        self.saved[key] = data
        return key

def make_image(color="red"):
    return Image.new("RGB", (2, 2), color=color)


def test_encode_image_base64_roundtrip():
    gen = vg.VQAGenerator.__new__(vg.VQAGenerator)
    img = make_image()
    data = gen.encode_image_base64(img)
    assert isinstance(data, str)
    import base64
    img_bytes = base64.b64decode(data)
    assert len(img_bytes) > 0


def test_transform_updates_messages(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(vg, "_check_optional_deps", lambda: None)
    monkeypatch.setattr(vg, "get_generation_config", lambda cfg: SimpleNamespace(temperature=0.0, max_tokens=5, batch_size=2))
    gen = vg.VQAGenerator(client)

    messages = {"image": [make_image()], "query": ["Q"], "label": ["A"]}
    result = gen.transform(messages)
    assert result["label"][0] == "resp0"
    assert client.calls


def test_process_dataset_file(monkeypatch, tmp_path):
    client = FakeClient()
    monkeypatch.setattr(vg, "_check_optional_deps", lambda: None)
    monkeypatch.setattr(vg, "get_generation_config", lambda cfg: SimpleNamespace(temperature=0.0, max_tokens=5, batch_size=1))
    gen = vg.VQAGenerator(client)

    path = tmp_path / "ds.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump({"image": ["i"], "query": ["q"], "label": ["l"]}, f)

    fake_ds_module = SimpleNamespace(Dataset=FakeDataset, load_dataset=lambda x: FakeDataset({"image": [], "query": [], "label": []}))
    fake_hub = SimpleNamespace(HfApi=lambda: SimpleNamespace(repo_exists=lambda repo_id, repo_type=None: False))
    monkeypatch.setitem(sys.modules, "datasets", fake_ds_module)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    ds = gen.process_dataset(str(path), num_examples=1, input_split="train")
    assert isinstance(ds, FakeDataset)
    assert ds.data["label"][0] == "resp0"


def test_process_dataset_backend(monkeypatch, tmp_path):
    client = FakeClient()
    monkeypatch.setattr(vg, "_check_optional_deps", lambda: None)
    monkeypatch.setattr(vg, "get_generation_config", lambda cfg: SimpleNamespace(temperature=0.0, max_tokens=5, batch_size=1))
    path = tmp_path / "ds.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump({"image": ["i"], "query": ["q"], "label": ["l"]}, f)
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(Dataset=FakeDataset))
    gen = vg.VQAGenerator(client)
    backend = FakeBackend()
    result = gen.process_dataset(str(path), input_split="train", backend=backend, redis_key="k")
    assert result == "k"
    assert backend.saved["k"]


def test_process_dataset_hub_redis(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(vg, "_check_optional_deps", lambda: None)
    monkeypatch.setattr(vg, "get_generation_config", lambda cfg: SimpleNamespace(temperature=0.0, max_tokens=5, batch_size=1))
    fake_ds_module = SimpleNamespace(
        Dataset=FakeDataset,
        load_dataset=lambda name: FakeDataset({"image": [make_image()], "query": ["q"], "label": ["l"]}),
    )
    fake_hub = SimpleNamespace(HfApi=lambda: SimpleNamespace(repo_exists=lambda repo_id, repo_type=None: True))
    monkeypatch.setitem(sys.modules, "datasets", fake_ds_module)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    import fakeredis

    gen = vg.VQAGenerator(client)
    redis_client = fakeredis.FakeStrictRedis()
    key = gen.process_dataset("ds-id", input_split="train", redis_client=redis_client, redis_key="rk")
    assert key == "rk"
    assert redis_client.get("rk")


def test_process_dataset_missing(monkeypatch, tmp_path):
    client = FakeClient()
    monkeypatch.setattr(vg, "_check_optional_deps", lambda: None)
    monkeypatch.setattr(vg, "get_generation_config", lambda cfg: SimpleNamespace(temperature=0.0, max_tokens=5, batch_size=1))
    fake_ds_module = SimpleNamespace(Dataset=FakeDataset, load_dataset=lambda name: FakeDataset({}))
    fake_hub = SimpleNamespace(HfApi=lambda: SimpleNamespace(repo_exists=lambda repo_id, repo_type=None: False))
    monkeypatch.setitem(sys.modules, "datasets", fake_ds_module)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    gen = vg.VQAGenerator(client)
    with pytest.raises(FileNotFoundError):
        gen.process_dataset(str(tmp_path / "missing.json"), input_split="train")


def test_init_optional(monkeypatch):
    def fake_deps():
        raise ImportError("missing")

    monkeypatch.setattr(vg, "_check_optional_deps", fake_deps)
    with pytest.raises(ImportError):
        vg.VQAGenerator(FakeClient())

    monkeypatch.setattr(vg, "_check_optional_deps", lambda: None)
    monkeypatch.setattr(vg, "get_generation_config", lambda cfg: SimpleNamespace(temperature=0.0, max_tokens=5, batch_size=1))
    monkeypatch.setattr(vg, "load_config", lambda path: {"prompt": "p"})
    gen = vg.VQAGenerator(FakeClient(), config_overrides={"a": 1})
    assert gen.config
