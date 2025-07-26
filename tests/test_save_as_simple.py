import json
import fakeredis
import pytest

from datacreek.core import save_as
from datacreek.storage import StorageBackend


class DummyBackend(StorageBackend):
    def __init__(self):
        self.saved = None

    def save(self, key: str, data: str) -> str:
        self.saved = (key, data)
        return key


def sample_pairs():
    return [{"question": "q", "answer": "a"}]


def test_format_pairs_variants():
    pairs = sample_pairs()
    assert save_as._format_pairs(pairs, "jsonl") == '{"question": "q", "answer": "a"}'
    alpaca = json.loads(save_as._format_pairs(pairs, "alpaca"))
    assert alpaca[0]["instruction"] == "q"
    chat = json.loads(save_as._format_pairs(pairs, "chatml"))
    assert chat[0]["messages"][1]["content"] == "q"
    with pytest.raises(ValueError):
        save_as._format_pairs(pairs, "bad")


def test_convert_format_file(tmp_path):
    data = {"qa_pairs": sample_pairs()}
    f = tmp_path / "pairs.json"
    f.write_text(json.dumps(data))
    out = save_as.convert_format(str(f), "jsonl")
    assert "question" in out


def test_convert_format_conversations():
    conv = {"conversations": [["sys", {"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]]}
    res = save_as.convert_format(conv, "alpaca")
    assert "instruction" in res


def test_convert_format_hf(monkeypatch):
    backend = DummyBackend()
    monkeypatch.setattr(save_as, "to_hf_dataset", lambda pairs, key: "HF")
    res = save_as.convert_format({"qa_pairs": sample_pairs()}, "jsonl", storage_format="hf", backend=backend, redis_key="k")
    assert res == "HF"


def test_convert_format_redis():
    client = fakeredis.FakeStrictRedis()
    key = "k1"
    res = save_as.convert_format({"qa_pairs": sample_pairs()}, "jsonl", redis_client=client, redis_key=key)
    assert res == key
    assert client.get(key)


def test_convert_format_unrecognized():
    with pytest.raises(ValueError):
        save_as.convert_format({"foo": 1}, "jsonl")

def test_convert_format_filtered():
    data = {"filtered_pairs": sample_pairs()}
    out = save_as.convert_format(data, "jsonl")
    assert "question" in out


def test_convert_format_list_input():
    data = sample_pairs()
    out = save_as.convert_format(data, "jsonl")
    assert "question" in out


class FailingBackend(StorageBackend):
    def save(self, key: str, data: str) -> str:
        raise RuntimeError


def test_convert_format_backend_error():
    with pytest.raises(RuntimeError):
        save_as.convert_format(
            {"qa_pairs": sample_pairs()},
            "jsonl",
            backend=FailingBackend(),
            redis_key="k",
        )


def test_convert_format_hf_missing(monkeypatch):
    monkeypatch.setattr(save_as, "to_hf_dataset", lambda pairs, key: "HF")
    with pytest.raises(ValueError):
        save_as.convert_format({"qa_pairs": sample_pairs()}, "jsonl", storage_format="hf")
