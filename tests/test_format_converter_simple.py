import json
import sys
import types

import pytest

sys.path.insert(0, ".")

from datacreek.utils.format_converter import (
    to_alpaca,
    to_chatml,
    to_fine_tuning,
    to_hf_dataset,
    to_jsonl,
)


def sample_pairs():
    return [
        {"question": "q1", "answer": "a1"},
        {"question": "q2", "answer": "a2"},
    ]


def test_to_jsonl():
    pairs = sample_pairs()
    out = to_jsonl(pairs)
    lines = out.splitlines()
    assert json.loads(lines[0]) == pairs[0]
    assert json.loads(lines[1]) == pairs[1]


def test_to_alpaca():
    pairs = sample_pairs()
    data = json.loads(to_alpaca(pairs))
    assert data[0]["instruction"] == "q1"
    assert data[1]["output"] == "a2"


def test_to_fine_tuning():
    pairs = sample_pairs()
    data = json.loads(to_fine_tuning(pairs))
    msg = data[0]["messages"]
    assert msg[1]["content"] == "q1"
    assert msg[2]["content"] == "a1"


def test_to_chatml():
    pairs = sample_pairs()
    lines = to_chatml(pairs).splitlines()
    record = json.loads(lines[0])["messages"]
    assert record[1]["content"] == "q1"


def test_to_hf_dataset(monkeypatch):
    pairs = sample_pairs()

    class FakeDataset:
        def __init__(self, data):
            self.data = data

        @classmethod
        def from_dict(cls, data):
            return cls(data)

        def to_json(self):
            return json.dumps(self.data)

    monkeypatch.setitem(
        sys.modules, "datasets", types.SimpleNamespace(Dataset=FakeDataset)
    )
    res = to_hf_dataset(pairs)
    assert json.loads(res)["question"] == ["q1", "q2"]
