import json
import types
import sys

import datacreek.utils.format_converter as fc


def test_to_jsonl():
    data = [{"a": 1}, {"b": 2}]
    result = fc.to_jsonl(data)
    assert result == json.dumps(data[0], ensure_ascii=False) + "\n" + json.dumps(data[1], ensure_ascii=False)


def test_to_alpaca():
    qa = [{"question": "q1", "answer": "a1"}]
    expected = [{"instruction": "q1", "input": "", "output": "a1"}]
    assert json.loads(fc.to_alpaca(qa)) == expected


def test_to_fine_tuning():
    qa = [{"question": "q2", "answer": "a2"}]
    messages = [{"messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
    ]}]
    assert json.loads(fc.to_fine_tuning(qa)) == messages


def test_to_chatml():
    qa = [{"question": "q", "answer": "a"}]
    result = fc.to_chatml(qa)
    line = json.loads(result)
    assert line["messages"][1]["content"] == "q"
    assert line["messages"][2]["content"] == "a"


def test_to_hf_dataset(monkeypatch):
    class FakeDataset:
        def __init__(self, data):
            self.data = data
        @classmethod
        def from_dict(cls, d):
            assert d == {"question": ["q"], "answer": ["a"]}
            return cls(d)
        def to_json(self):
            return json.dumps(self.data)
    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(Dataset=FakeDataset))
    result = fc.to_hf_dataset([{"question": "q", "answer": "a"}])
    assert result == json.dumps({"question": ["q"], "answer": ["a"]})
