import json

import pytest

from datacreek.core import curate
from datacreek.models.qa import QAPair


class DummyClient:
    def __init__(self, *a, **k):
        self.config = {}


def fake_batch(client, message_batches, **kwargs):
    res = []
    for m in message_batches:
        pairs = json.loads(m[0]["content"])
        res.append(
            json.dumps(
                [{"question": p["question"], "answer": p["answer"], "rating": 10} for p in pairs]
            )
        )
    return res


def fake_parse(text, orig):
    data = json.loads(text)
    return [QAPair(question=d["question"], answer=d["answer"], rating=d["rating"]) for d in data]


class DummyProgress:
    def __init__(self, *a, **k):
        pass

    def start(self):
        pass

    def stop(self):
        pass

    def update(self, *a, **k):
        pass


def test_curate_deduplicate(monkeypatch):
    monkeypatch.setattr(curate, "LLMClient", DummyClient)
    monkeypatch.setattr("datacreek.utils.batch.process_batches", fake_batch)
    monkeypatch.setattr(curate, "parse_ratings", fake_parse)
    monkeypatch.setattr(
        "datacreek.utils.progress.create_progress", lambda *a, **k: (DummyProgress(), 0)
    )
    monkeypatch.setattr(curate, "get_prompt", lambda cfg, name: "{pairs}")
    data = {
        "summary": "",
        "qa_pairs": [{"question": "q", "answer": "a"}, {"question": "q", "answer": "a"}],
    }
    res = curate.curate_qa_pairs(data, batch_size=1, inference_batch=1)
    assert len(res["qa_pairs"]) == 1


def test_curate_parse_error(monkeypatch):
    """Parsing failures should raise CurationError."""

    monkeypatch.setattr(curate, "LLMClient", DummyClient)
    monkeypatch.setattr("datacreek.utils.batch.process_batches", fake_batch)

    def bad_parse(text, orig):
        raise ValueError("nope")

    monkeypatch.setattr(curate, "parse_ratings", bad_parse)
    monkeypatch.setattr(
        "datacreek.utils.progress.create_progress", lambda *a, **k: (DummyProgress(), 0)
    )
    monkeypatch.setattr(curate, "get_prompt", lambda cfg, name: "{pairs}")

    data = {"summary": "", "qa_pairs": [{"question": "q", "answer": "a"}]}

    with pytest.raises(curate.CurationError):
        curate.curate_qa_pairs(data, batch_size=1, inference_batch=1)
