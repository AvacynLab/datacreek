import json
from pathlib import Path

import pytest

from datacreek.utils import (
    fine_tune_from_feedback,
    propose_merge_split,
    record_feedback,
)


class DummyLLM:
    def predict(self, text: str) -> str:  # pragma: no cover - simple stub
        return '{"merge": [["A", "B"]], "split": {"C": ["C1", "C2"]}}'


def test_propose_merge_split_parses_response():
    res = propose_merge_split(["A desc", "B desc", "C desc"], llm=DummyLLM())
    assert res["merge"] == [["A", "B"]]
    assert "C" in res["split"]


def test_record_feedback_appends(tmp_path: Path):
    log = tmp_path / "fb.jsonl"
    record_feedback({"x": 1}, True, log)
    record_feedback({"x": 2}, False, log)
    lines = log.read_text().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["accepted"] is True
    assert first["suggestion"]["x"] == 1


def test_fine_tune_from_feedback(monkeypatch, tmp_path: Path):
    data = tmp_path / "fb.jsonl"
    data.write_text("{}\n")

    class FakeOpenAI:
        class FineTuningJob:
            @staticmethod
            def create(training_file: str, model: str):
                return {"id": "ft-1", "file": training_file, "model": model}

    monkeypatch.setattr("datacreek.utils.curation_agent.openai", FakeOpenAI)
    job = fine_tune_from_feedback(data, model="gpt-3")
    assert job["id"] == "ft-1"
    assert job["file"] == str(data)
    assert job["model"] == "gpt-3"
