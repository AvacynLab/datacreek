import sys
sys.path.insert(0, ".")
import json
import pytest
from types import SimpleNamespace

from datacreek.core import curate
from datacreek.models.qa import QAPair
from datacreek.models.results import CurationMetrics, CurationResult

class DummyClient:
    def __init__(self, *a, **k):
        self.config = {
            "curate": {"threshold": 5, "batch_size": 2, "inference_batch": 2, "temperature": 0.1},
            "prompts": {"qa_rating": "{pairs}", "kg_qa_rating": "{pairs}:{facts}"},
        }

    async def async_batch_completion(self, batch, *, temperature, batch_size):
        return [json.dumps([{"question": "q_async", "answer": "a_async", "rating": 8}]) for _ in batch]

    def batch_completion(self, batch, *, temperature, batch_size):
        return [json.dumps([{"question": "q_sync", "answer": "a_sync", "rating": 9}]) for _ in batch]

class DummyKG:
    def to_text(self):
        return "facts"

@pytest.mark.asyncio
async def test_curate_async_basic(monkeypatch):
    monkeypatch.setattr(curate, "LLMClient", DummyClient)
    data = {"summary": "s", "qa_pairs": [{"question": "q", "answer": "a"}]}
    res = await curate.curate_qa_pairs_async(data, keep_ratings=True, kg=DummyKG(), as_dataclass=True)
    assert isinstance(res, CurationResult)
    assert res.metrics.total == 1
    assert res.metrics.filtered == 1
    assert res.rated_pairs and res.rated_pairs[0].question == "q_async"

@pytest.mark.asyncio
async def test_curate_sync_resume(monkeypatch):
    monkeypatch.setattr(curate, "LLMClient", DummyClient)
    prev = {
        "summary": "p",
        "rated_pairs": [{"question": "old", "answer": "a", "rating": 6}],
        "metrics": {"total": 1, "filtered": 1, "retention_rate": 1.0, "avg_score": 6},
    }
    data = {"summary": "s", "qa_pairs": [{"question": "old", "answer": "a"}, {"question": "new", "answer": "b"}]}
    out = await curate._curate_qa_pairs_impl(
        data,
        threshold=5,
        api_base=None,
        model=None,
        config_path=None,
        verbose=False,
        provider=None,
        kg=None,
        batch_size=1,
        inference_batch=1,
        keep_ratings=True,
        temperature=0.1,
        resume=True,
        previous_result=json.dumps(prev),
        as_dataclass=False,
        use_async_handlers=False,
    )
    assert out["metrics"]["total"] == 1
    assert len(out["rated_pairs"]) == 2
    assert out["qa_pairs"][0]["answer"] in {"a_sync", "b"}


def test_filter_rated_pairs_validation():
    with pytest.raises(ValueError):
        curate.filter_rated_pairs([], -1)


def test_apply_curation_threshold_paths():
    base = CurationResult(
        summary="s",
        qa_pairs=[],
        conversations=[],
        metrics=CurationMetrics(total=2, filtered=2, retention_rate=1.0, avg_score=8.0),
        rated_pairs=[QAPair(question="q", answer="a", rating=8)],
    )
    updated = curate.apply_curation_threshold(base, 5)
    assert updated.metrics.total == 1
    assert updated.qa_pairs[0].rating == 8
    bad = CurationResult(summary="s", qa_pairs=[], conversations=[], metrics=base.metrics, rated_pairs=None)
    with pytest.raises(ValueError):
        curate.apply_curation_threshold(bad, 5)
