import os
import pytest
from datacreek.core.dataset import DatasetBuilder, DatasetType

class FakeGraph:
    def __init__(self):
        self.calls = []
        self.graph = {
            "fractal_sigma": 0,
            "gw_entropy": 0,
            "recall10": 0,
            "tpl_w1": 0,
            "j_cost": 0,
        }

    def __getattr__(self, name):
        def method(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            if name in {"fractal_coverage", "sheaf_consistency_score"}:
                return 0.0
            if name == "governance_metrics":
                return {"score": 1.0}
            if name in {"similar_by_hybrid", "ann_hybrid_search"}:
                return [("x", 0.1)]
            if name.startswith("get_") or name.startswith("search"):
                return []
            return 0.0
        return method


def make_builder():
    os.environ["DATACREEK_REQUIRE_PERSISTENCE"] = "0"
    os.environ["DATACREEK_LIGHT_DATASET"] = "1"
    fg = FakeGraph()
    ds = DatasetBuilder(
        DatasetType.TEXT,
        name="ds",
        graph=fg,
    )
    ds._persist = lambda: None
    ds.policy.loops = 0
    fg.calls.clear()
    return ds, fg


def test_validation_and_event(monkeypatch):
    ds, _ = make_builder()

    assert ds.validate_name("ok") == "ok"
    with pytest.raises(ValueError):
        ds.validate_name("bad name")

    captured = []

    def fake_update(k, v):
        captured.append((k, v))

    monkeypatch.setattr("datacreek.core.dataset_light.push_metrics", fake_update, raising=False)

    ds._record_event("test", "hello", a=1)
    assert ds.history[-1] == "hello"
    ds.log_cycle_metrics()
    assert captured == []


def test_wrapper_calls():
    ds, fg = make_builder()

    actions = [
        ("add_document", ("d", "src"), {"text": "x"}),
        ("add_section", ("d", "s"), {}),
        ("add_chunk", ("d", "c", "txt"), {}),
        ("export_prompts", (), {}),
    ]

    for name, args, kwargs in actions:
        getattr(ds, name)(*args, **kwargs)

    assert fg.calls

