import json
import sys
import types

import datacreek.utils as utils


def test_dynamic_extract_facts_entities(monkeypatch):
    text = "Alice is a researcher. Bob was born in Paris."

    stub = types.ModuleType("fact_extraction")
    stub.extract_facts = lambda txt: [
        {"subject": "Alice", "predicate": "is", "object": "a researcher"},
        {"subject": "Bob", "predicate": "born_in", "object": "Paris"},
    ]
    monkeypatch.setitem(sys.modules, "datacreek.utils.fact_extraction", stub)

    facts = utils.extract_facts(text)
    assert {tuple(f.values()) for f in facts} == {
        ("Alice", "is", "a researcher"),
        ("Bob", "born_in", "Paris"),
    }

    stub2 = types.ModuleType("entity_extraction")
    stub2.extract_entities = lambda txt, model=None: ["Alice", "Bob", "Paris"]
    monkeypatch.setitem(sys.modules, "datacreek.utils.entity_extraction", stub2)

    ents = utils.extract_entities(text)
    assert set(ents) == {"Alice", "Bob", "Paris"}


class DummyLLM:
    def predict(self, text: str) -> str:
        return json.dumps({"merge": ["a"], "split": ["b"]})


def test_dynamic_propose_merge_split(tmp_path):
    func = utils.propose_merge_split
    out = func(["a", "b"], llm=DummyLLM())
    assert out == {"merge": ["a"], "split": ["b"]}

    log_path = tmp_path / "log.jsonl"
    utils.record_feedback(out, True, log_path)
    assert json.loads(log_path.read_text()) == {"suggestion": out, "accepted": True}


def test_cache_and_progress_stubs():
    utils.cache_l1 = None
    assert utils.cache_l1 is None
    p, tid = utils.create_progress("desc", 1)
    assert p is None and tid == 0
    with utils.progress_context("desc", 1) as result:
        assert result == (None, 0)
