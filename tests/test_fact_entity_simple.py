import sys
import types
from importlib import reload


def test_extract_facts_both_paths(monkeypatch):
    # Stub heavy dependencies used during module import
    llm_mod = types.ModuleType("datacreek.models.llm_client")

    class DummyLLMClient:
        def chat_completion(self, messages, temperature=0):
            return '[{"subject":"Alice","predicate":"is","object":"a researcher"}]'

    llm_mod.LLMClient = DummyLLMClient
    monkeypatch.setitem(sys.modules, "datacreek.models.llm_client", llm_mod)

    cfg_mod = types.ModuleType("datacreek.utils.config")
    cfg_mod.get_prompt = lambda cfg, name: ""
    cfg_mod.load_config = lambda: {}
    monkeypatch.setitem(sys.modules, "datacreek.utils.config", cfg_mod)

    fe = reload(__import__("datacreek.utils.fact_extraction", fromlist=[""]))

    text = "Alice is a researcher."
    # regex path
    assert fe.extract_facts(text, client=None) == [
        {"subject": "Alice", "predicate": "is", "object": "a researcher"}
    ]
    # LLM path
    out = fe.extract_facts(text, client=DummyLLMClient())
    assert out == [{"subject": "Alice", "predicate": "is", "object": "a researcher"}]


def test_extract_entities_both_paths(monkeypatch):
    spacy_mod = types.ModuleType("spacy")

    class DummyDoc:
        def __init__(self, text):
            self.ents = [
                types.SimpleNamespace(text=p) for p in text.split() if p.istitle()
            ]

    class DummyModel:
        def __call__(self, text):
            return DummyDoc(text)

    spacy_mod.load = lambda model: DummyModel()
    monkeypatch.setitem(sys.modules, "spacy", spacy_mod)

    ee = reload(__import__("datacreek.utils.entity_extraction", fromlist=[""]))

    text = "Alice went to Paris with Bob"
    ents = ee.extract_entities(text, model="en")
    assert set(ents) >= {"Alice", "Paris", "Bob"}
    # regex fallback when model None
    ents2 = ee.extract_entities(text, model=None)
    assert set(ents2) >= {"Alice", "Paris", "Bob"}
