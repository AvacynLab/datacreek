import json
import sys
import types
import pytest

from datacreek.core import create
from datacreek.models.content_type import ContentType

class DummyClient:
    def __init__(self):
        self.provider = "dummy"
        self.config = {}

class DummyKG:
    def __init__(self, text="kg"):
        self.text = text

    def to_text(self):
        return self.text

class DummyGenCfg:
    num_pairs = 2
    num_cot_examples = 1
    num_cot_enhance_examples = 1

def patch_common(monkeypatch):
    monkeypatch.setattr(create, "init_llm_client", lambda *a, **k: DummyClient())
    monkeypatch.setattr(create, "get_generation_config", lambda cfg: DummyGenCfg)
    monkeypatch.setattr(create, "QAGenerator", QAGen, raising=False)

class QAGen:
    def __init__(self, *a, **k):
        pass
    def process_document(self, text, num_pairs, verbose=False):
        return types.SimpleNamespace(to_dict=lambda: {"qa": text})
    async def process_document_async(self, text, num_pairs, verbose=False):
        return types.SimpleNamespace(to_dict=lambda: {"qa_async": text})
    def generate_summary(self, text, verbose=False):
        return "sum:" + text

class CotGen:
    def __init__(self, *a, **k):
        pass
    def process_document(self, text, num_examples, include_simple_steps=False):
        return types.SimpleNamespace(to_dict=lambda: {"cot": num_examples})
    def enhance_with_cot(self, messages, include_simple_steps=True):
        return ["enhanced"] + list(messages)

class KGGen:
    def __init__(self, *a, **k):
        pass
    def process_graph(self, kg, num_pairs, verbose=False, multi_answer=False):
        return {"graph": kg.to_text(), "pairs": num_pairs, "multi": multi_answer}

@pytest.fixture(autouse=True)
def _reload(monkeypatch):
    qa_mod = types.ModuleType("qa_generator")
    qa_mod.QAGenerator = QAGen
    cot_mod = types.ModuleType("cot_generator")
    cot_mod.COTGenerator = CotGen
    kg_mod = types.ModuleType("kg_generator")
    kg_mod.KGGenerator = KGGen
    monkeypatch.setitem(sys.modules, "datacreek.generators.qa_generator", qa_mod)
    monkeypatch.setitem(sys.modules, "datacreek.generators.cot_generator", cot_mod)
    monkeypatch.setitem(sys.modules, "datacreek.generators.kg_generator", kg_mod)
    yield


def test_summary_generation(monkeypatch):
    patch_common(monkeypatch)
    kg = DummyKG()
    result = create.process_file(None, content_type=ContentType.SUMMARY, kg=kg)
    assert result == {"summary": "sum:kg"}





def test_from_kg(monkeypatch):
    patch_common(monkeypatch)
    kg = DummyKG()
    result = create.process_file(None, content_type=ContentType.FROM_KG, kg=kg)
    assert result == {"graph": "kg", "pairs": 2, "multi": False}

@pytest.mark.asyncio
async def test_async_qa(monkeypatch):
    patch_common(monkeypatch)
    kg = DummyKG()
    out = await create.process_file_async(None, kg=kg)
    assert out == {"qa_async": "kg"}
