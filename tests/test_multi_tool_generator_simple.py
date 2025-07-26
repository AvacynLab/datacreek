import sys
import types
import pytest

from datacreek.generators import multi_tool_generator as mt
from datacreek.models.qa import QAPair
from datacreek.models.results import QAGenerationResult


class DummyClient:
    config = {"generation": {"temperature": 0}}


calls = {}


class DummyQAGenerator:
    def __init__(self, *_, **__):
        calls.clear()

    def process_document(self, text, num_pairs=25, verbose=False):
        calls.update({"mode": "sync", "num_pairs": num_pairs, "verbose": verbose})
        pair = QAPair(question=f"Q{num_pairs}", answer="A", chunk="C", source="S")
        return QAGenerationResult("SUM", [pair])

    async def process_document_async(self, text, num_pairs=25, verbose=False):
        calls.update({"mode": "async", "num_pairs": num_pairs, "verbose": verbose})
        pair = QAPair(question=f"Q{num_pairs}", answer="A", chunk="C", source="S")
        return QAGenerationResult("SUM", [pair])


def setup_dummy(monkeypatch):
    module = types.ModuleType("datacreek.generators.qa_generator")
    module.QAGenerator = DummyQAGenerator
    monkeypatch.setitem(sys.modules, "datacreek.generators.qa_generator", module)


def _check_result(result, q_value):
    record = result.conversations[0]
    conv = record["conversations"]
    assert conv[1]["content"] == q_value
    assert conv[2]["tool_call"]["name"] == "search"
    assert conv[3]["name"] == "search"
    assert conv[4]["tool_call"]["name"] == "calculator"
    assert conv[5]["name"] == "calculator"
    assert conv[6]["content"] == "A"
    assert record["chunk"] == "C"
    assert record["source"] == "S"


def test_process_document_sync(monkeypatch):
    setup_dummy(monkeypatch)
    gen = mt.MultiToolGenerator(DummyClient())
    result = gen.process_document("text", num_pairs=2, verbose=True)
    assert calls == {"mode": "sync", "num_pairs": 2, "verbose": True}
    assert result.summary == "SUM"
    _check_result(result, "Q2")


@pytest.mark.asyncio
async def test_process_document_async(monkeypatch):
    setup_dummy(monkeypatch)
    gen = mt.MultiToolGenerator(DummyClient())
    result = await gen.process_document_async("text", num_pairs=3, verbose=False)
    assert calls == {"mode": "async", "num_pairs": 3, "verbose": False}
    _check_result(result, "Q3")
