import sys
import types
import pytest

from datacreek.generators import conversation_generator as cg
from datacreek.models.qa import QAPair
from datacreek.models.results import QAGenerationResult

class DummyClient:
    config = {"generation": {"temperature": 0}}

calls = {}

class DummyQAGenerator:
    def __init__(self, *args, **kwargs):
        calls.clear()
    def process_document(self, text, num_pairs=25, verbose=False):
        calls['mode'] = 'sync'
        calls['num_pairs'] = num_pairs
        calls['verbose'] = verbose
        pair = QAPair(question=f"Q{num_pairs}", answer="A", chunk="C", source="S")
        return QAGenerationResult("SUM", [pair])
    async def process_document_async(self, text, num_pairs=25, verbose=False):
        calls['mode'] = 'async'
        calls['num_pairs'] = num_pairs
        calls['verbose'] = verbose
        pair = QAPair(question=f"Q{num_pairs}", answer="A", chunk="C", source="S")
        return QAGenerationResult("SUM", [pair])


def test_process_document_sync(monkeypatch):
    module = types.ModuleType("datacreek.generators.qa_generator")
    module.QAGenerator = DummyQAGenerator
    monkeypatch.setitem(sys.modules, "datacreek.generators.qa_generator", module)
    gen = cg.ConversationGenerator(DummyClient())
    result = gen.process_document("text", num_pairs=2, verbose=True)
    assert calls == {'mode': 'sync', 'num_pairs': 2, 'verbose': True}
    assert result.summary == "SUM"
    record = result.conversations[0]
    assert record['chunk'] == 'C'
    assert record['source'] == 'S'
    assert record['conversations'][1]['content'] == 'Q2'


@pytest.mark.asyncio
async def test_process_document_async(monkeypatch):
    module = types.ModuleType("datacreek.generators.qa_generator")
    module.QAGenerator = DummyQAGenerator
    monkeypatch.setitem(sys.modules, "datacreek.generators.qa_generator", module)
    gen = cg.ConversationGenerator(DummyClient())
    result = await gen.process_document_async("text", num_pairs=3, verbose=False)
    assert calls == {'mode': 'async', 'num_pairs': 3, 'verbose': False}
    assert result.conversations[0]['conversations'][1]['content'] == 'Q3'
