import asyncio
import types

import pytest

from datacreek.generators import cot_generator as cg
from datacreek.models.cot import COTExample


class DummyClient:
    """LLM client returning canned responses."""

    def __init__(self):
        self.config = {
            "generation": {"temperature": 0.0, "max_tokens": 16, "num_cot_examples": 1},
            "prompts": {
                "cot_generation": "gen {num_examples} {text}",
                "cot_enhancement": "enh {conversations} {include_simple_steps}",
            },
        }
        self.calls = []

    def chat_completion(self, messages, temperature=None, max_tokens=None):
        self.calls.append(messages)
        return '[{"question":"Q","reasoning":"R","answer":"A"}]'


def _make_gen():
    return cg.COTGenerator(DummyClient())


def test_parse_json_output(monkeypatch):
    gen = _make_gen()
    assert gen.parse_json_output('[{"a":1}]') == [{"a": 1}]
    assert gen.parse_json_output('{"a":1}') is None


def test_generate_cot_examples_sync():
    gen = _make_gen()
    examples = gen.generate_cot_examples("doc", num_examples=1)
    assert isinstance(examples[0], COTExample)
    assert examples[0].question == "Q"


@pytest.mark.asyncio
async def test_generate_cot_examples_async():
    gen = _make_gen()
    examples = await gen.generate_cot_examples_async("doc", num_examples=1)
    assert len(examples) == 1 and examples[0].answer == "A"


def test_enhance_with_cot(monkeypatch):
    client = DummyClient()
    client.chat_completion = lambda *a, **kw: '[{"enh":1}]'
    gen = cg.COTGenerator(client)
    res = gen.enhance_with_cot([{"x": 1}], include_simple_steps=True)
    assert res == [{"enh": 1}]


def test_process_document_sync(monkeypatch):
    client = DummyClient()

    def fake_chat(messages, **kw):
        if len(client.calls) == 0:
            client.calls.append("sum")
            return "SUM"
        return '[{"question":"Q","reasoning":"R","answer":"A"}]'

    client.chat_completion = fake_chat
    gen = cg.COTGenerator(client)
    result = gen.process_document("doc", num_examples=1)
    assert result.summary == "SUM"
    assert result.cot_examples[0].answer == "A"


@pytest.mark.asyncio
async def test_process_document_async():
    client = DummyClient()

    def fake_chat(messages, **kw):
        if len(client.calls) == 0:
            client.calls.append("sum")
            return "SUM"
        return '[{"question":"Q","reasoning":"R","answer":"A"}]'

    client.chat_completion = fake_chat
    gen = cg.COTGenerator(client)
    result = await gen.process_document_async("doc", num_examples=1)
    assert result.cot_examples[0].question == "Q"

