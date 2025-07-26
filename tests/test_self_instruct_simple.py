import types
import pytest
from datacreek.utils import self_instruct


def test_generate_success(monkeypatch):
    monkeypatch.setattr(self_instruct, "validate_output", lambda t, s: s == "ok")
    monkeypatch.setattr(self_instruct, "parse_qa_pairs", lambda s: None)

    def llm(prompt: str) -> str:
        assert prompt == "Do"
        return "ok"

    out = self_instruct.generate_with_self_instruct(llm, "Do", template="tpl")
    assert out == "ok"


def test_generate_fallback(monkeypatch):
    monkeypatch.setattr(self_instruct, "validate_output", lambda t, s: s == '[{"q": "Q", "a": "A"}]')
    monkeypatch.setattr(self_instruct, "parse_qa_pairs", lambda s: [types.SimpleNamespace(q="Q", a="A")])

    def llm(_: str) -> str:
        return "bad"

    out = self_instruct.generate_with_self_instruct(llm, "Do", template="tpl", retries=1)
    assert out == '[{"q": "Q", "a": "A"}]'


def test_generate_error(monkeypatch):
    monkeypatch.setattr(self_instruct, "validate_output", lambda *a: False)
    monkeypatch.setattr(self_instruct, "parse_qa_pairs", lambda s: None)

    with pytest.raises(RuntimeError):
        self_instruct.generate_with_self_instruct(lambda _: "never", "go", template="tpl", retries=2)


@pytest.mark.asyncio
async def test_generate_async(monkeypatch):
    async def llm(_: str) -> str:
        return "ok"

    monkeypatch.setattr(self_instruct, "validate_output", lambda t, s: True)
    monkeypatch.setattr(self_instruct, "parse_qa_pairs", lambda s: None)

    out = await self_instruct.generate_with_self_instruct_async(llm, "Do", template="tpl")
    assert out == "ok"


def test_auto_tool_calls():
    def insert(txt: str, tools):
        return txt + "{" + ",".join(n for n, _ in tools) + "}"

    assert self_instruct.auto_tool_calls("hi", [("t", "ex")], insert) == "hi{t}"
    assert self_instruct.auto_tool_calls("hi", [], lambda t, s: "") == "hi"

def test_generate_json_error(monkeypatch):
    monkeypatch.setattr(self_instruct, "validate_output", lambda *a: False)
    bad = types.SimpleNamespace(x=b"b")
    monkeypatch.setattr(self_instruct, "parse_qa_pairs", lambda s: [bad])

    with pytest.raises(RuntimeError):
        self_instruct.generate_with_self_instruct(lambda _: "bad", "p", template="t", retries=1)


@pytest.mark.asyncio
async def test_generate_async_fallback(monkeypatch):
    async def llm(_: str) -> str:
        return "x"

    monkeypatch.setattr(self_instruct, "validate_output", lambda t, s: s == '[{"q": "Q", "a": "A"}]')
    monkeypatch.setattr(self_instruct, "parse_qa_pairs", lambda s: [types.SimpleNamespace(q="Q", a="A")])

    out = await self_instruct.generate_with_self_instruct_async(llm, "do", template="tpl", retries=1)
    assert out == '[{"q": "Q", "a": "A"}]'
