import pytest
import types

import datacreek.models.llm_service as ls

class DummyClient:
    def __init__(self, *a, **k):
        self.seen = []
    def chat_completion(self, messages):
        self.seen.append(("chat", messages))
        return "reply"
    def batch_completion(self, batches):
        self.seen.append(("batch", batches))
        return [m[0]["content"].upper() for m in batches]
    async def async_batch_completion(self, batches):
        self.seen.append(("async", batches))
        return [m[0]["content"] * 2 for m in batches]

def make_service(monkeypatch):
    monkeypatch.setattr(ls, "LLMClient", lambda *a, **k: DummyClient())
    return ls.LLMService()

def test_call(monkeypatch):
    svc = make_service(monkeypatch)
    res = svc("hello")
    assert res == "reply"
    assert svc.client.seen[0][0] == "chat"

def test_batch(monkeypatch):
    svc = make_service(monkeypatch)
    res = svc.batch(["a", "b"])
    assert res == ["A", "B"]
    assert svc.client.seen[0][0] == "batch"

@pytest.mark.asyncio
async def test_acomplete(monkeypatch):
    svc = make_service(monkeypatch)
    out = await svc.acomplete(["x"])
    assert out == ["xx"]
    assert svc.client.seen[0][0] == "async"

@pytest.mark.asyncio
async def test_abatch_alias(monkeypatch):
    svc = make_service(monkeypatch)
    out = await svc.abatch(["y"])
    assert out == ["yy"]
    assert svc.client.seen[0][0] == "async"
