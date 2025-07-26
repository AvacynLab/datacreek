import types
from types import SimpleNamespace
import datacreek.models.llm_client as lc

class FakeResponse:
    def __init__(self, data, status=200):
        self._data = data
        self.status_code = status
    def json(self):
        return self._data
    def raise_for_status(self):
        if self.status_code != 200:
            raise lc.requests.exceptions.RequestException("error")


def make_client(monkeypatch, cfg=None):
    if cfg is None:
        cfg = {
            "llm": {"provider": "vllm"},
            "vllm": {"api_base": "http://test", "model": "x", "max_retries": 2, "retry_delay": 0},
            "generation": {},
        }
    monkeypatch.setattr(lc, "load_config_with_overrides", lambda *a, **k: cfg)
    stub = SimpleNamespace(
        get=lambda *a, **k: FakeResponse({}),
        post=lambda *a, **k: FakeResponse({}),
        exceptions=lc.requests.exceptions,
    )
    monkeypatch.setattr(lc, "requests", stub)
    return lc.LLMClient()


def test_check_vllm_server(monkeypatch):
    calls = {}
    c = make_client(monkeypatch)
    def fake_get(url, timeout):
        calls["url"] = url
        return FakeResponse({"ok": True})
    monkeypatch.setattr(lc, "requests", SimpleNamespace(get=fake_get, post=lambda *a, **k: FakeResponse({}), exceptions=lc.requests.exceptions))
    ok, data = c._check_vllm_server()
    assert ok and data == {"ok": True}
    assert calls["url"].endswith("/models")


def test_check_vllm_server_error(monkeypatch):
    c = make_client(monkeypatch)
    def bad_get(*a, **k):
        raise lc.requests.exceptions.RequestException("fail")
    monkeypatch.setattr(lc, "requests", SimpleNamespace(get=bad_get, post=lambda *a, **k: FakeResponse({}), exceptions=lc.requests.exceptions))
    ok, info = c._check_vllm_server()
    assert not ok and "fail" in info


def test_vllm_chat_completion(monkeypatch):
    c = make_client(monkeypatch)
    resp = FakeResponse({"choices": [{"message": {"content": "hi"}}]})
    monkeypatch.setattr(lc, "requests", SimpleNamespace(get=lambda *a, **k: FakeResponse({}), post=lambda *a, **k: resp, exceptions=lc.requests.exceptions))
    out = c._vllm_chat_completion([{"role": "user", "content": "hi"}], 0.1, 10, 1.0, 0.0, 0.0, None, False)
    assert out == "hi"


def test_vllm_chat_completion_retry(monkeypatch):
    c = make_client(monkeypatch)
    calls = {"count": 0}
    def flaky_post(*a, **k):
        calls["count"] += 1
        if calls["count"] == 1:
            raise lc.requests.exceptions.RequestException("boom")
        return FakeResponse({"choices": [{"message": {"content": "ok"}}]})
    monkeypatch.setattr(lc, "requests", SimpleNamespace(get=lambda *a, **k: FakeResponse({}), post=flaky_post, exceptions=lc.requests.exceptions))
    c.max_retries = 2
    c.retry_delay = 0
    out = c._vllm_chat_completion([{"role": "user", "content": "hi"}], 0.1, 10, 1.0, 0.0, 0.0, None, False)
    assert calls["count"] == 2
    assert out == "ok"


def test_vllm_batch_completion(monkeypatch):
    c = make_client(monkeypatch)
    posted = []
    def post(url, headers, data, timeout):
        posted.append(data)
        return FakeResponse({"choices": [{"message": {"content": "ans"}}]})
    monkeypatch.setattr(lc, "requests", SimpleNamespace(get=lambda *a, **k: FakeResponse({}), post=post, exceptions=lc.requests.exceptions))
    out = c._vllm_batch_completion([[{"role": "u", "content": "a"}], [{"role": "u", "content": "b"}]], 0.1, 10, 1.0, 1, 0.0, 0.0, None, False)
    assert out == ["ans", "ans"]
    assert len(posted) == 2


def test_chat_completion_wrapper(monkeypatch):
    c = make_client(monkeypatch)
    called = {}
    def fake(messages, *a, **k):
        called['ok'] = True
        return "wrapped"
    monkeypatch.setattr(lc.LLMClient, "_vllm_chat_completion", fake)
    out = c.chat_completion([{"role": "u", "content": "x"}])
    assert out == "wrapped" and called.get('ok')


def test_batch_completion_wrapper(monkeypatch):
    c = make_client(monkeypatch)
    called = {}
    def fake(self, msg_batches, *a, **k):
        called['ok'] = len(msg_batches)
        return ["r" for _ in msg_batches]
    monkeypatch.setattr(lc.LLMClient, "_vllm_batch_completion", fake)
    out = c.batch_completion([[{"role": "u", "content": "a"}], [{"role": "u", "content": "b"}]])
    assert out == ["r", "r"] and called['ok'] == 2


def test_from_config(monkeypatch, tmp_path):
    cfg = {
        "llm": {"provider": "vllm"},
        "vllm": {"api_base": "http://x", "model": "y", "max_retries": 1, "retry_delay": 0},
        "generation": {},
    }
    monkeypatch.setattr(lc, "load_config_with_overrides", lambda path, overrides=None: cfg)
    monkeypatch.setattr(lc, "requests", SimpleNamespace(get=lambda *a, **k: FakeResponse({}), post=lambda *a, **k: FakeResponse({}), exceptions=lc.requests.exceptions))
    c = lc.LLMClient.from_config(tmp_path / "c.yaml")
    assert isinstance(c, lc.LLMClient)
