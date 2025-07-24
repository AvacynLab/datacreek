import asyncio
import logging
import sys
import types

import pytest

# Patch heavy dependency before importing the module under test.
fake_llm_client = types.ModuleType("datacreek.models.llm_client")
fake_llm_client.LLMClient = object  # minimal stub
sys.modules["datacreek.models.llm_client"] = fake_llm_client

from datacreek.utils import batch


class DummyClient:
    def __init__(self):
        self.calls = []

    def batch_completion(self, batches, *, temperature=None, batch_size=None):
        self.calls.append(("sync", batches, temperature, batch_size))
        return [m[0]["content"] for m in batches]

    async def async_batch_completion(
        self, batches, *, temperature=None, batch_size=None
    ):
        self.calls.append(("async", batches, temperature, batch_size))
        return [m[0]["content"] for m in batches]


def test_process_batches_success():
    client = DummyClient()
    messages = [[{"role": "user", "content": "a"}], [{"role": "user", "content": "b"}]]
    res = batch.process_batches(
        client, messages, batch_size=1, temperature=0.2, parse_fn=lambda s: s.upper()
    )
    assert res == ["A", "B"]
    # ensure both batches processed
    assert len(client.calls) == 2
    assert client.calls[0][0] == "sync"


def test_process_batches_parse_error(caplog):
    client = DummyClient()

    def bad(_s: str) -> str:
        raise ValueError("boom")

    with caplog.at_level(logging.ERROR):
        res = batch.process_batches(
            client,
            [[{"role": "user", "content": "x"}]],
            batch_size=1,
            temperature=0.0,
            parse_fn=bad,
        )
    assert res == []
    assert "Failed to parse response" in caplog.text


def test_process_batches_raise_on_error():
    class ErrClient:
        def batch_completion(self, *a, **k):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        batch.process_batches(
            ErrClient(),
            [[{"role": "user", "content": "x"}]],
            batch_size=1,
            temperature=0.0,
            parse_fn=str,
            raise_on_error=True,
        )


@pytest.mark.asyncio
async def test_async_process_batches_success():
    client = DummyClient()
    messages = [[{"role": "user", "content": "a"}], [{"role": "user", "content": "b"}]]
    res = await batch.async_process_batches(
        client, messages, batch_size=1, temperature=0.1, parse_fn=lambda s: s.upper()
    )
    assert res == ["A", "B"]
    assert any(c[0] == "async" for c in client.calls)


@pytest.mark.asyncio
async def test_async_process_batches_error():
    class ErrClient:
        async def async_batch_completion(self, *a, **k):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        await batch.async_process_batches(
            ErrClient(),
            [[{"role": "user", "content": "z"}]],
            batch_size=1,
            temperature=0.0,
            parse_fn=str,
            raise_on_error=True,
        )


def test_process_batches_client_error(caplog):
    class ErrClient:
        def batch_completion(self, *a, **k):
            raise RuntimeError("boom")

    with caplog.at_level(logging.ERROR):
        res = batch.process_batches(
            ErrClient(),
            [[{"role": "user", "content": "x"}]],
            batch_size=1,
            temperature=0.0,
            parse_fn=str,
        )
    assert res == []
    assert "Error processing batch" in caplog.text


@pytest.mark.asyncio
async def test_async_process_batches_parse_error(caplog):
    class Dummy:
        async def async_batch_completion(
            self, batches, *, temperature=None, batch_size=None
        ):
            return ["foo"] * len(batches)

    def parse(_s: str) -> str:
        raise ValueError("bad")

    with caplog.at_level(logging.ERROR):
        res = await batch.async_process_batches(
            Dummy(),
            [[{"role": "user", "content": "a"}]],
            batch_size=1,
            temperature=0.0,
            parse_fn=parse,
        )
    assert res == []
    assert "Failed to parse response" in caplog.text


@pytest.mark.asyncio
async def test_async_process_batches_client_error(caplog):
    class ErrClient:
        async def async_batch_completion(self, *a, **k):
            raise RuntimeError("fail")

    with caplog.at_level(logging.ERROR):
        res = await batch.async_process_batches(
            ErrClient(),
            [[{"role": "user", "content": "a"}]],
            batch_size=1,
            temperature=0.0,
            parse_fn=str,
        )
    assert res == []
    assert "Error processing batch" in caplog.text
