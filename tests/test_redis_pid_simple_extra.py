import asyncio
import types
import pytest
import datacreek.utils.redis_pid as redis_pid

class DummyRedis:
    def __init__(self, hits=0, miss=0):
        self.hits = hits
        self.miss = miss
    async def get(self, key):
        return self.hits if key == "hits" else self.miss

@pytest.mark.asyncio
async def test_update_ttl_increases(monkeypatch):
    monkeypatch.setattr(redis_pid, "redis_hit_ratio_stdev", None)
    monkeypatch.setattr(redis_pid, "update_metric", lambda *a, **k: None)
    monkeypatch.setattr(redis_pid, "K_P", 1.0)
    monkeypatch.setattr(redis_pid, "K_I", 1.0)
    monkeypatch.setattr(redis_pid, "I_MAX", 100.0)
    redis_pid.set_current_ttl(3600)
    redis_pid._integral_err = 0.0
    redis_pid._kp_dynamic = redis_pid.K_P
    redis_pid._err_mean = 0.0
    redis_pid._err_var = 1.0
    redis_pid._sigma_e0 = 1.0
    client = DummyRedis(hits=0, miss=10)
    await redis_pid.update_ttl(client)
    assert redis_pid.get_current_ttl() > 3600

@pytest.mark.asyncio
async def test_update_ttl_network_error(monkeypatch):
    class Bad:
        async def get(self, key):
            raise RuntimeError
    before = redis_pid.get_current_ttl()
    await redis_pid.update_ttl(Bad())
    assert redis_pid.get_current_ttl() == before

@pytest.mark.asyncio
async def test_pid_loop_and_start(monkeypatch):
    events = []
    async def fake_update(client):
        events.append("u")
        stop.set()
    monkeypatch.setattr(redis_pid, "update_ttl", fake_update)
    monkeypatch.setattr(redis_pid, "aioredis", types.SimpleNamespace(Redis=lambda: DummyRedis()))
    async def fake_sleep(_):
        return None
    monkeypatch.setattr(redis_pid.asyncio, "sleep", fake_sleep)
    stop = asyncio.Event()
    await redis_pid.pid_loop(stop_event=stop)
    assert events == ["u"]
    created = {}
    def fake_create_task(coro):
        created["coro"] = coro
        # close the coroutine without awaiting to avoid warnings
        coro.close()
        return types.SimpleNamespace(stop_event=None)
    monkeypatch.setattr(asyncio, "create_task", fake_create_task)
    task = redis_pid.start_pid_controller(DummyRedis())
    assert "coro" in created
    assert hasattr(task, "stop_event")
