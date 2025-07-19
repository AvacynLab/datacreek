import asyncio
import importlib.abc
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

utils_pkg = ModuleType("datacreek.utils")
utils_pkg.__path__ = [str(Path(__file__).resolve().parents[1] / "datacreek" / "utils")]
sys.modules["datacreek.utils"] = utils_pkg
config_stub = ModuleType("datacreek.utils.config")
config_stub.load_config = lambda: {"cache": {}}
sys.modules["datacreek.utils.config"] = config_stub

spec = importlib.util.spec_from_file_location(
    "datacreek.utils.redis_pid",
    Path(__file__).resolve().parents[1] / "datacreek" / "utils" / "redis_pid.py",
)
redis_pid = importlib.util.module_from_spec(spec)
assert isinstance(spec.loader, importlib.abc.Loader)
spec.loader.exec_module(redis_pid)


def test_pid_controller_increase(monkeypatch):
    class DummyRedis:
        def __init__(self):
            self.store = {}

        async def set(self, k, v):
            self.store[k] = str(v)

        async def get(self, k):
            return self.store.get(k)

    client = DummyRedis()

    async def prepare():
        await client.set("hits", 9)
        await client.set("miss", 1)

    asyncio.run(prepare())

    monkeypatch.setattr(redis_pid, "INTERVAL", 100.0)
    redis_pid.set_current_ttl(600)
    redis_pid._integral_err = 0.0

    asyncio.run(redis_pid.update_ttl(client))
    assert redis_pid.get_current_ttl() < 600


def test_pid_controller_decrease(monkeypatch):
    class DummyRedis:
        def __init__(self):
            self.store = {}

        async def set(self, k, v):
            self.store[k] = str(v)

        async def get(self, k):
            return self.store.get(k)

    client = DummyRedis()

    async def prepare():
        await client.set("hits", 1)
        await client.set("miss", 9)

    asyncio.run(prepare())

    monkeypatch.setattr(redis_pid, "INTERVAL", 100.0)
    redis_pid.set_current_ttl(600)
    redis_pid._integral_err = 0.0

    asyncio.run(redis_pid.update_ttl(client))
    assert redis_pid.get_current_ttl() >= 600


def test_pid_burst_no_overshoot(monkeypatch):
    class DummyRedis:
        def __init__(self):
            self.store = {}

        async def set(self, k, v):
            self.store[k] = str(v)

        async def get(self, k):
            return self.store.get(k)

    client = DummyRedis()

    async def run_trace(ratios, ttls):
        for r in ratios:
            hits = int(r * 100)
            miss = 100 - hits
            await client.set("hits", hits)
            await client.set("miss", miss)
            await redis_pid.update_ttl(client)
            ttls.append(redis_pid.get_current_ttl())

    monkeypatch.setattr(redis_pid, "INTERVAL", 1.0)
    redis_pid.set_current_ttl(300)
    redis_pid._integral_err = 0.0

    ratios = [0.2, 0.2, 0.2, 0.9, 0.45, 0.45, 0.45]
    ttls: list[int] = []
    asyncio.run(run_trace(ratios, ttls))

    final_ttl = ttls[-1]
    peak = max(ttls)
    overshoot = abs(peak - final_ttl)
    assert overshoot / final_ttl <= 0.05


def test_pid_converges_to_target(monkeypatch):
    class DummyRedis:
        def __init__(self):
            self.store = {}

        async def set(self, k, v):
            self.store[k] = str(v)

        async def get(self, k):
            return self.store.get(k)

    client = DummyRedis()

    async def run_trace(ratios, ttls):
        for r in ratios:
            hits = int(r * 100)
            miss = 100 - hits
            await client.set("hits", hits)
            await client.set("miss", miss)
            await redis_pid.update_ttl(client)
            ttls.append(redis_pid.get_current_ttl())

    monkeypatch.setattr(redis_pid, "INTERVAL", 1.0)
    redis_pid.set_current_ttl(300)
    redis_pid._integral_err = 0.0

    ratios = [0.2] * 3 + [0.9] * 3 + [0.45] * 6
    ttls: list[int] = []
    asyncio.run(run_trace(ratios, ttls))

    stable = ttls[-5:]
    assert max(stable) - min(stable) <= stable[-1] * 0.05
