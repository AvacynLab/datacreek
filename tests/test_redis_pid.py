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
    fakeredis = pytest.importorskip("fakeredis.aioredis")
    client = fakeredis.FakeRedis()

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
    fakeredis = pytest.importorskip("fakeredis.aioredis")
    client = fakeredis.FakeRedis()

    async def prepare():
        await client.set("hits", 1)
        await client.set("miss", 9)

    asyncio.run(prepare())

    monkeypatch.setattr(redis_pid, "INTERVAL", 100.0)
    redis_pid.set_current_ttl(600)
    redis_pid._integral_err = 0.0

    asyncio.run(redis_pid.update_ttl(client))
    assert redis_pid.get_current_ttl() > 600
