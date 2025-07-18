import asyncio
import importlib.abc
import importlib.util
import sys
import time
import warnings
from pathlib import Path
from types import ModuleType

import pytest

utils_pkg = ModuleType("datacreek.utils")
utils_pkg.__path__ = [str(Path(__file__).resolve().parents[1] / "datacreek" / "utils")]
sys.modules["datacreek.utils"] = utils_pkg
config_stub = ModuleType("datacreek.utils.config")
config_stub.load_config = lambda: {"cache": {}}
sys.modules["datacreek.utils.config"] = config_stub


@pytest.fixture(autouse=True, scope="module")
def _cleanup():
    yield
    loop = asyncio.new_event_loop()
    loop.run_until_complete(cache.ttl_manager.stop())
    loop.close()


spec = importlib.util.spec_from_file_location(
    "datacreek.utils.cache",
    Path(__file__).resolve().parents[1] / "datacreek" / "utils" / "cache.py",
)
cache = importlib.util.module_from_spec(spec)
assert isinstance(spec.loader, importlib.abc.Loader)
spec.loader.exec_module(cache)


def test_ttl_adaptive(monkeypatch):
    cache.ttl_manager.current_ttl = 600
    if cache.hits is None or cache.hit_ratio_g is None:
        pytest.skip("prometheus not available")
    cache.hit_ratio_g.set(0.1)
    cache.hits._value.set(0)
    cache.miss._value.set(10)
    cache.ttl_manager.run_once()
    assert cache.ttl_manager.current_ttl == 300


def test_ttl_manager_async_task():
    """Manager starts an asyncio task on import."""
    for _ in range(10):
        if cache.ttl_manager._task is not None:
            break
        time.sleep(0.01)
    assert cache.ttl_manager._task is not None


def test_ttl_manager_stop():
    mgr = cache.TTLManager()
    assert mgr._task is not None
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            loop.run_until_complete(mgr.stop())
        assert not w
    finally:
        loop.close()
    assert mgr._task is None
    assert mgr._thread is None
    # also stop global instance to avoid task leakage
    loop = asyncio.new_event_loop()
    loop.run_until_complete(cache.ttl_manager.stop())
    loop.close()


def test_ttl_pid_convergence():
    if cache.hits is None or cache.hit_ratio_g is None:
        pytest.skip("prometheus not available")
    cache.ttl_manager.current_ttl = 300
    cache.ttl_manager._integral = 0.0
    cache.hits._value.set(10)
    cache.miss._value.set(1)
    values = []
    for _ in range(3):
        cache.ttl_manager.run_once()
        values.append(cache.ttl_manager.current_ttl)
        cache.hits._value.set(cache.hits._value.get() + 10)
    assert values == sorted(values)
