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

    assert cache.ttl_manager._task is not None
