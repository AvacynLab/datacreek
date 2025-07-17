import pytest

import datacreek.utils.cache as cache


def test_ttl_adaptive(monkeypatch):
    cache.ttl_manager.current_ttl = 600
    if cache.hits is None or cache.hit_ratio_g is None:
        pytest.skip("prometheus not available")
    cache.hit_ratio_g.set(0.1)
    cache.hits._value.set(0)
    cache.miss._value.set(10)
    cache.ttl_manager.run_once()
    assert cache.ttl_manager.current_ttl == 300
