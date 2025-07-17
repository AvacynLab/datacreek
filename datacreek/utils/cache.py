"""L1 cache utilities with Prometheus counters."""

from __future__ import annotations

from functools import wraps
from threading import Thread
from typing import Callable, Optional

import time

try:  # optional dependency
    import redis  # type: ignore
except Exception:  # pragma: no cover - optional
    redis = None  # type: ignore

try:  # optional Prometheus metrics
    from prometheus_client import CollectorRegistry, Counter, Gauge
except Exception:  # pragma: no cover - optional
    Counter = None  # type: ignore
    Gauge = None  # type: ignore

from .config import load_config

cfg = load_config()
cache_cfg = cfg.get("cache", {})

if Counter is not None and Gauge is not None:
    _REGISTRY = CollectorRegistry()
    hits = Counter("redis_hits_total", "Redis L1 hits", registry=_REGISTRY)
    miss = Counter("redis_miss_total", "Redis L1 misses", registry=_REGISTRY)
    hit_ratio_g = Gauge("redis_hit_ratio", "L1 hit ratio", registry=_REGISTRY)
else:  # pragma: no cover - metrics disabled
    hits = miss = hit_ratio_g = None  # type: ignore


def cache_l1(func: Callable) -> Callable:
    """Decorator injecting a default Redis client if available."""

    @wraps(func)
    def wrapper(
        key: str,
        *args,
        redis_client: Optional["redis.Redis"] = None,
        **kwargs,
    ):
        client = redis_client
        if client is None and redis is not None:
            try:
                client = redis.Redis()
            except Exception:  # pragma: no cover - connection errors
                client = None
        result = func(key, *args, redis_client=client, **kwargs)
        return result

    return wrapper


class TTLManager(Thread):
    """Background thread adjusting TTL from hit ratio."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.current_ttl = int(cache_cfg.get("l1_ttl_init", 3600))
        self._alpha = 0.3
        self._ema = 0.5
        self.start()

    def run(self) -> None:  # pragma: no cover - loop
        while True:
            time.sleep(300)
            self.run_once()

    def run_once(self) -> None:
        if hits is None or miss is None or hit_ratio_g is None:
            return
        h = hits._value.get()
        m = miss._value.get()
        if hit_ratio_g._value.get() > 0:
            self._ema = hit_ratio_g._value.get()
        else:
            ratio = h / max(1, h + m)
            self._ema = self._alpha * ratio + (1 - self._alpha) * self._ema
        hit_ratio_g.set(self._ema)
        ttl_min = int(cache_cfg.get("l1_ttl_min", 300))
        ttl_max = int(cache_cfg.get("l1_ttl_max", 7200))
        if self._ema < 0.2:
            self.current_ttl = max(int(self.current_ttl * 0.5), ttl_min)
        elif self._ema > 0.8:
            self.current_ttl = min(int(self.current_ttl * 1.2), ttl_max)


ttl_manager = TTLManager()


def l1_cache(key_fn: Callable[..., str]) -> Callable[[Callable], Callable]:
    """Simple Redis memoization with adaptive TTL."""

    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if redis is not None:
                key = key_fn(*args, **kwargs)
                try:
                    if redis.exists(key):
                        if hits is not None:
                            hits.inc()
                        return redis.get(key)
                except Exception:
                    pass
            result = fn(*args, **kwargs)
            if redis is not None:
                try:
                    if miss is not None:
                        miss.inc()
                    redis.setex(key, ttl_manager.current_ttl, result)
                except Exception:
                    pass
            return result

        return wrapper

    return decorator
