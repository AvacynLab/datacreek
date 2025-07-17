"""L1 cache utilities with Prometheus counters."""

from __future__ import annotations

from functools import wraps
from typing import Callable, Optional

try:  # optional dependency
    import redis  # type: ignore
except Exception:  # pragma: no cover - optional
    redis = None  # type: ignore

from .config import load_config

cfg = load_config()
cache_cfg = cfg.get("cache", {})


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
