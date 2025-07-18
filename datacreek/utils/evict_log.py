from __future__ import annotations

"""Ring buffer for LMDB eviction logs."""

from collections import deque
from dataclasses import dataclass
from typing import Deque, Literal

from .config import load_config


@dataclass
class EvictLog:
    """Record a single LMDB eviction."""

    key: str
    ts: float
    cause: Literal["ttl", "quota", "manual"]


cfg = load_config()
cache_cfg = cfg.get("cache", {})
max_len = int(cache_cfg.get("l2_log_len", 100_000))

evict_logs: Deque[EvictLog] = deque(maxlen=max_len)


def log_eviction(key: str, ts: float, cause: Literal["ttl", "quota", "manual"]) -> None:
    """Append an eviction record to the global log."""

    evict_logs.append(EvictLog(key, ts, cause))


def clear_eviction_logs() -> None:
    """Remove all eviction entries."""

    evict_logs.clear()
