from __future__ import annotations

"""Ring buffer for LMDB eviction logs."""

import json
import logging
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
logger = logging.getLogger("datacreek.eviction")


def log_eviction(key: str, ts: float, cause: Literal["ttl", "quota", "manual"]) -> None:
    """Append an eviction record to the global log."""

    evict_logs.append(EvictLog(key, ts, cause))
    logger.info(
        json.dumps({"event": "lmdb_eviction", "key": key, "ts": ts, "cause": cause})
    )
    try:
        from ..analysis.monitoring import lmdb_eviction_last_ts, lmdb_evictions_total

        if lmdb_evictions_total is not None:
            lmdb_evictions_total.labels(cause=cause).inc()
        if lmdb_eviction_last_ts is not None:
            lmdb_eviction_last_ts.labels(cause=cause).set(ts)
    except Exception:  # pragma: no cover - optional metrics
        pass


def clear_eviction_logs() -> None:
    """Remove all eviction entries."""

    evict_logs.clear()
