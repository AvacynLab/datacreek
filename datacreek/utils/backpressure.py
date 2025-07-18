"""Back-pressure controls for ingestion queue."""

from __future__ import annotations

import os
import threading

__all__ = [
    "has_capacity",
    "acquire_slot",
    "release_slot",
    "set_limit",
    "active_count",
]

_limit = int(os.getenv("INGEST_QUEUE_LIMIT", "10000"))
_active = 0
_lock = threading.Lock()


def set_limit(limit: int) -> None:
    """Set queue limit (testing only)."""
    global _limit, _active
    with _lock:
        _limit = limit
        _active = 0


def has_capacity() -> bool:
    """Return ``True`` if the queue is below its limit."""
    with _lock:
        return _active < _limit


def acquire_slot() -> bool:
    """Try to reserve one slot for an ingest task."""
    global _active
    with _lock:
        if _active >= _limit:
            return False
        _active += 1
        return True


def release_slot() -> None:
    """Release a previously acquired slot."""
    global _active
    with _lock:
        if _active > 0:
            _active -= 1


def active_count() -> int:
    """Return the number of active ingest tasks."""
    with _lock:
        return _active
