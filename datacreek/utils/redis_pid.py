from __future__ import annotations

"""Asynchronous PID controller for Redis TTL.

This module adjusts the L1 cache TTL using a PI controller to
maintain a target hit ratio. It relies on Redis counters ``hits``
and ``miss`` updated elsewhere in the application.
"""

import asyncio
import logging
from typing import Optional

try:  # optional dependency
    import redis.asyncio as aioredis  # type: ignore
except Exception:  # pragma: no cover - optional
    aioredis = None  # type: ignore

from .config import load_config

cfg = load_config()
cache_cfg = cfg.get("cache", {})

TARGET_HIT = float(cache_cfg.get("pid_target", 0.45))
K_P = float(cache_cfg.get("pid_kp", 0.4))
K_I = float(cache_cfg.get("pid_ki", 0.05))
INTERVAL = float(cache_cfg.get("pid_interval", 60.0))
_TTL_MIN = int(cache_cfg.get("l1_ttl_min", 300))
_TTL_MAX = int(cache_cfg.get("l1_ttl_max", 7200))
_current_ttl = int(cache_cfg.get("l1_ttl_init", 3600))

_integral_err = 0.0


async def update_ttl(client: "aioredis.Redis") -> None:
    """Single PID update step."""

    global _current_ttl, _integral_err
    try:
        hits = int(await client.get("hits") or 0)
        miss = int(await client.get("miss") or 0)
    except Exception:  # pragma: no cover - network
        return
    total = hits + miss
    ratio = hits / total if total else 0.0
    err = TARGET_HIT - ratio
    _integral_err += err * INTERVAL
    delta = K_P * err + K_I * _integral_err
    new_ttl = max(_TTL_MIN, min(_TTL_MAX, int(_current_ttl + delta)))
    if new_ttl != _current_ttl:
        logging.getLogger(__name__).debug("PID TTL updated to %d", new_ttl)
        _current_ttl = new_ttl


def get_current_ttl() -> int:
    """Return the current TTL managed by the PID controller."""

    return _current_ttl


def set_current_ttl(value: int) -> None:
    """Set the TTL value, useful for tests."""

    global _current_ttl
    _current_ttl = value


async def pid_loop(
    client: Optional["aioredis.Redis"] = None,
    *,
    stop_event: Optional[asyncio.Event] = None,
) -> None:
    """Run PID updates periodically in an asyncio task."""

    if aioredis is None:  # pragma: no cover - missing dependency
        return
    if client is None:
        try:
            client = aioredis.Redis()
        except Exception:  # pragma: no cover - connection
            return
    while stop_event is None or not stop_event.is_set():
        await asyncio.sleep(INTERVAL)
        await update_ttl(client)


def start_pid_controller(client: Optional["aioredis.Redis"] = None) -> asyncio.Task:
    """Launch :func:`pid_loop` and return the task."""

    event = asyncio.Event()
    task = asyncio.create_task(pid_loop(client, stop_event=event))
    task.stop_event = event  # type: ignore[attr-defined]
    return task
