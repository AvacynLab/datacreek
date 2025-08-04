"""Utilities for streaming hypergraph updates.

This module provides two small helpers used to update a hot ``RedisGraph`` layer
from a stream of edge events.  The goal is to approximate the behaviour of a
Flink job with 30 second tumbling windows without pulling in heavy
infrastructure.

The function :func:`process_edge_stream` consumes an iterable of edge events.
Each event must provide a ``ts`` timestamp, a ``src`` node and a ``dst`` node.
Events are grouped into 30 second windows; when a window closes, the edges are
written to the provided hot layer via :meth:`RedisGraphHotLayer.add_edge`.

``RedisGraphHotLayer`` stores edges in memory and tracks the latency of each
operation so that the p95 latency can be monitored.  The latency target from the
checklist is 500 ms:

.. math::
   \text{p95 latency} < 0.5\,\text{s}
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List


def process_edge_stream(
    events: Iterable[Dict[str, Any]],
    hot_layer: "RedisGraphHotLayer",
    window: timedelta = timedelta(seconds=30),
) -> None:
    """Group events into ``window`` and write edges incrementally.

    Parameters
    ----------
    events:
        Iterable of event dictionaries with ``src``, ``dst`` and ``ts`` (``datetime``).
    hot_layer:
        Instance of :class:`RedisGraphHotLayer` that receives edge updates.
    window:
        Length of the tumbling window.  The default is 30 seconds as required
        by the specification.
    """
    sorted_events: List[Dict[str, Any]] = sorted(events, key=lambda e: e["ts"])
    batch: List[Dict[str, Any]] = []
    window_start: datetime | None = None

    for event in sorted_events:
        if window_start is None:
            window_start = event["ts"]
        if event["ts"] - window_start < window:
            batch.append(event)
        else:
            _flush_batch(batch, hot_layer)
            batch = [event]
            window_start = event["ts"]
    if batch:
        _flush_batch(batch, hot_layer)


def _flush_batch(batch: List[Dict[str, Any]], hot_layer: "RedisGraphHotLayer") -> None:
    """Write a batch of events to the hot layer."""
    for event in batch:
        hot_layer.add_edge(event["src"], event["dst"])


@dataclass
class RedisGraphHotLayer:
    """Simplistic in-memory stand-in for a RedisGraph hot layer.

    The class records the latency of each write/read operation to enable
    monitoring of the p95 latency.
    """

    edges: Dict[str, set] = None
    latencies: List[float] = None  # seconds

    def __post_init__(self) -> None:  # pragma: no cover - simple initialisation
        self.edges = defaultdict(set)
        self.latencies = []

    def add_edge(self, src: str, dst: str) -> None:
        """Insert an edge ``src -> dst``.

        The operation time is recorded to allow p95 latency computation.
        """
        start = time.perf_counter()
        self.edges[src].add(dst)
        self.latencies.append(time.perf_counter() - start)

    def neighbours(self, src: str) -> List[str]:
        """Return neighbours of ``src`` and record query latency."""
        start = time.perf_counter()
        result = list(self.edges.get(src, []))
        self.latencies.append(time.perf_counter() - start)
        return result

    def p95_latency_ms(self) -> float:
        """Compute the 95th percentile latency in milliseconds."""
        if not self.latencies:
            return 0.0
        sorted_lat = sorted(self.latencies)
        idx = int(0.95 * (len(sorted_lat) - 1))
        return sorted_lat[idx] * 1000
