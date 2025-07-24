import json
import types
from collections import deque

import pytest

from datacreek.utils import evict_log


def test_key_type_and_ring_buffer(monkeypatch):
    monkeypatch.setattr(evict_log, "evict_logs", deque(maxlen=3))
    evict_log.clear_eviction_logs()
    for i in range(5):
        evict_log.log_eviction(f"img:{i}", float(i), "ttl")
    assert len(evict_log.evict_logs) == 3
    assert evict_log.evict_logs[0].key == "img:2"
    assert evict_log._key_type("audio:foo") == "audio"
    assert evict_log._key_type("foo") == "raw"


def test_metrics_and_json_logging(monkeypatch, caplog):
    counter_calls = []
    gauge_values = []

    class DummyCounter:
        def labels(self, *, cause, type):
            def inc():
                counter_calls.append((cause, type))

            return types.SimpleNamespace(inc=inc)

    class DummyGauge:
        def labels(self, *, cause):
            def set(value):
                gauge_values.append((cause, value))

            return types.SimpleNamespace(set=set)

    monkeypatch.setattr(
        "datacreek.analysis.monitoring.lmdb_evictions_total",
        DummyCounter(),
        raising=False,
    )
    monkeypatch.setattr(
        "datacreek.analysis.monitoring.lmdb_eviction_last_ts",
        DummyGauge(),
        raising=False,
    )

    monkeypatch.setattr(evict_log, "evict_logs", deque(maxlen=10))
    evict_log.clear_eviction_logs()
    caplog.set_level("INFO", logger="datacreek.eviction")
    evict_log.log_eviction("img:1", 4.2, "quota")

    assert counter_calls == [("quota", "img")]
    assert gauge_values == [("quota", 4.2)]
    assert evict_log.evict_logs[-1].key == "img:1"
    assert any("lmdb_eviction" in r.message for r in caplog.records)
    data = json.loads(caplog.records[0].message)
    assert data["key"] == "img:1"
