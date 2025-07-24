import sys
import types

from datacreek.utils.metrics import push_metrics


def test_push_metrics_no_client(monkeypatch):
    monkeypatch.setitem(sys.modules, "statsd", None)
    push_metrics({"a": 1.0})


def test_push_metrics_with_client(monkeypatch):
    calls = []

    class Dummy:
        def __init__(self, *_, **__):
            pass

        def gauge(self, key, value):
            calls.append((key, value))

    monkeypatch.setitem(sys.modules, "statsd", types.SimpleNamespace(StatsClient=Dummy))
    push_metrics({"a": 2.0}, prefix="x")
    assert calls == [("a", 2.0)]
