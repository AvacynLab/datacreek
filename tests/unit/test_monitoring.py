import importlib
import sys
import types

import pytest


class DummyGauge:
    def __init__(self, name, desc, *, labelnames=None, registry=None):
        self.name = name
        self.desc = desc
        self.labelnames = labelnames
        self.registry = registry
        self.value = None
        self.labels_dict = None
        if registry is not None:
            registry.store[name] = None

    def labels(self, **labels):
        self.labels_dict = labels
        return self

    def set(self, value):
        self.value = value
        if self.registry is not None:
            self.registry.store[self.name] = value


class DummyRegistry:
    def __init__(self):
        self.store = {}


@pytest.fixture
def monitoring(monkeypatch):
    """Reload monitoring module with dummy prometheus client."""
    ports = []
    pushes = []

    def start_http_server(port):
        ports.append(port)

    def push_to_gateway(gateway, job, registry):
        pushes.append((gateway, job, registry.store.copy()))

    prom = types.SimpleNamespace(
        REGISTRY=types.SimpleNamespace(_names_to_collectors={}),
        CollectorRegistry=DummyRegistry,
        Counter=DummyGauge,
        Gauge=DummyGauge,
        push_to_gateway=push_to_gateway,
        start_http_server=start_http_server,
    )
    monkeypatch.setitem(sys.modules, "prometheus_client", prom)
    # avoid automatic startup during import
    monkeypatch.setitem(sys.modules, "datacreek.utils.config", types.SimpleNamespace(load_config=lambda: {}))
    import datacreek.analysis.monitoring as mon
    mon = importlib.reload(mon)
    ports.clear()
    mon._SERVER_STARTED = False
    return mon, ports, pushes


def test_start_metrics_server_idempotent(monitoring):
    mon, ports, _ = monitoring
    mon.start_metrics_server(1234)
    mon.start_metrics_server(1234)
    assert ports == [1234]
    assert mon._SERVER_STARTED is True
    assert all(mon._METRICS[name] is not None for name in mon._METRICS)


def test_update_and_push_metrics(monitoring):
    mon, _, pushes = monitoring
    mon.start_metrics_server(0)
    mon.update_metric("ingest_queue_fill_ratio", 0.5)
    assert mon._METRICS["ingest_queue_fill_ratio"].value == 0.5
    mon.update_metric("whisper_xrt", 1.1, labels={"device": "cpu"})
    gauge = mon._METRICS["whisper_xrt"]
    assert gauge.labels_dict == {"device": "cpu"}
    assert gauge.value == 1.1
    mon.push_metrics_gateway({"a": 2.0}, gateway="gw")
    assert pushes == [("gw", "datacreek", {"a": 2.0})]
