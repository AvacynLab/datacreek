import importlib
import importlib.abc
import importlib.util
import sys
from types import SimpleNamespace

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


def load_module(monkeypatch):
    ports = []
    pushes = []

    class Gauge:
        def __init__(self, name, desc, **kw):
            self.name = name
            self.desc = desc
            self.value = None
            self.kw = kw

        def labels(self, **labels):
            self.labels_dict = labels
            return self

        def set(self, value):
            self.value = value

    prom = SimpleNamespace(
        REGISTRY=SimpleNamespace(_names_to_collectors={}),
        CollectorRegistry=lambda: "reg",
        Counter=Gauge,
        Gauge=Gauge,
        push_to_gateway=lambda gateway, job, registry: pushes.append(
            (gateway, job, registry)
        ),
        start_http_server=lambda port: ports.append(port),
    )
    monkeypatch.setitem(sys.modules, "prometheus_client", prom)
    spec = importlib.util.spec_from_file_location(
        "monitoring", ROOT / "datacreek" / "analysis" / "monitoring.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert isinstance(spec.loader, importlib.abc.Loader)
    spec.loader.exec_module(module)
    return module, ports, pushes


def test_start_metrics_server_idempotent(monkeypatch):
    mod, ports, _ = load_module(monkeypatch)
    mod.start_metrics_server(1234)
    mod.start_metrics_server(5678)
    assert ports == [1234]
    assert all(g is not None for g in mod._METRICS.values())


def test_push_and_update(monkeypatch):
    mod, _, pushes = load_module(monkeypatch)
    mod.start_metrics_server(9999)
    mod.update_metric("tpl_w1", 1.5)
    assert mod._METRICS["tpl_w1"].value == 1.5
    mod.push_metrics_gateway({"foo": 2.0}, gateway="gw:9091")

    assert pushes == [("gw:9091", "datacreek", "reg")]
