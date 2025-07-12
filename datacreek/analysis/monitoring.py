"""Prometheus monitoring utilities."""

from __future__ import annotations

from typing import Dict

try:
    from prometheus_client import CollectorRegistry, Gauge, push_to_gateway, start_http_server
except Exception:  # pragma: no cover - optional
    CollectorRegistry = None
    Gauge = None
    push_to_gateway = None
    start_http_server = None


_METRICS = {
    "sigma_db": None,
    "recall10": None,
    "sheaf_score": None,
    "gw_entropy": None,
    "tpl_w1": None,
    "j_cost": None,
}


def start_metrics_server(port: int = 8000) -> None:
    """Start an HTTP server exposing Prometheus gauges."""
    if start_http_server is None or Gauge is None:
        return
    for name in _METRICS:
        _METRICS[name] = Gauge(name, f"{name} metric")
    start_http_server(port)


def push_metrics_gateway(metrics: Dict[str, float], gateway: str = "localhost:9091") -> None:
    """Push ``metrics`` to a Prometheus pushgateway if available."""
    if CollectorRegistry is None or Gauge is None or push_to_gateway is None:
        return
    registry = CollectorRegistry()
    for key, value in metrics.items():
        g = Gauge(key, f"{key} metric", registry=registry)
        g.set(value)
    try:
        push_to_gateway(gateway, job="datacreek", registry=registry)
    except Exception:  # pragma: no cover - network errors
        pass


def update_metric(name: str, value: float) -> None:
    """Update one of the default gauges."""
    g = _METRICS.get(name)
    if g is not None:
        try:
            g.set(value)
        except Exception:  # pragma: no cover
            pass
