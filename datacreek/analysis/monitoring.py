"""Prometheus monitoring utilities."""

from __future__ import annotations

from typing import Dict

try:
    from prometheus_client import (
        CollectorRegistry,
        Counter,
        Gauge,
        push_to_gateway,
        start_http_server,
    )
except Exception:  # pragma: no cover - optional
    CollectorRegistry = None
    Gauge = None
    push_to_gateway = None
    start_http_server = None

if Gauge is not None:
    tpl_w1 = Gauge("tpl_w1", "Wasserstein-1 TPL")
    sheaf_score = Gauge("sheaf_score", "Sheaf consistency score")
    gw_entropy = Gauge("gw_entropy", "GraphWave entropy")
    autotune_cost = Gauge("autotune_cost", "Current J(theta)")
    bias_wasserstein_last = Gauge(
        "bias_wasserstein_last", "Latest Wasserstein distance used"
    )
    haa_edges_total = Counter("haa_edges_total", "Hyper-AA edges written")
    j_cost = autotune_cost
else:  # pragma: no cover - optional dependency missing
    tpl_w1 = None
    sheaf_score = None
    gw_entropy = None
    autotune_cost = None
    bias_wasserstein_last = None
    haa_edges_total = None
    j_cost = None


_METRICS = {
    "sigma_db": None,
    "recall10": None,
    "sheaf_score": sheaf_score,
    "gw_entropy": gw_entropy,
    "tpl_w1": tpl_w1,
    "j_cost": j_cost,
    "autotune_cost": autotune_cost,
    "bias_wasserstein_last": bias_wasserstein_last,
    "haa_edges_total": haa_edges_total,
    # ingestion statistics
    "atoms_total": None,
    "avg_chunk_len": None,
    # embedding statistics
    "n2v_var_norm": None,
}


def start_metrics_server(port: int = 8000) -> None:
    """Start an HTTP server exposing Prometheus gauges."""
    if start_http_server is None or Gauge is None:
        return
    for name, g in _METRICS.items():
        if g is None:
            _METRICS[name] = Gauge(name, f"{name} metric")
    start_http_server(port)


def push_metrics_gateway(
    metrics: Dict[str, float], gateway: str = "localhost:9091"
) -> None:
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


from ..utils.config import load_config

cfg = load_config()
start_metrics_server(int(cfg.get("monitor", {}).get("port", 8000)))
