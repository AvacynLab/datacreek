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
    tpl_w1_g = Gauge("tpl_w1", "Wasserstein-1 TPL")
    sheaf_score_g = Gauge("sheaf_score", "Sheaf consistency score")
    gw_entropy = Gauge("gw_entropy", "GraphWave entropy")
    autotune_cost_g = Gauge("autotune_cost", "Current J(theta)")
    bias_wasserstein_last = Gauge(
        "bias_wasserstein_last", "Latest Wasserstein distance used"
    )
    haa_edges_total = Counter("haa_edges_total", "Hyper-AA edges written")
    gp_jitter_restarts_total = Counter(
        "gp_jitter_restarts_total", "SVGP restarts due to jitter"
    )
    prune_reverts_total = Counter("prune_reverts_total", "FractalNet pruning rollbacks")
    redis_hit_ratio = Gauge("redis_hit_ratio", "Redis L1 hit ratio")
    eigsh_timeouts_total = Counter(
        "eigsh_timeouts_total",
        "Number of eigsh timeouts triggering Lanczos fallback",
    )
    eigsh_last_duration = Gauge(
        "eigsh_last_runtime_seconds",
        "Duration of the last eigsh call in seconds",
    )
    ann_backend = Gauge(
        "ann_backend",
        "Approximate NN backend in use",
    )
    redis_evictions_l2_total = Gauge(
        "redis_evictions_l2_total",
        "LMDB L2 evictions performed",
    )
    lmdb_evictions_total = Counter(
        "lmdb_evictions_total",
        "LMDB evictions by cause",
        labelnames=["cause"],
    )
    lmdb_eviction_last_ts = Gauge(
        "lmdb_eviction_last_ts",
        "Timestamp of last LMDB eviction per cause",
        labelnames=["cause"],
    )
    lang_mismatch_total = Counter(
        "lang_mismatch_total",
        "Cross-language merge attempts rejected",
    )
    whisper_xrt = Gauge(
        "whisper_xrt",
        "Realtime factor of Whisper batch transcription",
    )
    whisper_fallback_total = Counter(
        "whisper_fallback_total",
        "Whisper GPU fallbacks to CPU",
    )
    ingest_queue_fill_ratio = Gauge(
        "ingest_queue_fill_ratio",
        "Ratio of active ingest tasks to queue size",
    )
    j_cost = autotune_cost_g
else:  # pragma: no cover - optional dependency missing
    tpl_w1_g = None
    sheaf_score_g = None
    gw_entropy = None
    autotune_cost_g = None
    bias_wasserstein_last = None
    haa_edges_total = None
    gp_jitter_restarts_total = None  # type: ignore
    prune_reverts_total = None  # type: ignore
    redis_hit_ratio = None  # type: ignore
    eigsh_timeouts_total = None  # type: ignore
    eigsh_last_duration = None  # type: ignore
    redis_evictions_l2_total = None  # type: ignore
    lmdb_evictions_total = None  # type: ignore
    lmdb_eviction_last_ts = None  # type: ignore
    lang_mismatch_total = None  # type: ignore
    ann_backend = None  # type: ignore
    whisper_xrt = None  # type: ignore
    whisper_fallback_total = None  # type: ignore
    ingest_queue_fill_ratio = None  # type: ignore
    j_cost = None


_METRICS = {
    "sigma_db": None,
    "recall10": None,
    "sheaf_score": sheaf_score_g,
    "gw_entropy": gw_entropy,
    "tpl_w1": tpl_w1_g,
    "j_cost": j_cost,
    "autotune_cost": autotune_cost_g,
    "bias_wasserstein_last": bias_wasserstein_last,
    "haa_edges_total": haa_edges_total,
    "gp_jitter_restarts_total": gp_jitter_restarts_total,
    "prune_reverts_total": prune_reverts_total,
    "redis_hit_ratio": redis_hit_ratio,
    "eigsh_timeouts_total": eigsh_timeouts_total,
    "eigsh_last_duration": eigsh_last_duration,
    "redis_evictions_l2_total": redis_evictions_l2_total,
    "ann_backend": ann_backend,
    "lang_mismatch_total": lang_mismatch_total,
    "whisper_xrt": whisper_xrt,
    "whisper_fallback_total": whisper_fallback_total,
    "ingest_queue_fill_ratio": ingest_queue_fill_ratio,
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
