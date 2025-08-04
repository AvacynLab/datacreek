"""Tests for GPU credit Prometheus exporter, cost metrics and quota controller."""

import pytest
from prometheus_client import REGISTRY

from metrics_prometheus.gpu_billing import (
    QuotaController,
    QuotaExceededError,
    calculate_cost,
    gpu_cost_total,
    gpu_minutes_total,
)


def reset_metrics() -> None:
    """Utility to clear counter state between tests."""
    gpu_minutes_total._metrics.clear()  # type: ignore[attr-defined]
    gpu_cost_total._metrics.clear()  # type: ignore[attr-defined]


def test_consume_increments_counter_and_deducts_credit():
    reset_metrics()
    controller = QuotaController({"acme": 30}, prices={"acme": 0.5})
    remaining = controller.consume("acme", 10)
    assert remaining == 20
    minutes_value = REGISTRY.get_sample_value("gpu_minutes_total", {"tenant": "acme"})
    cost_value = REGISTRY.get_sample_value("gpu_cost_total", {"tenant": "acme"})
    assert minutes_value == 10.0
    assert cost_value == 5.0


def test_exceeding_quota_raises_without_increment():
    reset_metrics()
    controller = QuotaController({"acme": 5})
    with pytest.raises(QuotaExceededError):
        controller.consume("acme", 10)
    minutes_value = REGISTRY.get_sample_value("gpu_minutes_total", {"tenant": "acme"})
    cost_value = REGISTRY.get_sample_value("gpu_cost_total", {"tenant": "acme"})
    assert minutes_value is None
    assert cost_value is None


def test_calculate_cost_helper():
    assert calculate_cost(2.0, 3.5) == 7.0
