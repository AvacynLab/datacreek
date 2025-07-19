import os

import pybreaker

from datacreek.analysis import monitoring

__all__ = ["neo4j_breaker", "CircuitBreakerError", "reconfigure"]

CircuitBreakerError = pybreaker.CircuitBreakerError


_fail_max = int(os.getenv("NEO4J_CB_FAIL_MAX", "5"))
_reset_timeout = int(os.getenv("NEO4J_CB_TIMEOUT", "30"))


class _PrometheusListener(pybreaker.CircuitBreakerListener):
    """Update Prometheus gauge whenever the circuit state changes."""

    def state_change(self, cb, old_state, new_state):
        val = 0 if new_state.name == pybreaker.STATE_CLOSED else 1
        try:  # pragma: no cover - metrics optional
            monitoring.update_metric("breaker_state", val)
        except Exception:
            pass


neo4j_breaker = pybreaker.CircuitBreaker(
    fail_max=_fail_max,
    reset_timeout=_reset_timeout,
    name="neo4j",
    listeners=[_PrometheusListener()],
)
monitoring.update_metric("breaker_state", 0)


def reconfigure(fail_max: int | None = None, timeout: int | None = None) -> None:
    """Adjust breaker parameters (testing)."""
    if fail_max is not None:
        neo4j_breaker.fail_max = fail_max
    if timeout is not None:
        neo4j_breaker.reset_timeout = timeout
    neo4j_breaker.close()
    monitoring.update_metric("breaker_state", 0)
