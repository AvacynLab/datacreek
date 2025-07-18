import os

import pybreaker

__all__ = ["neo4j_breaker", "CircuitBreakerError", "reconfigure"]

CircuitBreakerError = pybreaker.CircuitBreakerError

_fail_max = int(os.getenv("NEO4J_CB_FAIL_MAX", "5"))
_reset_timeout = int(os.getenv("NEO4J_CB_TIMEOUT", "30"))

neo4j_breaker = pybreaker.CircuitBreaker(
    fail_max=_fail_max,
    reset_timeout=_reset_timeout,
    name="neo4j",
)


def reconfigure(fail_max: int | None = None, timeout: int | None = None) -> None:
    """Adjust breaker parameters (testing)."""
    if fail_max is not None:
        neo4j_breaker.fail_max = fail_max
    if timeout is not None:
        neo4j_breaker.reset_timeout = timeout
    neo4j_breaker.close()
