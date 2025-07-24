import importlib
from types import SimpleNamespace
import pytest

pybreaker = pytest.importorskip("pybreaker")

import datacreek.utils.neo4j_breaker as nb


def test_reconfigure_and_listener(monkeypatch):
    calls = []
    monkeypatch.setattr(nb.monitoring, "update_metric", lambda n, v: calls.append((n, v)))
    importlib.reload(nb)
    nb.reconfigure(fail_max=2, timeout=1)
    assert nb.neo4j_breaker.fail_max == 2
    assert nb.neo4j_breaker.reset_timeout == 1
    # listener open/close
    listener = nb.neo4j_breaker.listeners[0]
    listener.state_change(nb.neo4j_breaker, None, SimpleNamespace(name=pybreaker.STATE_OPEN))
    listener.state_change(nb.neo4j_breaker, None, SimpleNamespace(name=pybreaker.STATE_CLOSED))
    assert ("breaker_state", 0) in calls
    assert ("breaker_state", 1) in calls
