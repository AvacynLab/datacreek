import os

os.environ.setdefault("DATACREEK_REQUIRE_PERSISTENCE", "0")


import pytest


@pytest.fixture(autouse=True)
def _no_neo4j(monkeypatch):
    """Disable Neo4j access during tests."""
    try:
        monkeypatch.setattr("datacreek.api.get_neo4j_driver", lambda: None)
    except Exception:
        # API may require optional deps like fastapi
        pass
    try:
        monkeypatch.setattr("datacreek.core.dataset.InvariantPolicy.loops", 0)
    except Exception:
        # dataset module may be missing heavy deps
        pass
    yield
