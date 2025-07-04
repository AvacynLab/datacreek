import os

os.environ.setdefault("DATACREEK_REQUIRE_PERSISTENCE", "0")


import pytest


@pytest.fixture(autouse=True)
def _no_neo4j(monkeypatch):
    """Disable Neo4j access during tests."""
    monkeypatch.setattr("datacreek.api.get_neo4j_driver", lambda: None)
    yield
