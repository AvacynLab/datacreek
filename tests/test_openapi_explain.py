import os
import sys

from fastapi.testclient import TestClient

os.environ.setdefault("DATABASE_URL", "sqlite:///test_openapi.db")
os.environ.setdefault("DATACREEK_REQUIRE_PERSISTENCE", "0")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datacreek.api import app


def test_explain_route_in_openapi():
    client = TestClient(app)
    schema = client.get("/openapi.json").json()
    assert "/explain/{node}" in schema["paths"]
