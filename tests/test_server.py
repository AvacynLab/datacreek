import importlib
import os
import sys

from werkzeug.security import generate_password_hash

os.environ["DATABASE_URL"] = "sqlite:///test_server.db"
if os.path.exists("test_server.db"):
    os.remove("test_server.db")
if "datacreek.db" in sys.modules:
    importlib.reload(sys.modules["datacreek.db"])
import datacreek.db as db

db.init_db()
with db.SessionLocal() as session:
    user = db.User(
        username="alice",
        api_key="key",
        password_hash=generate_password_hash("pw"),
    )
    session.add(user)
    session.commit()

import datacreek.server.app as app_module
from datacreek.core.dataset import DatasetBuilder
from datacreek.core.knowledge_graph import KnowledgeGraph
from datacreek.db import verify_password
from datacreek.pipelines import DatasetType
from datacreek.server.app import DATASETS, app

app.config["WTF_CSRF_ENABLED"] = False


def _login(client):
    return client.post(
        "/api/login",
        json={"username": "alice", "password": "pw"},
    )


def test_register_and_login():
    with app.test_client() as client:
        res = client.post(
            "/api/register",
            json={"username": "bob", "password": "pw"},
        )
        data = res.get_json()
        assert "api_key" in data
        # Now login with new user
        res = client.post(
            "/api/login",
            json={"username": "bob", "password": "pw"},
        )
        assert res.status_code == 200


def test_login_required_redirect():
    with app.test_client() as client:
        res = client.get("/datasets")
        assert res.status_code == 401


def test_dataset_graph_route():
    ds = DatasetBuilder(DatasetType.QA, name="demo")
    ds.add_document("d1", source="s")
    ds.add_chunk("d1", "c1", "text")
    DATASETS["demo"] = ds

    with app.test_client() as client:
        _login(client)
        res = client.get("/datasets/demo/graph")
        assert res.status_code == 200
        data = res.get_json()
        assert len(data["nodes"]) == 2
        assert len(data["edges"]) == 1
    DATASETS.clear()


def test_dataset_search_route():
    ds = DatasetBuilder(DatasetType.QA, name="demo")
    ds.add_document("d1", source="s")
    ds.add_chunk("d1", "c1", "hello world")
    DATASETS["demo"] = ds

    with app.test_client() as client:
        _login(client)
        res = client.get("/datasets/demo/search", query_string={"q": "hello"})
        assert res.status_code == 200
        data = res.get_json()
        assert data == ["c1"]
    DATASETS.clear()


def test_dataset_ingest_route(tmp_path):
    ds = DatasetBuilder(DatasetType.TEXT, name="demo")
    DATASETS["demo"] = ds

    f = tmp_path / "doc.txt"
    f.write_text("hello world")

    with app.test_client() as client:
        _login(client)
        res = client.post(
            "/datasets/demo/ingest",
            data={"input_path": str(f), "doc_id": "doc1"},
            follow_redirects=True,
        )
        assert res.status_code == 200
    assert ds.search_chunks("hello") == ["doc1_chunk_0"]
    DATASETS.clear()


def test_api_dataset_ingest_resolves_path(tmp_path):
    ds = DatasetBuilder(DatasetType.TEXT, name="demo")
    DATASETS["demo"] = ds

    f = tmp_path / "doc.txt"
    f.write_text("hello world")

    cfg = {"paths": {"input": {"txt": str(tmp_path), "default": str(tmp_path)}}}
    orig = app_module.config
    app_module.config = cfg
    try:
        with app.test_client() as client:
            _login(client)
            res = client.post(
                "/api/datasets/demo/ingest",
                json={"path": "doc.txt"},
            )
            assert res.status_code == 200
    finally:
        app_module.config = orig

    assert ds.search_chunks("hello") == ["doc_chunk_0"]
    DATASETS.clear()


def test_dataset_detail_route():
    ds = DatasetBuilder(DatasetType.QA, name="demo")
    ds.add_document("d1", source="s")
    ds.add_chunk("d1", "c1", "hello")
    DATASETS["demo"] = ds

    with app.test_client() as client:
        _login(client)
        res = client.get("/api/datasets/demo")
        assert res.status_code == 200
        data = res.get_json()
        assert data["id"] == ds.id
        assert data["size"] == len("hello")
    DATASETS.clear()


def test_save_dataset_neo4j(monkeypatch):
    ds = DatasetBuilder(DatasetType.QA, name="demo")
    ds.add_document("d1", source="s")
    DATASETS["demo"] = ds

    called = {}

    def fake_to_neo4j(self, driver, clear=True):
        called["called"] = True

    monkeypatch.setattr(KnowledgeGraph, "to_neo4j", fake_to_neo4j)

    class DummyDriver:
        def close(self):
            pass

    monkeypatch.setattr(app_module, "get_neo4j_driver", lambda: DummyDriver())

    with app.test_client() as client:
        _login(client)
        res = client.post("/datasets/demo/save_neo4j")
        assert res.status_code == 302
    assert called.get("called")
    DATASETS.clear()


def test_load_dataset_neo4j(monkeypatch):
    ds = DatasetBuilder(DatasetType.QA, name="demo")
    ds.add_document("d1", source="s")
    DATASETS["demo"] = ds

    new_graph = KnowledgeGraph()
    new_graph.add_document("n", source="a")
    monkeypatch.setattr(KnowledgeGraph, "from_neo4j", staticmethod(lambda driver: new_graph))

    class DummyDriver:
        def close(self):
            pass

    monkeypatch.setattr(app_module, "get_neo4j_driver", lambda: DummyDriver())

    with app.test_client() as client:
        _login(client)
        res = client.post("/datasets/demo/load_neo4j")
        assert res.status_code == 302
    assert ds.graph is new_graph
    DATASETS.clear()


def test_api_search_endpoints():
    ds = DatasetBuilder(DatasetType.TEXT, name="demo")
    ds.add_document("d", source="s")
    ds.add_chunk("d", "c1", "hello world")
    ds.add_chunk("d", "c2", "hello planet")
    ds.graph.index.build()
    ds.link_similar_chunks(k=1)
    DATASETS["demo"] = ds

    with app.test_client() as client:
        _login(client)
        res = client.get(
            "/api/datasets/demo/search_hybrid", query_string={"q": "hello", "k": 1}
        )
        assert res.status_code == 200
        assert res.get_json()[0] == "c1"

        res = client.get(
            "/api/datasets/demo/search_links",
            query_string={"q": "hello", "k": 1, "hops": 1},
        )
        assert res.status_code == 200
        ids = [r["id"] for r in res.get_json()]
        assert "c1" in ids and "c2" in ids
    DATASETS.clear()


def test_dataset_ops_endpoints():
    ds = DatasetBuilder(DatasetType.QA, name="demo")
    ds.add_document("d1", source="s")
    ds.add_chunk("d1", "c1", "hello")
    DATASETS["demo"] = ds

    with app.test_client() as client:
        _login(client)
        for op in [
            "consolidate",
            "communities",
            "summaries",
            "trust",
            "similarity",
            "entity_groups",
            "entity_group_summaries",
        ]:
            res = client.post(f"/api/datasets/demo/{op}")
            assert res.status_code == 200
    DATASETS.clear()
