import importlib
import os
import sys

import requests
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

import fakeredis
import pytest

import datacreek.server.app as app_module
from datacreek.core.dataset import DatasetBuilder
from datacreek.core.knowledge_graph import KnowledgeGraph
from datacreek.db import verify_password
from datacreek.pipelines import DatasetType
from datacreek.server.app import DATASETS, app


@pytest.fixture(autouse=True)
def _patch_persistence(monkeypatch):
    monkeypatch.setattr(
        app_module,
        "get_redis_client",
        lambda: fakeredis.FakeStrictRedis(),
    )

    class _FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def run(self, *a, **kw):
            return None

        def execute_write(self, fn, *args, **kwargs):
            return fn(self, *args, **kwargs)

    class _FakeDriver:
        def close(self):
            pass

        def session(self):
            return _FakeSession()

    monkeypatch.setattr(app_module, "get_neo4j_driver", lambda: _FakeDriver())

    monkeypatch.setattr(app_module, "persist_dataset", lambda ds: None)
    # Ensure server uses the test database
    monkeypatch.setattr(app_module, "SessionLocal", db.SessionLocal)


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
    ds.add_section("d", "s1", title="Intro")
    ds.add_section("d", "s2", title="Introduction")
    ds.add_section("d", "s3", title="Other")
    ds.add_chunk("d", "c1", "hello world")
    ds.add_chunk("d", "c2", "hello planet")
    ds.add_chunk("d", "c3", "other text")
    ds.graph.index.build()
    ds.link_similar_chunks(k=1)
    DATASETS["demo"] = ds

    with app.test_client() as client:
        _login(client)
        res = client.get("/api/datasets/demo/search_hybrid", query_string={"q": "hello", "k": 1})
        assert res.status_code == 200
        assert res.get_json()[0] == "c1"

        res = client.get(
            "/api/datasets/demo/search_links",
            query_string={"q": "hello", "k": 1, "hops": 1},
        )
        assert res.status_code == 200
        ids = [r["id"] for r in res.get_json()]
        assert "c1" in ids and "c2" in ids

        res = client.get(
            "/api/datasets/demo/similar_chunks",
            query_string={"cid": "c1", "k": 2},
        )
        assert res.status_code == 200
        sims = res.get_json()
        assert "c1" not in sims
        assert "c2" in sims

        res = client.get(
            "/api/datasets/demo/similar_chunks_data",
            query_string={"cid": "c1", "k": 2},
        )
        assert res.status_code == 200
        data = res.get_json()
        ids = [d["id"] for d in data]
        assert "c1" not in ids
        assert "c2" in ids

        res = client.get(
            "/api/datasets/demo/chunk_neighbors",
            query_string={"k": 1},
        )
        assert res.status_code == 200
        neighbors = res.get_json()
        assert neighbors["c1"][0] == "c2"

        res = client.get(
            "/api/datasets/demo/chunk_neighbors_data",
            query_string={"k": 1},
        )
        assert res.status_code == 200
        data = res.get_json()
        assert data["c1"][0]["id"] in {"c2", "c3"}

        res = client.get(
            "/api/datasets/demo/chunk_context",
            query_string={"cid": "c2", "before": 1, "after": 1},
        )
        assert res.status_code == 200
        assert res.get_json() == ["c1", "c2", "c3"]

        res = client.get(
            "/api/datasets/demo/similar_sections",
            query_string={"sid": "s1", "k": 2},
        )
        assert res.status_code == 200
        sims = res.get_json()
        assert "s1" not in sims
        assert "s2" in sims

        res = client.get(
            "/api/datasets/demo/similar_documents",
            query_string={"did": "d", "k": 1},
        )
        assert res.status_code == 200
        assert res.get_json() == []

        res = client.post(
            "/api/datasets/demo/section_similarity",
            query_string={"k": 1},
        )
        assert res.status_code == 200
        res = client.post(
            "/api/datasets/demo/document_similarity",
            query_string={"k": 1},
        )
        assert res.status_code == 200
        assert ("s1", "s2") in ds.graph.graph.edges
    DATASETS.clear()


def test_dataset_ops_endpoints(monkeypatch):
    ds = DatasetBuilder(DatasetType.QA, name="demo")
    ds.add_document("d1", source="s")
    ds.add_chunk("d1", "c1", "hello")
    ds.add_chunk("d1", "c2", "hello")
    ds.add_document("d2", source="bad")
    ds.add_chunk("d2", "c3", "bad text")
    ds.add_entity("e1", "Beethoven")
    ds.add_entity("e2", "Ludwig van Beethoven")
    ds.graph.graph.nodes["e1"]["birth_date"] = "2024-01-01"
    ds.graph.graph.nodes["e2"]["birth_date"] = "2023-01-01"
    ds.graph.graph.add_edge("e1", "e2", relation="parent_of")
    DATASETS["demo"] = ds

    class FakeResponse:
        def __init__(self, data):
            self._data = data

        def json(self):
            return self._data

    def fake_get(url, params=None, headers=None, timeout=10):
        if "lookup.dbpedia.org" in url:
            return FakeResponse(
                {
                    "results": [
                        {"id": "http://dbpedia.org/resource/Beethoven", "description": "desc"}
                    ]
                }
            )
        return FakeResponse({"search": [{"id": "Q1", "description": "composer"}]})

    with app.test_client() as client:
        _login(client)
        ops = [
            "consolidate",
            "communities",
            "summaries",
            "trust",
            "similarity",
            "co_mentions",
            "doc_co_mentions",
            "section_co_mentions",
            "author_org_links",
            "entity_groups",
            "entity_group_summaries",
            "deduplicate",
            "clean_chunks",
            "normalize_dates",
            "prune",
            "graph_embeddings",
            "mark_conflicts",
            "validate",
            "resolve_entities",
            "predict_links?graph=true",
            "centrality",
        ]
        for op in ops:
            if op == "prune":
                res = client.post(f"/api/datasets/demo/{op}", json={"sources": ["bad"]})
            elif op == "resolve_entities":
                res = client.post(
                    f"/api/datasets/demo/{op}",
                    json={"aliases": {"Beethoven": ["Ludwig van Beethoven"]}},
                )
            elif op == "graph_embeddings":
                res = client.post(
                    f"/api/datasets/demo/{op}",
                    json={
                        "dimensions": 8,
                        "walk_length": 4,
                        "num_walks": 5,
                        "seed": 42,
                        "workers": 1,
                    },
                )
            else:
                res = client.post(f"/api/datasets/demo/{op}")
            assert res.status_code == 200
            if op == "validate":
                assert res.get_json()["marked"] == 1
        # Prune removed the bad document and chunk
        assert "d2" not in ds.graph.graph.nodes
        assert "c3" not in ds.graph.graph.nodes

        assert len(ds.graph.graph.nodes["e1"]["embedding"]) == 8

        monkeypatch.setattr(requests, "get", fake_get)
        res = client.post("/api/datasets/demo/enrich_entity/e1")
        assert res.status_code == 200
        res = client.post("/api/datasets/demo/enrich_entity_dbpedia/e1")
        assert res.status_code == 200
        res = client.post("/api/datasets/demo/extract_entities", json={"model": None})
        assert res.status_code == 200
    DATASETS.clear()


def test_lookup_endpoints():
    ds = DatasetBuilder(DatasetType.QA, name="demo")
    ds.add_document("doc", source="s")
    ds.add_section("doc", "sec1")
    ds.add_chunk("doc", "c1", "A is B", section_id="sec1")
    ds.add_entity("A", "A")
    ds.add_entity("B", "B")
    ds.link_entity("c1", "A")
    ds.link_entity("c1", "B")
    fid = ds.graph.add_fact("A", "is", "B")
    ds.graph.graph.add_edge("c1", fid, relation="has_fact")
    DATASETS["demo"] = ds

    with app.test_client() as client:
        _login(client)
        res = client.get("/api/datasets/demo/chunk_document", query_string={"cid": "c1"})
        assert res.status_code == 200
        assert res.get_json() == "doc"

        res = client.get("/api/datasets/demo/chunk_page", query_string={"cid": "c1"})
        assert res.status_code == 200
        assert res.get_json() == 1

        res = client.get(
            "/api/datasets/demo/section_page",
            query_string={"sid": "sec1"},
        )
        assert res.status_code == 200
        assert res.get_json() == 1

        res = client.get("/api/datasets/demo/section_document", query_string={"sid": "sec1"})
        assert res.status_code == 200
        assert res.get_json() == "doc"

        res = client.get("/api/datasets/demo/chunk_entities", query_string={"cid": "c1"})
        assert set(res.get_json()) == {"A", "B"}

        res = client.get("/api/datasets/demo/chunk_facts", query_string={"cid": "c1"})
        assert res.get_json() == [fid]

        res = client.get("/api/datasets/demo/fact_sections", query_string={"fid": fid})
        assert res.get_json() == ["sec1"]

        res = client.get("/api/datasets/demo/fact_documents", query_string={"fid": fid})
        assert res.get_json() == ["doc"]

        res = client.get("/api/datasets/demo/fact_pages", query_string={"fid": fid})
        assert res.get_json() == [1]

        res = client.get("/api/datasets/demo/entity_documents", query_string={"eid": "A"})
        assert res.get_json() == ["doc"]

        res = client.get("/api/datasets/demo/entity_chunks", query_string={"eid": "A"})
        assert res.get_json() == ["c1"]

        res = client.get("/api/datasets/demo/entity_facts", query_string={"eid": "A"})
        assert res.get_json() == [fid]

        res = client.get("/api/datasets/demo/entity_pages", query_string={"eid": "A"})
        assert res.get_json() == [1]
    DATASETS.clear()


def test_conflicts_endpoint():
    ds = DatasetBuilder(DatasetType.TEXT, name="demo")
    ds.graph.add_fact("A", "likes", "B", source="s1")
    ds.graph.add_fact("A", "likes", "C", source="s2")
    DATASETS["demo"] = ds
    with app.test_client() as client:
        _login(client)
        res = client.get("/api/datasets/demo/conflicts")
        assert res.status_code == 200
        data = res.get_json()
        assert data == [["A", "likes", {"B": ["s1"], "C": ["s2"]}]]
    DATASETS.clear()
