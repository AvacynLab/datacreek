import json
import os
import time
from pathlib import Path

import fakeredis
import pytest
from fastapi.testclient import TestClient

os.environ["DATABASE_URL"] = "sqlite:///test.db"

import datacreek.server.app as app_module
import datacreek.tasks as tasks_module


@pytest.fixture(autouse=True)
def _patch_persistence(monkeypatch):
    """Use fake Redis/Neo4j and disable persistence during tests."""

    monkeypatch.setattr(
        tasks_module,
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

    monkeypatch.setattr(tasks_module, "get_neo4j_driver", lambda: _FakeDriver())
    monkeypatch.setattr(tasks_module, "persist_dataset", lambda ds: None)
    # API routes use the same helpers
    monkeypatch.setattr(
        app_module,
        "get_redis_client",
        lambda: fakeredis.FakeStrictRedis(),
    )
    monkeypatch.setattr(app_module, "get_neo4j_driver", lambda: _FakeDriver())
    monkeypatch.setattr(app_module, "persist_dataset", lambda ds: None)
    from datacreek.db import SessionLocal as DBSession

    monkeypatch.setattr(app_module, "SessionLocal", DBSession)


os.environ.setdefault("CELERY_TASK_ALWAYS_EAGER", "true")

os.environ["DATABASE_URL"] = "sqlite:///test.db"
from datacreek.api import app
from datacreek.db import Dataset, SessionLocal, SourceData, User
from datacreek.services import hash_key

client = TestClient(app)


def teardown_module(module):
    if os.path.exists("test.db"):
        os.remove("test.db")


def _create_user() -> tuple[int, str]:
    name = f"u{int(time.time()*1000)}"
    res = client.post("/users", json={"username": name})
    assert res.status_code == 200
    body = res.json()
    return body["id"], body["api_key"]


def test_create_user():
    uname = f"alice_{int(time.time()*1000)}"
    res = client.post("/users", json={"username": uname})
    assert res.status_code == 200
    body = res.json()
    key = body["api_key"]
    user_id = body["id"]
    with SessionLocal() as db:
        user = db.get(User, user_id)
        assert user is not None
        assert user.username == uname
        assert user.api_key == hash_key(key)


def _wait_task(task_id: str) -> dict:
    for _ in range(200):
        res = client.get(f"/tasks/{task_id}")
        body = res.json()
        if body["status"] == "finished":
            return body["result"]
        time.sleep(0.05)
    raise AssertionError("task did not finish")


def test_async_pipeline(monkeypatch, tmp_path):
    def dummy_generate(path, output_dir, *args, document_text=None, **kwargs):
        assert kwargs.get("config_overrides") == {"generation": {"temperature": 0.1}}
        assert kwargs.get("provider") == "api-endpoint"
        return {"qa_pairs": [{"question": "q", "answer": "a"}]}

    def dummy_curate(data, output_path=None, *args, **kwargs):
        return data

    def dummy_save(data, output_path, fmt, cfg, storage):
        return "converted"

    monkeypatch.setattr("datacreek.tasks.generate_data", dummy_generate)
    monkeypatch.setattr("datacreek.tasks.curate_qa_pairs", dummy_curate)
    monkeypatch.setattr("datacreek.tasks.convert_format", dummy_save)

    user_id, key = _create_user()
    headers = {"X-API-Key": key}

    src_file = tmp_path / "src.txt"
    src_file.write_text("Paris is the capital of France.")

    res = client.post(
        "/tasks/ingest",
        json={"path": str(src_file), "extract_entities": True, "extract_facts": True},
        headers=headers,
    )
    task_id = res.json()["task_id"]
    result = _wait_task(task_id)
    src_id = result["id"]
    with SessionLocal() as db:
        src = db.get(SourceData, src_id)
        assert src.entities and src.facts

    res = client.post(
        "/tasks/generate",
        json={
            "src_id": src_id,
            "provider": "api-endpoint",
            "generation": {"temperature": 0.1},
        },
        headers=headers,
    )
    task_id = res.json()["task_id"]
    result = _wait_task(task_id)
    ds_id = result["id"]

    res = client.post("/tasks/curate", json={"ds_id": ds_id}, headers=headers)
    task_id = res.json()["task_id"]
    _wait_task(task_id)

    res = client.post("/tasks/save", json={"ds_id": ds_id, "fmt": "jsonl"}, headers=headers)
    task_id = res.json()["task_id"]
    _wait_task(task_id)

    res = client.get("/datasets", headers=headers)
    assert len(res.json()) == 1

    # download dataset
    res = client.get(f"/datasets/{ds_id}/download", headers=headers)
    assert res.status_code == 200

    # update dataset path
    new_path = str(tmp_path / "new.json")
    res = client.patch(f"/datasets/{ds_id}", json={"path": new_path}, headers=headers)
    assert res.json()["path"] == new_path

    res = client.delete(f"/datasets/{ds_id}", headers=headers)
    assert res.json()["status"] == "deleted"

    with SessionLocal() as db:
        ds = db.get(Dataset, ds_id)
        assert ds is None
