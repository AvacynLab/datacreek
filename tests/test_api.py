import os
import json
import time
from pathlib import Path
from fastapi.testclient import TestClient

os.environ.setdefault("CELERY_TASK_ALWAYS_EAGER", "true")

os.environ["DATABASE_URL"] = "sqlite:///test.db"
from datacreek.api import app
from datacreek.db import SessionLocal, User, Dataset
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
    res = client.post("/users", json={"username": "alice"})
    assert res.status_code == 200
    body = res.json()
    key = body["api_key"]
    user_id = body["id"]
    with SessionLocal() as db:
        user = db.get(User, user_id)
        assert user is not None
        assert user.username == "alice"
        assert user.api_key == hash_key(key)




def _wait_task(task_id: str) -> dict:
    for _ in range(50):
        res = client.get(f"/tasks/{task_id}")
        body = res.json()
        if body["status"] == "finished":
            return body["result"]
        time.sleep(0.01)
    raise AssertionError("task did not finish")


def test_async_pipeline(monkeypatch, tmp_path):
    def dummy_generate(path, output_dir, *args, document_text=None, **kwargs):
        assert kwargs.get("config_overrides") == {"generation": {"temperature": 0.1}}
        assert kwargs.get("provider") == "api-endpoint"
        out = Path(output_dir) / "gen.json"
        with open(out, "w") as f:
            json.dump({"qa_pairs": [{"question": "q", "answer": "a"}]}, f)
        return str(out)

    def dummy_curate(input_path, output_path, *args, **kwargs):
        Path(output_path).write_text(Path(input_path).read_text())
        return output_path

    def dummy_save(input_path, output_path, fmt, cfg, storage):
        out = Path(tmp_path) / f"final.{fmt}"
        Path(out).write_text(Path(input_path).read_text())
        return str(out)

    monkeypatch.setattr("datacreek.tasks.generate_data", dummy_generate)
    monkeypatch.setattr("datacreek.tasks.curate_qa_pairs", dummy_curate)
    monkeypatch.setattr("datacreek.tasks.convert_format", dummy_save)

    user_id, key = _create_user()
    headers = {"X-API-Key": key}

    src_file = tmp_path / "src.txt"
    src_file.write_text("hi")

    res = client.post("/tasks/ingest", json={"path": str(src_file)}, headers=headers)
    task_id = res.json()["task_id"]
    result = _wait_task(task_id)
    src_id = result["id"]

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
