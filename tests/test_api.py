import os
import json
from pathlib import Path
from fastapi.testclient import TestClient

os.environ["DATABASE_URL"] = "sqlite:///test.db"
from datacreek.api import app
from datacreek.db import SessionLocal, User, SourceData, Dataset
from datacreek.services import hash_key

client = TestClient(app)

def teardown_module(module):
    if os.path.exists("test.db"):
        os.remove("test.db")

def test_create_user():
    res = client.post("/users", json={"username": "alice", "api_key": "key"})
    assert res.status_code == 200
    user_id = res.json()["id"]
    with SessionLocal() as db:
        user = db.get(User, user_id)
        assert user is not None
        assert user.username == "alice"
        assert user.api_key == hash_key("key")

def test_ingest(tmp_path):
    text_file = tmp_path / "example.txt"
    text_file.write_text("hello")
    res = client.post("/ingest", json={"path": str(text_file)})
    assert res.status_code == 200
    src_id = res.json()["id"]
    with SessionLocal() as db:
        src = db.get(SourceData, src_id)
        assert src is not None
        assert src.path == str(text_file)


def test_full_pipeline(monkeypatch, tmp_path):
    # stub generation to avoid real LLM calls
    def dummy_generate(path, output_dir, *args, document_text=None, **kwargs):
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

    monkeypatch.setattr("datacreek.api.generate_data", dummy_generate)
    monkeypatch.setattr("datacreek.api.curate_qa_pairs", dummy_curate)
    monkeypatch.setattr("datacreek.api.convert_format", dummy_save)

    src_file = tmp_path / "src.txt"
    src_file.write_text("hi")

    res = client.post("/ingest", json={"path": str(src_file)})
    src_id = res.json()["id"]

    res = client.post("/generate", json={"src_id": src_id})
    assert res.status_code == 200
    ds_id = res.json()["id"]

    res = client.post("/curate", json={"ds_id": ds_id})
    assert res.status_code == 200

    res = client.post("/save", json={"ds_id": ds_id, "fmt": "jsonl"})
    assert res.status_code == 200

    with SessionLocal() as db:
        ds = db.get(Dataset, ds_id)
        assert ds is not None
        assert ds.path.endswith("jsonl")
