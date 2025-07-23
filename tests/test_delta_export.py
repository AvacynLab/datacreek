import json
import os
from pathlib import Path

import pytest

pytest.importorskip("fakeredis")

import fakeredis

from datacreek.core.dataset import DatasetBuilder, DatasetType
from datacreek.models.export_format import ExportFormat
from datacreek.tasks import (
    dataset_export_task,
    get_neo4j_driver,
    get_redis_client,
    get_s3_storage,
)


def setup_ds(tmp_path, monkeypatch):
    client = fakeredis.FakeStrictRedis()
    monkeypatch.setattr("datacreek.tasks.get_redis_client", lambda: client)
    monkeypatch.setattr("datacreek.tasks.get_neo4j_driver", lambda: None)
    monkeypatch.setattr("datacreek.tasks.get_s3_storage", lambda: None)
    ds = DatasetBuilder(DatasetType.TEXT, name="demo", owner_id=42)
    ds.versions.append({"result": [{"question": "q", "answer": "a"}]})
    ds.redis_client = client
    ds.to_redis(client, "dataset:demo")
    client.sadd("datasets", "demo")
    return ds, client


def test_delta_export_lakefs(monkeypatch, tmp_path):
    ds, client = setup_ds(tmp_path, monkeypatch)
    calls = []

    def fake_run(cmd, check):
        calls.append(cmd)

    monkeypatch.setenv("DELTA_EXPORT_ROOT", str(tmp_path))
    monkeypatch.setenv("LAKEFS_REPO", "repo")
    monkeypatch.setattr("subprocess.run", fake_run)

    result = dataset_export_task.delay("demo", ExportFormat.DELTA, 42).get()

    date = os.listdir(tmp_path / "org_id=42" / "kind=text")[0]
    file_path = Path(tmp_path) / "org_id=42" / "kind=text" / date / "data.jsonl"
    assert file_path.exists()
    assert any("lakefs" in c[0] for c in calls)
    assert result["key"] == str(file_path)
