import importlib
import sys

import fakeredis
import pytest

from datacreek.core.dataset import DatasetBuilder
from datacreek.pipelines import DatasetType


@pytest.fixture(autouse=True)
def reload_backpressure(monkeypatch):
    """Ensure backpressure module is reloaded with test limit."""
    if "datacreek.utils.backpressure" in sys.modules:
        importlib.reload(sys.modules["datacreek.utils.backpressure"])
    bp = importlib.import_module("datacreek.utils.backpressure")
    bp.set_limit(1)
    yield bp
    bp.set_limit(10000)


def test_dataset_ingest_task_queue_full(tmp_path, monkeypatch, reload_backpressure):
    bp = reload_backpressure
    client = fakeredis.FakeStrictRedis()
    monkeypatch.setattr("datacreek.tasks.get_redis_client", lambda: client)
    monkeypatch.setattr("datacreek.tasks.get_neo4j_driver", lambda: None)
    import datacreek.tasks as tasks_mod

    tasks_mod.celery_app.conf.task_always_eager = True
    ds = DatasetBuilder(DatasetType.TEXT, name="demo")
    ds.redis_client = client
    ds.to_redis(client, "dataset:demo")
    client.sadd("datasets", "demo")
    bp.acquire_slot()
    f = tmp_path / "doc.txt"
    f.write_text("x")
    with pytest.raises(RuntimeError):
        tasks_mod.dataset_ingest_task.delay("demo", str(f)).get()
    bp.release_slot()


def test_api_ingest_returns_429(tmp_path, monkeypatch, reload_backpressure):
    bp = reload_backpressure
    client = fakeredis.FakeStrictRedis()
    monkeypatch.setattr("datacreek.tasks.get_redis_client", lambda: client)
    monkeypatch.setattr("datacreek.tasks.get_neo4j_driver", lambda: None)
    import datacreek.server.app as app_module

    app_module.REDIS = client
    import datacreek.tasks as tasks_mod

    tasks_mod.celery_app.conf.task_always_eager = True
    ds = DatasetBuilder(DatasetType.TEXT, name="demo")
    ds.redis_client = client
    ds.to_redis(client, "dataset:demo")
    app_module.DATASETS["demo"] = ds
    f = tmp_path / "doc.txt"
    f.write_text("hi")
    bp.acquire_slot()
    with app_module.app.test_client() as cl:
        cl.post("/api/login", json={"username": "alice", "password": "pw"})
        res = cl.post("/api/datasets/demo/ingest", json={"path": str(f)})
        assert res.status_code == 429
    bp.release_slot()
    app_module.DATASETS.clear()
