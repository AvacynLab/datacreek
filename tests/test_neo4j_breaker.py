import json
import fakeredis
import pytest
from werkzeug.security import generate_password_hash
import datacreek.db as db
from datacreek.core.dataset import DatasetBuilder, DatasetType
from datacreek.server import app as app_module
from datacreek.tasks import dataset_save_neo4j_task
from datacreek.utils.neo4j_breaker import neo4j_breaker, reconfigure

dataset_save_neo4j_task.app.conf.task_always_eager = True
db.init_db()


@pytest.fixture(autouse=True)
def create_user():
    with db.SessionLocal() as session:
        if not session.query(db.User).filter_by(username="alice").first():
            user = db.User(
                username="alice",
                api_key="key_neo",
                password_hash=generate_password_hash("pw"),
            )
            session.add(user)
            session.commit()
    yield


@pytest.fixture(autouse=True)
def reset_breaker():
    reconfigure(fail_max=1, timeout=30)
    yield
    neo4j_breaker.close()


def setup_ds(client, monkeypatch):
    ds = DatasetBuilder(DatasetType.TEXT, name="demo")
    ds.redis_client = client
    ds.add_document("d", source="s")
    ds.to_redis(client, "dataset:demo")
    app_module.DATASETS["demo"] = ds
    monkeypatch.setattr("datacreek.tasks.get_redis_client", lambda: client)
    return ds


def test_api_save_dataset_neo4j_circuit_open(monkeypatch):
    client = fakeredis.FakeStrictRedis()
    app_module.REDIS = client
    ds = setup_ds(client, monkeypatch)

    class FailDriver:
        def close(self):
            pass

    # first call fails, opening circuit
    monkeypatch.setattr("datacreek.tasks.get_neo4j_driver", lambda: FailDriver())
    monkeypatch.setattr(ds.graph.__class__, "to_neo4j", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("fail")))
    with pytest.raises(RuntimeError):
        dataset_save_neo4j_task.delay("demo").get()
    assert neo4j_breaker.current_state == "open"

    with app_module.app.test_client() as cl:
        cl.post("/api/login", json={"username": "alice", "password": "pw"})
        res = cl.post("/api/datasets/demo/save_neo4j")
        assert res.status_code == 429
    app_module.DATASETS.clear()
