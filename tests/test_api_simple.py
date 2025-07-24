import pytest
import types
import json
from datacreek import api
from datacreek.api import ExportFormat
from fastapi import HTTPException

class DummyRedis:
    def __init__(self):
        self.sets = {}
        self.hashes = {}
        self.values = {}
        self.lists = {}
    def smembers(self, key):
        return self.sets.get(key, set())
    def hgetall(self, key):
        return self.hashes.get(key, {})
    def hget(self, key, field):
        return self.hashes.get(key, {}).get(field)
    def get(self, key):
        return self.values.get(key)
    def lrange(self, key, start, end):
        lst = self.lists.get(key, [])
        if end == -1:
            end = len(lst) - 1
        return lst[start:end+1]

class DummyBuilder:
    def __init__(self, owner_id=1):
        self.owner_id = owner_id
        self.stage = "ready"
        self.redis_client = None
        self.versions = [{"v":1},{"v":2}]
    @classmethod
    def from_redis(cls, client, key, driver):
        inst = cls()
        inst.redis_client = client
        return inst
    def delete_version(self, index):
        self.versions.pop(index-1)
    def restore_version(self, index):
        self.versions.append(self.versions[index-1])
    def explain_node(self, node_id, hops=3):
        return {"nodes":[],"edges":[],"attention":{}}

class DummyUser:
    def __init__(self, uid=1):
        self.id = uid


def test_list_user_datasets(monkeypatch):
    redis = DummyRedis()
    redis.sets["user:1:datasets"] = {b"foo", "bar"}
    monkeypatch.setattr(api, "get_redis_client", lambda: redis)
    assert api.list_user_datasets(DummyUser()) == ["bar", "foo"]

def test_list_user_datasets_details(monkeypatch):
    redis = DummyRedis()
    redis.sets["user:1:datasets"] = {"ds"}
    redis.hashes["dataset:ds:progress"] = {"pct": "50"}
    monkeypatch.setattr(api, "get_redis_client", lambda: redis)
    monkeypatch.setattr(api, "get_neo4j_driver", lambda: None)
    monkeypatch.setattr(api, "DatasetBuilder", DummyBuilder)
    res = api.list_user_datasets_details(DummyUser())
    assert res == [{"name": "ds", "stage": "ready", "progress": {"pct": 50}}]

def test_load_dataset_owner(monkeypatch):
    redis = DummyRedis()
    monkeypatch.setattr(api, "get_redis_client", lambda: redis)
    monkeypatch.setattr(api, "get_neo4j_driver", lambda: None)
    monkeypatch.setattr(api, "DatasetBuilder", DummyBuilder)
    ds = api._load_dataset("name", DummyUser())
    assert ds.redis_client is redis

    # owner mismatch
    class Other(DummyBuilder):
        def __init__(self):
            super().__init__(owner_id=2)
    monkeypatch.setattr(api, "DatasetBuilder", Other)
    with pytest.raises(HTTPException):
        api._load_dataset("name", DummyUser())

def test_dataset_version_routes(monkeypatch):
    dummy = DummyBuilder()
    redis = DummyRedis()
    dummy.redis_client = redis
    monkeypatch.setattr(api, "_load_dataset", lambda name, u: dummy)
    assert api.dataset_versions("ds", DummyUser()) == [
        {"index": 1, **dummy.versions[0]},
        {"index": 2, **dummy.versions[1]},
    ]
    assert api.dataset_version_item("ds", 1, DummyUser()) == dummy.versions[0]
    with pytest.raises(HTTPException):
        api.dataset_version_item("ds", 3, DummyUser())
    api.delete_dataset_version_item("ds", 1, DummyUser())
    assert len(dummy.versions) == 1
    api.restore_dataset_version_item("ds", 1, DummyUser())
    assert len(dummy.versions) == 2

def test_dataset_export_result(monkeypatch):
    redis = DummyRedis()
    redis.hashes["dataset:x:progress"] = {"export": json.dumps({"key": "k"})}
    redis.values["k"] = "{}"
    dummy = DummyBuilder()
    dummy.redis_client = redis
    monkeypatch.setattr(api, "_load_dataset", lambda n, u: dummy)
    resp = api.dataset_export_result("x", ExportFormat.JSONL, DummyUser())
    assert resp.media_type == "application/json"
    assert resp.body == b"{}"
    with pytest.raises(HTTPException):
        redis.values.clear()
        api.dataset_export_result("x", ExportFormat.JSONL, DummyUser())

def test_dataset_progress(monkeypatch):
    redis = DummyRedis()
    redis.hashes["dataset:x:progress"] = {"a": "1"}
    dummy = DummyBuilder()
    dummy.redis_client = redis
    monkeypatch.setattr(api, "_load_dataset", lambda n, u: dummy)
    assert api.dataset_progress("x", DummyUser()) == {"a": 1}

def test_dataset_progress_history(monkeypatch):
    redis = DummyRedis()
    redis.lists["dataset:x:progress:history"] = ["{}", b"{}"]
    dummy = DummyBuilder()
    dummy.redis_client = redis
    monkeypatch.setattr(api, "_load_dataset", lambda n, u: dummy)
    assert len(api.dataset_progress_history("x", DummyUser())) == 2

def test_dp_sample():
    assert api.dp_sample(DummyUser()) == {"ok": True}

@pytest.mark.asyncio
async def test_async_task_endpoints(monkeypatch):
    class DummyTask:
        def apply_async(self, *a, **k):
            return types.SimpleNamespace(id="42")
    monkeypatch.setattr(api, "ingest_task", DummyTask())
    payload = types.SimpleNamespace(path="p", high_res=False, ocr=False, use_unstructured=False, extract_entities=False, extract_facts=False)
    assert await api.ingest_async(payload, DummyUser()) == {"task_id": "42"}

    monkeypatch.setattr(api, "generate_task", DummyTask())
    params = types.SimpleNamespace(src_id=1, content_type="t", num_pairs=1, provider="p", profile="pr", model="m", api_base="b", generation=None, prompts=None)
    assert await api.generate_async(params, DummyUser(), None) == {"task_id": "42"}

    monkeypatch.setattr(api, "curate_task", DummyTask())
    cparams = types.SimpleNamespace(ds_id=1, threshold=0.5)
    assert await api.curate_async(cparams, DummyUser()) == {"task_id": "42"}

    monkeypatch.setattr(api, "save_task", DummyTask())
    sparams = types.SimpleNamespace(ds_id=1, fmt=types.SimpleNamespace(value="jsonl"))
    assert await api.save_async(sparams, DummyUser()) == {"task_id": "42"}

    monkeypatch.setattr(api, "celery_app", types.SimpleNamespace(AsyncResult=lambda tid: types.SimpleNamespace(state="FAILURE", result="err")))
    assert await api.get_task("x") == {"status": "failed", "error": "err"}
