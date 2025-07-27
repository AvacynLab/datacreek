import json

import pytest

pytest.importorskip("fakeredis")
pytest.importorskip("pydantic")
import fakeredis
from pydantic import ValidationError

pytest.importorskip("torch")

from datacreek.analysis.monitoring import ingest_validation_fail_total
from datacreek.core.dataset import DatasetBuilder, DatasetType
from datacreek.tasks import dataset_ingest_task, get_redis_client


def setup_fake(monkeypatch):
    client = fakeredis.FakeStrictRedis()
    monkeypatch.setattr("datacreek.tasks.get_redis_client", lambda: client)
    return client


def test_dataset_ingest_validation_metric(tmp_path, monkeypatch):
    client = setup_fake(monkeypatch)
    ds = DatasetBuilder(DatasetType.TEXT, name="demo")
    ds.redis_client = client
    ds.to_redis(client, "dataset:demo")
    client.sadd("datasets", "demo")
    monkeypatch.setattr("datacreek.tasks.get_neo4j_driver", lambda: None)
    start = 0.0
    if ingest_validation_fail_total is not None:
        start = ingest_validation_fail_total._value.get()
    with pytest.raises(ValidationError):
        dataset_ingest_task.delay("demo", "", high_res=True).get()
    if ingest_validation_fail_total is not None:
        assert ingest_validation_fail_total._value.get() == start + 1
