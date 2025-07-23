import types

import pytest

from datacreek.utils import kafka_queue


def test_get_producer_config(monkeypatch):
    created = {}

    class FakeProducer:
        def __init__(self, **kwargs):
            created.update(kwargs)

        def send(self, topic, value):
            return types.SimpleNamespace()

    monkeypatch.setattr(kafka_queue, "KafkaProducer", FakeProducer)
    kafka_queue._PRODUCER = None
    producer = kafka_queue.get_producer()
    assert producer is not None
    assert created["acks"] == "all"
    assert created["compression_type"] == "lz4"


def test_enqueue_ingest(monkeypatch):
    sent = {}

    class Stub:
        def send(self, topic, value):
            sent["topic"] = topic
            sent["value"] = value
            return types.SimpleNamespace(get=lambda timeout=None: None)

    monkeypatch.setattr(kafka_queue, "get_producer", lambda: Stub())
    res = kafka_queue.enqueue_ingest("demo", "/tmp/x", user_id=1, extra=True)
    assert sent["topic"] == "datacreek_ingest"
    assert sent["value"]["extra"]
    assert hasattr(res, "get")
