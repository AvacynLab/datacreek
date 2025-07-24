import sys

sys.path.insert(0, ".")

from datacreek.utils.kafka_queue import enqueue_ingest, get_producer


def test_get_producer_stub():
    prod = get_producer()
    assert hasattr(prod, "send")
    assert prod.sent == []


def test_enqueue_ingest_default():
    prod = get_producer()
    result = enqueue_ingest("name", "/tmp/x", user_id=1)
    assert prod.sent[0][0] == "datacreek_ingest"
    payload = prod.sent[0][1]
    assert payload["name"] == "name"
    assert payload["path"] == "/tmp/x"
    assert payload["user_id"] == 1
    assert hasattr(result, "get")
    assert result.get() is None
