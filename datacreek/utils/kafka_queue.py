import json
import os
from typing import Any

try:  # optional kafka dependency
    from kafka import KafkaProducer
except Exception:  # pragma: no cover - when kafka not available
    KafkaProducer = None  # type: ignore

__all__ = ["get_producer", "enqueue_ingest"]

_PRODUCER = None


def get_producer() -> Any:
    """Return a Kafka producer configured for ingestion.

    The producer uses ``acks='all'`` and ``lz4`` compression. When the
    ``kafka`` package is missing, a lightweight stub is returned that just
    records messages in memory for tests.
    """

    global _PRODUCER
    if _PRODUCER is not None:
        return _PRODUCER
    if KafkaProducer is None:
        # very small stub used during tests when kafka-python is missing
        class _Stub:
            def __init__(self):
                self.sent = []

            def send(self, topic: str, value: bytes):
                self.sent.append((topic, value))

        _PRODUCER = _Stub()
        return _PRODUCER
    servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092").split(",")
    _PRODUCER = KafkaProducer(  # pragma: no cover - heavy dependency
        bootstrap_servers=servers,  # pragma: no cover - heavy dependency
        acks="all",  # pragma: no cover - heavy dependency
        compression_type="lz4",  # pragma: no cover - heavy dependency
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),  # pragma: no cover - heavy
    )
    return _PRODUCER


def enqueue_ingest(
    name: str, path: str, user_id: int | None = None, **kwargs: Any
) -> Any:
    """Send an ingestion request to the Kafka topic.

    The topic defaults to ``datacreek_ingest`` unless ``KAFKA_INGEST_TOPIC`` is
    set. The returned object mimics the :class:`kafka.producer.future.Future`
    interface minimally with a ``get`` method returning ``None``.
    """

    topic = os.getenv("KAFKA_INGEST_TOPIC", "datacreek_ingest")
    producer = get_producer()
    payload = {"name": name, "path": path, "user_id": user_id}
    payload.update(kwargs)
    future = producer.send(topic, payload)

    # Ensure the returned object has a ``get`` method for compatibility
    class _Result:
        def __init__(self, f: Any):
            self._f = f
            self.id = None  # Celery-style attribute

        def get(self, timeout: float | None = None) -> Any:
            if hasattr(self._f, "get"):
                return self._f.get(timeout=timeout)
            return None

    return _Result(future)
