import pytest
from fastapi import FastAPI

from datacreek.telemetry import init_tracing


def test_init_tracing():
    app = FastAPI()
    init_tracing(app)
    try:
        from opentelemetry import trace  # type: ignore
    except Exception:
        pytest.skip("opentelemetry not installed")

    assert isinstance(trace.get_tracer_provider(), trace.TracerProvider)
