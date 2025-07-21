from fastapi import FastAPI

from datacreek.telemetry import init_tracing


def test_init_tracing():
    app = FastAPI()
    init_tracing(app)
    # Should complete without error; tracer provider should be set
    from opentelemetry import trace

    assert isinstance(trace.get_tracer_provider(), trace.TracerProvider)
