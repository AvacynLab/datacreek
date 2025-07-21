"""OpenTelemetry configuration utilities."""

from __future__ import annotations

import os
from typing import Optional

from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter as SpanExporter,
)
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

try:
    from opentelemetry.instrumentation.aio_pika import AioPikaInstrumentor
except Exception:  # pragma: no cover - optional dependency
    AioPikaInstrumentor = None


def init_tracing(
    app: FastAPI,
    service_name: str = "datacreek",
    endpoint: Optional[str] = None,
) -> None:
    """Initialize tracing for FastAPI and optional AioPika instrumentation."""

    endpoint = endpoint or os.getenv(
        "OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318"
    )
    resource = Resource.create({SERVICE_NAME: service_name})
    provider = TracerProvider(resource=resource)
    processor = BatchSpanProcessor(SpanExporter(endpoint=endpoint))
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)

    FastAPIInstrumentor().instrument_app(app)
    if AioPikaInstrumentor is not None:
        AioPikaInstrumentor().instrument()
