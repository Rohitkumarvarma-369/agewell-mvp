"""Tracing checks for the Phase 0 FastAPI stub."""

from fastapi.testclient import TestClient
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from agewell.service import create_app


def test_health_route_emits_span() -> None:
    """The /health route emits a server span."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    client = TestClient(create_app(tracer_provider=provider))
    response = client.get("/health")

    assert response.status_code == 200
    provider.force_flush()
    span_names = {span.name for span in exporter.get_finished_spans()}
    assert "GET /health" in span_names
