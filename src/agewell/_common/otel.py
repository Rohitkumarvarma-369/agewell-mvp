"""OpenTelemetry setup helpers."""

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor, SpanExporter


def build_tracer_provider(service_name: str, exporter: SpanExporter) -> TracerProvider:
    """Build a tracer provider for a named AgeWell service."""
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider


def configure_console_tracing(service_name: str = "agewell") -> TracerProvider:
    """Configure a console span exporter and return the active provider."""
    current_provider = trace.get_tracer_provider()
    if isinstance(current_provider, TracerProvider):
        return current_provider

    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)
    return provider
