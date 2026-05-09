"""Phase 0 stub FastAPI service."""

from fastapi import FastAPI
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.trace import TracerProvider

from agewell._common.otel import configure_console_tracing
from agewell.healthcheck import router as health_router


def create_app(tracer_provider: TracerProvider | None = None) -> FastAPI:
    """Create the Phase 0 inference stub app."""
    provider = tracer_provider or configure_console_tracing("agewell-inference")
    app = FastAPI(title="AgeWell-IN Inference Stub", version="0.1.0")
    FastAPIInstrumentor.instrument_app(app, tracer_provider=provider)
    app.include_router(health_router)

    @app.get("/")
    async def root() -> dict[str, str]:
        """Return a small service descriptor."""
        return {"service": "agewell-inference-stub", "status": "ok"}

    return app


app = create_app()
