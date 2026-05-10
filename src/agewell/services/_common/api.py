"""Shared FastAPI application factory."""

from __future__ import annotations

from fastapi import FastAPI
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from agewell.healthcheck import router as health_router
from agewell.logging import configure_logging


def build_app(name: str) -> FastAPI:
    """Build a consistently instrumented FastAPI service."""
    configure_logging()
    app = FastAPI(title=name)
    app.include_router(health_router)
    FastAPIInstrumentor.instrument_app(app)
    return app
