"""Phase 0 stub FastAPI service."""

from fastapi import FastAPI

from agewell.healthcheck import router as health_router

app = FastAPI(title="AgeWell-IN Inference Stub", version="0.1.0")
app.include_router(health_router)


@app.get("/")
async def root() -> dict[str, str]:
    """Return a small service descriptor."""
    return {"service": "agewell-inference-stub", "status": "ok"}
