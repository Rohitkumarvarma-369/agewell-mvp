"""Shared health-check routes for AgeWell FastAPI services."""

from datetime import UTC, datetime

from fastapi import APIRouter

from agewell import __version__

router = APIRouter()


@router.get("/health")
async def health() -> dict[str, str]:
    """Return the service liveness payload."""
    return {
        "status": "ok",
        "version": __version__,
        "ts": datetime.now(UTC).isoformat(),
    }
