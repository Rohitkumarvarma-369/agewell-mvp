"""FastAPI service for Phase 2 imaging QC."""

from __future__ import annotations

from pydantic import BaseModel, Field

from agewell.services._common.api import build_app
from agewell.services.qc_svc.checks import run_qc

app = build_app("qc-svc")


class QCRequest(BaseModel):
    """Request payload for imaging QC."""

    mask_uri: str | None = None
    brain_volume_ml: float | None = None
    registration_mi: float | None = None
    registration_required: bool = False
    features_uri: str
    normalized_mean: float
    normalized_std: float


class QCResponse(BaseModel):
    """Response payload from imaging QC."""

    qc_status: str
    qc_reasons: list[str] = Field(default_factory=list)


@app.post("/qc", response_model=QCResponse)
def qc(req: QCRequest) -> QCResponse:
    """Run imaging QC checks."""
    result = run_qc(
        mask_uri=req.mask_uri,
        brain_volume_ml=req.brain_volume_ml,
        registration_mi=req.registration_mi,
        registration_required=req.registration_required,
        features_uri=req.features_uri,
        normalized_mean=req.normalized_mean,
        normalized_std=req.normalized_std,
    )
    return QCResponse(qc_status=result.status, qc_reasons=list(result.reasons))
