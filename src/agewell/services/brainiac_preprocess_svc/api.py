"""FastAPI service for BrainIAC offline preprocessing."""

from __future__ import annotations

from pydantic import BaseModel, Field

from agewell.services._common.api import build_app
from agewell.services.brainiac_preprocess_svc.preprocess import preprocess_scan

app = build_app("brainiac-preprocess-svc")


class PreprocessRequest(BaseModel):
    """Request payload for preprocessing one MRI scan."""

    subject_id: str
    visit_idx: int = 0
    cohort: str
    nifti_uri: str


class PreprocessResponse(BaseModel):
    """Response payload from preprocessing one MRI scan."""

    preprocessed_uri: str
    preprocess_version: str
    registered_uri: str | None = None
    mask_uri: str | None = None
    brain_volume_ml: float | None = None
    registration_mi: float | None = None
    cache_hit: bool = False
    qc_reasons: list[str] = Field(default_factory=list)


@app.post("/preprocess", response_model=PreprocessResponse)
def preprocess(req: PreprocessRequest) -> PreprocessResponse:
    """Run cohort-specific BrainIAC preprocessing."""
    out = preprocess_scan(cohort=req.cohort, nifti_uri=req.nifti_uri)
    return PreprocessResponse(
        preprocessed_uri=out.preprocessed_uri,
        preprocess_version=out.preprocess_version,
        registered_uri=out.registered_uri,
        mask_uri=out.mask_uri,
        brain_volume_ml=out.brain_volume_ml,
        registration_mi=out.registration_mi,
        cache_hit=out.cache_hit,
        qc_reasons=list(out.qc_reasons),
    )
