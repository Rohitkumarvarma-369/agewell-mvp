"""FastAPI wrapper for dcm2niix."""

from __future__ import annotations

import subprocess
from pathlib import Path

from pydantic import BaseModel

from agewell.services._common.api import build_app

app = build_app("dcm2niix-svc")


class ConvertRequest(BaseModel):
    """Request payload for DICOM conversion."""

    subject_id: str
    visit_idx: int = 0
    dicom_dir: str
    output_root: str = "/data/raw/manual"


class ConvertResponse(BaseModel):
    """Response payload for DICOM conversion."""

    nifti_uri: str
    json_uri: str
    n_volumes: int


@app.post("/convert", response_model=ConvertResponse)
def convert(req: ConvertRequest) -> ConvertResponse:
    """Convert one DICOM directory into BIDS-like T1w NIfTI output."""
    out = (
        Path(req.output_root)
        / f"sub-{_safe_id(req.subject_id)}"
        / f"ses-{req.visit_idx:02d}"
        / "anat"
    )
    out.mkdir(parents=True, exist_ok=True)
    subject = f"sub-{_safe_id(req.subject_id)}_ses-{req.visit_idx:02d}_T1w"
    subprocess.run(
        ["dcm2niix", "-z", "y", "-f", subject, "-o", str(out), req.dicom_dir],
        check=True,
        capture_output=True,
        text=True,
    )
    nifti = next(out.glob(f"{subject}*.nii.gz"))
    sidecar = next(out.glob(f"{subject}*.json"))
    return ConvertResponse(nifti_uri=str(nifti), json_uri=str(sidecar), n_volumes=1)


def _safe_id(value: str) -> str:
    return value.replace(":", "")
