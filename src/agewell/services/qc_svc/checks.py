"""Quality checks for BrainIAC imaging derivatives."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import nibabel as nib
import numpy as np


@dataclass(frozen=True)
class QCResult:
    """QC status and reason list."""

    status: str
    reasons: tuple[str, ...]


def run_qc(
    *,
    mask_uri: str | None,
    brain_volume_ml: float | None,
    registration_mi: float | None,
    registration_required: bool,
    features_uri: str,
    normalized_mean: float,
    normalized_std: float,
) -> QCResult:
    """Run Phase 2 BrainIAC QC checks."""
    reasons: list[str] = []
    failures: list[str] = []

    if mask_uri:
        voxel_count = _mask_voxel_count(Path(mask_uri))
        if voxel_count <= 100_000:
            failures.append("mask_too_small")
        if brain_volume_ml is None or not 800.0 <= brain_volume_ml <= 1900.0:
            failures.append("brain_volume_out_of_range")
    elif brain_volume_ml is not None:
        reasons.append("brain_volume_without_mask")

    if registration_required and (registration_mi is None or registration_mi <= 0.5):
        failures.append("registration_mi_low")

    embedding = np.load(features_uri)
    norm = float(np.linalg.norm(embedding))
    if not np.isfinite(embedding).all() or norm == 0.0:
        failures.append("brainiac_embedding_invalid")
    elif not 20.0 <= norm <= 500.0:
        failures.append("brainiac_embedding_norm_out_of_range")

    if not np.isfinite([normalized_mean, normalized_std]).all():
        failures.append("normalized_intensity_invalid")
    elif abs(normalized_mean) > 0.2 or not 0.35 <= normalized_std <= 1.5:
        failures.append("normalized_intensity_distribution_out_of_range")

    status = "fail" if failures else "pass"
    return QCResult(status=status, reasons=tuple(reasons + failures))


def _mask_voxel_count(mask_path: Path) -> int:
    mask = cast(Any, nib.load(str(mask_path))).get_fdata()
    return int((mask > 0).sum())
