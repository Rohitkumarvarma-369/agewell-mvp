"""Tests for Phase 2 BrainIAC imaging QC checks."""

import numpy as np

from agewell.services.qc_svc.checks import run_qc


def test_qc_accepts_valid_brainiac_embedding_without_mask(tmp_path) -> None:
    """ADNI pass-through rows can pass QC without an HD-BET mask."""
    features = np.ones(768, dtype=np.float32) * 4.0
    path = tmp_path / "features.npy"
    np.save(path, features)

    result = run_qc(
        mask_uri=None,
        brain_volume_ml=None,
        registration_mi=None,
        registration_required=False,
        features_uri=str(path),
        normalized_mean=0.0,
        normalized_std=1.0,
    )

    assert result.status == "pass"
    assert result.reasons == ()


def test_qc_rejects_degenerate_brainiac_embedding(tmp_path) -> None:
    """Degenerate encoder output is a hard QC failure."""
    path = tmp_path / "features.npy"
    np.save(path, np.zeros(768, dtype=np.float32))

    result = run_qc(
        mask_uri=None,
        brain_volume_ml=None,
        registration_mi=None,
        registration_required=False,
        features_uri=str(path),
        normalized_mean=0.0,
        normalized_std=1.0,
    )

    assert result.status == "fail"
    assert "brainiac_embedding_invalid" in result.reasons


def test_qc_requires_registration_mi_when_registration_ran(tmp_path) -> None:
    """Registration MI is enforced only for full raw-cohort preprocessing."""
    features = np.ones(768, dtype=np.float32) * 4.0
    path = tmp_path / "features.npy"
    np.save(path, features)

    result = run_qc(
        mask_uri=None,
        brain_volume_ml=None,
        registration_mi=0.1,
        registration_required=True,
        features_uri=str(path),
        normalized_mean=0.0,
        normalized_std=1.0,
    )

    assert result.status == "fail"
    assert "registration_mi_low" in result.reasons
