"""Tests for BrainIAC preprocessing branch and cache behavior."""

import nibabel as nib
import numpy as np

from agewell.services._common.cache import imaging_cache_stem
from agewell.services.brainiac_preprocess_svc.preprocess import (
    ADNI_PASSTHROUGH_VERSION,
    FULL_PREPROCESS_VERSION,
    IXI_HDBET_VERSION,
    _write_nifti_gz,
    plan_for_cohort,
    preprocess_scan,
)


def test_preprocess_branch_rules_are_cohort_specific() -> None:
    """ADNI_NIFTI is pass-through, IXI is HD-BET only, future raw cohorts are full."""
    adni = plan_for_cohort("ADNI_NIFTI")
    assert adni.version == ADNI_PASSTHROUGH_VERSION
    assert adni.passthrough
    assert not adni.run_hdbet

    ixi = plan_for_cohort("IXI")
    assert ixi.version == IXI_HDBET_VERSION
    assert ixi.run_hdbet
    assert not ixi.run_n4
    assert not ixi.run_registration

    future = plan_for_cohort("CBR")
    assert future.version == FULL_PREPROCESS_VERSION
    assert future.run_n4
    assert future.run_registration
    assert future.run_hdbet


def test_adni_nifti_preprocess_is_passthrough() -> None:
    """Already preprocessed ADNI_NIFTI rows should not create heavy derivatives."""
    out = preprocess_scan(cohort="ADNI_NIFTI", nifti_uri="/tmp/example.nii")
    assert out.preprocessed_uri == "/tmp/example.nii"
    assert out.preprocess_version == ADNI_PASSTHROUGH_VERSION
    assert out.cache_hit
    assert out.mask_uri is None
    assert out.registered_uri is None


def test_cache_key_includes_preprocess_version() -> None:
    """Changing preprocessing recipe invalidates the BrainIAC feature cache."""
    source = "/data/source/sub-001_T1w.nii.gz"
    assert imaging_cache_stem(source, ADNI_PASSTHROUGH_VERSION) != imaging_cache_stem(
        source, FULL_PREPROCESS_VERSION
    )


def test_write_nifti_gz_recompresses_plain_nii(tmp_path) -> None:
    """The HD-BET staging path should not copy plain NIfTI bytes under a .gz suffix."""
    source = tmp_path / "scan.nii"
    dest = tmp_path / "scan_0000.nii.gz"
    image = nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.float32), np.eye(4))
    nib.save(image, source)

    _write_nifti_gz(source, dest)

    assert dest.read_bytes()[:2] == b"\x1f\x8b"
    assert nib.load(str(dest)).shape == (2, 2, 2)
