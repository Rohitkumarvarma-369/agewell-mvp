"""Prefect imaging flow for BrainIAC derivative generation."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any

from prefect import flow, get_run_logger, task
from prefect.tasks import task_input_hash

from agewell.config import load_cfg
from agewell.pipelines.tasks.dispatch import ServiceClient
from agewell.pipelines.tasks.upsert_master import default_master_path, upsert_imaging_uris
from agewell.services.brainiac_preprocess_svc.preprocess import (
    FULL_PREPROCESS_VERSION,
    plan_for_cohort,
)


@task(retries=3, retry_delay_seconds=10, cache_key_fn=task_input_hash)
def step_dcm2niix(
    subject_id: str,
    visit_idx: int,
    dicom_dir: str,
    service_url: str,
) -> dict[str, Any]:
    """Convert DICOM to NIfTI."""
    return ServiceClient(service_url).post(
        "/convert",
        {"subject_id": subject_id, "visit_idx": visit_idx, "dicom_dir": dicom_dir},
    )


@task(
    retries=2,
    retry_delay_seconds=30,
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(days=7),
)
def step_preprocess(
    subject_id: str,
    visit_idx: int,
    cohort: str,
    nifti_uri: str,
    service_url: str,
) -> dict[str, Any]:
    """Run cohort-specific BrainIAC preprocessing."""
    return ServiceClient(service_url, timeout=3600.0).post(
        "/preprocess",
        {
            "subject_id": subject_id,
            "visit_idx": visit_idx,
            "cohort": cohort,
            "nifti_uri": nifti_uri,
        },
    )


@task(retries=2, retry_delay_seconds=10, cache_key_fn=task_input_hash)
def step_brainiac(
    subject_id: str,
    visit_idx: int,
    source_nifti_uri: str,
    preprocess: dict[str, Any],
    service_url: str,
) -> dict[str, Any]:
    """Run BrainIAC feature extraction."""
    return ServiceClient(service_url, timeout=900.0).post(
        "/encode",
        {
            "subject_id": subject_id,
            "visit_idx": visit_idx,
            "preprocessed_uri": preprocess["preprocessed_uri"],
            "preprocess_version": preprocess["preprocess_version"],
            "source_nifti_uri": source_nifti_uri,
        },
    )


@task
def step_qc(
    cohort: str,
    preprocess: dict[str, Any],
    brainiac: dict[str, Any],
    service_url: str,
) -> dict[str, Any]:
    """Run imaging QC checks."""
    registration_required = plan_for_cohort(cohort).version == FULL_PREPROCESS_VERSION
    return ServiceClient(service_url, timeout=120.0).post(
        "/qc",
        {
            "mask_uri": preprocess.get("mask_uri"),
            "brain_volume_ml": preprocess.get("brain_volume_ml"),
            "registration_mi": preprocess.get("registration_mi"),
            "registration_required": registration_required,
            "features_uri": brainiac["features_uri"],
            "normalized_mean": brainiac["normalized_mean"],
            "normalized_std": brainiac["normalized_std"],
        },
    )


@flow(name="imaging_flow", log_prints=True)
def imaging_flow(
    *,
    subject_id: str,
    cohort: str,
    visit_idx: int = 0,
    dicom_dir: str | None = None,
    nifti_uri: str | None = None,
    master_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run one subject-visit through the Phase 2 imaging pipeline."""
    cfg = load_cfg()
    log = get_run_logger()
    urls = cfg.pipeline.services

    if dicom_dir and not nifti_uri:
        converted = step_dcm2niix(
            subject_id,
            visit_idx,
            dicom_dir,
            str(urls.dcm2niix),
        )
        nifti_uri = str(converted["nifti_uri"])
    if not nifti_uri:
        raise ValueError("nifti_uri or dicom_dir is required")

    pre = step_preprocess(subject_id, visit_idx, cohort, nifti_uri, str(urls.brainiac_preprocess))
    brain = step_brainiac(subject_id, visit_idx, nifti_uri, pre, str(urls.brainiac))
    qc = step_qc(cohort, pre, brain, str(urls.qc))
    updates = {
        "mri_stripped_uri": pre["preprocessed_uri"],
        "mri_brainiac_uri": brain["features_uri"],
        "qc_status": qc["qc_status"],
        "qc_reasons": qc["qc_reasons"],
    }
    upsert_imaging_uris(
        master_path=master_path or default_master_path(),
        subject_id=subject_id,
        visit_idx=visit_idx,
        updates=updates,
    )
    log.info("imaging_flow done: %s/%s", subject_id, visit_idx)
    return {"preprocess": pre, "brainiac": brain, "qc": qc}
