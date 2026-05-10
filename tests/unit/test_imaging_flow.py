"""Tests for Phase 2 imaging-flow safety policies."""

from __future__ import annotations

import json

import pandas as pd

from agewell.pipelines.imaging_flow import build_imaging_updates
from agewell.pipelines.tasks.upsert_master import upsert_imaging_uris
from agewell.scripts import run_imaging


def test_build_imaging_updates_quarantines_qc_failed_brainiac_features() -> None:
    """QC-failed embeddings should be traceable but unavailable to Phase 5 training."""
    updates = build_imaging_updates(
        preprocess={"preprocessed_uri": "/tmp/preprocessed.nii.gz"},
        brainiac={"features_uri": "/tmp/bad.npy"},
        qc={"qc_status": "fail", "qc_reasons": ["brainiac_embedding_invalid"]},
    )

    assert updates["mri_brainiac_uri"] is None
    assert updates["mri_stripped_uri"] == "/tmp/preprocessed.nii.gz"
    assert updates["qc_status"] == "fail"
    assert updates["qc_reasons"] == ["brainiac_embedding_invalid"]


def test_upsert_imaging_uris_removes_mri_raw_when_brainiac_uri_is_quarantined(
    tmp_path,
) -> None:
    """Master updates should make QC-failed MRI raw features honestly missing."""
    master_path = tmp_path / "master.parquet"
    quality_path = tmp_path / "quality_report.json"
    df = pd.DataFrame(
        [
            {
                "subject_id": "ADNI:0001",
                "visit_idx": 0,
                "available_modalities": ["clinical_demo", "mri_raw"],
                "qc_reasons": ["old_reason"],
                "mri_brainiac_uri": "/tmp/old.npy",
                "mri_stripped_uri": "/tmp/old.nii.gz",
                "qc_status": "pass",
                "record_version": "old",
            }
        ]
    )
    df.to_parquet(master_path, index=False)
    quality_path.write_text(json.dumps({"mri_brainiac_uri_count": 1}), encoding="utf-8")

    upsert_imaging_uris(
        master_path=master_path,
        subject_id="ADNI:0001",
        visit_idx=0,
        updates={
            "mri_stripped_uri": "/tmp/preprocessed.nii.gz",
            "mri_brainiac_uri": None,
            "qc_status": "fail",
            "qc_reasons": ["brainiac_embedding_invalid"],
        },
    )

    out = pd.read_parquet(master_path).iloc[0]
    assert pd.isna(out["mri_brainiac_uri"])
    assert "mri_raw" not in out["available_modalities"]
    assert out["qc_status"] == "fail"
    assert list(out["qc_reasons"]) == ["brainiac_embedding_invalid", "old_reason"]
    assert json.loads(quality_path.read_text(encoding="utf-8"))["mri_brainiac_uri_count"] == 0


def test_run_imaging_continues_after_failed_scan(monkeypatch, tmp_path) -> None:
    """One bad scan should be logged and skipped instead of killing the back-fill."""
    calls = {"count": 0}

    def fake_imaging_flow(**kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("corrupt nifti")
        return {"subject_id": kwargs["subject_id"]}

    monkeypatch.setattr(run_imaging, "imaging_flow", fake_imaging_flow)
    progress_path = tmp_path / "progress.jsonl"
    todo = pd.DataFrame(
        [
            {
                "subject_id": "ADNI:0001",
                "visit_idx": 0,
                "cohort": "ADNI_NIFTI",
                "mri_t1_uri": "/tmp/one.nii",
            },
            {
                "subject_id": "ADNI:0002",
                "visit_idx": 0,
                "cohort": "ADNI_NIFTI",
                "mri_t1_uri": "/tmp/two.nii",
            },
        ]
    )

    summary = run_imaging._run_rows(
        todo,
        master_path=tmp_path / "master.parquet",
        progress_path=progress_path,
        dry_run=False,
        fail_fast=False,
    )

    assert summary == {"queued": 2, "dry_run": 0, "done": 1, "failed": 1}
    events = [json.loads(line) for line in progress_path.read_text(encoding="utf-8").splitlines()]
    assert [event["status"] for event in events].count("failed") == 1
    assert [event["status"] for event in events].count("done") == 1
    assert events[-1]["status"] == "summary"
