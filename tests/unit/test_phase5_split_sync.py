"""Tests for Phase 5 split/master synchronization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agewell.data.split_sync import refresh_splits_from_master, verify_splits_synced


def test_refresh_splits_from_master_preserves_membership_and_updates_payload(
    tmp_path: Path,
) -> None:
    vector = tmp_path / "brainiac.npy"
    np.save(vector, np.ones(2, dtype=np.float32))
    master = pd.DataFrame(
        [
            _row("A", 0, "CN", str(vector)),
            _row("B", 0, "AD", None),
            _row("C", 0, "CN", str(vector)),
        ]
    )
    splits = tmp_path / "splits"
    splits.mkdir()
    master.to_parquet(tmp_path / "master.parquet", index=False)
    master.iloc[[0]].assign(mri_brainiac_uri=None).to_parquet(splits / "train.parquet", index=False)
    master.iloc[[1]].to_parquet(splits / "calib.parquet", index=False)
    master.iloc[[2]].to_parquet(splits / "test.parquet", index=False)

    with pytest.raises(AssertionError, match="stale"):
        verify_splits_synced(master_path=tmp_path / "master.parquet", splits_dir=splits)

    report = refresh_splits_from_master(
        master_path=tmp_path / "master.parquet",
        splits_dir=splits,
        write=True,
    )

    train = pd.read_parquet(splits / "train.parquet")
    assert report.refreshed == ["train"]
    assert train["subject_id"].tolist() == ["A"]
    assert train["mri_brainiac_uri"].tolist() == [str(vector)]
    verify_splits_synced(master_path=tmp_path / "master.parquet", splits_dir=splits)


def test_refresh_splits_rejects_duplicate_subject_visit_keys(tmp_path: Path) -> None:
    master = pd.DataFrame([_row("A", 0, "CN", None), _row("A", 0, "AD", None)])
    splits = tmp_path / "splits"
    splits.mkdir()
    master.to_parquet(tmp_path / "master.parquet", index=False)
    master.iloc[[0]].to_parquet(splits / "train.parquet", index=False)
    master.iloc[[0]].to_parquet(splits / "calib.parquet", index=False)
    master.iloc[[0]].to_parquet(splits / "test.parquet", index=False)

    with pytest.raises(AssertionError, match="duplicate"):
        refresh_splits_from_master(master_path=tmp_path / "master.parquet", splits_dir=splits)


def _row(
    subject_id: str,
    visit_idx: int,
    diagnosis: str,
    brainiac: str | None,
) -> dict[str, object]:
    return {
        "subject_id": subject_id,
        "visit_idx": visit_idx,
        "cohort": "TEST",
        "diagnosis": diagnosis,
        "mri_brainiac_uri": brainiac,
    }
