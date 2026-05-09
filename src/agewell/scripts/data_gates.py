"""Phase 1 data sanity gates."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from agewell.config import load_cfg, repo_root


def main() -> None:
    """CLI entrypoint."""
    cfg = load_cfg()
    master_path = _repo_path(str(cfg.data.master_path))
    splits_dir = _repo_path(str(cfg.data.splits_dir))
    report_path = master_path.parent / "quality_report.json"
    profile_path = repo_root() / "docs" / "data_profile.html"

    df = pd.read_parquet(master_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    train = pd.read_parquet(splits_dir / "train.parquet")
    calib = pd.read_parquet(splits_dir / "calib.parquet")
    test = pd.read_parquet(splits_dir / "test.parquet")

    assert 5900 <= len(df) <= 6200, f"unexpected master row count: {len(df)}"
    assert 1300 <= int(df["is_rich"].sum()) <= 1600, "unexpected rich-row count"
    assert df[["subject_id", "visit_idx"]].duplicated().sum() == 0
    assert df["diagnosis"].notna().all()
    assert df["available_modalities"].apply(len).gt(0).all()
    assert int(df["mri_t1_uri"].notna().sum()) >= 1800
    assert sum(1 for count in report["modality_pattern_counts"].values() if count >= 100) >= 5
    assert profile_path.exists()
    assert report_path.exists()

    split_subjects = {
        "train": set(train["subject_id"]),
        "calib": set(calib["subject_id"]),
        "test": set(test["subject_id"]),
    }
    assert split_subjects["train"].isdisjoint(split_subjects["calib"])
    assert split_subjects["train"].isdisjoint(split_subjects["test"])
    assert split_subjects["calib"].isdisjoint(split_subjects["test"])

    total = len(df)
    ratios = {
        name: len(split) / total
        for name, split in {"train": train, "calib": calib, "test": test}.items()
    }
    assert 0.65 <= ratios["train"] <= 0.75, ratios
    assert 0.10 <= ratios["calib"] <= 0.20, ratios
    assert 0.10 <= ratios["test"] <= 0.20, ratios
    print("Phase 1 data gates passed")


def _repo_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else repo_root() / candidate


if __name__ == "__main__":
    main()
