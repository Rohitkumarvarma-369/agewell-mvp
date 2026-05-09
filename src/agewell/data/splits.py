"""Subject-disjoint split generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

from agewell.config import repo_root
from agewell.data.master import update_quality_report_splits


def subject_disjoint_split(df: pd.DataFrame, seed: int = 42) -> dict[str, pd.DataFrame]:
    """Split rows into 70/15/15 partitions grouped by subject_id."""
    subjects = (
        df.groupby("subject_id", as_index=False)
        .agg(cohort=("cohort", "first"), diagnosis=("diagnosis", "first"))
        .assign(strat=lambda x: x["cohort"].astype(str) + "_" + x["diagnosis"].astype(str))
    )
    train_subjects, rest_subjects = train_test_split(
        subjects,
        test_size=0.30,
        random_state=seed,
        stratify=_safe_stratify(subjects["strat"]),
    )
    calib_subjects, test_subjects = train_test_split(
        rest_subjects,
        test_size=0.50,
        random_state=seed + 1,
        stratify=_safe_stratify(rest_subjects["strat"]),
    )
    subject_sets = {
        "train": set(train_subjects["subject_id"]),
        "calib": set(calib_subjects["subject_id"]),
        "test": set(test_subjects["subject_id"]),
    }
    return {
        name: df[df["subject_id"].isin(subject_ids)].reset_index(drop=True)
        for name, subject_ids in subject_sets.items()
    }


def write_splits(data_cfg: DictConfig) -> dict[str, Path]:
    """Read master parquet, write split parquets, and update quality report."""
    master_path = _repo_path(str(data_cfg.master_path))
    splits_dir = _repo_path(str(data_cfg.splits_dir))
    seed = int(data_cfg.split.seed)
    df = pd.read_parquet(master_path)
    splits = subject_disjoint_split(df, seed=seed)
    splits_dir.mkdir(parents=True, exist_ok=True)
    out_paths: dict[str, Path] = {}
    for name, split_df in splits.items():
        path = splits_dir / f"{name}.parquet"
        split_df.to_parquet(path, index=False)
        out_paths[name] = path
    update_quality_report_splits(data_cfg, split_ratios(splits))
    return out_paths


def split_ratios(splits: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Return high-level split ratios for the quality report."""
    total = sum(len(split) for split in splits.values())
    return {
        name: {
            "rows": len(split),
            "subjects": int(split["subject_id"].nunique()),
            "row_ratio": 0.0 if total == 0 else round(len(split) / total, 4),
            "cohort_counts": {
                str(key): int(value) for key, value in split["cohort"].value_counts().items()
            },
            "diagnosis_counts": {
                str(key): int(value) for key, value in split["diagnosis"].value_counts().items()
            },
        }
        for name, split in splits.items()
    }


def assert_subject_disjoint(splits: dict[str, pd.DataFrame]) -> None:
    """Raise if any split shares subjects with another split."""
    train = set(splits["train"]["subject_id"])
    calib = set(splits["calib"]["subject_id"])
    test = set(splits["test"]["subject_id"])
    assert train.isdisjoint(calib)
    assert train.isdisjoint(test)
    assert calib.isdisjoint(test)


def _safe_stratify(labels: pd.Series) -> pd.Series | None:
    counts = labels.value_counts()
    if len(counts) <= 1 or counts.min() < 2:
        return None
    return labels


def _repo_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else repo_root() / candidate
