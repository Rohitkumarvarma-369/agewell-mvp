"""Keep split parquet files synchronized with the current master parquet."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from agewell.config import repo_root

KEY_COLUMNS: tuple[str, str] = ("subject_id", "visit_idx")
SPLIT_NAMES: tuple[str, str, str] = ("train", "calib", "test")


@dataclass(frozen=True)
class SplitSyncReport:
    """Summary of split synchronization state."""

    master_rows: int
    master_brainiac_non_null: int
    split_rows: dict[str, int]
    split_brainiac_non_null: dict[str, int]
    split_brainiac_existing: dict[str, int]
    refreshed: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "master_rows": self.master_rows,
            "master_brainiac_non_null": self.master_brainiac_non_null,
            "split_rows": self.split_rows,
            "split_brainiac_non_null": self.split_brainiac_non_null,
            "split_brainiac_existing": self.split_brainiac_existing,
            "refreshed": self.refreshed,
        }


def refresh_splits_from_master(
    *,
    master_path: str | Path = "data/master.parquet",
    splits_dir: str | Path = "data/splits",
    write: bool = False,
    strict_paths: bool = True,
) -> SplitSyncReport:
    """Refresh split row payloads from master while preserving split membership/order."""
    root = repo_root()
    master_file = _repo_path(master_path, root)
    split_root = _repo_path(splits_dir, root)
    master = pd.read_parquet(master_file)
    _assert_unique_keys(master, "master")

    refreshed: list[str] = []
    split_rows: dict[str, int] = {}
    split_brainiac_non_null: dict[str, int] = {}
    split_brainiac_existing: dict[str, int] = {}

    for name in SPLIT_NAMES:
        split_path = split_root / f"{name}.parquet"
        existing = pd.read_parquet(split_path)
        _assert_unique_keys(existing, name)
        updated = _rebuild_split(existing, master, name)
        if not _frames_equal(existing, updated):
            refreshed.append(name)
            if write:
                updated.to_parquet(split_path, index=False)
        frame = updated if write or name in refreshed else existing
        split_rows[name] = len(frame)
        split_brainiac_non_null[name] = _non_null_count(frame, "mri_brainiac_uri")
        split_brainiac_existing[name] = _existing_path_count(frame, "mri_brainiac_uri")

    report = SplitSyncReport(
        master_rows=len(master),
        master_brainiac_non_null=_non_null_count(master, "mri_brainiac_uri"),
        split_rows=split_rows,
        split_brainiac_non_null=split_brainiac_non_null,
        split_brainiac_existing=split_brainiac_existing,
        refreshed=refreshed,
    )
    if strict_paths and sum(report.split_brainiac_non_null.values()) != sum(
        report.split_brainiac_existing.values()
    ):
        raise FileNotFoundError(
            "One or more split mri_brainiac_uri paths do not exist: "
            f"{report.split_brainiac_existing} existing for "
            f"{report.split_brainiac_non_null} non-null"
        )
    return report


def verify_splits_synced(
    *,
    master_path: str | Path = "data/master.parquet",
    splits_dir: str | Path = "data/splits",
    strict_paths: bool = True,
) -> SplitSyncReport:
    """Raise if split parquet files differ from the matching rows in master."""
    report = refresh_splits_from_master(
        master_path=master_path,
        splits_dir=splits_dir,
        write=False,
        strict_paths=strict_paths,
    )
    if report.refreshed:
        raise AssertionError(f"Split parquet files are stale: {report.refreshed}")
    return report


def _rebuild_split(existing: pd.DataFrame, master: pd.DataFrame, name: str) -> pd.DataFrame:
    keys = existing.loc[:, list(KEY_COLUMNS)]
    updated = keys.merge(master, on=list(KEY_COLUMNS), how="left", sort=False)
    if len(updated) != len(existing):
        raise AssertionError(f"{name} split row count changed during refresh")
    missing = updated["cohort"].isna() if "cohort" in updated else pd.Series([False])
    if bool(missing.any()):
        raise KeyError(f"{name} split contains keys missing from master")
    return updated.reset_index(drop=True)


def _assert_unique_keys(frame: pd.DataFrame, label: str) -> None:
    missing_cols = [column for column in KEY_COLUMNS if column not in frame]
    if missing_cols:
        raise KeyError(f"{label} is missing split key columns: {missing_cols}")
    duplicates = int(frame.duplicated(list(KEY_COLUMNS)).sum())
    if duplicates:
        raise AssertionError(f"{label} contains {duplicates} duplicate subject/visit keys")


def _frames_equal(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    if list(left.columns) != list(right.columns):
        return False
    return left.reset_index(drop=True).equals(right.reset_index(drop=True))


def _non_null_count(frame: pd.DataFrame, column: str) -> int:
    return int(frame[column].notna().sum()) if column in frame else 0


def _existing_path_count(frame: pd.DataFrame, column: str) -> int:
    if column not in frame:
        return 0
    return int(sum(Path(str(path)).exists() for path in frame[column].dropna()))


def _repo_path(path: str | Path, root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate
