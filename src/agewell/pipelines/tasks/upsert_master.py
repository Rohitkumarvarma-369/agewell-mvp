"""Update master.parquet with imaging derivative URIs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from agewell.config import repo_root
from agewell.data.master import _record_version, _repo_path, _sorted_list


def upsert_imaging_uris(
    *,
    master_path: str | Path,
    subject_id: str,
    visit_idx: int,
    updates: dict[str, Any],
) -> None:
    """Update one subject-visit row in master.parquet with imaging outputs."""
    path = _repo_path(str(master_path))
    df = pd.read_parquet(path)
    mask = df["subject_id"].eq(subject_id) & df["visit_idx"].eq(visit_idx)
    if int(mask.sum()) != 1:
        raise ValueError(f"expected exactly one master row for {subject_id}/{visit_idx}")

    idx = df.index[mask][0]
    for column, value in updates.items():
        if column == "qc_reasons":
            prior = set(_sorted_list(df.at[idx, "qc_reasons"]))
            prior.update(str(reason) for reason in value)
            df.at[idx, column] = sorted(prior)
        else:
            df.at[idx, column] = value
    modalities = set(_sorted_list(df.at[idx, "available_modalities"]))
    if "mri_brainiac_uri" in updates:
        if updates["mri_brainiac_uri"] is None:
            modalities.discard("mri_raw")
        else:
            modalities.add("mri_raw")
    df.at[idx, "available_modalities"] = sorted(modalities)
    df.at[idx, "record_version"] = _record_version(df.loc[idx])
    df.to_parquet(path, index=False)
    _update_quality_report(path)


def _update_quality_report(master_path: Path) -> None:
    report_path = master_path.parent / "quality_report.json"
    if not report_path.exists():
        return
    report = json.loads(report_path.read_text(encoding="utf-8"))
    df = pd.read_parquet(master_path, columns=["mri_brainiac_uri"])
    report["mri_brainiac_uri_count"] = int(df["mri_brainiac_uri"].notna().sum())
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def default_master_path() -> Path:
    """Return the canonical master parquet path."""
    return repo_root() / "data" / "master.parquet"
