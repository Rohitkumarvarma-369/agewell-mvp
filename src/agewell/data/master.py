"""Build and validate the Phase 1 master dataframe."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from agewell.config import repo_root
from agewell.data.label_harmonization import compute_conversion_labels
from agewell.data.registry import ADAPTERS
from agewell.data.schema import PUBLIC_RICH_MODALITIES, CanonicalRecord


def build_master_dataframe(
    data_cfg: DictConfig,
    datasets: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the harmonized master dataframe and quality metadata."""
    selected = datasets or list(ADAPTERS)
    excluded = set(OmegaConf.to_container(data_cfg.get("exclude", []), resolve=True) or [])
    records: list[CanonicalRecord] = []
    adapter_counts: dict[str, int] = {}
    skipped: dict[str, dict[str, int]] = {}

    kaggle_root = Path(str(data_cfg.kaggle_root)).expanduser()
    subdirs = OmegaConf.to_container(data_cfg.subdir, resolve=True)
    if not isinstance(subdirs, dict):
        raise TypeError("data.subdir must be a mapping")

    for name in selected:
        if name in excluded:
            continue
        adapter_cls = ADAPTERS[name]
        source_root = kaggle_root / str(subdirs[name])
        adapter = adapter_cls(source_root=source_root)
        emitted = list(adapter.iter_records())
        records.extend(emitted)
        adapter_counts[name] = len(emitted)
        skipped[name] = _explicit_skipped_counts(name, adapter.skipped_counts)

    raw_df = canonical_records_to_dataframe(records)
    merged_df, merge_stats = merge_adni_nifti(raw_df)
    final_df = finalize_master_dataframe(merged_df)
    quality = build_quality_report(final_df, adapter_counts, skipped, merge_stats)
    return final_df, quality


def canonical_records_to_dataframe(records: list[CanonicalRecord]) -> pd.DataFrame:
    """Convert records to a wide dataframe with flattened MRI volume columns."""
    rows: list[dict[str, Any]] = []
    for record in records:
        row = record.model_dump(mode="json")
        features = row.pop("mri_vol_features", {}) or {}
        for key, value in features.items():
            row[f"mri_vol__{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def finalize_master_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Finalize derived columns and stable row versions."""
    out = compute_conversion_labels(df)
    out["available_modalities"] = out["available_modalities"].apply(_sorted_list)
    out["is_rich"] = out["available_modalities"].apply(
        lambda values: PUBLIC_RICH_MODALITIES.issubset(set(values))
    )
    out = out.sort_values(["cohort", "subject_id", "visit_idx"]).reset_index(drop=True)
    out["record_version"] = out.apply(_record_version, axis=1)
    return out


def write_master(
    data_cfg: DictConfig,
    datasets: list[str] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Build and write master parquet plus quality report."""
    df, quality = build_master_dataframe(data_cfg, datasets=datasets)
    master_path = _repo_path(str(data_cfg.master_path))
    master_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(master_path, index=False)

    quality_path = master_path.parent / "quality_report.json"
    quality_path.write_text(json.dumps(quality, indent=2, sort_keys=True), encoding="utf-8")
    return master_path, quality


def merge_adni_nifti(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Collapse ADNI tabular/NIfTI duplicate keys into single rich rows."""
    if df.empty:
        return df, {"joined_adni_nifti": 0, "standalone_adni_nifti": 0}

    rows: list[dict[str, Any]] = []
    joined = 0
    standalone = 0
    for _, group in df.groupby(["subject_id", "visit_idx"], sort=False):
        if len(group) == 1:
            row = group.iloc[0].to_dict()
            if row.get("cohort") == "ADNI_NIFTI":
                standalone += 1
            rows.append(row)
            continue

        base_candidates = group[group["cohort"].eq("ADNI_TAB")]
        base = (
            base_candidates.iloc[0].to_dict()
            if not base_candidates.empty
            else group.iloc[0].to_dict()
        )
        modalities = set(_sorted_list(base.get("available_modalities")))
        qc_reasons = set(_sorted_list(base.get("qc_reasons")))
        for _, other_row in group.iterrows():
            other = other_row.to_dict()
            if other is base:
                continue
            if base.get("cohort") == "ADNI_TAB" and other.get("cohort") == "ADNI_NIFTI":
                joined += 1
                qc_reasons.add("merged_adni_nifti")
            modalities.update(_sorted_list(other.get("available_modalities")))
            qc_reasons.update(_sorted_list(other.get("qc_reasons")))
            for column, value in other.items():
                if column in {"cohort", "diagnosis", "diagnosis_source", "diagnosis_confidence"}:
                    continue
                if column == "available_modalities":
                    continue
                if column == "qc_reasons":
                    continue
                if _is_missing(base.get(column)) and not _is_missing(value):
                    base[column] = value
        base["available_modalities"] = sorted(modalities)
        base["qc_reasons"] = sorted(qc_reasons)
        rows.append(base)

    return pd.DataFrame(rows), {
        "joined_adni_nifti": joined,
        "standalone_adni_nifti": standalone,
    }


def build_quality_report(
    df: pd.DataFrame,
    adapter_counts: dict[str, int],
    skipped: dict[str, dict[str, int]],
    merge_stats: dict[str, int],
) -> dict[str, Any]:
    """Create the Phase 1 quality report dictionary."""
    modality_patterns = (
        df["available_modalities"]
        .apply(lambda values: "+".join(_sorted_list(values)))
        .value_counts()
        .to_dict()
    )
    split_ratio_placeholder: dict[str, Any] = {}
    return {
        "row_count": len(df),
        "unique_subject_visits": int(df[["subject_id", "visit_idx"]].drop_duplicates().shape[0]),
        "is_rich_count": int(df["is_rich"].sum()),
        "mri_t1_uri_count": int(df["mri_t1_uri"].notna().sum()),
        "adapter_counts": adapter_counts,
        "skipped_counts": skipped,
        "adni_merge": merge_stats,
        "cohort_counts": _nested_counts(df, "cohort"),
        "diagnosis_counts": _nested_counts(df, "diagnosis"),
        "cohort_diagnosis_counts": _nested_counts(df, ["cohort", "diagnosis"]),
        "modality_pattern_counts": modality_patterns,
        "split_ratios": split_ratio_placeholder,
    }


def update_quality_report_splits(data_cfg: DictConfig, split_ratios: dict[str, Any]) -> None:
    """Attach split ratios to the existing quality report."""
    report_path = _repo_path(str(data_cfg.master_path)).parent / "quality_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["split_ratios"] = split_ratios
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def _explicit_skipped_counts(adapter_name: str, skipped_counts: dict[str, int]) -> dict[str, int]:
    """Return skip counts with known zero-count reasons made explicit."""
    out = dict(skipped_counts)
    if adapter_name in {"brsdincer", "oasis_cross", "oasis_long"}:
        out.setdefault("blank_cdr", 0)
    return out


def _repo_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else repo_root() / candidate


def _nested_counts(df: pd.DataFrame, columns: str | list[str]) -> dict[str, int]:
    counts = df.groupby(columns, dropna=False).size()
    return {
        "|".join(str(part) for part in key if part is not pd.NA): int(value)
        for key, value in _iter_counts(counts)
    }


def _iter_counts(counts: pd.Series) -> list[tuple[tuple[Any, ...], int]]:
    out: list[tuple[tuple[Any, ...], int]] = []
    for key, value in counts.items():
        key_tuple = key if isinstance(key, tuple) else (key,)
        out.append((key_tuple, int(value)))
    return out


def _sorted_list(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return sorted(str(item) for item in value if not _is_missing(item))
    if isinstance(value, tuple | set):
        return sorted(str(item) for item in value if not _is_missing(item))
    return [str(value)]


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, list | tuple | set | dict):
        return len(value) == 0
    try:
        return bool(pd.isna(value))
    except ValueError:
        return False


def _record_version(row: pd.Series) -> str:
    payload = {
        key: _jsonable(value) for key, value in row.to_dict().items() if key != "record_version"
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _jsonable(value: Any) -> Any:
    if _is_missing(value):
        return None
    if isinstance(value, dict):
        return {str(key): _jsonable(child) for key, child in sorted(value.items())}
    if isinstance(value, list | tuple | set):
        return [_jsonable(child) for child in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            return str(value)
    return value
