"""Convert master dataframe rows into Phase 3 encoder batch dictionaries."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from agewell._common.paths import models_root
from agewell.data.cognitive_columns import COGNITIVE_COLUMNS
from agewell.data.comorbid_columns import COMORBID_COLUMNS, LIFESTYLE_COLUMNS

DIAG_LABELS: tuple[str, ...] = ("CN", "SMC", "EMCI", "LMCI", "AD")
DIAG_TO_INDEX: dict[str, int] = {label: idx for idx, label in enumerate(DIAG_LABELS)}

CLINICAL_DEMO_FEATURES: tuple[str, ...] = ("age", "sex", "education_years", "ses", "bmi")
BLOOD_FEATURES: tuple[str, ...] = (
    "cholesterol_total",
    "cholesterol_ldl",
    "cholesterol_hdl",
    "cholesterol_triglycerides",
    "hba1c",
    "fasting_glucose",
    "crp",
    "hemoglobin",
)
MRI_RAW_DIM = 768
LIPID_DIM = 213


def mri_vol_feature_names(path: Path | None = None) -> tuple[str, ...]:
    """Return fixed-order FreeSurfer feature column names from master parquet."""
    source = (
        path or Path(__file__).resolve().parents[1] / "data" / "freesurfer_columns_canonical.txt"
    )
    return tuple(line.strip() for line in source.read_text(encoding="utf-8").splitlines() if line)


FEATURE_NAMES: dict[str, tuple[str, ...]] = {
    "clinical_demo": CLINICAL_DEMO_FEATURES,
    "clinical_lifestyle": LIFESTYLE_COLUMNS,
    "clinical_comorbid": COMORBID_COLUMNS,
    "cognitive": COGNITIVE_COLUMNS,
    "blood": BLOOD_FEATURES,
    "mri_vol": mri_vol_feature_names(),
    "mri_raw": tuple(f"brainiac_{idx}" for idx in range(MRI_RAW_DIM)),
    "lipid": tuple(f"lipid_{idx}" for idx in range(LIPID_DIM)),
}


def compose_batch(
    rows: pd.DataFrame,
    *,
    imputation_stats: dict[str, dict[str, float]] | None = None,
    strict_mri_paths: bool = True,
) -> dict[str, Any]:
    """Compose a dataframe slice into the Phase 3 encoder batch format."""
    df = rows.reset_index(drop=True)
    batch: dict[str, Any] = {}
    for modality in (
        "clinical_demo",
        "clinical_lifestyle",
        "clinical_comorbid",
        "cognitive",
        "blood",
        "mri_vol",
        "lipid",
    ):
        features, presence = extract_modality_features(
            df,
            modality,
            imputation_stats=imputation_stats,
        )
        batch[f"{modality}_features"] = torch.from_numpy(features)
        batch[f"{modality}_presence"] = torch.from_numpy(presence)

    mri_raw, mri_presence = extract_mri_raw_features(df, strict_paths=strict_mri_paths)
    batch["mri_raw_features"] = torch.from_numpy(mri_raw)
    batch["mri_raw_presence"] = torch.from_numpy(mri_presence)

    genetic_values, genetic_presence = extract_genetic_apoe4(df)
    batch["genetic_apoe4"] = torch.from_numpy(genetic_values)
    batch["genetic_presence"] = torch.from_numpy(genetic_presence)

    targets = extract_targets(df)
    batch.update(targets)
    batch["is_rich"] = torch.as_tensor(_bool_column(df, "is_rich"), dtype=torch.bool)
    if "available_modalities" in df:
        batch["available_modalities"] = [
            sorted(_as_string_set(value)) for value in df["available_modalities"].tolist()
        ]
    else:
        batch["available_modalities"] = [[] for _ in range(len(df))]
    return batch


def extract_modality_features(
    rows: pd.DataFrame,
    modality: str,
    *,
    imputation_stats: dict[str, dict[str, float]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract one non-MRI/non-genetic modality matrix and presence mask."""
    if modality == "clinical_demo":
        features = _clinical_demo_features(rows)
    elif modality == "mri_vol":
        features = _mri_vol_features(rows)
    elif modality == "lipid":
        features = _lipid_features(rows)
    else:
        names = FEATURE_NAMES[modality]
        features = _numeric_matrix(rows, names)

    presence = _presence_from_available(rows, modality)
    features = features.astype(np.float32, copy=False)
    features[~presence] = 0.0
    if imputation_stats is not None:
        features = apply_imputation(
            features, FEATURE_NAMES[modality], imputation_stats.get(modality, {})
        )
        features[~presence] = 0.0
    return features, presence


def extract_mri_raw_features(
    rows: pd.DataFrame,
    *,
    strict_paths: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Load cached BrainIAC vectors from ``mri_brainiac_uri``."""
    features = np.zeros((len(rows), MRI_RAW_DIM), dtype=np.float32)
    presence = np.zeros(len(rows), dtype=bool)
    if "mri_brainiac_uri" not in rows:
        return features, presence
    for idx, uri in enumerate(rows["mri_brainiac_uri"].tolist()):
        if _is_missing(uri):
            continue
        path = Path(str(uri))
        if not path.exists():
            if strict_paths:
                raise FileNotFoundError(path)
            continue
        arr = np.load(path, mmap_mode="r").astype(np.float32).reshape(-1)
        if arr.shape[0] != MRI_RAW_DIM:
            raise ValueError(
                f"Expected BrainIAC vector dim {MRI_RAW_DIM}, got {arr.shape[0]}: {path}"
            )
        if not np.isfinite(arr).all():
            raise ValueError(f"BrainIAC vector contains non-finite values: {path}")
        features[idx] = arr
        presence[idx] = True
    return features, presence


def extract_genetic_apoe4(rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Extract APOE4 count, mapping binary high-risk to the conservative 0/2 scale."""
    values = np.zeros(len(rows), dtype=np.int64)
    presence = _presence_from_available(rows, "genetic")
    for idx, (_, row) in enumerate(rows.iterrows()):
        apoe4 = row.get("apoe4")
        high_risk = row.get("high_risk_apoe4")
        if not _is_missing(apoe4):
            values[idx] = int(np.clip(int(apoe4), 0, 2))
        elif not _is_missing(high_risk):
            values[idx] = 2 if int(high_risk) > 0 else 0
        else:
            presence[idx] = False
    return values, presence


def extract_targets(rows: pd.DataFrame) -> dict[str, Tensor]:
    """Extract diagnosis, cognitive, and conversion targets."""
    labels = np.array(
        [DIAG_TO_INDEX.get(str(value), -1) for value in rows["diagnosis"]], dtype=np.int64
    )
    confidence = _float_column(rows, "diagnosis_confidence", default=1.0)
    mmse = _float_column(rows, "mmse", default=np.nan)
    cdr = _float_column(rows, "cdr", default=np.nan)
    surv_bin, censored, has_survival = _survival_targets(rows)
    return {
        "diag_label": torch.from_numpy(labels),
        "label_confidence_weight": torch.from_numpy(confidence.astype(np.float32)),
        "mmse": torch.from_numpy(np.nan_to_num(mmse, nan=0.0).astype(np.float32)),
        "has_mmse": torch.from_numpy(np.isfinite(mmse)),
        "cdr": torch.from_numpy(np.nan_to_num(cdr, nan=0.0).astype(np.float32)),
        "has_cdr": torch.from_numpy(np.isfinite(cdr)),
        "surv_bin": torch.from_numpy(surv_bin),
        "censored": torch.from_numpy(censored),
        "has_survival": torch.from_numpy(has_survival),
    }


def compute_imputation_stats(rows: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Compute per-modality median imputation values from a training split."""
    stats: dict[str, dict[str, float]] = {}
    for modality in (
        "clinical_demo",
        "clinical_lifestyle",
        "clinical_comorbid",
        "cognitive",
        "blood",
        "mri_vol",
        "lipid",
    ):
        features, presence = extract_modality_features(rows, modality, imputation_stats=None)
        present_features = features[presence]
        names = FEATURE_NAMES[modality]
        stats[modality] = _feature_medians(present_features, names)
    return stats


def write_imputation_stats(
    rows: pd.DataFrame,
    path: Path | None = None,
) -> Path:
    """Write train-split imputation medians to JSON."""
    out = path or models_root() / "imputation_medians.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(compute_imputation_stats(rows), indent=2, sort_keys=True), encoding="utf-8"
    )
    return out


def load_imputation_stats(path: Path | None = None) -> dict[str, dict[str, float]]:
    """Load serialized imputation medians."""
    source = path or models_root() / "imputation_medians.json"
    return json.loads(source.read_text(encoding="utf-8"))


def apply_imputation(
    features: np.ndarray,
    names: tuple[str, ...],
    stats: dict[str, float],
) -> np.ndarray:
    """Fill NaN feature values with precomputed medians."""
    out = features.copy()
    for idx, name in enumerate(names):
        value = float(stats.get(name, 0.0))
        mask = ~np.isfinite(out[:, idx])
        out[mask, idx] = value
    return out


def build_tabpfn_context(
    rows: pd.DataFrame,
    *,
    modality: str,
    ctx_size: int,
    seed: int = 1337,
    imputation_stats: dict[str, dict[str, float]] | None = None,
) -> tuple[Tensor, Tensor]:
    """Build a deterministic context set for one TabPFN modality."""
    if imputation_stats is None:
        base_features, presence = extract_modality_features(rows, modality, imputation_stats=None)
        modality_stats = _feature_medians(base_features[presence], FEATURE_NAMES[modality])
        features = apply_imputation(base_features, FEATURE_NAMES[modality], modality_stats)
        features[~presence] = 0.0
    else:
        features, presence = extract_modality_features(
            rows, modality, imputation_stats=imputation_stats
        )
    labels = np.array(
        [DIAG_TO_INDEX.get(str(value), -1) for value in rows["diagnosis"]], dtype=np.int64
    )
    valid = presence & (labels >= 0)
    valid_idx = np.flatnonzero(valid)
    if len(valid_idx) == 0:
        raise ValueError(f"No valid TabPFN context rows for modality {modality}")
    if len(valid_idx) > ctx_size:
        rng = np.random.default_rng(seed)
        valid_idx = np.sort(rng.choice(valid_idx, size=ctx_size, replace=False))
    return torch.from_numpy(features[valid_idx].astype(np.float32)), torch.from_numpy(
        labels[valid_idx]
    )


def _feature_medians(features: np.ndarray, names: tuple[str, ...]) -> dict[str, float]:
    stats: dict[str, float] = {}
    for idx, name in enumerate(names):
        values = features[:, idx] if len(features) else np.array([], dtype=np.float32)
        finite = values[np.isfinite(values)]
        stats[name] = float(np.median(finite)) if len(finite) else 0.0
    return stats


def _clinical_demo_features(rows: pd.DataFrame) -> np.ndarray:
    features = np.zeros((len(rows), len(CLINICAL_DEMO_FEATURES)), dtype=np.float32)
    for idx, (_, row) in enumerate(rows.iterrows()):
        features[idx, 0] = _to_float(row.get("age"))
        features[idx, 1] = _sex_code(row.get("sex"))
        features[idx, 2] = _to_float(row.get("education_years"))
        features[idx, 3] = _to_float(row.get("ses"))
        features[idx, 4] = _to_float(row.get("bmi"))
    return features


def _mri_vol_features(rows: pd.DataFrame) -> np.ndarray:
    names = FEATURE_NAMES["mri_vol"]
    return _numeric_matrix(rows, names)


def _lipid_features(rows: pd.DataFrame) -> np.ndarray:
    out = np.full((len(rows), LIPID_DIM), np.nan, dtype=np.float32)
    for idx, (_, row) in enumerate(rows.iterrows()):
        out[idx, 0] = _to_float(row.get("csf_amyloid"))
        out[idx, 1] = _to_float(row.get("csf_total_tau"))
        out[idx, 2] = _to_float(row.get("csf_p_tau"))
        panel = row.get("plasma_lipid_panel")
        if isinstance(panel, np.ndarray):
            panel_values = panel.tolist()
        elif isinstance(panel, list | tuple):
            panel_values = list(panel)
        else:
            panel_values = []
        for offset, value in enumerate(panel_values[: LIPID_DIM - 3], start=3):
            out[idx, offset] = _to_float(value)
    return out


def _numeric_matrix(rows: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    out = np.full((len(rows), len(columns)), np.nan, dtype=np.float32)
    for col_idx, column in enumerate(columns):
        if column not in rows:
            continue
        out[:, col_idx] = [_to_float(value) for value in rows[column].tolist()]
    return out


def _survival_targets(rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bins = np.full(len(rows), 4, dtype=np.int64)
    censored = np.zeros(len(rows), dtype=bool)
    has_survival = np.zeros(len(rows), dtype=bool)
    for idx, (_, row) in enumerate(rows.iterrows()):
        c2 = row.get("converted_within_2y")
        c5 = row.get("converted_within_5y")
        c10 = row.get("converted_within_10y")
        cens = row.get("censored")
        values = (c2, c5, c10, cens, row.get("time_to_event_y"))
        has_survival[idx] = any(not _is_missing(value) for value in values)
        censored[idx] = bool(int(cens)) if not _is_missing(cens) else False
        if not has_survival[idx]:
            continue
        if _truthy(c2):
            bins[idx] = 0
        elif _truthy(c5):
            bins[idx] = 1
        elif _truthy(c10):
            bins[idx] = 2
        elif censored[idx]:
            bins[idx] = 4
        else:
            bins[idx] = 3
    return bins, censored, has_survival


def _presence_from_available(rows: pd.DataFrame, modality: str) -> np.ndarray:
    if "available_modalities" not in rows:
        return np.zeros(len(rows), dtype=bool)
    return np.array(
        [modality in _as_string_set(value) for value in rows["available_modalities"].tolist()],
        dtype=bool,
    )


def _float_column(rows: pd.DataFrame, column: str, *, default: float) -> np.ndarray:
    if column not in rows:
        return np.full(len(rows), default, dtype=np.float32)
    return np.array(
        [_to_float(value, default=default) for value in rows[column].tolist()], dtype=np.float32
    )


def _bool_column(rows: pd.DataFrame, column: str) -> np.ndarray:
    if column not in rows:
        return np.zeros(len(rows), dtype=bool)
    return np.array(
        [False if _is_missing(value) else bool(value) for value in rows[column].tolist()],
        dtype=bool,
    )


def _to_float(value: object, *, default: float = np.nan) -> float:
    if _is_missing(value):
        return default
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _sex_code(value: object) -> float:
    if _is_missing(value):
        return np.nan
    text = str(value).strip().upper()
    if text == "M":
        return 0.0
    if text == "F":
        return 1.0
    return np.nan


def _truthy(value: object) -> bool:
    numeric = _to_float(value, default=0.0)
    return bool(int(numeric)) if math.isfinite(numeric) else False


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    return bool(pd.isna(value)) if not isinstance(value, list | tuple | set | np.ndarray) else False


def _as_string_set(value: object) -> set[str]:
    if _is_missing(value):
        return set()
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, list | tuple | set):
        return {str(item) for item in value if not _is_missing(item)}
    return {str(value)}
