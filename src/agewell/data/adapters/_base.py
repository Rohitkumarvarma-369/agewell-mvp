"""Base classes and parsing helpers for Phase 1 adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from math import isnan
from pathlib import Path
from typing import Any

from agewell.data.cognitive_columns import COGNITIVE_COLUMNS
from agewell.data.comorbid_columns import COMORBID_COLUMNS, LIFESTYLE_COLUMNS
from agewell.data.schema import CanonicalRecord, ModalityName

DEMO_FIELDS: tuple[str, ...] = ("age", "sex", "education_years", "ses", "bmi")
BLOOD_FIELDS: tuple[str, ...] = (
    "cholesterol_total",
    "cholesterol_ldl",
    "cholesterol_hdl",
    "cholesterol_triglycerides",
    "hba1c",
    "fasting_glucose",
    "crp",
    "hemoglobin",
)
MRI_VOL_FIELDS: tuple[str, ...] = (
    "etiv",
    "n_wbv",
    "asf",
    "hippocampus_l",
    "hippocampus_r",
    "ventricles",
    "whole_brain",
    "entorhinal",
    "fusiform",
    "mid_temp",
)
MRI_RAW_FIELDS: tuple[str, ...] = ("mri_t1_uri", "mri_stripped_uri", "mri_seg_uri")
GENETIC_FIELDS: tuple[str, ...] = ("apoe4", "high_risk_apoe4")
LIPID_FIELDS: tuple[str, ...] = ("csf_amyloid", "csf_total_tau", "csf_p_tau")


class BaseAdapter(ABC):
    """Base interface for adapters that yield canonical subject-visit records."""

    cohort: str
    populates: tuple[ModalityName, ...]

    def __init__(self, source_root: Path):
        self.source_root = source_root
        self.skipped_counts: dict[str, int] = {}

    @abstractmethod
    def iter_records(self) -> Iterable[CanonicalRecord]:
        """Yield one canonical record per source subject-visit."""

    def populate_modalities(self, record: CanonicalRecord) -> CanonicalRecord:
        """Set sorted modality availability and public-rich status."""
        record.available_modalities = [
            modality for modality in self.populates if self._modality_present(record, modality)
        ]
        record.is_rich = {"clinical_demo", "cognitive", "mri_vol", "mri_raw"}.issubset(
            set(record.available_modalities)
        )
        return record

    def _modality_present(self, record: CanonicalRecord, modality: ModalityName) -> bool:
        if modality == "clinical_demo":
            return _any_field(record, DEMO_FIELDS)
        if modality == "clinical_lifestyle":
            return _any_field(record, LIFESTYLE_COLUMNS)
        if modality == "clinical_comorbid":
            return _any_field(record, COMORBID_COLUMNS)
        if modality == "cognitive":
            return _any_field(record, COGNITIVE_COLUMNS)
        if modality == "blood":
            return _any_field(record, BLOOD_FIELDS)
        if modality == "mri_vol":
            return _any_field(record, MRI_VOL_FIELDS) or bool(record.mri_vol_features)
        if modality == "mri_raw":
            return _any_field(record, MRI_RAW_FIELDS)
        if modality == "genetic":
            return _any_field(record, GENETIC_FIELDS)
        if modality == "lipid":
            return _any_field(record, LIPID_FIELDS) or bool(record.plasma_lipid_panel)
        raise ValueError(f"Unhandled modality {modality}")

    def _skip(self, reason: str) -> None:
        self.skipped_counts[reason] = self.skipped_counts.get(reason, 0) + 1


def _any_field(record: CanonicalRecord, fields: tuple[str, ...]) -> bool:
    for field in fields:
        value = getattr(record, field)
        if is_present(value):
            return True
    return False


def is_present(value: object) -> bool:
    """Return whether a value is meaningful for modality detection."""
    if value is None:
        return False
    if isinstance(value, float) and isnan(value):
        return False
    return not (isinstance(value, str) and value.strip() == "")


def as_float(value: Any) -> float | None:
    """Parse a nullable float from CSV-like values."""
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if value == "" or value.upper() == "N/A":
            return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return None if isnan(parsed) else parsed


def as_int(value: Any) -> int | None:
    """Parse a nullable int from CSV-like values."""
    parsed = as_float(value)
    return None if parsed is None else int(parsed)


def sex_from_value(value: Any) -> str | None:
    """Normalize common sex/gender encodings."""
    raw = str(value).strip()
    mapping = {
        "M": "M",
        "Male": "M",
        "0": "M",
        "0.0": "M",
        "F": "F",
        "Female": "F",
        "1": "F",
        "1.0": "F",
    }
    return mapping.get(raw)


def local_path(path: Path) -> str:
    """Return an absolute local path string without requiring the file to exist."""
    return str(path.expanduser().resolve())
