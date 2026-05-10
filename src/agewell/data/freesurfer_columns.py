"""FreeSurfer-style MRI volume feature naming helpers."""

import re
from functools import lru_cache
from pathlib import Path

FREESURFER_PREFIXES: tuple[str, ...] = (
    "Volume (Cortical Parcellation)",
    "Volume (WM Parcellation)",
    "Surface Area",
    "Cortical Thickness Average",
    "Cortical Thickness Standard Deviation",
)
CANONICAL_FREESURFER_COLUMNS_PATH = Path(__file__).with_name("freesurfer_columns_canonical.txt")


def is_freesurfer_column(column: str) -> bool:
    """Return whether a source column is a FreeSurfer-derived feature."""
    return any(column.startswith(prefix) for prefix in FREESURFER_PREFIXES)


def canonicalize_freesurfer_column(column: str) -> str:
    """Convert a source FreeSurfer column name into a stable flat column suffix."""
    cleaned = column.strip().lower()
    cleaned = cleaned.replace("(", "").replace(")", "")
    cleaned = cleaned.replace("/", "_")
    cleaned = re.sub(r"[^a-z0-9]+", "_", cleaned)
    return cleaned.strip("_")


@lru_cache
def canonical_freesurfer_columns() -> tuple[str, ...]:
    """Load the canonical Phase 1 FreeSurfer-derived feature list."""
    columns = [
        line.strip()
        for line in CANONICAL_FREESURFER_COLUMNS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(columns) != len(set(columns)):
        raise ValueError("canonical FreeSurfer column list contains duplicates")
    return tuple(columns)
