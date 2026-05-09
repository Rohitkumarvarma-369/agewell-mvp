"""FreeSurfer-style MRI volume feature naming helpers."""

import re

FREESURFER_PREFIXES: tuple[str, ...] = (
    "Volume (Cortical Parcellation)",
    "Volume (WM Parcellation)",
    "Surface Area",
    "Cortical Thickness Average",
    "Cortical Thickness Standard Deviation",
)


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
