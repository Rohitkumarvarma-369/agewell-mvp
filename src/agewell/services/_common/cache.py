"""Derivative cache path helpers for imaging services."""

from __future__ import annotations

import hashlib
from pathlib import Path

from agewell._common.paths import data_root


def sha1_text(value: str) -> str:
    """Return the stable SHA1 digest for a text value."""
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def imaging_cache_stem(source_uri: str, preprocess_version: str) -> str:
    """Return the versioned cache stem for an imaging derivative."""
    return f"{sha1_text(source_uri)}__{preprocess_version}"


def derivative_root(kind: str, root: Path | None = None) -> Path:
    """Return the derivative root for a named imaging artifact kind."""
    return (root or data_root()) / "derivatives" / kind
