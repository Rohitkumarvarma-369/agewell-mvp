"""Stable identifier factories for AgeWell artifacts."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4


def _new_id(prefix: str) -> str:
    ts = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    return f"{prefix}_{ts}_{uuid4().hex[:12]}"


def prediction_id() -> str:
    """Create a prediction artifact id."""
    return _new_id("pred")


def run_id() -> str:
    """Create a pipeline or training run id."""
    return _new_id("run")


def subject_id(cohort: str, raw_id: str | int) -> str:
    """Create a namespaced subject id."""
    return f"{cohort}:{str(raw_id).strip()}"
