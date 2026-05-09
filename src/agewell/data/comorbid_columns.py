"""Canonical comorbidity and lifestyle columns for Phase 1."""

COMORBID_COLUMNS: tuple[str, ...] = (
    "family_history_ad",
    "cardiovascular",
    "diabetes",
    "depression",
    "head_injury",
    "hypertension",
    "systolic_bp",
    "diastolic_bp",
)

LIFESTYLE_COLUMNS: tuple[str, ...] = (
    "smoking",
    "alcohol",
    "physical_activity",
    "diet_quality",
    "sleep_quality",
)
