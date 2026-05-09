"""Tests for Phase 1 schema contracts."""

import pytest

from agewell.data.schema import CanonicalRecord


def test_subject_id_must_be_namespaced() -> None:
    """Canonical records require globally namespaced IDs."""
    with pytest.raises(ValueError, match="namespaced"):
        CanonicalRecord(subject_id="0001", cohort="ADNI_TAB")


def test_available_modalities_sorted_unique() -> None:
    """Modality lists are stable for downstream pattern grouping."""
    record = CanonicalRecord(
        subject_id="ADNI:0001",
        cohort="ADNI_TAB",
        available_modalities=["mri_vol", "clinical_demo", "mri_vol"],
    )
    assert record.available_modalities == ["clinical_demo", "mri_vol"]
