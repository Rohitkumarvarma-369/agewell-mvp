"""Tests for Phase 1 diagnosis and conversion rules."""

import pytest


def test_lipidomics_actual_labels_map_to_canonical() -> None:
    """The actual lipidomics labels are mapped without guessing aliases."""
    pytest.importorskip("pandas")
    from agewell.data.label_harmonization import canonicalize_diagnosis

    assert canonicalize_diagnosis("Control", "LIPID") == ("CN", 0.90, "explicit")
    assert canonicalize_diagnosis("Mild Cognitive Impairment", "LIPID") == (
        "EMCI",
        0.90,
        "explicit",
    )
    assert canonicalize_diagnosis("Alzheimer's Disease", "LIPID") == ("AD", 0.90, "explicit")


def test_oasis_cdr_confidence_rules() -> None:
    """OASIS CDR-derived labels retain per-label confidence."""
    pytest.importorskip("pandas")
    from agewell.data.label_harmonization import canonicalize_diagnosis

    assert canonicalize_diagnosis(0.0, "OASIS1") == ("CN", 0.95, "cdr_derived")
    assert canonicalize_diagnosis(0.5, "OASIS1") == ("EMCI", 0.85, "cdr_derived")
    assert canonicalize_diagnosis(1.0, "OASIS1") == ("AD", 0.90, "cdr_derived")
    assert canonicalize_diagnosis(2.0, "OASIS1") == ("AD", 0.95, "cdr_derived")


def test_conversion_labels_are_per_cohort() -> None:
    """Conversion labels preserve ADNI approximation and lipid progression rules."""
    pd = pytest.importorskip("pandas")
    from agewell.data.label_harmonization import compute_conversion_labels

    df = pd.DataFrame(
        [
            {"cohort": "ADNI_TAB", "subject_id": "ADNI:0001", "diagnosis": "AD"},
            {"cohort": "ADNI_TAB", "subject_id": "ADNI:0002", "diagnosis": "CN"},
            {
                "cohort": "LIPID",
                "subject_id": "LIPID:0001",
                "diagnosis": "EMCI",
                "progression_to_ad": "Yes",
                "progression_time_months": 30.0,
            },
        ]
    )
    out = compute_conversion_labels(df)
    assert out.loc[0, "converted_within_5y"] == 1
    assert out.loc[0, "conversion_confidence"] == 0.85
    assert out.loc[1, "converted_within_5y"] == 0
    assert out.loc[2, "converted_within_2y"] == 0
    assert out.loc[2, "converted_within_5y"] == 1
