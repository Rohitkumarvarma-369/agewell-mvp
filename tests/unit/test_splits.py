"""Tests for Phase 1 split generation."""

import pytest


def test_subject_disjoint_split_is_deterministic() -> None:
    """Splits are deterministic and never share subjects."""
    pd = pytest.importorskip("pandas")
    from agewell.data.splits import assert_subject_disjoint, subject_disjoint_split

    rows = []
    for idx in range(60):
        rows.append(
            {
                "subject_id": f"S:{idx:03d}",
                "visit_idx": 0,
                "cohort": "A" if idx < 30 else "B",
                "diagnosis": "CN" if idx % 2 == 0 else "AD",
            }
        )
    df = pd.DataFrame(rows)
    first = subject_disjoint_split(df, seed=7)
    second = subject_disjoint_split(df, seed=7)
    assert_subject_disjoint(first)
    assert {name: list(split["subject_id"]) for name, split in first.items()} == {
        name: list(split["subject_id"]) for name, split in second.items()
    }
