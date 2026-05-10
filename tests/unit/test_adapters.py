"""Smoke tests for Phase 1 dataset adapters."""

from pathlib import Path

import pytest

from agewell.data.schema import CanonicalRecord

KAGGLE_ROOT = Path("/home/rohit/kaggle-iisc/kaggle-cli/downloads")
ADAPTER_NAMES = (
    "adni_nifti",
    "adni_tabular",
    "brsdincer",
    "ixi",
    "lipidomics",
    "oasis_cross",
    "oasis_long",
    "rabieelkharoua",
)
SUBDIRS = {
    "adni_tabular": "tabular/sarthakkanjariya_alzheimer-dataset",
    "adni_nifti": "imaging/afaqkhan012_adni-nifti-3d",
    "oasis_cross": "tabular/jboysen_mri-and-alzheimers",
    "oasis_long": "tabular/jboysen_mri-and-alzheimers",
    "brsdincer": "tabular/brsdincer_alzheimer-features",
    "rabieelkharoua": "tabular/rabieelkharoua_alzheimers-disease-dataset",
    "lipidomics": "tabular/fereshtehjozaghkar_plasma-lipidomics",
    "ixi": "imaging/kingpowa_preprocessed-ixi-fs8",
}


@pytest.mark.parametrize("adapter_name", ADAPTER_NAMES)
def test_adapter_emits_well_formed_record(adapter_name: str) -> None:
    """Each adapter emits at least one valid canonical row on local Kaggle data."""
    pytest.importorskip("pandas")
    from agewell.data.registry import ADAPTERS

    source_root = KAGGLE_ROOT / SUBDIRS[adapter_name]
    if not source_root.exists():
        pytest.skip(f"missing local dataset: {source_root}")
    adapter = ADAPTERS[adapter_name](source_root)
    record = next(iter(adapter.iter_records()))
    assert isinstance(record, CanonicalRecord)
    assert ":" in record.subject_id
    assert record.diagnosis is not None
    assert record.available_modalities


def test_oasis_cross_skips_blank_cdr_rows() -> None:
    """The OASIS-1 adapter skips rows that cannot satisfy non-null diagnosis gates."""
    pytest.importorskip("pandas")
    from agewell.data.registry import ADAPTERS

    source_root = KAGGLE_ROOT / SUBDIRS["oasis_cross"]
    if not source_root.exists():
        pytest.skip(f"missing local dataset: {source_root}")
    adapter = ADAPTERS["oasis_cross"](source_root)
    records = list(adapter.iter_records())
    assert len(records) == 235
    assert adapter.skipped_counts["blank_cdr"] == 201


def test_adni_tabular_marks_binary_apoe_qc_reason() -> None:
    """ADNI exposes APOE as binary high-risk status, not allele count."""
    pytest.importorskip("pandas")
    from agewell.data.registry import ADAPTERS

    source_root = KAGGLE_ROOT / SUBDIRS["adni_tabular"]
    if not source_root.exists():
        pytest.skip(f"missing local dataset: {source_root}")
    adapter = ADAPTERS["adni_tabular"](source_root)
    apoe_records = [record for record in adapter.iter_records() if record.apoe4 is not None]
    assert apoe_records
    assert all("apoe_binary_collapsed" in record.qc_reasons for record in apoe_records)
