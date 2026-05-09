"""Adapter for the IXI healthy-control MRI dataset."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from agewell.data.adapters._base import BaseAdapter, as_float, local_path, sex_from_value
from agewell.data.label_harmonization import canonicalize_diagnosis
from agewell.data.schema import CanonicalRecord, ModalityName

CSV = "subjects.csv"


class IXIAdapter(BaseAdapter):
    """Emit canonical healthy-control rows from IXI."""

    cohort = "IXI"
    populates: tuple[ModalityName, ...] = ("clinical_demo", "mri_raw")

    def iter_records(self) -> Iterable[CanonicalRecord]:
        df = pd.read_csv(self.source_root / CSV)
        for _, row in df.iterrows():
            subject = str(row["subject_id"]).strip()
            t1_path = _resolve_t1_path(self.source_root, subject)
            if t1_path is None:
                self._skip("missing_registered_t1")
                continue
            seg_path = _resolve_seg_path(self.source_root, subject)
            diagnosis, confidence, source = canonicalize_diagnosis("CN", self.cohort)
            record = CanonicalRecord(
                subject_id=f"IXI:{subject}",
                visit_idx=0,
                cohort="IXI",
                age=as_float(row.get("age")),
                sex=sex_from_value(row.get("sex")),  # type: ignore[arg-type]
                mri_t1_uri=local_path(t1_path),
                mri_seg_uri=None if seg_path is None else local_path(seg_path),
                diagnosis=diagnosis,
                diagnosis_source=source,
                diagnosis_confidence=confidence,
                qc_status="pass",
            )
            yield self.populate_modalities(record)


def _subject_anat_dir(source_root: Path, subject: str) -> Path:
    return (
        source_root
        / "T1w_Processed_IXI_with_csv"
        / "IXI"
        / f"sub-{subject}"
        / "ses-1"
        / "run-1"
        / "anat"
    )


def _resolve_t1_path(source_root: Path, subject: str) -> Path | None:
    anat_dir = _subject_anat_dir(source_root, subject)
    candidates = sorted(
        path for path in anat_dir.glob("*mni_registered_T1w.nii*") if "segmask" not in path.name
    )
    return candidates[0] if candidates else None


def _resolve_seg_path(source_root: Path, subject: str) -> Path | None:
    anat_dir = _subject_anat_dir(source_root, subject)
    candidates = sorted(anat_dir.glob("*segmask_mni_registered_T1w.nii*"))
    return candidates[0] if candidates else None
