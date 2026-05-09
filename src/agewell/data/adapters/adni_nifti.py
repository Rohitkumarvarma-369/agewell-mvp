"""Adapter for ADNI 3D NIfTI files and metadata."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from agewell.data.adapters._base import BaseAdapter, as_float, local_path, sex_from_value
from agewell.data.label_harmonization import canonicalize_diagnosis
from agewell.data.schema import CanonicalRecord, ModalityName

CSV = "metadata.csv"


class ADNINiftiAdapter(BaseAdapter):
    """Emit canonical MRI rows from ADNI NIfTI metadata."""

    cohort = "ADNI_NIFTI"
    populates: tuple[ModalityName, ...] = (
        "clinical_demo",
        "clinical_comorbid",
        "cognitive",
        "mri_raw",
    )

    def iter_records(self) -> Iterable[CanonicalRecord]:
        df = pd.read_csv(self.source_root / CSV)
        for _, row in df.iterrows():
            label = str(row["label"]).strip()
            ptid = str(row["PTID"]).strip()
            rid = _rid_from_ptid(ptid)
            nii_path = self.source_root / label / f"{ptid}.nii"
            if not nii_path.exists():
                self._skip("missing_nifti_file")
                continue
            diagnosis, confidence, source = canonicalize_diagnosis(label, self.cohort)
            record = CanonicalRecord(
                subject_id=f"ADNI:{rid:04d}",
                visit_idx=0,
                cohort="ADNI_NIFTI",
                age=as_float(row.get("AGE")),
                sex=sex_from_value(row.get("GENDER")),  # type: ignore[arg-type]
                education_years=as_float(row.get("EDUCATION")),
                cdrsb=as_float(row.get("CDRSB")),
                mmse=as_float(row.get("MMSE")),
                log_mem_delayed=as_float(row.get("LogMem_Delayed")),
                log_mem_immediate=as_float(row.get("LogMem_Immediate")),
                gds_total=as_float(row.get("GDS_TOTAL")),
                systolic_bp=as_float(row.get("BP_Systolic")),
                mri_t1_uri=local_path(Path(nii_path)),
                diagnosis=diagnosis,
                diagnosis_source=source,
                diagnosis_confidence=confidence,
                qc_status="pass",
            )
            yield self.populate_modalities(record)


def _rid_from_ptid(ptid: str) -> int:
    raw = ptid.split("_")[-1]
    return int(raw)
