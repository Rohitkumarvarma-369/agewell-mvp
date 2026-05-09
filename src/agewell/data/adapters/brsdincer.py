"""Adapter for the Brsdincer OASIS-style Alzheimer features CSV."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from agewell.data.adapters._base import BaseAdapter, as_float, as_int, sex_from_value
from agewell.data.label_harmonization import canonicalize_diagnosis
from agewell.data.schema import CanonicalRecord, ModalityName

CSV = "alzheimer.csv"


class BrsdincerAdapter(BaseAdapter):
    """Emit canonical rows from the Brsdincer OASIS-style CSV."""

    cohort = "BRSDINCER"
    populates: tuple[ModalityName, ...] = ("clinical_demo", "cognitive", "mri_vol")

    def iter_records(self) -> Iterable[CanonicalRecord]:
        df = pd.read_csv(self.source_root / CSV)
        for idx, row in df.iterrows():
            cdr = as_float(row.get("CDR"))
            if cdr is None:
                self._skip("blank_cdr")
                continue
            diagnosis, confidence, source = canonicalize_diagnosis(cdr, self.cohort)
            record = CanonicalRecord(
                subject_id=f"BRSDINCER:row_{idx:04d}",
                visit_idx=0,
                cohort="BRSDINCER",
                age=as_float(row.get("Age")),
                sex=sex_from_value(row.get("M/F")),  # type: ignore[arg-type]
                education_years=as_float(row.get("EDUC")),
                ses=as_int(row.get("SES")),
                mmse=as_float(row.get("MMSE")),
                cdr=cdr,
                etiv=as_float(row.get("eTIV")),
                n_wbv=as_float(row.get("nWBV")),
                asf=as_float(row.get("ASF")),
                diagnosis=diagnosis,
                diagnosis_source=source,
                diagnosis_confidence=confidence,
                qc_status="pass",
                source_group=str(row.get("Group", "")).strip(),
            )
            yield self.populate_modalities(record)
