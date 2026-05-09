"""Adapter for OASIS-2 longitudinal rows."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from agewell.data.adapters._base import BaseAdapter, as_float, as_int, sex_from_value
from agewell.data.label_harmonization import canonicalize_diagnosis
from agewell.data.schema import CanonicalRecord, ModalityName

CSV = "oasis_longitudinal.csv"


class OASISLongAdapter(BaseAdapter):
    """Emit canonical rows from OASIS-2 longitudinal CSV."""

    cohort = "OASIS2"
    populates: tuple[ModalityName, ...] = ("clinical_demo", "cognitive", "mri_vol")

    def iter_records(self) -> Iterable[CanonicalRecord]:
        df = pd.read_csv(self.source_root / CSV)
        for _, row in df.iterrows():
            cdr = as_float(row.get("CDR"))
            if cdr is None:
                self._skip("blank_cdr")
                continue
            diagnosis, confidence, source = canonicalize_diagnosis(cdr, self.cohort)
            record = CanonicalRecord(
                subject_id=f"OASIS2:{str(row['Subject ID']).strip()}",
                visit_idx=max((as_int(row.get("Visit")) or 1) - 1, 0),
                cohort="OASIS2",
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
                source_visit_delay_days=as_float(row.get("MR Delay")),
            )
            yield self.populate_modalities(record)
