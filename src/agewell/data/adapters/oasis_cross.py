"""Adapter for OASIS-1 cross-sectional rows."""

from __future__ import annotations

import re
from collections.abc import Iterable

import pandas as pd

from agewell.data.adapters._base import BaseAdapter, as_float, as_int, sex_from_value
from agewell.data.label_harmonization import canonicalize_diagnosis
from agewell.data.schema import CanonicalRecord, ModalityName

CSV = "oasis_cross-sectional.csv"


class OASISCrossAdapter(BaseAdapter):
    """Emit canonical rows from OASIS-1 cross-sectional CSV."""

    cohort = "OASIS1"
    populates: tuple[ModalityName, ...] = ("clinical_demo", "cognitive", "mri_vol")

    def iter_records(self) -> Iterable[CanonicalRecord]:
        df = pd.read_csv(self.source_root / CSV)
        for _, row in df.iterrows():
            cdr = as_float(row.get("CDR"))
            if cdr is None:
                self._skip("blank_cdr")
                continue
            diagnosis, confidence, source = canonicalize_diagnosis(cdr, self.cohort)
            subject, visit_idx = _parse_oasis1_id(str(row["ID"]))
            record = CanonicalRecord(
                subject_id=f"OASIS1:{subject}",
                visit_idx=visit_idx,
                cohort="OASIS1",
                age=as_float(row.get("Age")),
                sex=sex_from_value(row.get("M/F")),  # type: ignore[arg-type]
                education_years=as_float(row.get("Educ")),
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
            )
            yield self.populate_modalities(record)


def _parse_oasis1_id(raw: str) -> tuple[str, int]:
    match = re.match(r"(?P<subject>OAS1_\d+)_MR(?P<visit>\d+)", raw)
    if not match:
        return raw, 0
    return match.group("subject"), int(match.group("visit")) - 1
