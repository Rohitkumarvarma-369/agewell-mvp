"""Adapter for the public CSF/APOE lipidomics-named dataset."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from agewell.data.adapters._base import BaseAdapter, as_float, sex_from_value
from agewell.data.label_harmonization import canonicalize_diagnosis
from agewell.data.schema import CanonicalRecord, ModalityName

CSV = "Plasma lipidomics in Alzheimers disease and its progression-1.csv"


class LipidomicsAdapter(BaseAdapter):
    """Emit canonical rows from the public lipidomics CSV."""

    cohort = "LIPID"
    populates: tuple[ModalityName, ...] = ("clinical_demo", "cognitive", "genetic", "lipid")

    def iter_records(self) -> Iterable[CanonicalRecord]:
        df = pd.read_csv(self.source_root / CSV)
        for _, row in df.iterrows():
            diagnosis, confidence, source = canonicalize_diagnosis(row["Diagnostic"], self.cohort)
            apoe4 = _apoe4(row.get("APOE4"))
            record = CanonicalRecord(
                subject_id=f"LIPID:{int(row['Sample']):04d}",
                visit_idx=0,
                cohort="LIPID",
                age=as_float(row.get("Age")),
                sex=sex_from_value(row.get("Sex")),  # type: ignore[arg-type]
                mmse=as_float(row.get("MMSE")),
                apoe4=apoe4,
                high_risk_apoe4=None if apoe4 is None else int(apoe4 > 0),
                csf_amyloid=as_float(row.get("CSF Amyloid (pg/mL)")),
                csf_total_tau=as_float(row.get("CSF Total tau (pg/mL)")),
                csf_p_tau=as_float(row.get("CSF Phosphorylated tau (pg/mL)")),
                plasma_lipid_panel=[],
                diagnosis=diagnosis,
                diagnosis_source=source,
                diagnosis_confidence=confidence,
                qc_status="pass",
                progression_to_ad=str(row.get("Progression to Alzheimer's Disease", "")).strip(),
                progression_time_months=as_float(row.get("Progression time (months)")),
            )
            yield self.populate_modalities(record)


def _apoe4(value: object) -> int | None:
    raw = str(value).strip()
    if raw == "Yes":
        return 2
    if raw == "No":
        return 0
    return None
