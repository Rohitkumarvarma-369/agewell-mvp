"""Adapter for the ADNI tabular Kaggle snapshot."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from agewell.data.adapters._base import BaseAdapter, as_float, as_int, sex_from_value
from agewell.data.freesurfer_columns import canonicalize_freesurfer_column, is_freesurfer_column
from agewell.data.label_harmonization import canonicalize_diagnosis
from agewell.data.schema import CanonicalRecord, ModalityName

CSV = "Alzheimer_DataSet.csv"


class ADNITabularAdapter(BaseAdapter):
    """Emit canonical rows from the ADNI tabular CSV."""

    cohort = "ADNI_TAB"
    populates: tuple[ModalityName, ...] = ("clinical_demo", "cognitive", "mri_vol", "genetic")

    def iter_records(self) -> Iterable[CanonicalRecord]:
        df = pd.read_csv(self.source_root / CSV, low_memory=False)
        for _, row in df.iterrows():
            yield self.populate_modalities(self._row_to_record(row))

    def _row_to_record(self, row: pd.Series) -> CanonicalRecord:
        diagnosis, confidence, source = canonicalize_diagnosis(row["Diagnosis"], self.cohort)
        apoe4 = as_int(row.get("High_risk_ApoE4"))
        qc_reasons = []
        if apoe4 is not None:
            qc_reasons.append("apoe_binary_collapsed")
        return CanonicalRecord(
            subject_id=f"ADNI:{int(row['RID']):04d}",
            visit_idx=0,
            cohort="ADNI_TAB",
            age=as_float(row.get("Age")),
            sex=sex_from_value(row.get("Gender")),  # type: ignore[arg-type]
            education_years=as_float(row.get("Year_education")),
            mmse=as_float(row.get("MMSE")),
            cdrsb=as_float(row.get("CDRSB")),
            adas11=as_float(row.get("ADAS11")),
            adas13=as_float(row.get("ADAS13")),
            ravlt_immediate=as_float(row.get("RAVLT_immediate")),
            ravlt_learning=as_float(row.get("RAVLT_learning")),
            ravlt_forgetting=as_float(row.get("RAVLT_forgetting")),
            ravlt_perc_forgetting=as_float(row.get("RAVLT_perc_forgetting")),
            apoe4=apoe4,
            high_risk_apoe4=None if apoe4 is None else int(apoe4 > 0),
            etiv=as_float(row.get("Intra cranial volume")),
            ventricles=as_float(row.get("Ventricles")),
            hippocampus_l=as_float(row.get("Volume (WM Parcellation) of LeftHippocampus")),
            hippocampus_r=as_float(row.get("Volume (WM Parcellation) of RightHippocampus")),
            whole_brain=as_float(row.get("WholeBrain")),
            entorhinal=as_float(row.get("Entorhinal")),
            fusiform=as_float(row.get("Fusiform")),
            mid_temp=as_float(row.get("MidTemp")),
            mri_vol_features=_flatten_freesurfer(row),
            diagnosis=diagnosis,
            diagnosis_source=source,
            diagnosis_confidence=confidence,
            qc_status="pass",
            qc_reasons=qc_reasons,
        )


def _flatten_freesurfer(row: pd.Series) -> dict[str, float]:
    features: dict[str, float] = {}
    for column, value in row.items():
        if not is_freesurfer_column(str(column)):
            continue
        parsed = as_float(value)
        if parsed is not None:
            features[canonicalize_freesurfer_column(str(column))] = parsed
    return features
