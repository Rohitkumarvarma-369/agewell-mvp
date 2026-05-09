"""Adapter for the Rabie El Kharoua Alzheimer's disease dataset."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from agewell.data.adapters._base import BaseAdapter, as_float, as_int, sex_from_value
from agewell.data.label_harmonization import canonicalize_diagnosis
from agewell.data.schema import CanonicalRecord, ModalityName

CSV = "alzheimers_disease_data.csv"


class RabieElKharouaAdapter(BaseAdapter):
    """Emit canonical rows from the dense tabular RABIE dataset."""

    cohort = "RABIE"
    populates: tuple[ModalityName, ...] = (
        "clinical_demo",
        "clinical_lifestyle",
        "clinical_comorbid",
        "cognitive",
        "blood",
    )

    def iter_records(self) -> Iterable[CanonicalRecord]:
        df = pd.read_csv(self.source_root / CSV)
        for _, row in df.iterrows():
            diagnosis, confidence, source = canonicalize_diagnosis(row["Diagnosis"], self.cohort)
            record = CanonicalRecord(
                subject_id=f"RABIE:{int(row['PatientID'])}",
                visit_idx=0,
                cohort="RABIE",
                age=as_float(row.get("Age")),
                sex=sex_from_value(row.get("Gender")),  # type: ignore[arg-type]
                education_years=as_float(row.get("EducationLevel")),
                bmi=as_float(row.get("BMI")),
                smoking=as_float(row.get("Smoking")),
                alcohol=as_float(row.get("AlcoholConsumption")),
                physical_activity=as_float(row.get("PhysicalActivity")),
                diet_quality=as_float(row.get("DietQuality")),
                sleep_quality=as_float(row.get("SleepQuality")),
                family_history_ad=as_int(row.get("FamilyHistoryAlzheimers")),
                cardiovascular=as_int(row.get("CardiovascularDisease")),
                diabetes=as_int(row.get("Diabetes")),
                depression=as_int(row.get("Depression")),
                head_injury=as_int(row.get("HeadInjury")),
                hypertension=as_int(row.get("Hypertension")),
                systolic_bp=as_float(row.get("SystolicBP")),
                diastolic_bp=as_float(row.get("DiastolicBP")),
                cholesterol_total=as_float(row.get("CholesterolTotal")),
                cholesterol_ldl=as_float(row.get("CholesterolLDL")),
                cholesterol_hdl=as_float(row.get("CholesterolHDL")),
                cholesterol_triglycerides=as_float(row.get("CholesterolTriglycerides")),
                mmse=as_float(row.get("MMSE")),
                functional_assessment=as_float(row.get("FunctionalAssessment")),
                adl=as_float(row.get("ADL")),
                memory_complaints=as_int(row.get("MemoryComplaints")),
                behavioral_problems=as_int(row.get("BehavioralProblems")),
                confusion=as_int(row.get("Confusion")),
                disorientation=as_int(row.get("Disorientation")),
                personality_changes=as_int(row.get("PersonalityChanges")),
                difficulty_completing_tasks=as_int(row.get("DifficultyCompletingTasks")),
                forgetfulness=as_int(row.get("Forgetfulness")),
                diagnosis=diagnosis,
                diagnosis_source=source,
                diagnosis_confidence=confidence,
                qc_status="pass",
            )
            yield self.populate_modalities(record)
