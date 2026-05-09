"""Canonical Phase 1 data schema."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

ModalityName = Literal[
    "clinical_demo",
    "clinical_lifestyle",
    "clinical_comorbid",
    "cognitive",
    "blood",
    "mri_vol",
    "mri_raw",
    "genetic",
    "lipid",
]

DiagnosisLabel = Literal["CN", "SMC", "EMCI", "LMCI", "AD"]

CohortName = Literal[
    "ADNI_TAB",
    "ADNI_NIFTI",
    "OASIS1",
    "OASIS2",
    "BRSDINCER",
    "RABIE",
    "LIPID",
    "IXI",
]

DiagnosisSource = Literal["explicit", "cdr_derived", "binary_derived"]
QCStatus = Literal["pass", "warn", "fail"]

MODALITIES: tuple[ModalityName, ...] = (
    "clinical_demo",
    "clinical_lifestyle",
    "clinical_comorbid",
    "cognitive",
    "blood",
    "mri_vol",
    "mri_raw",
    "genetic",
    "lipid",
)

# Public Kaggle data cannot populate the original 6-modality rich definition:
# ADNI has mri_vol but no RABIE-style lifestyle/comorbid/blood columns, while
# RABIE has those columns but no MRI. The public teacher-rich subset is the
# ADNI tabular/NIfTI overlap.
PUBLIC_RICH_MODALITIES: frozenset[ModalityName] = frozenset(
    {"clinical_demo", "cognitive", "mri_vol", "mri_raw"}
)


class CanonicalRecord(BaseModel):
    """One harmonized subject-visit row.

    The public "lipid" modality currently contains CSF amyloid/tau biomarkers
    and APOE4 only. The plasma lipid panel is an empty fixed-order vector until
    CBR data or another true lipidomics source is integrated.
    """

    subject_id: str
    visit_idx: int = 0
    cohort: CohortName

    age: float | None = None
    sex: Literal["M", "F"] | None = None
    education_years: float | None = None
    ses: int | None = None
    bmi: float | None = None

    smoking: float | None = None
    alcohol: float | None = None
    physical_activity: float | None = None
    diet_quality: float | None = None
    sleep_quality: float | None = None

    family_history_ad: int | None = None
    cardiovascular: int | None = None
    diabetes: int | None = None
    depression: int | None = None
    head_injury: int | None = None
    hypertension: int | None = None
    systolic_bp: float | None = None
    diastolic_bp: float | None = None

    mmse: float | None = None
    adas11: float | None = None
    adas13: float | None = None
    cdr: float | None = None
    cdrsb: float | None = None
    adl: float | None = None
    functional_assessment: float | None = None
    memory_complaints: int | None = None
    behavioral_problems: int | None = None
    confusion: int | None = None
    disorientation: int | None = None
    personality_changes: int | None = None
    difficulty_completing_tasks: int | None = None
    forgetfulness: int | None = None
    log_mem_immediate: float | None = None
    log_mem_delayed: float | None = None
    gds_total: float | None = None
    ravlt_immediate: float | None = None
    ravlt_learning: float | None = None
    ravlt_forgetting: float | None = None
    ravlt_perc_forgetting: float | None = None
    hmse: float | None = None

    cholesterol_total: float | None = None
    cholesterol_ldl: float | None = None
    cholesterol_hdl: float | None = None
    cholesterol_triglycerides: float | None = None
    hba1c: float | None = None
    fasting_glucose: float | None = None
    crp: float | None = None
    hemoglobin: float | None = None

    etiv: float | None = None
    n_wbv: float | None = None
    asf: float | None = None
    hippocampus_l: float | None = None
    hippocampus_r: float | None = None
    ventricles: float | None = None
    whole_brain: float | None = None
    entorhinal: float | None = None
    fusiform: float | None = None
    mid_temp: float | None = None
    mri_vol_features: dict[str, float] = Field(default_factory=dict)

    mri_t1_uri: str | None = None
    mri_stripped_uri: str | None = None
    mri_seg_uri: str | None = None
    mri_brainiac_uri: str | None = None

    apoe4: int | None = None
    high_risk_apoe4: int | None = None

    csf_amyloid: float | None = None
    csf_total_tau: float | None = None
    csf_p_tau: float | None = None
    plasma_lipid_panel: list[float] = Field(default_factory=list)

    diagnosis: DiagnosisLabel | None = None
    diagnosis_source: DiagnosisSource | None = None
    diagnosis_confidence: float = 1.0
    converted_within_2y: int | None = None
    converted_within_5y: int | None = None
    converted_within_10y: int | None = None
    conversion_confidence: float | None = None
    time_to_event_y: float | None = None
    censored: int | None = None

    available_modalities: list[ModalityName] = Field(default_factory=list)
    is_rich: bool = False
    qc_status: QCStatus | None = None
    qc_reasons: list[str] = Field(default_factory=list)
    record_version: str = ""

    source_group: str | None = None
    source_visit_delay_days: float | None = None
    progression_to_ad: str | None = None
    progression_time_months: float | None = None

    @field_validator("subject_id")
    @classmethod
    def _id_is_namespaced(cls, value: str) -> str:
        if ":" not in value:
            raise ValueError("subject_id must be namespaced like 'ADNI:0002'")
        return value

    @field_validator("visit_idx")
    @classmethod
    def _visit_is_nonnegative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("visit_idx must be non-negative")
        return value

    @field_validator("available_modalities")
    @classmethod
    def _modalities_are_sorted(cls, value: list[ModalityName]) -> list[ModalityName]:
        return sorted(set(value))
