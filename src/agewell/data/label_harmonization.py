"""Diagnosis and survival-label harmonization rules."""

from __future__ import annotations

import pandas as pd

from agewell.data.schema import DiagnosisLabel, DiagnosisSource


def _norm(raw: object) -> str:
    return str(raw).strip()


def canonicalize_diagnosis(
    raw: object, cohort: str
) -> tuple[DiagnosisLabel, float, DiagnosisSource]:
    """Return canonical diagnosis, confidence, and source kind for a cohort."""
    value = _norm(raw)

    if cohort == "ADNI_TAB":
        mapping: dict[str, DiagnosisLabel] = {
            "CN": "CN",
            "SMC": "SMC",
            "EMCI": "EMCI",
            "LMCI": "LMCI",
            "AD": "AD",
        }
        return mapping[value], 1.0, "explicit"

    if cohort == "ADNI_NIFTI":
        mapping = {"CN": "CN", "AD": "AD", "MCI": "EMCI"}
        confidence = 0.85 if value == "MCI" else 1.0
        return mapping[value], confidence, "explicit"

    if cohort in ("OASIS1", "OASIS2", "BRSDINCER"):
        cdr = float(value)
        if cdr == 0.0:
            return "CN", 0.95, "cdr_derived"
        if cdr == 0.5:
            return "EMCI", 0.85, "cdr_derived"
        if cdr == 1.0:
            return "AD", 0.90, "cdr_derived"
        if cdr >= 2.0:
            return "AD", 0.95, "cdr_derived"
        raise ValueError(f"Unhandled CDR value {cdr}")

    if cohort == "RABIE":
        return ("AD" if int(float(value)) == 1 else "CN"), 0.70, "binary_derived"

    if cohort == "LIPID":
        mapping = {
            "Control": "CN",
            "CN": "CN",
            "Mild Cognitive Impairment": "EMCI",
            "MCI": "EMCI",
            "Alzheimer's Disease": "AD",
        }
        return mapping[value], 0.90, "explicit"

    if cohort == "IXI":
        return "CN", 1.0, "explicit"

    raise ValueError(f"Unhandled cohort {cohort}")


def compute_conversion_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Populate conversion/survival labels with explicit per-cohort rules."""
    out = df.copy()
    for column in (
        "converted_within_2y",
        "converted_within_5y",
        "converted_within_10y",
        "conversion_confidence",
        "time_to_event_y",
        "censored",
    ):
        if column not in out:
            out[column] = pd.NA

    adni = out["cohort"].eq("ADNI_TAB")
    adni_ad = adni & out["diagnosis"].eq("AD")
    adni_cn = adni & out["diagnosis"].eq("CN")
    out.loc[adni_ad, ["converted_within_5y", "censored", "conversion_confidence"]] = [
        1,
        0,
        0.85,
    ]
    out.loc[adni_cn, ["converted_within_5y", "censored", "conversion_confidence"]] = [
        0,
        1,
        0.95,
    ]
    out.loc[adni & ~(adni_ad | adni_cn), ["censored", "conversion_confidence"]] = [1, 0.85]

    oasis2 = out["cohort"].eq("OASIS2")
    if oasis2.any():
        for _, group in out[oasis2].groupby("subject_id", sort=False):
            converted = group["source_group"].astype("string").eq("Converted")
            subject_index = group.index
            if converted.any():
                converted_rows = group[converted]
                conversion_delay = float(converted_rows["source_visit_delay_days"].min())
                for idx, row in group.iterrows():
                    current_delay = row.get("source_visit_delay_days")
                    if pd.isna(current_delay):
                        continue
                    years = max((conversion_delay - float(current_delay)) / 365.25, 0.0)
                    out.loc[idx, "time_to_event_y"] = years
                    out.loc[idx, "censored"] = 0
                    out.loc[idx, "conversion_confidence"] = 0.90
                    for horizon in (2, 5, 10):
                        out.loc[idx, f"converted_within_{horizon}y"] = int(years <= horizon)
            else:
                out.loc[subject_index, ["censored", "conversion_confidence"]] = [1, 0.90]

    lipid = out["cohort"].eq("LIPID")
    if lipid.any():
        for idx, row in out[lipid].iterrows():
            progression = str(row.get("progression_to_ad") or "").strip()
            months = row.get("progression_time_months")
            out.loc[idx, "conversion_confidence"] = 0.90
            if progression == "Yes" and pd.notna(months):
                years = float(months) / 12.0
                out.loc[idx, "time_to_event_y"] = years
                out.loc[idx, "censored"] = 0
                for horizon in (2, 5, 10):
                    out.loc[idx, f"converted_within_{horizon}y"] = int(years <= horizon)
            elif progression == "No":
                out.loc[idx, "censored"] = 1
                for horizon in (2, 5, 10):
                    out.loc[idx, f"converted_within_{horizon}y"] = 0

    remaining = out["censored"].isna()
    out.loc[remaining, "censored"] = 1
    return out
