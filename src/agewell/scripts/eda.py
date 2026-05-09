"""Generate the Phase 1 data profile HTML."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import pandas as pd
from ydata_profiling import ProfileReport

from agewell.config import load_cfg, repo_root


def main() -> None:
    """CLI entrypoint."""
    cfg = load_cfg()
    master_path = _repo_path(str(cfg.data.master_path))
    report_path = master_path.parent / "quality_report.json"
    out_path = repo_root() / "docs" / "data_profile.html"
    df = pd.read_parquet(master_path)
    quality = json.loads(report_path.read_text(encoding="utf-8"))
    profile_df = _profile_frame(df)
    profile = ProfileReport(
        profile_df,
        title="AgeWell-IN Phase 1 Data Profile",
        minimal=True,
        explorative=False,
    )
    profile_html = profile.to_html()
    summary = _summary_html(quality)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(profile_html.replace("<body>", f"<body>{summary}", 1), encoding="utf-8")
    print(f"wrote {out_path}")


def _profile_frame(df: pd.DataFrame) -> pd.DataFrame:
    profile = df.copy()
    profile["modality_pattern"] = profile["available_modalities"].apply(lambda v: "+".join(v))
    profile["n_modalities"] = profile["available_modalities"].apply(len)
    profile["mri_vol_feature_count"] = profile.filter(regex=r"^mri_vol__").notna().sum(axis=1)
    drop_prefixes = ("mri_vol__",)
    drop_exact = {
        "available_modalities",
        "qc_reasons",
        "plasma_lipid_panel",
        "mri_t1_uri",
        "mri_stripped_uri",
        "mri_seg_uri",
        "mri_brainiac_uri",
        "record_version",
    }
    columns = [
        column
        for column in profile.columns
        if column not in drop_exact and not column.startswith(drop_prefixes)
    ]
    return profile[columns]


def _summary_html(quality: dict[str, Any]) -> str:
    rows = [
        ("Rows", quality["row_count"]),
        ("Rich rows", quality["is_rich_count"]),
        ("MRI T1 rows", quality["mri_t1_uri_count"]),
        ("ADNI tabular/NIfTI joins", quality["adni_merge"]["joined_adni_nifti"]),
        ("ADNI NIfTI-only rows", quality["adni_merge"]["standalone_adni_nifti"]),
    ]
    table = "".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>"
        for key, value in rows
    )
    return (
        "<section style='max-width: 1100px; margin: 24px auto; font-family: sans-serif;'>"
        "<h1>AgeWell-IN Phase 1 Quality Summary</h1>"
        f"<table>{table}</table>"
        "<p>See <code>data/quality_report.json</code> for full counts, skipped rows, "
        "modality patterns, and split ratios.</p>"
        "</section>"
    )


def _repo_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else repo_root() / candidate


if __name__ == "__main__":
    main()
