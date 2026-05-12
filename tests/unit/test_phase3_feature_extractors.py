"""Tests for Phase 3 master-row feature extraction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from agewell.ml.feature_extractors import (
    FEATURE_NAMES,
    MRI_RAW_DIM,
    build_tabpfn_context,
    compose_batch,
    compute_imputation_stats,
)


def test_compose_batch_shapes_presence_and_targets(tmp_path) -> None:
    vector_path = tmp_path / "brainiac.npy"
    np.save(vector_path, np.arange(MRI_RAW_DIM, dtype=np.float32))
    mri_vol_name = FEATURE_NAMES["mri_vol"][0]

    rows = pd.DataFrame(
        [
            {
                "subject_id": "ADNI:0001",
                "cohort": "ADNI_TAB",
                "age": 72.0,
                "sex": "M",
                "education_years": 16.0,
                "mmse": 29.0,
                "diagnosis": "CN",
                "mri_brainiac_uri": str(vector_path),
                mri_vol_name: 123.0,
                "apoe4": 2,
                "available_modalities": [
                    "clinical_demo",
                    "cognitive",
                    "mri_vol",
                    "mri_raw",
                    "genetic",
                ],
                "is_rich": True,
                "converted_within_2y": 1,
                "censored": 0,
            },
            {
                "subject_id": "LIPID:0002",
                "cohort": "LIPID",
                "diagnosis": "AD",
                "high_risk_apoe4": 0,
                "csf_amyloid": 10.0,
                "plasma_lipid_panel": [1.0, 2.0],
                "available_modalities": ["genetic", "lipid"],
                "is_rich": False,
            },
        ],
        index=[10, 20],
    )

    stats = compute_imputation_stats(rows)
    batch = compose_batch(rows, imputation_stats=stats)

    assert batch["clinical_demo_features"].shape == (2, 5)
    assert batch["cognitive_features"].shape == (2, 22)
    assert batch["mri_vol_features"].shape == (2, 328)
    assert batch["mri_raw_features"].shape == (2, MRI_RAW_DIM)
    assert batch["lipid_features"].shape == (2, 213)
    assert batch["clinical_demo_presence"].tolist() == [True, False]
    assert batch["mri_raw_presence"].tolist() == [True, False]
    assert batch["genetic_presence"].tolist() == [True, True]
    assert batch["genetic_apoe4"].tolist() == [2, 0]
    assert batch["diag_label"].tolist() == [0, 4]
    assert batch["surv_bin"].tolist() == [0, 4]
    assert batch["is_rich"].tolist() == [True, False]
    assert torch.isfinite(batch["clinical_demo_features"]).all()


def test_compose_batch_without_available_modalities_stays_row_aligned() -> None:
    rows = pd.DataFrame(
        [
            {"subject_id": "RABIE:1", "cohort": "RABIE", "diagnosis": "SMC"},
            {"subject_id": "RABIE:2", "cohort": "RABIE", "diagnosis": "LMCI"},
        ]
    )

    batch = compose_batch(rows, strict_mri_paths=False)

    assert batch["available_modalities"] == [[], []]
    assert batch["clinical_demo_presence"].tolist() == [False, False]
    assert batch["diag_label"].tolist() == [1, 3]


def test_build_tabpfn_context_is_deterministic_and_bounded() -> None:
    rows = pd.DataFrame(
        [
            {
                "age": 70.0 + idx,
                "sex": "F" if idx % 2 else "M",
                "education_years": 12.0,
                "diagnosis": "CN" if idx % 2 else "AD",
                "available_modalities": ["clinical_demo"],
            }
            for idx in range(8)
        ]
    )

    x1, y1 = build_tabpfn_context(rows, modality="clinical_demo", ctx_size=3, seed=11)
    x2, y2 = build_tabpfn_context(rows, modality="clinical_demo", ctx_size=3, seed=11)

    assert x1.shape == (3, 5)
    assert y1.shape == (3,)
    assert torch.equal(x1, x2)
    assert torch.equal(y1, y2)
