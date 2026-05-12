"""Tests for Phase 3 modality registry wiring."""

from __future__ import annotations

import numpy as np
import torch

from agewell.data.schema import MODALITIES
from agewell.ml.encoders import (
    APOEEncoder,
    BrainIACCachedEncoder,
    LipidEncoder,
    TabPFNFrozenEncoder,
)
from agewell.ml.modality_registry import (
    MOD_IDX,
    MODALITY_LIST,
    N_MODALITIES,
    build_default_registry,
    registry_by_name,
)


def test_registry_matches_schema_order_and_feature_dimensions() -> None:
    registry = build_default_registry()
    by_name = registry_by_name(registry)

    assert tuple(MODALITIES) == MODALITY_LIST
    assert [config.name for config in registry] == list(MODALITIES)
    assert N_MODALITIES == 9
    assert MOD_IDX["clinical_demo"] == 0
    assert MOD_IDX["lipid"] == 8
    assert by_name["clinical_demo"].n_features == 5
    assert by_name["clinical_lifestyle"].n_features == 5
    assert by_name["clinical_comorbid"].n_features == 8
    assert by_name["cognitive"].n_features == 22
    assert by_name["blood"].n_features == 8
    assert by_name["mri_vol"].n_features == 328
    assert by_name["mri_raw"].n_features == 768
    assert by_name["genetic"].n_features is None
    assert by_name["lipid"].n_features == 213


def test_registry_builds_non_tabpfn_encoders() -> None:
    by_name = registry_by_name()

    assert isinstance(by_name["mri_raw"].build_encoder(), BrainIACCachedEncoder)
    assert isinstance(by_name["genetic"].build_encoder(), APOEEncoder)
    assert isinstance(by_name["lipid"].build_encoder(), LipidEncoder)


def test_registry_builds_tabpfn_encoder_from_explicit_context() -> None:
    by_name = registry_by_name()

    encoder = by_name["clinical_demo"].build_encoder(
        ctx_X=torch.zeros(4, 5),
        ctx_y=torch.tensor([0, 1, 0, 1]),
        embedding_dim=3,
        out_dim=4,
        embedding_backend=lambda x: np.ones((x.shape[0], 3), dtype=np.float32),
    )

    assert isinstance(encoder, TabPFNFrozenEncoder)
    out = encoder({"clinical_demo_features": torch.zeros(2, 5)})
    assert out.shape == (2, 4)
    assert torch.isfinite(out).all()
