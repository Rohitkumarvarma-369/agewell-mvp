"""Opt-in real TabPFN V2.5 smoke test.

This is intentionally gated because TabPFN local inference requires one-time
license acceptance plus either TABPFN_TOKEN or a cached checkpoint.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest
import torch

from agewell.ml.encoders.tabular_tabpfn import TabPFNFrozenEncoder
from agewell.ml.feature_extractors import build_tabpfn_context, compute_imputation_stats

CHECKPOINT = Path.home() / ".cache/tabpfn/tabpfn-v2.5-classifier-v2.5_default.ckpt"


@pytest.mark.skipif(
    os.getenv("AGEWELL_TABPFN_REAL_SMOKE") != "1",
    reason="set AGEWELL_TABPFN_REAL_SMOKE=1 to run the real TabPFN smoke",
)
@pytest.mark.skipif(
    not os.getenv("TABPFN_TOKEN") and not CHECKPOINT.exists(),
    reason="TabPFN V2.5 local weights require TABPFN_TOKEN or cached checkpoint",
)
def test_real_tabpfn_v25_encoder_smoke() -> None:
    train = pd.read_parquet("data/splits/train.parquet")
    stats = compute_imputation_stats(train)
    ctx_x, ctx_y = build_tabpfn_context(
        train,
        modality="clinical_demo",
        ctx_size=64,
        imputation_stats=stats,
    )
    encoder = TabPFNFrozenEncoder(
        modality="clinical_demo",
        n_features=5,
        ctx_X=ctx_x,
        ctx_y=ctx_y,
        n_estimators=1,
        device="cpu",
    )

    out = encoder({"clinical_demo_features": ctx_x[:4]})

    assert out.shape == (4, 256)
    assert torch.isfinite(out).all()
