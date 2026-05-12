"""Tests for Phase 3 modality encoders."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from agewell.ml.encoders import (
    APOEEncoder,
    BrainIACCachedEncoder,
    LipidEncoder,
    ModalityIDEmbedding,
    TabPFNFrozenEncoder,
)


def test_brainiac_cached_encoder_projects_finite_tokens() -> None:
    encoder = BrainIACCachedEncoder(in_dim=4, out_dim=6)
    batch = {"mri_raw_features": torch.tensor([[1.0, float("nan"), 3.0, 4.0]])}

    out = encoder(batch)

    assert out.shape == (1, 6)
    assert torch.isfinite(out).all()


def test_lipid_encoder_projects_finite_tokens() -> None:
    encoder = LipidEncoder(n_features=5, hidden=8, out_dim=6, dropout=0.0)
    batch = {"lipid_features": torch.tensor([[1.0, 2.0, float("nan"), 4.0, 5.0]])}

    out = encoder(batch)

    assert out.shape == (1, 6)
    assert torch.isfinite(out).all()


def test_apoe_encoder_clamps_allele_counts() -> None:
    encoder = APOEEncoder(out_dim=4)
    batch = {"genetic_apoe4": torch.tensor([-2, 0, 1, 2, 5])}

    out = encoder(batch)

    assert out.shape == (5, 4)
    assert torch.equal(out[0], out[1])
    assert torch.equal(out[3], out[4])


def test_modality_id_embedding_returns_ordered_tokens() -> None:
    embedding = ModalityIDEmbedding(n_modalities=3, out_dim=4)

    out = embedding(batch_size=2)

    assert out.shape == (2, 3, 4)
    assert torch.equal(out[0], out[1])


def test_tabpfn_encoder_uses_backend_and_averages_estimators() -> None:
    def backend(x: np.ndarray) -> np.ndarray:
        assert x.shape == (2, 5)
        low = np.ones((2, 3), dtype=np.float32)
        high = np.full((2, 3), 3.0, dtype=np.float32)
        return np.stack([low, high], axis=0)

    encoder = TabPFNFrozenEncoder(
        modality="clinical_demo",
        n_features=5,
        ctx_X=torch.zeros(4, 5),
        ctx_y=torch.tensor([0, 1, 0, 1]),
        embedding_dim=3,
        out_dim=7,
        embedding_backend=backend,
    )

    out = encoder({"clinical_demo_features": torch.zeros(2, 5)})

    assert out.shape == (2, 7)
    assert torch.isfinite(out).all()


def test_tabpfn_encoder_accepts_batch_first_estimator_embeddings() -> None:
    def backend(x: np.ndarray) -> np.ndarray:
        assert x.shape == (1, 5)
        low = np.ones((1, 3), dtype=np.float32)
        high = np.full((1, 3), 5.0, dtype=np.float32)
        return np.stack([low, high], axis=1)

    encoder = TabPFNFrozenEncoder(
        modality="clinical_demo",
        n_features=5,
        ctx_X=torch.zeros(4, 5),
        ctx_y=torch.tensor([0, 1, 0, 1]),
        embedding_dim=3,
        out_dim=7,
        embedding_backend=backend,
    )

    out = encoder({"clinical_demo_features": torch.zeros(1, 5)})

    assert out.shape == (1, 7)
    assert torch.isfinite(out).all()


def test_tabpfn_encoder_accepts_single_sample_squeezed_estimators() -> None:
    def backend(x: np.ndarray) -> np.ndarray:
        assert x.shape == (1, 5)
        return np.stack(
            [
                np.ones(3, dtype=np.float32),
                np.full(3, 5.0, dtype=np.float32),
            ],
            axis=0,
        )

    encoder = TabPFNFrozenEncoder(
        modality="clinical_demo",
        n_features=5,
        ctx_X=torch.zeros(4, 5),
        ctx_y=torch.tensor([0, 1, 0, 1]),
        embedding_dim=3,
        out_dim=7,
        embedding_backend=backend,
    )

    out = encoder({"clinical_demo_features": torch.zeros(1, 5)})

    assert out.shape == (1, 7)
    assert torch.isfinite(out).all()


def test_tabpfn_encoder_validates_embedding_dim() -> None:
    encoder = TabPFNFrozenEncoder(
        modality="clinical_demo",
        n_features=5,
        ctx_X=torch.zeros(4, 5),
        ctx_y=torch.tensor([0, 1, 0, 1]),
        embedding_dim=3,
        embedding_backend=lambda x: np.zeros((x.shape[0], 2), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="embedding dim mismatch"):
        encoder({"clinical_demo_features": torch.zeros(2, 5)})
