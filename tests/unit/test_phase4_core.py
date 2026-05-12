"""Tests for Phase 4 fusion primitives."""

from __future__ import annotations

import torch

from agewell.ml.losses import cdr_to_class, compute_phase4_loss
from agewell.ml.modality_dropout import sample_modality_mask, scheduled_modality_dropout_p
from agewell.ml.permod_moe import PerModMoEBlock
from agewell.ml.transformer import ModalityMaskedTransformer


def test_modality_dropout_preserves_one_present_modality_per_row() -> None:
    presence = torch.tensor(
        [
            [True, False, True],
            [False, False, False],
            [True, True, True],
        ]
    )

    kept = sample_modality_mask(presence, p_drop=1.0, generator=torch.Generator().manual_seed(4))

    assert kept[0].sum().item() == 1
    assert kept[1].sum().item() == 0
    assert kept[2].sum().item() == 1
    assert (kept & ~presence).sum().item() == 0
    assert (
        scheduled_modality_dropout_p(
            global_step=50,
            total_steps=100,
            warmup_steps=0,
            max_p=0.4,
        )
        == 0.2
    )


def test_permod_moe_routes_only_unmasked_tokens() -> None:
    block = PerModMoEBlock(
        d_model=8,
        n_modalities=3,
        n_experts=4,
        top_k=2,
        expert_ff_mult=2,
        dropout=0.0,
    )
    tokens = torch.randn(2, 3, 8)
    key_padding_mask = torch.tensor([[False, True, False], [False, False, True]])

    update, aux = block(
        tokens,
        modality_ids=torch.tensor([0, 1, 2]),
        key_padding_mask=key_padding_mask,
    )

    assert update.shape == tokens.shape
    assert torch.isfinite(update).all()
    assert torch.isfinite(aux)
    assert torch.equal(update[0, 1], torch.zeros(8))
    assert torch.equal(update[1, 2], torch.zeros(8))
    assert block.last_stats is not None
    assert block.last_stats.active_tokens == 4


def test_masked_transformer_keeps_cls_and_masks_absent_modalities() -> None:
    transformer = ModalityMaskedTransformer(
        n_modalities=3,
        d_model=8,
        n_layers=2,
        n_heads=2,
        n_experts=4,
        top_k=2,
        expert_ff_mult=2,
        dropout=0.0,
    )
    presence = torch.tensor([[True, False, True], [False, True, False]])

    out = transformer(
        torch.randn(2, 3, 8),
        modality_ids=torch.tensor([0, 1, 2]),
        presence_mask=presence,
    )

    assert out["cls"].shape == (2, 8)
    assert out["tokens"].shape == (2, 3, 8)
    assert out["key_padding_mask"].tolist() == [
        [False, False, True, False],
        [False, True, False, True],
    ]
    assert torch.isfinite(out["aux_loss"])


def test_phase4_loss_handles_partial_targets_reconstruction_and_distillation() -> None:
    batch = {
        "diag_label": torch.tensor([0, 4, -1]),
        "label_confidence_weight": torch.tensor([1.0, 0.5, 1.0]),
        "surv_bin": torch.tensor([0, 4, 1]),
        "has_survival": torch.tensor([True, False, True]),
        "mmse": torch.tensor([29.0, 0.0, 24.0]),
        "has_mmse": torch.tensor([True, False, True]),
        "cdr": torch.tensor([0.0, 0.5, 2.0]),
        "has_cdr": torch.tensor([True, True, False]),
    }
    outputs = {
        "cls": torch.randn(3, 8),
        "diag_logits": torch.randn(3, 5),
        "surv_logits": torch.randn(3, 5),
        "mmse_pred": torch.randn(3),
        "cdr_logits": torch.randn(3, 5),
        "recon": {"clinical_demo": torch.randn(3, 8)},
        "original_tokens": {"clinical_demo": torch.randn(3, 8)},
        "presence_orig": torch.tensor([[True], [True], [False]]),
        "presence_now": torch.tensor([[False], [True], [False]]),
        "modality_names": ("clinical_demo",),
        "aux_loss": torch.tensor(0.3),
    }
    teacher = {
        "cls": torch.randn(3, 8),
        "diag_logits": torch.randn(3, 5),
        "surv_logits": torch.randn(3, 5),
    }
    weights = {
        "diag": 1.0,
        "surv": 1.0,
        "mmse": 0.2,
        "cdr": 0.5,
        "recon": 0.1,
        "distill_logits": 0.3,
        "distill_embedding": 0.2,
        "load_balance": 0.01,
    }

    loss, parts = compute_phase4_loss(outputs, batch, weights=weights, teacher_outputs=teacher)

    assert torch.isfinite(loss)
    assert parts["diag"] > 0
    assert parts["surv"] > 0
    assert parts["recon"] > 0
    assert parts["distill_logits"] >= 0
    assert cdr_to_class(torch.tensor([0.0, 0.5, 1.0, 2.0, 3.0])).tolist() == [
        0,
        1,
        2,
        3,
        4,
    ]
