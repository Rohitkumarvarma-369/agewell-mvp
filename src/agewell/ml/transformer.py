"""Masked transformer fusion stack for Phase 4."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from agewell.ml.permod_moe import PerModMoEBlock


class FusionTransformerLayer(nn.Module):
    """One pre-norm attention plus sparse MoE layer."""

    def __init__(
        self,
        *,
        d_model: int = 256,
        n_heads: int = 8,
        n_modalities: int,
        n_experts: int = 8,
        top_k: int = 2,
        expert_ff_mult: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_drop = nn.Dropout(dropout)
        self.moe_norm = nn.LayerNorm(d_model)
        self.moe = PerModMoEBlock(
            d_model=d_model,
            n_modalities=n_modalities,
            n_experts=n_experts,
            top_k=top_k,
            expert_ff_mult=expert_ff_mult,
            dropout=dropout,
        )
        self.moe_drop = nn.Dropout(dropout)

    def forward(
        self,
        tokens: Tensor,
        *,
        modality_ids: Tensor,
        key_padding_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Apply one masked transformer fusion layer."""
        normed = self.attn_norm(tokens)
        attn_out, _ = self.attn(
            normed,
            normed,
            normed,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        tokens = tokens + self.attn_drop(attn_out)
        moe_update, aux_loss = self.moe(
            self.moe_norm(tokens),
            modality_ids=modality_ids,
            key_padding_mask=key_padding_mask,
        )
        tokens = tokens + self.moe_drop(moe_update)
        return tokens, aux_loss


class ModalityMaskedTransformer(nn.Module):
    """Fuse observed modality tokens with a learnable CLS token."""

    def __init__(
        self,
        *,
        n_modalities: int,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        n_experts: int = 8,
        top_k: int = 2,
        expert_ff_mult: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        self.n_modalities = n_modalities
        self.cls_modality_id = n_modalities
        self.d_model = d_model
        router_modalities = n_modalities + 1
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.modality_embedding = nn.Embedding(router_modalities, d_model)
        self.layers = nn.ModuleList(
            FusionTransformerLayer(
                d_model=d_model,
                n_heads=n_heads,
                n_modalities=router_modalities,
                n_experts=n_experts,
                top_k=top_k,
                expert_ff_mult=expert_ff_mult,
                dropout=dropout,
            )
            for _ in range(n_layers)
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        modality_tokens: Tensor,
        *,
        modality_ids: Tensor,
        presence_mask: Tensor,
    ) -> dict[str, Any]:
        """Fuse modality tokens.

        Args:
            modality_tokens: ``(B, M, D)`` encoded modality tokens.
            modality_ids: ``(M,)`` integer modality ids matching token order.
            presence_mask: ``(B, M)`` where ``True`` means observed.
        """
        if modality_tokens.ndim != 3:
            raise ValueError(
                f"modality_tokens must have shape (B, M, D); got {tuple(modality_tokens.shape)}"
            )
        batch_size, n_modalities, d_model = modality_tokens.shape
        if n_modalities != self.n_modalities:
            raise ValueError(f"expected {self.n_modalities} modalities, got {n_modalities}")
        if d_model != self.d_model:
            raise ValueError(f"expected d_model={self.d_model}, got {d_model}")
        if modality_ids.shape != (n_modalities,):
            raise ValueError(
                f"modality_ids must have shape ({n_modalities},); got {tuple(modality_ids.shape)}"
            )
        if presence_mask.shape != (batch_size, n_modalities):
            raise ValueError(
                f"presence_mask must have shape ({batch_size}, {n_modalities}); "
                f"got {tuple(presence_mask.shape)}"
            )

        device = modality_tokens.device
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, modality_tokens], dim=1)
        ids_with_cls = torch.cat(
            [
                torch.tensor([self.cls_modality_id], device=device, dtype=torch.long),
                modality_ids.to(device=device, dtype=torch.long),
            ],
            dim=0,
        )
        tokens = tokens + self.modality_embedding(ids_with_cls).unsqueeze(0)
        key_padding_mask = torch.cat(
            [
                torch.zeros(batch_size, 1, dtype=torch.bool, device=device),
                ~presence_mask.to(device=device, dtype=torch.bool),
            ],
            dim=1,
        )

        aux_losses: list[Tensor] = []
        for layer in self.layers:
            tokens, aux_loss = layer(
                tokens,
                modality_ids=ids_with_cls,
                key_padding_mask=key_padding_mask,
            )
            aux_losses.append(aux_loss)
        tokens = self.final_norm(tokens)
        aux = torch.stack(aux_losses).mean() if aux_losses else tokens.new_zeros(())
        return {
            "cls": tokens[:, 0],
            "tokens": tokens[:, 1:],
            "key_padding_mask": key_padding_mask,
            "aux_loss": aux,
        }
