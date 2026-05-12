"""Full Phase 4 AgeWell-IN fusion model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd
import torch
from torch import Tensor, nn

from agewell.ml.config_utils import cfg_select
from agewell.ml.encoders._base import D_TOKEN, BaseEncoder
from agewell.ml.heads import HeadStack
from agewell.ml.modality_dropout import sample_modality_mask
from agewell.ml.modality_registry import (
    MOD_IDX,
    N_MODALITIES,
    ModalityConfig,
    build_default_registry,
)
from agewell.ml.transformer import ModalityMaskedTransformer


class AgeWellINModel(nn.Module):
    """Sparse-modality transformer with frozen external encoders and trainable fusion."""

    def __init__(
        self,
        *,
        model_cfg: Any | None = None,
        registry: Sequence[ModalityConfig] | None = None,
        context_df: pd.DataFrame | None = None,
        imputation_stats: dict[str, dict[str, float]] | None = None,
        encoders: Mapping[str, BaseEncoder] | None = None,
        tabpfn_overrides: Mapping[str, Any] | None = None,
        seed: int = 1337,
    ) -> None:
        super().__init__()
        self.registry = list(registry or build_default_registry())
        self.modality_names = tuple(config.name for config in self.registry)
        self.d_model = int(cfg_select(model_cfg, "transformer.d", D_TOKEN))
        if self.d_model != D_TOKEN:
            raise ValueError(f"Phase 4 expects D_TOKEN={D_TOKEN}; got {self.d_model}")
        self.encoders = self._build_encoders(
            context_df=context_df,
            imputation_stats=imputation_stats,
            encoders=encoders,
            tabpfn_overrides=tabpfn_overrides,
            seed=seed,
        )
        self.transformer = ModalityMaskedTransformer(
            n_modalities=len(self.registry),
            d_model=self.d_model,
            n_layers=int(cfg_select(model_cfg, "transformer.n_layers", 4)),
            n_heads=int(cfg_select(model_cfg, "transformer.n_heads", 8)),
            n_experts=int(cfg_select(model_cfg, "moe.n_experts", 8)),
            top_k=int(cfg_select(model_cfg, "moe.top_k", 2)),
            expert_ff_mult=int(cfg_select(model_cfg, "moe.expert_ff_mult", 4)),
            dropout=float(cfg_select(model_cfg, "transformer.dropout", 0.1)),
        )
        self.heads = HeadStack(
            d_model=self.d_model,
            n_classes=int(cfg_select(model_cfg, "heads.n_classes", 5)),
            n_surv_bins=int(cfg_select(model_cfg, "heads.n_surv_bins", 4)),
            modality_names=self.modality_names,
            recon_hidden=int(cfg_select(model_cfg, "heads.recon_hidden", 256)),
        )
        self.default_modality_dropout_p = 0.0

    def forward(
        self,
        batch: Mapping[str, Any],
        *,
        modality_dropout_p: float | None = None,
        presence_override: Tensor | str | None = None,
    ) -> dict[str, Any]:
        """Encode available modalities, apply optional masking, and run heads."""
        encoded = self._encode_modalities(batch)
        tokens = encoded["tokens"]
        presence_orig = encoded["presence"]
        if presence_override is None:
            p_drop = (
                self.default_modality_dropout_p
                if modality_dropout_p is None
                else modality_dropout_p
            )
            presence_now = (
                sample_modality_mask(presence_orig, p_drop=p_drop)
                if self.training and p_drop > 0.0
                else presence_orig.clone()
            )
        elif isinstance(presence_override, Tensor):
            presence_now = presence_override.to(device=tokens.device, dtype=torch.bool)
        elif presence_override in {"observed", "full"}:
            presence_now = presence_orig.clone()
        else:
            raise ValueError(f"Unsupported presence_override: {presence_override!r}")
        if presence_now.shape != presence_orig.shape:
            raise ValueError(
                f"presence override shape mismatch: expected {tuple(presence_orig.shape)}, "
                f"got {tuple(presence_now.shape)}"
            )

        masked_tokens = tokens * presence_now.unsqueeze(-1).to(dtype=tokens.dtype)
        modality_ids = torch.tensor(
            [MOD_IDX[name] for name in self.modality_names],
            device=tokens.device,
            dtype=torch.long,
        )
        fused = self.transformer(
            masked_tokens,
            modality_ids=modality_ids,
            presence_mask=presence_now,
        )
        head_out = self.heads(fused["cls"], fused["tokens"])
        out: dict[str, Any] = {
            **head_out,
            "cls": fused["cls"],
            "tokens": fused["tokens"],
            "key_padding_mask": fused["key_padding_mask"],
            "aux_loss": fused["aux_loss"],
            "presence_orig": presence_orig,
            "presence_now": presence_now,
            "original_tokens": encoded["tokens_by_name"],
            "modality_names": self.modality_names,
        }
        return out

    def _build_encoders(
        self,
        *,
        context_df: pd.DataFrame | None,
        imputation_stats: dict[str, dict[str, float]] | None,
        encoders: Mapping[str, BaseEncoder] | None,
        tabpfn_overrides: Mapping[str, Any] | None,
        seed: int,
    ) -> nn.ModuleDict:
        if encoders is not None:
            missing = sorted(set(self.modality_names) - set(encoders))
            if missing:
                raise ValueError(f"Missing encoders for modalities: {missing}")
            return nn.ModuleDict({name: encoders[name] for name in self.modality_names})

        built: dict[str, BaseEncoder] = {}
        for config in self.registry:
            overrides = (
                dict(tabpfn_overrides or {}) if config.encoder_type == "tabpfn_frozen" else {}
            )
            built[config.name] = config.build_encoder(
                context_df=context_df,
                imputation_stats=imputation_stats,
                seed=seed,
                **overrides,
            )
        return nn.ModuleDict(built)

    def _encode_modalities(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        tokens_by_name: dict[str, Tensor] = {}
        token_list: list[Tensor] = []
        presence_list: list[Tensor] = []
        device: torch.device | None = None
        for config in self.registry:
            token = self.encoders[config.name](dict(batch))
            if token.ndim != 2 or token.shape[1] != self.d_model:
                raise ValueError(
                    f"{config.name} encoder returned {tuple(token.shape)}, "
                    f"expected (B, {self.d_model})"
                )
            device = token.device if device is None else device
            token = token.to(device=device)
            presence = _batch_tensor(batch, config.presence_key).to(device=device, dtype=torch.bool)
            if presence.shape != (token.shape[0],):
                raise ValueError(
                    f"{config.presence_key} must have shape ({token.shape[0]},); "
                    f"got {tuple(presence.shape)}"
                )
            tokens_by_name[config.name] = token
            token_list.append(token)
            presence_list.append(presence)
        return {
            "tokens": torch.stack(token_list, dim=1),
            "presence": torch.stack(presence_list, dim=1),
            "tokens_by_name": tokens_by_name,
        }


def count_trainable_parameters(module: nn.Module) -> int:
    """Return trainable parameter count."""
    return sum(param.numel() for param in module.parameters() if param.requires_grad)


def _batch_tensor(batch: Mapping[str, Any], key: str) -> Tensor:
    value = batch[key]
    return value if isinstance(value, Tensor) else torch.as_tensor(value)


__all__ = ["N_MODALITIES", "AgeWellINModel", "count_trainable_parameters"]
