"""Lightning training module for Phase 4."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import lightning as L
import pandas as pd
import torch
from torch import Tensor, nn

from agewell.ml.config_utils import cfg_mapping, cfg_select
from agewell.ml.encoders._base import BaseEncoder
from agewell.ml.losses import compute_phase4_loss
from agewell.ml.modality_dropout import scheduled_modality_dropout_p
from agewell.ml.model import AgeWellINModel


class AgeWellINLightning(L.LightningModule):
    """Full Phase 4 trainer module."""

    def __init__(
        self,
        cfg: Any,
        *,
        context_df: pd.DataFrame | None = None,
        imputation_stats: dict[str, dict[str, float]] | None = None,
        model: AgeWellINModel | None = None,
        teacher: nn.Module | None = None,
        encoders: Mapping[str, BaseEncoder] | None = None,
        tabpfn_overrides: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.loss_weights = cfg_mapping(
            cfg,
            "training.loss_weights",
            {
                "diag": 1.0,
                "surv": 1.0,
                "mmse": 0.2,
                "cdr": 0.5,
                "recon": 0.1,
                "distill_logits": 0.0,
                "distill_embedding": 0.0,
                "load_balance": 0.01,
            },
        )
        self.teacher = teacher
        self.model = model or AgeWellINModel(
            model_cfg=cfg_select(cfg, "model", cfg),
            context_df=context_df,
            imputation_stats=imputation_stats,
            encoders=encoders,
            tabpfn_overrides=tabpfn_overrides,
            seed=int(cfg_select(cfg, "seed", 1337)),
        )

    def forward(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        """Run the wrapped Phase 4 model without training-time dropout."""
        return self.model(batch, modality_dropout_p=0.0, presence_override="observed")

    def training_step(self, batch: Mapping[str, Any], batch_idx: int) -> Tensor:
        """Run one supervised/distillation training step."""
        p_drop = scheduled_modality_dropout_p(
            global_step=int(self.global_step),
            total_steps=int(cfg_select(self.cfg, "training.total_steps", 1)),
            warmup_steps=int(cfg_select(self.cfg, "training.modality_dropout.warmup_steps", 0)),
            max_p=float(cfg_select(self.cfg, "training.modality_dropout.max_p", 0.0)),
        )
        outputs = self.model(batch, modality_dropout_p=p_drop)
        teacher_outputs = self._teacher_outputs(batch)
        loss, parts = compute_phase4_loss(
            outputs,
            batch,
            weights=self.loss_weights,
            teacher_outputs=teacher_outputs,
            distill_temperature=float(cfg_select(self.cfg, "training.distill_temperature", 2.0)),
        )
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/modality_dropout_p", p_drop, on_step=True, on_epoch=False)
        self._log_parts("train", parts)
        return loss

    def validation_step(self, batch: Mapping[str, Any], batch_idx: int) -> Tensor:
        """Run one validation step without modality dropout."""
        outputs = self.model(batch, modality_dropout_p=0.0, presence_override="observed")
        loss, parts = compute_phase4_loss(
            outputs,
            batch,
            weights=self.loss_weights,
            teacher_outputs=None,
            distill_temperature=float(cfg_select(self.cfg, "training.distill_temperature", 2.0)),
        )
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self._log_parts("val", parts)
        return loss

    def configure_optimizers(self) -> Any:
        """Use AdamW with cosine decay over configured total steps."""
        optimizer = torch.optim.AdamW(
            [param for param in self.parameters() if param.requires_grad],
            lr=float(cfg_select(self.cfg, "training.lr", 3.0e-4)),
            weight_decay=float(cfg_select(self.cfg, "training.wd", 1.0e-4)),
        )
        total_steps = max(int(cfg_select(self.cfg, "training.total_steps", 1)), 1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def _teacher_outputs(self, batch: Mapping[str, Any]) -> Mapping[str, Any] | None:
        if self.teacher is None:
            return None
        with torch.no_grad():
            teacher_out = self.teacher(batch)
        if not isinstance(teacher_out, Mapping):
            raise TypeError("teacher must return a mapping of Phase 4 outputs")
        return teacher_out

    def _log_parts(self, prefix: str, parts: Mapping[str, Tensor]) -> None:
        for name, value in parts.items():
            self.log(f"{prefix}/{name}", value, on_step=prefix == "train", on_epoch=True)
