"""Local Phase 4 end-to-end smoke runner."""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping
from typing import Any

import lightning as L
import torch
from torch import Tensor, nn

from agewell.config import load_cfg
from agewell.ml.config_utils import cfg_select
from agewell.ml.datamodule import AgeWellDataModule
from agewell.ml.encoders._base import D_TOKEN, BaseEncoder
from agewell.ml.lightning_module import AgeWellINLightning
from agewell.ml.modality_registry import ModalityConfig, build_default_registry
from agewell.ml.model import count_trainable_parameters


class _SmokeEncoder(BaseEncoder):
    """Tiny trainable encoder used only for fast Phase 4 integration smoke tests."""

    def __init__(self, *, modality: str, presence_key: str, out_dim: int = D_TOKEN) -> None:
        super().__init__()
        self.modality = modality
        self.presence_key = presence_key
        self.out_dim = out_dim
        self.proj = nn.Linear(1, out_dim)

    def forward(self, batch: dict[str, Any]) -> Tensor:
        presence = batch[self.presence_key]
        tensor = presence if isinstance(presence, Tensor) else torch.as_tensor(presence)
        return self.proj(tensor.float().unsqueeze(-1))


def main() -> None:
    """Run a local Phase 4 train-step smoke."""
    args = _parse_args()
    if args.allow_cpu_large_tabpfn:
        os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"
    overrides = [
        f"training.batch_size={args.batch_size}",
        f"training.total_steps={args.max_steps}",
        f"training.precision={args.precision}",
        f"training.subset={args.subset}",
    ]
    cfg = load_cfg(overrides=overrides)
    L.seed_everything(int(cfg_select(cfg, "seed", 1337)), workers=True)

    dm = AgeWellDataModule.from_cfg(cfg, strict_mri_paths=not args.relaxed_mri_paths)
    dm.setup("fit")
    registry = build_default_registry()
    encoders: Mapping[str, BaseEncoder] | None = (
        _build_smoke_encoders(registry) if args.fake_encoders else None
    )
    tabpfn_overrides: dict[str, Any] = {}
    if args.tabpfn_n_estimators is not None:
        tabpfn_overrides["n_estimators"] = args.tabpfn_n_estimators
    if args.tabpfn_device is not None:
        tabpfn_overrides["device"] = args.tabpfn_device

    module = AgeWellINLightning(
        cfg,
        context_df=dm.context_df,
        imputation_stats=dm.imputation_stats,
        encoders=encoders,
        tabpfn_overrides=tabpfn_overrides or None,
    )
    trainable = count_trainable_parameters(module.model)
    max_trainable = int(cfg_select(cfg, "model.max_trainable_params", 25_000_000))
    if trainable > max_trainable:
        raise RuntimeError(f"trainable params {trainable:,} exceed gate {max_trainable:,}")
    trainer = L.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_steps=args.max_steps,
        limit_val_batches=args.limit_val_batches,
        enable_checkpointing=False,
        enable_model_summary=True,
        logger=False,
        precision=args.precision,
        gradient_clip_val=float(cfg_select(cfg, "training.gradient_clip_val", 1.0)),
        deterministic=False,
    )
    trainer.fit(module, datamodule=dm)
    print(
        "phase4 smoke OK "
        f"trainable_params={trainable} "
        f"fake_encoders={int(args.fake_encoders)} "
        f"steps={args.max_steps}"
    )


def _build_smoke_encoders(registry: list[ModalityConfig]) -> dict[str, BaseEncoder]:
    return {
        config.name: _SmokeEncoder(modality=config.name, presence_key=config.presence_key)
        for config in registry
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--subset", default="all")
    parser.add_argument("--accelerator", default="cpu")
    parser.add_argument("--devices", default="1")
    parser.add_argument("--precision", default="32-true")
    parser.add_argument("--limit-val-batches", type=int, default=1)
    parser.add_argument("--fake-encoders", action="store_true")
    parser.add_argument("--relaxed-mri-paths", action="store_true")
    parser.add_argument("--tabpfn-n-estimators", type=int, default=None)
    parser.add_argument("--tabpfn-device", default=None)
    parser.add_argument("--allow-cpu-large-tabpfn", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
