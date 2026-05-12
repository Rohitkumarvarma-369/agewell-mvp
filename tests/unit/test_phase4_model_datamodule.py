"""Tests for the full Phase 4 model/data/trainer path."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch import Tensor, nn

from agewell.config import load_cfg
from agewell.ml.datamodule import AgeWellDataModule
from agewell.ml.encoders._base import D_TOKEN, BaseEncoder
from agewell.ml.feature_extractors import FEATURE_NAMES, MRI_RAW_DIM
from agewell.ml.lightning_module import AgeWellINLightning
from agewell.ml.modality_registry import build_default_registry
from agewell.ml.model import AgeWellINModel, count_trainable_parameters


class FakeEncoder(BaseEncoder):
    """Fast deterministic test encoder keyed only by modality presence."""

    def __init__(self, *, modality: str, presence_key: str, offset: float = 0.0) -> None:
        super().__init__()
        self.modality = modality
        self.presence_key = presence_key
        self.proj = nn.Linear(1, D_TOKEN)
        with torch.no_grad():
            self.proj.weight.fill_(0.01 + offset)
            self.proj.bias.fill_(offset)

    def forward(self, batch: dict[str, Any]) -> Tensor:
        presence = batch[self.presence_key]
        tensor = presence if isinstance(presence, Tensor) else torch.as_tensor(presence)
        return self.proj(tensor.float().unsqueeze(-1))


def test_full_default_phase4_model_forward_and_parameter_gate() -> None:
    cfg = load_cfg()
    registry = build_default_registry()
    model = AgeWellINModel(
        model_cfg=cfg.model,
        encoders=_fake_encoders(registry),
    )
    batch = _fake_tensor_batch(batch_size=3)

    out = model.train()(batch, modality_dropout_p=0.5)
    trainable = count_trainable_parameters(model)

    assert len(model.transformer.layers) == 4
    assert model.transformer.layers[0].moe.n_experts == 8
    assert model.transformer.layers[0].moe.top_k == 2
    assert out["diag_logits"].shape == (3, 5)
    assert out["surv_logits"].shape == (3, 5)
    assert out["tokens"].shape == (3, 9, D_TOKEN)
    assert out["presence_now"].any(dim=1).all()
    assert 18_000_000 < trainable < int(cfg.model.max_trainable_params)


def test_datamodule_loads_splits_and_collates_phase4_batch(tmp_path: Path) -> None:
    split_dir, _ = _write_split_data(tmp_path)
    dm = AgeWellDataModule(
        master_path=tmp_path / "master.parquet",
        splits_dir=split_dir,
        batch_size=2,
        strict_mri_paths=True,
    )

    dm.setup("fit")
    batch = next(iter(dm.train_dataloader()))

    assert len(dm.context_df) == 4
    assert batch["clinical_demo_features"].shape == (2, 5)
    assert batch["mri_vol_features"].shape == (2, 328)
    assert batch["mri_raw_features"].shape == (2, MRI_RAW_DIM)
    assert batch["diag_label"].shape == (2,)


def test_lightning_module_runs_one_train_step(tmp_path: Path) -> None:
    split_dir, _ = _write_split_data(tmp_path)
    cfg = OmegaConf.create(
        {
            "seed": 7,
            "data": {
                "master_path": str(tmp_path / "master.parquet"),
                "splits_dir": str(split_dir),
            },
            "model": {
                "transformer": {"n_layers": 1, "n_heads": 8, "d": D_TOKEN, "dropout": 0.0},
                "moe": {"n_experts": 2, "top_k": 1, "expert_ff_mult": 1},
                "heads": {"n_classes": 5, "n_surv_bins": 4, "recon_hidden": 32},
                "max_trainable_params": 25_000_000,
            },
            "training": {
                "subset": "all",
                "batch_size": 2,
                "num_workers": 0,
                "total_steps": 1,
                "lr": 1.0e-3,
                "wd": 0.0,
                "gradient_clip_val": 1.0,
                "strict_mri_paths": True,
                "modality_dropout": {"max_p": 0.2, "warmup_steps": 0},
                "loss_weights": {
                    "diag": 1.0,
                    "surv": 1.0,
                    "mmse": 0.2,
                    "cdr": 0.5,
                    "recon": 0.1,
                    "distill_logits": 0.0,
                    "distill_embedding": 0.0,
                    "load_balance": 0.01,
                },
            },
        }
    )
    dm = AgeWellDataModule.from_cfg(cfg)
    dm.setup("fit")
    registry = build_default_registry()
    module = AgeWellINLightning(
        cfg,
        context_df=dm.context_df,
        imputation_stats=dm.imputation_stats,
        encoders=_fake_encoders(registry),
    )
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        max_steps=1,
        limit_val_batches=0,
        enable_checkpointing=False,
        logger=False,
        enable_model_summary=False,
        precision="32-true",
    )

    trainer.fit(module, datamodule=dm)

    assert trainer.global_step == 1


def _fake_encoders(registry: list[Any]) -> dict[str, BaseEncoder]:
    return {
        config.name: FakeEncoder(
            modality=config.name,
            presence_key=config.presence_key,
            offset=idx * 0.01,
        )
        for idx, config in enumerate(registry)
    }


def _fake_tensor_batch(batch_size: int) -> dict[str, Any]:
    batch: dict[str, Any] = {}
    for config in build_default_registry():
        batch[config.presence_key] = torch.ones(batch_size, dtype=torch.bool)
    batch["diag_label"] = torch.tensor([0, 1, 4][:batch_size])
    batch["label_confidence_weight"] = torch.ones(batch_size)
    batch["surv_bin"] = torch.tensor([0, 1, 4][:batch_size])
    batch["has_survival"] = torch.ones(batch_size, dtype=torch.bool)
    batch["mmse"] = torch.full((batch_size,), 28.0)
    batch["has_mmse"] = torch.ones(batch_size, dtype=torch.bool)
    batch["cdr"] = torch.zeros(batch_size)
    batch["has_cdr"] = torch.ones(batch_size, dtype=torch.bool)
    return batch


def _write_split_data(tmp_path: Path) -> tuple[Path, Path]:
    vector_path = tmp_path / "brainiac.npy"
    np.save(vector_path, np.ones(MRI_RAW_DIM, dtype=np.float32))
    rows = pd.DataFrame([_row(idx, vector_path) for idx in range(8)])
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    rows.to_parquet(tmp_path / "master.parquet", index=False)
    rows.iloc[:4].to_parquet(split_dir / "train.parquet", index=False)
    rows.iloc[4:6].to_parquet(split_dir / "calib.parquet", index=False)
    rows.iloc[6:].to_parquet(split_dir / "test.parquet", index=False)
    return split_dir, vector_path


def _row(idx: int, vector_path: Path) -> dict[str, Any]:
    mri_vol_name = FEATURE_NAMES["mri_vol"][0]
    return {
        "subject_id": f"T:{idx:03d}",
        "visit_idx": 0,
        "cohort": "ADNI_TAB",
        "age": 70.0 + idx,
        "sex": "M" if idx % 2 else "F",
        "education_years": 16.0,
        "mmse": 29.0 - idx * 0.1,
        "cdr": 0.0 if idx % 2 else 0.5,
        "diagnosis": "CN" if idx % 2 else "AD",
        "diagnosis_confidence": 1.0,
        "converted_within_2y": 0,
        "converted_within_5y": int(idx % 2 == 0),
        "converted_within_10y": 0,
        "censored": int(idx % 2 == 1),
        "mri_brainiac_uri": str(vector_path),
        mri_vol_name: float(idx),
        "apoe4": idx % 3,
        "csf_amyloid": 10.0 + idx,
        "plasma_lipid_panel": [float(idx), float(idx + 1)],
        "available_modalities": [
            "clinical_demo",
            "cognitive",
            "mri_vol",
            "mri_raw",
            "genetic",
            "lipid",
        ],
        "is_rich": True,
    }
