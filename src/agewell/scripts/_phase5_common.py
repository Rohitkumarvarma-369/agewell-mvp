"""Shared helpers for Phase 5 training, calibration, and evaluation scripts."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import lightning as L
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn

from agewell.config import load_cfg, repo_root
from agewell.ml.config_utils import cfg_select
from agewell.ml.datamodule import AgeWellDataModule
from agewell.ml.encoders._base import D_TOKEN, BaseEncoder
from agewell.ml.lightning_module import AgeWellINLightning
from agewell.ml.modality_registry import build_default_registry
from agewell.ml.model import count_trainable_parameters


@dataclass(frozen=True)
class RunArtifacts:
    """Filesystem layout for one Phase 5 run."""

    run_dir: Path
    checkpoints_dir: Path
    metrics_dir: Path
    predictions_dir: Path


class SmokeEncoder(BaseEncoder):
    """Fast deterministic encoder used only by explicit smoke-test CLIs."""

    def __init__(self, *, presence_key: str, offset: float) -> None:
        super().__init__()
        self.modality = presence_key.replace("_presence", "")
        self.presence_key = presence_key
        self.proj = nn.Linear(1, D_TOKEN)
        with torch.no_grad():
            self.proj.weight.fill_(0.01 + offset)
            self.proj.bias.fill_(offset)

    def forward(self, batch: dict[str, Any]) -> Tensor:
        """Encode modality presence into a deterministic token."""
        value = batch[self.presence_key]
        tensor = value if isinstance(value, Tensor) else torch.as_tensor(value)
        return self.proj(tensor.float().unsqueeze(-1))


def add_train_args(parser: argparse.ArgumentParser) -> None:
    """Add common training CLI arguments."""
    parser.add_argument("--run-name", default="")
    parser.add_argument("--output-root", default="")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--accelerator", default=None)
    parser.add_argument("--devices", default=None)
    parser.add_argument("--precision", default=None)
    parser.add_argument("--tabpfn-device", default=None)
    parser.add_argument("--tabpfn-estimators", type=int, default=None)
    parser.add_argument("--limit-val-batches", type=float, default=1.0)
    parser.add_argument("--subset", default=None)
    parser.add_argument("--fake-encoders", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides such as training.lr=1e-4")


def add_eval_args(parser: argparse.ArgumentParser) -> None:
    """Add common evaluation/calibration CLI arguments."""
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("train", "calib", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--tabpfn-device", default=None)
    parser.add_argument("--tabpfn-estimators", type=int, default=None)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--fake-encoders", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides")


def load_phase5_cfg(args: argparse.Namespace, *, default_run_name: str) -> DictConfig:
    """Load Hydra config and apply CLI overrides."""
    cfg = load_cfg(list(getattr(args, "overrides", [])))
    run_name = str(getattr(args, "run_name", "") or cfg_select(cfg, "run_name", default_run_name))
    OmegaConf.update(cfg, "run_name", run_name, merge=True)
    if getattr(args, "output_root", ""):
        OmegaConf.update(cfg, "output_root", str(args.output_root), merge=True)
    for arg_name, cfg_key in (
        ("batch_size", "training.batch_size"),
        ("num_workers", "training.num_workers"),
        ("subset", "training.subset"),
        ("accelerator", "training.trainer.accelerator"),
        ("devices", "training.trainer.devices"),
        ("precision", "training.precision"),
    ):
        value = getattr(args, arg_name, None)
        if value is not None:
            OmegaConf.update(cfg, cfg_key, _devices_value(value), merge=True)
    return cfg


def prepare_run(cfg: DictConfig, *, overwrite: bool = False) -> RunArtifacts:
    """Create the local output layout for a run."""
    root = _repo_path(str(cfg_select(cfg, "output_root", "outputs")))
    run_name = str(cfg_select(cfg, "run_name", _timestamped_name("phase5")))
    run_dir = root / "runs" / run_name
    if run_dir.exists() and not overwrite:
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts = RunArtifacts(
        run_dir=run_dir,
        checkpoints_dir=run_dir / "checkpoints",
        metrics_dir=run_dir / "metrics",
        predictions_dir=run_dir / "predictions",
    )
    for path in (artifacts.checkpoints_dir, artifacts.metrics_dir, artifacts.predictions_dir):
        path.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, artifacts.run_dir / "config.yaml", resolve=True)
    write_json(artifacts.run_dir / "environment.json", environment_record())
    return artifacts


def build_datamodule(cfg: DictConfig) -> AgeWellDataModule:
    """Build and set up the Phase 5 datamodule."""
    dm = AgeWellDataModule.from_cfg(cfg)
    dm.setup("fit")
    return dm


def update_total_steps(cfg: DictConfig, dm: AgeWellDataModule, args: argparse.Namespace) -> None:
    """Set scheduler steps from CLI max steps or estimated epoch length."""
    if getattr(args, "max_steps", None) is not None:
        total_steps = max(int(args.max_steps), 1)
    else:
        train_rows = 0 if dm.train_df is None else len(dm.train_df)
        batch_size = int(cfg_select(cfg, "training.batch_size", 1))
        max_epochs = int(
            getattr(args, "max_epochs", None) or cfg_select(cfg, "training.total_epochs", 1)
        )
        total_steps = max(math.ceil(train_rows / max(batch_size, 1)) * max_epochs, 1)
    OmegaConf.update(cfg, "training.total_steps", total_steps, merge=True)


def tabpfn_overrides_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Return TabPFN encoder overrides from CLI args."""
    overrides: dict[str, Any] = {}
    if getattr(args, "tabpfn_device", None):
        overrides["device"] = str(args.tabpfn_device)
    if getattr(args, "tabpfn_estimators", None) is not None:
        overrides["n_estimators"] = int(args.tabpfn_estimators)
    return overrides


def build_module(
    cfg: DictConfig,
    dm: AgeWellDataModule,
    args: argparse.Namespace,
    *,
    teacher: nn.Module | None = None,
) -> AgeWellINLightning:
    """Instantiate a Phase 5 Lightning module."""
    encoders = fake_encoders() if bool(getattr(args, "fake_encoders", False)) else None
    return AgeWellINLightning(
        cfg,
        context_df=dm.context_df,
        imputation_stats=dm.imputation_stats,
        teacher=teacher,
        encoders=encoders,
        tabpfn_overrides=tabpfn_overrides_from_args(args),
    )


def fake_encoders() -> dict[str, BaseEncoder]:
    """Return deterministic fake encoders for explicit smoke tests."""
    return {
        config.name: SmokeEncoder(presence_key=config.presence_key, offset=idx * 0.01)
        for idx, config in enumerate(build_default_registry())
    }


def build_trainer(
    cfg: DictConfig,
    args: argparse.Namespace,
    artifacts: RunArtifacts,
) -> tuple[L.Trainer, ModelCheckpoint | None]:
    """Build a Lightning trainer with local-only logging and checkpointing."""
    limit_val_batches = float(getattr(args, "limit_val_batches", 1.0))
    checkpoint: ModelCheckpoint | None
    callbacks: list[Any] = [LearningRateMonitor(logging_interval="step")]
    if limit_val_batches == 0.0:
        checkpoint = ModelCheckpoint(dirpath=artifacts.checkpoints_dir, save_last=True)
    else:
        checkpoint = ModelCheckpoint(
            dirpath=artifacts.checkpoints_dir,
            monitor="val/diag_macro_f1",
            mode="max",
            filename="epoch{epoch:03d}-diagf1{val/diag_macro_f1:.4f}",
            auto_insert_metric_name=False,
            save_top_k=3,
            save_last=True,
        )
    callbacks.append(checkpoint)
    precision = cast(Any, str(cfg_select(cfg, "training.precision", "bf16-mixed")))
    trainer = L.Trainer(
        accelerator=str(cfg_select(cfg, "training.trainer.accelerator", "auto")),
        devices=cfg_select(cfg, "training.trainer.devices", "auto"),
        precision=precision,
        max_epochs=int(
            getattr(args, "max_epochs", None) or cfg_select(cfg, "training.total_epochs", 1)
        ),
        max_steps=int(getattr(args, "max_steps", None) or -1),
        gradient_clip_val=float(cfg_select(cfg, "training.gradient_clip_val", 1.0)),
        log_every_n_steps=int(cfg_select(cfg, "training.trainer.log_every_n_steps", 10)),
        limit_val_batches=limit_val_batches,
        callbacks=callbacks,
        logger=CSVLogger(save_dir=str(artifacts.run_dir), name="logs"),
    )
    return trainer, checkpoint


def copy_best_checkpoint(checkpoint: ModelCheckpoint | None, artifacts: RunArtifacts) -> str:
    """Copy the best or last checkpoint to a stable ``best.ckpt`` path."""
    if checkpoint is None:
        return ""
    source = checkpoint.best_model_path or checkpoint.last_model_path
    if not source:
        return ""
    target = artifacts.checkpoints_dir / "best.ckpt"
    if Path(source).resolve() != target.resolve():
        shutil.copy2(source, target)
    return str(target)


def load_module_from_checkpoint(
    checkpoint_path: str | Path,
    cfg: DictConfig,
    dm: AgeWellDataModule,
    args: argparse.Namespace,
    *,
    strict: bool = True,
) -> AgeWellINLightning:
    """Instantiate a module and load a Lightning checkpoint state dict."""
    module = build_module(cfg, dm, args)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    module.load_state_dict(state, strict=strict)
    return module


def initialize_student_from_teacher(student: AgeWellINLightning, teacher_ckpt: str | Path) -> None:
    """Load teacher model weights into the student model only."""
    checkpoint = torch.load(teacher_ckpt, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    model_state = {
        key.removeprefix("model."): value
        for key, value in state.items()
        if key.startswith("model.")
    }
    student.model.load_state_dict(model_state, strict=True)


def freeze_module(module: nn.Module) -> nn.Module:
    """Freeze a module in eval mode."""
    module.eval()
    for param in module.parameters():
        param.requires_grad = False
    return module


def resolve_device(accelerator: str | None) -> torch.device:
    """Resolve a manual inference device from a CLI accelerator string."""
    if accelerator in {None, "auto"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if accelerator == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("accelerator='gpu' requested but CUDA is unavailable")
        return torch.device("cuda")
    if accelerator == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported accelerator: {accelerator}")


def collect_predictions(
    module: AgeWellINLightning,
    dm: AgeWellDataModule,
    *,
    split: str,
    batch_size: int,
    device: torch.device,
    mode: str = "observed",
    progress: bool = False,
    progress_label: str = "",
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Run batched inference for one split and return arrays plus metadata."""
    rows = _split_df(dm, split).reset_index(drop=True)
    total_batches = max(math.ceil(len(rows) / max(batch_size, 1)), 1)
    started = time.monotonic()
    module.to(device)
    module.eval()
    collected: dict[str, list[np.ndarray]] = {
        "diag_logits": [],
        "surv_logits": [],
        "mmse_pred": [],
        "cdr_logits": [],
        "diag_label": [],
        "surv_bin": [],
        "has_survival": [],
        "mmse": [],
        "has_mmse": [],
        "cdr": [],
        "has_cdr": [],
    }
    with torch.no_grad():
        for batch_idx, start in enumerate(range(0, len(rows), batch_size), start=1):
            frame = rows.iloc[start : start + batch_size]
            batch = move_batch_to_device(dm.collate_rows(frame.to_dict(orient="records")), device)
            outputs = module.model(
                batch,
                modality_dropout_p=0.0,
                presence_override=_presence_override(module.model.modality_names, batch, mode),
            )
            _append_output(collected, "diag_logits", outputs["diag_logits"])
            _append_output(collected, "surv_logits", outputs["surv_logits"])
            _append_output(collected, "mmse_pred", outputs["mmse_pred"])
            _append_output(collected, "cdr_logits", outputs["cdr_logits"])
            for key in (
                "diag_label",
                "surv_bin",
                "has_survival",
                "mmse",
                "has_mmse",
                "cdr",
                "has_cdr",
            ):
                _append_output(collected, key, batch[key])
            if progress:
                _print_progress(
                    label=progress_label or f"{split}:{mode}",
                    batch_idx=batch_idx,
                    total_batches=total_batches,
                    started=started,
                )
    arrays = {key: np.concatenate(value, axis=0) for key, value in collected.items()}
    return arrays, rows


def predictions_frame(
    arrays: Mapping[str, np.ndarray],
    rows: pd.DataFrame,
    *,
    diag_temperature: float = 1.0,
) -> pd.DataFrame:
    """Build a compact predictions dataframe for the observed-mode run."""
    from agewell.ml.metrics import softmax_np

    probs = softmax_np(arrays["diag_logits"], temperature=diag_temperature)
    out = rows.loc[:, ["subject_id", "visit_idx", "cohort", "diagnosis"]].reset_index(drop=True)
    out["diag_pred"] = probs.argmax(axis=1)
    out["mmse_pred"] = arrays["mmse_pred"]
    for idx in range(probs.shape[1]):
        out[f"diag_prob_{idx}"] = probs[:, idx]
    return out


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write stable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def environment_record() -> dict[str, Any]:
    """Return minimal reproducibility metadata."""
    return {
        "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
    }


def training_summary(
    *,
    cfg: DictConfig,
    dm: AgeWellDataModule,
    module: AgeWellINLightning,
    best_checkpoint: str,
) -> dict[str, Any]:
    """Return a compact run summary."""
    return {
        "run_name": str(cfg_select(cfg, "run_name", "")),
        "train_rows": 0 if dm.train_df is None else len(dm.train_df),
        "calib_rows": 0 if dm.val_df is None else len(dm.val_df),
        "test_rows": 0 if dm.test_df is None else len(dm.test_df),
        "trainable_params": count_trainable_parameters(module.model),
        "total_steps": int(cfg_select(cfg, "training.total_steps", 0)),
        "best_checkpoint": best_checkpoint,
    }


def move_batch_to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    """Move tensor values in a batch mapping to a device."""
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if isinstance(value, Tensor) else value
    return moved


def _presence_override(
    modality_names: tuple[str, ...],
    batch: Mapping[str, Any],
    mode: str,
) -> Tensor | str:
    if mode == "observed":
        return "observed"
    presence = torch.stack(
        [batch[f"{name}_presence"].bool() for name in modality_names],
        dim=1,
    )
    if mode == "no_mri":
        for idx, name in enumerate(modality_names):
            if name in {"mri_raw", "mri_vol"}:
                presence[:, idx] = False
        return presence
    if mode == "mri_only":
        for idx, name in enumerate(modality_names):
            if name not in {"mri_raw", "mri_vol"}:
                presence[:, idx] = False
        return presence
    if mode.startswith("drop:"):
        target = mode.split(":", maxsplit=1)[1]
        if target not in modality_names:
            raise ValueError(f"Unknown modality for robustness drop: {target}")
        presence[:, modality_names.index(target)] = False
        return presence
    raise ValueError(f"Unsupported evaluation mode: {mode}")


def _append_output(collected: dict[str, list[np.ndarray]], key: str, value: Any) -> None:
    tensor = value if isinstance(value, Tensor) else torch.as_tensor(value)
    collected[key].append(tensor.detach().cpu().numpy())


def _print_progress(
    *,
    label: str,
    batch_idx: int,
    total_batches: int,
    started: float,
) -> None:
    elapsed = time.monotonic() - started
    seconds_per_batch = elapsed / max(batch_idx, 1)
    remaining = max(total_batches - batch_idx, 0) * seconds_per_batch
    sys.stderr.write(
        f"[phase5] {label} batch {batch_idx}/{total_batches} "
        f"elapsed={_format_duration(elapsed)} eta={_format_duration(remaining)}\n"
    )
    sys.stderr.flush()


def _format_duration(seconds: float) -> str:
    total = max(int(seconds), 0)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


def _split_df(dm: AgeWellDataModule, split: str) -> pd.DataFrame:
    if split == "train" and dm.train_df is not None:
        return dm.train_df
    if split == "calib" and dm.val_df is not None:
        return dm.val_df
    if split == "test" and dm.test_df is not None:
        return dm.test_df
    raise RuntimeError(f"Datamodule has not loaded split {split!r}")


def _devices_value(value: Any) -> Any:
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return value


def _repo_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else repo_root() / candidate


def _timestamped_name(prefix: str) -> str:
    return f"{prefix}-{dt.datetime.now(dt.UTC).strftime('%Y%m%dT%H%M%SZ')}"


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo_root(), text=True).strip()
