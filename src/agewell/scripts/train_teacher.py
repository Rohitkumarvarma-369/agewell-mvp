"""Train the Phase 5 supervised teacher model."""

from __future__ import annotations

import argparse
import json

import torch
from omegaconf import DictConfig, OmegaConf

from agewell.scripts._phase5_common import (
    add_train_args,
    build_datamodule,
    build_module,
    build_trainer,
    copy_best_checkpoint,
    load_phase5_cfg,
    prepare_run,
    training_summary,
    update_total_steps,
    write_json,
)


def main() -> None:
    """Run teacher training."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_train_args(parser)
    args = parser.parse_args()
    cfg = load_phase5_cfg(args, default_run_name="phase5_teacher_full")
    _apply_teacher_defaults(cfg, args)

    dm = build_datamodule(cfg)
    update_total_steps(cfg, dm, args)
    artifacts = prepare_run(cfg, overwrite=bool(args.overwrite))
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    module = build_module(cfg, dm, args)
    trainer, checkpoint = build_trainer(cfg, args, artifacts)
    trainer.fit(module, datamodule=dm)

    best_checkpoint = copy_best_checkpoint(checkpoint, artifacts)
    summary = training_summary(
        cfg=cfg,
        dm=dm,
        module=module,
        best_checkpoint=best_checkpoint,
    )
    write_json(artifacts.run_dir / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True))


def _apply_teacher_defaults(cfg: DictConfig, args: argparse.Namespace) -> None:
    max_epochs = int(args.max_epochs or 30)
    OmegaConf.update(cfg, "training.total_epochs", max_epochs, merge=True)
    OmegaConf.update(cfg, "training.modality_dropout.max_p", 0.0, merge=True)
    OmegaConf.update(cfg, "training.modality_dropout.warmup_steps", 0, merge=True)
    OmegaConf.update(cfg, "training.loss_weights.distill_logits", 0.0, merge=True)
    OmegaConf.update(cfg, "training.loss_weights.distill_embedding", 0.0, merge=True)


if __name__ == "__main__":
    main()
