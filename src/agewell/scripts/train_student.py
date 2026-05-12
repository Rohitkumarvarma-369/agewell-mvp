"""Train the Phase 5 robust student model from a supervised teacher."""

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
    freeze_module,
    initialize_student_from_teacher,
    load_module_from_checkpoint,
    load_phase5_cfg,
    prepare_run,
    training_summary,
    update_total_steps,
    write_json,
)


def main() -> None:
    """Run student training."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_train_args(parser)
    parser.add_argument("--teacher-checkpoint", required=True)
    args = parser.parse_args()
    cfg = load_phase5_cfg(args, default_run_name="phase5_student_full")
    _apply_student_defaults(cfg, args)

    dm = build_datamodule(cfg)
    update_total_steps(cfg, dm, args)
    artifacts = prepare_run(cfg, overwrite=bool(args.overwrite))
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    teacher = load_module_from_checkpoint(args.teacher_checkpoint, cfg, dm, args, strict=True)
    freeze_module(teacher)
    module = build_module(cfg, dm, args, teacher=teacher)
    initialize_student_from_teacher(module, args.teacher_checkpoint)

    trainer, checkpoint = build_trainer(cfg, args, artifacts)
    trainer.fit(module, datamodule=dm)

    best_checkpoint = copy_best_checkpoint(checkpoint, artifacts)
    summary = training_summary(
        cfg=cfg,
        dm=dm,
        module=module,
        best_checkpoint=best_checkpoint,
    )
    summary["teacher_checkpoint"] = str(args.teacher_checkpoint)
    write_json(artifacts.run_dir / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True))


def _apply_student_defaults(cfg: DictConfig, args: argparse.Namespace) -> None:
    max_epochs = int(args.max_epochs or 20)
    OmegaConf.update(cfg, "training.total_epochs", max_epochs, merge=True)
    OmegaConf.update(cfg, "training.modality_dropout.max_p", 0.35, merge=True)
    OmegaConf.update(cfg, "training.modality_dropout.warmup_steps", 250, merge=True)
    OmegaConf.update(cfg, "training.loss_weights.distill_logits", 0.5, merge=True)
    OmegaConf.update(cfg, "training.loss_weights.distill_embedding", 0.1, merge=True)


if __name__ == "__main__":
    main()
