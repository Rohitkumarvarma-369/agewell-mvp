"""Evaluate a Phase 5 checkpoint on held-out splits and robustness modes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from agewell.ml.config_utils import cfg_select
from agewell.ml.metrics import compute_phase5_metrics, json_sanitize
from agewell.scripts._phase5_common import (
    add_eval_args,
    build_datamodule,
    collect_predictions,
    load_module_from_checkpoint,
    load_phase5_cfg,
    predictions_frame,
    resolve_device,
    write_json,
)


def main() -> None:
    """Run evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_eval_args(parser)
    parser.add_argument("--calibration", default="")
    parser.add_argument(
        "--robustness",
        choices=("observed", "core", "all"),
        default="observed",
        help="robustness modes to evaluate",
    )
    args = parser.parse_args()

    cfg = load_phase5_cfg(args, default_run_name="phase5_evaluate")
    dm = build_datamodule(cfg)
    module = load_module_from_checkpoint(args.checkpoint, cfg, dm, args, strict=False)
    batch_size = int(args.batch_size or cfg_select(cfg, "training.batch_size", 32))
    device = resolve_device(args.accelerator)
    temperature = _temperature(args.calibration)
    modes = _modes(str(args.robustness), tuple(module.model.modality_names))

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.checkpoint).parent.parent
    all_metrics: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "split": str(args.split),
        "diag_temperature": temperature,
        "modes": {},
    }
    observed_arrays = None
    observed_rows = None
    for mode_idx, mode in enumerate(modes, start=1):
        if not bool(args.no_progress):
            sys.stderr.write(f"[phase5] evaluate mode {mode_idx}/{len(modes)}: {mode}\n")
            sys.stderr.flush()
        arrays, rows = collect_predictions(
            module,
            dm,
            split=str(args.split),
            batch_size=batch_size,
            device=device,
            mode=mode,
            progress=not bool(args.no_progress),
            progress_label=f"evaluate:{args.split}:{mode}",
        )
        all_metrics["modes"][mode] = compute_phase5_metrics(
            arrays,
            diag_temperature=temperature,
            prefix=mode,
        )
        if mode == "observed":
            observed_arrays = arrays
            observed_rows = rows

    metrics_path = output_dir / "metrics" / f"{args.split}_metrics.json"
    write_json(metrics_path, json_sanitize(all_metrics))
    if observed_arrays is not None and observed_rows is not None:
        predictions_path = output_dir / "predictions" / f"{args.split}_predictions.parquet"
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        predictions_frame(
            observed_arrays,
            observed_rows,
            diag_temperature=temperature,
        ).to_parquet(predictions_path, index=False)
    print(json.dumps({"metrics": str(metrics_path), **json_sanitize(all_metrics)}, sort_keys=True))


def _temperature(path: str) -> float:
    if not path:
        return 1.0
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return float(payload["temperature"])


def _modes(level: str, modality_names: tuple[str, ...]) -> list[str]:
    if level == "observed":
        return ["observed"]
    modes = ["observed", "no_mri", "mri_only"]
    if level == "all":
        modes.extend(f"drop:{name}" for name in modality_names)
    return modes


if __name__ == "__main__":
    main()
