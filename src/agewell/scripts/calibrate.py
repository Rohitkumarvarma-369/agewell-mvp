"""Fit Phase 5 diagnosis temperature calibration on a split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agewell.ml.calibration import fit_temperature
from agewell.ml.config_utils import cfg_select
from agewell.scripts._phase5_common import (
    add_eval_args,
    build_datamodule,
    collect_predictions,
    load_module_from_checkpoint,
    load_phase5_cfg,
    resolve_device,
    write_json,
)


def main() -> None:
    """Run calibration."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_eval_args(parser)
    parser.set_defaults(split="calib")
    args = parser.parse_args()

    cfg = load_phase5_cfg(args, default_run_name="phase5_calibration")
    dm = build_datamodule(cfg)
    module = load_module_from_checkpoint(args.checkpoint, cfg, dm, args, strict=False)
    batch_size = int(args.batch_size or cfg_select(cfg, "training.batch_size", 32))
    device = resolve_device(args.accelerator)
    arrays, _ = collect_predictions(
        module,
        dm,
        split=str(args.split),
        batch_size=batch_size,
        device=device,
        mode="observed",
    )
    result = fit_temperature(arrays["diag_logits"], arrays["diag_label"])
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.checkpoint).parent.parent
    path = output_dir / "calibration" / "diagnosis_temperature.json"
    payload = {
        "checkpoint": str(args.checkpoint),
        "split": str(args.split),
        **result.to_dict(),
    }
    write_json(path, payload)
    print(json.dumps({"calibration": str(path), **payload}, sort_keys=True))


if __name__ == "__main__":
    main()
