"""Run the Phase 2 imaging pipeline over master.parquet rows."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from agewell.config import repo_root
from agewell.pipelines.imaging_flow import imaging_flow


def main() -> None:
    """CLI entrypoint."""
    args = _parse_args()
    master_path = _repo_path(args.master_path)
    df = pd.read_parquet(master_path)
    todo = df[df["mri_t1_uri"].notna()].copy()
    if args.cohort:
        todo = todo[todo["cohort"].eq(args.cohort)]
    if not args.force:
        todo = todo[todo["mri_brainiac_uri"].isna()]
    if args.limit:
        todo = todo.head(args.limit)

    progress_path = _repo_path(args.progress_path)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    summary = _run_rows(
        todo,
        master_path=master_path,
        progress_path=progress_path,
        dry_run=bool(args.dry_run),
        fail_fast=bool(args.fail_fast),
    )
    print(json.dumps(summary, sort_keys=True))


def _run_rows(
    todo: pd.DataFrame,
    *,
    master_path: Path,
    progress_path: Path,
    dry_run: bool,
    fail_fast: bool,
) -> dict[str, int]:
    summary = {"queued": len(todo), "dry_run": 0, "done": 0, "failed": 0}
    for _, row in todo.iterrows():
        payload = {
            "ts": datetime.now(UTC).isoformat(),
            "subject_id": row["subject_id"],
            "visit_idx": int(row["visit_idx"]),
            "cohort": row["cohort"],
            "mri_t1_uri": row["mri_t1_uri"],
            "status": "dry_run" if dry_run else "started",
        }
        _append_jsonl(progress_path, payload)
        if dry_run:
            summary["dry_run"] += 1
            continue
        try:
            out = imaging_flow(
                subject_id=str(row["subject_id"]),
                visit_idx=int(row["visit_idx"]),
                cohort=str(row["cohort"]),
                nifti_uri=str(row["mri_t1_uri"]),
                master_path=master_path,
            )
            _append_jsonl(progress_path, payload | {"status": "done", "output": out})
            summary["done"] += 1
        except Exception as exc:
            _append_jsonl(
                progress_path,
                payload | {"status": "failed", "error": type(exc).__name__, "message": str(exc)},
            )
            summary["failed"] += 1
            if fail_fast:
                raise
            continue
    _append_jsonl(
        progress_path,
        {"ts": datetime.now(UTC).isoformat(), "status": "summary", "summary": summary},
    )
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 2 BrainIAC imaging backfill.")
    parser.add_argument("--cohort", choices=["ADNI_NIFTI", "IXI"], default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--master-path", default="data/master.parquet")
    parser.add_argument("--progress-path", default="logs/imaging_progress.jsonl")
    return parser.parse_args()


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else repo_root() / candidate


if __name__ == "__main__":
    main()
