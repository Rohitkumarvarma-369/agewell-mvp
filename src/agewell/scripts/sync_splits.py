"""Refresh or verify split parquet files against master parquet."""

from __future__ import annotations

import argparse
import json

from agewell.data.split_sync import refresh_splits_from_master, verify_splits_synced


def main() -> None:
    """Run split synchronization checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-path", default="data/master.parquet")
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--write", action="store_true", help="rewrite stale split parquet files")
    parser.add_argument(
        "--allow-missing-brainiac-paths",
        action="store_true",
        help="do not fail when cached BrainIAC paths are missing on disk",
    )
    args = parser.parse_args()
    strict_paths = not bool(args.allow_missing_brainiac_paths)
    if args.write:
        report = refresh_splits_from_master(
            master_path=args.master_path,
            splits_dir=args.splits_dir,
            write=True,
            strict_paths=strict_paths,
        )
    else:
        report = verify_splits_synced(
            master_path=args.master_path,
            splits_dir=args.splits_dir,
            strict_paths=strict_paths,
        )
    print(json.dumps(report.to_dict(), sort_keys=True))


if __name__ == "__main__":
    main()
