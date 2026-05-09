"""Build the Phase 1 master parquet."""

from __future__ import annotations

import argparse
import json

from agewell.config import load_cfg
from agewell.data.master import write_master


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        default="all",
        help="Comma-separated adapter names or 'all'.",
    )
    args = parser.parse_args()
    datasets = (
        None if args.datasets == "all" else [item.strip() for item in args.datasets.split(",")]
    )
    cfg = load_cfg()
    path, quality = write_master(cfg.data, datasets=datasets)
    print(f"wrote {path}")
    print(json.dumps({"row_count": quality["row_count"], "is_rich": quality["is_rich_count"]}))


if __name__ == "__main__":
    main()
