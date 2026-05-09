"""Write Phase 1 train/calib/test splits."""

from __future__ import annotations

from agewell.config import load_cfg
from agewell.data.splits import write_splits


def main() -> None:
    """CLI entrypoint."""
    cfg = load_cfg()
    out_paths = write_splits(cfg.data)
    for name, path in out_paths.items():
        print(f"wrote {name}: {path}")


if __name__ == "__main__":
    main()
