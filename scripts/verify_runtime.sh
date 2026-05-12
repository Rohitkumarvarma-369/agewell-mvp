#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/verify_runtime.sh [--real-tabpfn] [--skip-tests]

Verify a hydrated runtime on a fresh cloud machine. By default this checks
artifact integrity, lint/typecheck, unit tests, and a fast full-stack Phase 4
smoke. Add --real-tabpfn to verify actual TabPFN CPU embeddings with full
1500-row context and default 8 estimators.
USAGE
}

real_tabpfn=0
skip_tests=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --real-tabpfn)
      real_tabpfn=1
      shift
      ;;
    --skip-tests)
      skip_tests=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

repo_root="$(git rev-parse --show-toplevel)"
cd "${repo_root}"

uv run python -m agewell.scripts.runtime_artifacts verify

if [[ "${skip_tests}" != "1" ]]; then
  uv run ruff check .
  uv run ruff format --check .
  uv run mypy src/agewell
  uv run pytest tests/unit -q
fi

uv run python -m agewell.scripts._ml_smoke \
  --max-steps 2 \
  --batch-size 4 \
  --fake-encoders \
  --accelerator cpu \
  --devices 1 \
  --precision 32-true

if [[ "${real_tabpfn}" == "1" ]]; then
  TABPFN_ALLOW_CPU_LARGE_DATASET=1 uv run python - <<'PY'
import pandas as pd
import torch

from agewell.ml.encoders.tabular_tabpfn import TabPFNFrozenEncoder
from agewell.ml.feature_extractors import build_tabpfn_context, compute_imputation_stats

train = pd.read_parquet("data/splits/train.parquet")
stats = compute_imputation_stats(train)
ctx_x, ctx_y = build_tabpfn_context(
    train,
    modality="clinical_demo",
    ctx_size=1500,
    imputation_stats=stats,
)
encoder = TabPFNFrozenEncoder(
    modality="clinical_demo",
    n_features=5,
    ctx_X=ctx_x,
    ctx_y=ctx_y,
    n_estimators=8,
    device="cpu",
)
out = encoder({"clinical_demo_features": ctx_x[:4]})
assert out.shape == (4, 256)
assert torch.isfinite(out).all()
print({"real_tabpfn": "ok", "ctx": tuple(ctx_x.shape), "out": tuple(out.shape)})
PY
fi

echo "Runtime verification complete."
