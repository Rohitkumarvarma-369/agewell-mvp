#!/usr/bin/env bash
set -euo pipefail

include_mvp="${INCLUDE_MVP:-0}"
if [[ "$include_mvp" =~ ^(1|true|TRUE|yes|YES)$ ]]; then
  uv sync --group dev --extra mvp
else
  uv sync --group dev
fi

if git rev-parse --git-dir >/dev/null 2>&1; then
  uv run pre-commit install
else
  echo "No git repository detected; skipping pre-commit install."
fi
