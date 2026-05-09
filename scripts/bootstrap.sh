#!/usr/bin/env bash
set -euo pipefail

uv sync --group dev
if git rev-parse --git-dir >/dev/null 2>&1; then
  uv run pre-commit install
else
  echo "No git repository detected; skipping pre-commit install."
fi
