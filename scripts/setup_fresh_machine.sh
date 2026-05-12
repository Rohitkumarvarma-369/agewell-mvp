#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/setup_fresh_machine.sh [--artifact ARTIFACT_OR_URL] [--skip-system-packages]

Prepare a fresh Linux cloud machine for this repo:
  - install basic system packages when apt-get is available
  - install uv when missing
  - install Python 3.11 via uv
  - sync the locked dev + MVP Python environment
  - optionally hydrate runtime artifacts

Run from the agewell-mvp repo root after git clone.
USAGE
}

artifact=""
install_system=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --artifact)
      artifact="$2"
      shift 2
      ;;
    --skip-system-packages)
      install_system=0
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

if [[ "${install_system}" == "1" && -x "$(command -v apt-get || true)" ]]; then
  apt_cmd=(apt-get)
  if [[ "${EUID}" -ne 0 ]]; then
    apt_cmd=(sudo apt-get)
  fi
  "${apt_cmd[@]}" update
  "${apt_cmd[@]}" install -y ca-certificates curl git build-essential zstd unzip rsync
fi

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

uv python install 3.11
UV_LINK_MODE="${UV_LINK_MODE:-copy}" uv sync --group dev --extra mvp --frozen
uv run pre-commit install || true

if [[ -n "${artifact}" ]]; then
  ./scripts/hydrate_runtime.sh "${artifact}"
fi

echo "Fresh-machine setup complete."
