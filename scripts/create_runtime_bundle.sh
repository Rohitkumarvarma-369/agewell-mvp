#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/create_runtime_bundle.sh [OUT_DIR]

Create a portable runtime artifact archive for a fresh cloud clone. The archive
contains runtime data, derived BrainIAC artifacts, model files, TabPFN weights,
and a manifest pinned to the current git commit.

Environment:
  TABPFN_CHECKPOINT  Override TabPFN checkpoint path.
  ALLOW_DIRTY=1      Allow bundling from a dirty worktree.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

repo_root="$(git rev-parse --show-toplevel)"
out_dir="${1:-${repo_root}/../agewell-runtime-artifacts}"
tabpfn_checkpoint="${TABPFN_CHECKPOINT:-${HOME}/.cache/tabpfn/tabpfn-v2.5-classifier-v2.5_default.ckpt}"
git_commit="$(git -C "${repo_root}" rev-parse --short=12 HEAD)"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
base_name="agewell-runtime-${git_commit}-${timestamp}"

if [[ -n "$(git -C "${repo_root}" status --porcelain)" && "${ALLOW_DIRTY:-0}" != "1" ]]; then
  echo "Refusing to bundle dirty worktree. Commit changes first or set ALLOW_DIRTY=1." >&2
  exit 1
fi

required_paths=(
  "data/master.parquet"
  "data/splits"
  "data/derivatives/brainiac"
  "data/derivatives/brainiac_preprocess"
  "models"
)

for rel_path in "${required_paths[@]}"; do
  if [[ ! -e "${repo_root}/${rel_path}" ]]; then
    echo "Missing runtime artifact path: ${rel_path}" >&2
    exit 1
  fi
done
if [[ ! -f "${tabpfn_checkpoint}" ]]; then
  echo "Missing TabPFN checkpoint: ${tabpfn_checkpoint}" >&2
  exit 1
fi

mkdir -p "${out_dir}"
tmp_dir="$(mktemp -d)"
cleanup() {
  rm -rf "${tmp_dir}"
}
trap cleanup EXIT

mkdir -p "${tmp_dir}/data/derivatives" "${tmp_dir}/tabpfn_cache"
ln -s "${repo_root}/data/master.parquet" "${tmp_dir}/data/master.parquet"
if [[ -f "${repo_root}/data/quality_report.json" ]]; then
  ln -s "${repo_root}/data/quality_report.json" "${tmp_dir}/data/quality_report.json"
fi
ln -s "${repo_root}/data/splits" "${tmp_dir}/data/splits"
ln -s "${repo_root}/data/derivatives/brainiac" "${tmp_dir}/data/derivatives/brainiac"
ln -s "${repo_root}/data/derivatives/brainiac_preprocess" "${tmp_dir}/data/derivatives/brainiac_preprocess"
ln -s "${repo_root}/models" "${tmp_dir}/models"
ln -s "${tabpfn_checkpoint}" "${tmp_dir}/tabpfn_cache/$(basename "${tabpfn_checkpoint}")"

archive="${out_dir}/${base_name}.tar.zst"
if ! command -v zstd >/dev/null 2>&1; then
  archive="${out_dir}/${base_name}.tar.gz"
fi

uv run python -m agewell.scripts.runtime_artifacts manifest \
  --output "${tmp_dir}/runtime_manifest.json" \
  --archive-name "$(basename "${archive}")" \
  --tabpfn-checkpoint "${tabpfn_checkpoint}"

if [[ "${archive}" == *.tar.zst ]]; then
  tar --dereference -C "${tmp_dir}" -I 'zstd -T0 -19' -cf "${archive}" \
    data models tabpfn_cache runtime_manifest.json
else
  tar --dereference -C "${tmp_dir}" -czf "${archive}" \
    data models tabpfn_cache runtime_manifest.json
fi

(
  cd "${out_dir}"
  sha256sum "$(basename "${archive}")" > "$(basename "${archive}").sha256"
)

echo "Created ${archive}"
echo "Created ${archive}.sha256"
