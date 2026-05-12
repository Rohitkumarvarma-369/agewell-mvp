#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/hydrate_runtime.sh ARTIFACT_OR_URL

Hydrate a freshly cloned repo from a runtime artifact archive created by
scripts/create_runtime_bundle.sh. Supports local paths, http(s) URLs, s3:// URLs
when aws CLI is installed, and gs:// URLs when gsutil is installed.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || $# -lt 1 ]]; then
  usage
  exit 0
fi

repo_root="$(git rev-parse --show-toplevel)"
source_ref="$1"
cache_dir="${repo_root}/.runtime/artifact_cache"
mkdir -p "${cache_dir}" "${repo_root}/.runtime"

download() {
  local src="$1"
  local dst="$2"
  case "${src}" in
    http://*|https://*)
      curl -fL "${src}" -o "${dst}"
      ;;
    s3://*)
      aws s3 cp "${src}" "${dst}"
      ;;
    gs://*)
      gsutil cp "${src}" "${dst}"
      ;;
    *)
      cp -f "${src}" "${dst}"
      ;;
  esac
}

artifact_name="$(basename "${source_ref}")"
archive="${cache_dir}/${artifact_name}"
download "${source_ref}" "${archive}"

sidecar="${archive}.sha256"
if [[ "${source_ref}" == http://* || "${source_ref}" == https://* || "${source_ref}" == s3://* || "${source_ref}" == gs://* ]]; then
  if download "${source_ref}.sha256" "${sidecar}" 2>/dev/null; then
    echo "Downloaded checksum sidecar."
  else
    echo "No checksum sidecar found for ${source_ref}; continuing without archive hash check." >&2
  fi
elif [[ -f "${source_ref}.sha256" ]]; then
  cp -f "${source_ref}.sha256" "${sidecar}"
fi

if [[ -f "${sidecar}" ]]; then
  (
    cd "${cache_dir}"
    sha256sum -c "$(basename "${sidecar}")"
  )
fi

case "${archive}" in
  *.tar.zst|*.tzst)
    tar -C "${repo_root}" -I zstd -xf "${archive}"
    ;;
  *.tar.gz|*.tgz)
    tar -C "${repo_root}" -xzf "${archive}"
    ;;
  *)
    echo "Unsupported artifact extension: ${archive}" >&2
    exit 1
    ;;
esac

if [[ -d "${repo_root}/tabpfn_cache" ]]; then
  mkdir -p "${HOME}/.cache/tabpfn"
  cp -a "${repo_root}/tabpfn_cache/." "${HOME}/.cache/tabpfn/"
fi
if [[ -f "${repo_root}/runtime_manifest.json" ]]; then
  mv -f "${repo_root}/runtime_manifest.json" "${repo_root}/.runtime/runtime_manifest.json"
fi

uv run python -m agewell.scripts.runtime_artifacts rebase --repo-root "${repo_root}"
uv run python -m agewell.scripts.runtime_artifacts verify --manifest "${repo_root}/.runtime/runtime_manifest.json"

echo "Runtime hydrated and verified."
