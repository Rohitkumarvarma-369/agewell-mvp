# Reproducible Cloud Handoff

This runbook is for short-lived GPU or high-memory cloud machines. The target is:
clone, hydrate, verify, run the next phase, and shut the machine down without
manual state reconstruction.

## What Is Portable

The repo stores code, lockfiles, DVC metadata, configs, and tests. Runtime
artifacts are intentionally outside git and are moved as a single archive:

- `data/master.parquet`
- `data/splits/`
- `data/derivatives/brainiac/`
- `data/derivatives/brainiac_preprocess/`
- `models/`
- `~/.cache/tabpfn/tabpfn-v2.5-classifier-v2.5_default.ckpt`

The archive also contains `runtime_manifest.json`, which pins the git commit,
key file hashes, data row counts, derived-file counts, and the TabPFN checkpoint
hash.

## Create A Runtime Bundle

From a clean local checkout:

```bash
cd /home/rohit/kaggle-iisc/agewell-mvp
git status --short
./scripts/create_runtime_bundle.sh /home/rohit/kaggle-iisc/runtime-artifacts
```

This creates:

```text
agewell-runtime-<git_sha>-<timestamp>.tar.zst
agewell-runtime-<git_sha>-<timestamp>.tar.zst.sha256
```

If `zstd` is unavailable, the script falls back to `.tar.gz`.

## Fresh Cloud Machine

```bash
git clone https://github.com/Rohitkumarvarma-369/agewell-mvp
cd agewell-mvp
./scripts/setup_fresh_machine.sh \
  --artifact /path/to/agewell-runtime-<git_sha>-<timestamp>.tar.zst \
  --torch-cuda auto
./scripts/verify_runtime.sh --real-tabpfn
```

The setup script installs basic apt packages when available, installs `uv` when
missing, provisions Python 3.11, syncs `uv.lock` with `--extra mvp --frozen`, and
hydrates the runtime archive. `--torch-cuda auto` keeps CPU machines unchanged
and repairs CUDA 12.x cloud machines to the known working cu124 Torch wheel set
when the locked Torch wheel cannot initialize CUDA.

## Why Hydration Rebases Paths

The imported `master.parquet` and split parquets contain cached artifact paths.
Those paths may point to the original local clone. During hydration,
`agewell.scripts.runtime_artifacts rebase` rewrites paths under
`data/derivatives/` and `models/` to the new clone root, so BrainIAC feature
loading works from any cloud directory.

## Fast Verification

For quick code/data sanity without actual TabPFN CPU embedding:

```bash
./scripts/verify_runtime.sh
```

For full runtime confidence, including actual TabPFN with full 1500-row context
and default 8 estimators:

```bash
./scripts/verify_runtime.sh --real-tabpfn
```

## Upload Targets

The artifact path can be local or remote:

```bash
./scripts/hydrate_runtime.sh ./agewell-runtime-....tar.zst
./scripts/hydrate_runtime.sh https://.../agewell-runtime-....tar.zst
./scripts/hydrate_runtime.sh s3://bucket/path/agewell-runtime-....tar.zst
./scripts/hydrate_runtime.sh gs://bucket/path/agewell-runtime-....tar.zst
```

For remote URLs, keep the `.sha256` sidecar next to the archive.
