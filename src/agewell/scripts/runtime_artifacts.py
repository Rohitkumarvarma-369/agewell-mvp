"""Create, rebase, and verify portable runtime artifact manifests."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

from agewell.config import repo_root

TABPFN_CHECKPOINT = "tabpfn-v2.5-classifier-v2.5_default.ckpt"
PATH_COLUMNS = ("mri_t1_uri", "mri_stripped_uri", "mri_seg_uri", "mri_brainiac_uri")
REBASE_MARKERS = ("/data/derivatives/", "/models/")
KEY_REPO_FILES = (
    "pyproject.toml",
    "uv.lock",
    "data/master.parquet",
    "data/splits/train.parquet",
    "data/splits/calib.parquet",
    "data/splits/test.parquet",
    "models/brainiac-v1-simclr.pt",
    "models/brainiac-brainage.ckpt",
    "models/brainiac-vit-mci.ckpt",
    "models/brainiac/hdbet/0.model",
    "models/brainiac/atlases/nihpd_asym_13.0-18.5_t1w.nii",
)
REBASEABLE_PARQUET_FILES = {
    "data/master.parquet",
    "data/splits/train.parquet",
    "data/splits/calib.parquet",
    "data/splits/test.parquet",
}
DIRECTORY_STATS = (
    "data/derivatives/brainiac",
    "data/derivatives/brainiac_preprocess",
    "models",
)


def main() -> None:
    """Run the runtime artifact CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    manifest = subparsers.add_parser("manifest", help="write a runtime artifact manifest")
    manifest.add_argument("--output", required=True)
    manifest.add_argument("--archive-name", default="")
    manifest.add_argument("--tabpfn-checkpoint", default="")

    rebase = subparsers.add_parser("rebase", help="rebase cached paths in parquet files")
    rebase.add_argument("--repo-root", default="")

    verify = subparsers.add_parser("verify", help="verify hydrated runtime artifacts")
    verify.add_argument("--manifest", default=".runtime/runtime_manifest.json")
    verify.add_argument("--allow-commit-mismatch", action="store_true")
    verify.add_argument("--skip-key-hashes", action="store_true")

    args = parser.parse_args()
    if args.cmd == "manifest":
        write_manifest(
            output=Path(args.output),
            archive_name=str(args.archive_name),
            tabpfn_checkpoint=_tabpfn_path(str(args.tabpfn_checkpoint)),
        )
    elif args.cmd == "rebase":
        rebase_runtime_paths(Path(args.repo_root) if args.repo_root else repo_root())
    elif args.cmd == "verify":
        verify_runtime(
            manifest_path=_repo_path(Path(args.manifest)),
            allow_commit_mismatch=bool(args.allow_commit_mismatch),
            skip_key_hashes=bool(args.skip_key_hashes),
        )


def write_manifest(*, output: Path, archive_name: str, tabpfn_checkpoint: Path) -> None:
    """Write a JSON manifest for the current runtime artifacts."""
    root = repo_root()
    manifest = {
        "schema_version": 1,
        "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "archive_name": archive_name,
        "repo_key_files": _key_file_hashes(root),
        "tabpfn_checkpoint": _file_record(tabpfn_checkpoint),
        "directory_stats": {path: _directory_record(root / path) for path in DIRECTORY_STATS},
        "data_stats": _data_stats(root),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(output), "git_commit": manifest["git_commit"]}))


def rebase_runtime_paths(root: Path) -> None:
    """Rewrite cached artifact path columns to point at this clone root."""
    parquet_paths = [root / "data/master.parquet"]
    parquet_paths.extend(sorted((root / "data/splits").glob("*.parquet")))
    changed: list[str] = []
    for path in parquet_paths:
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        touched = False
        for column in PATH_COLUMNS:
            if column not in df:
                continue
            updated = df[column].map(lambda value: _rebase_value(value, root))
            if not updated.equals(df[column]):
                df[column] = updated
                touched = True
        if touched:
            df.to_parquet(path, index=False)
            changed.append(str(path.relative_to(root)))
    print(json.dumps({"rebased": changed}, sort_keys=True))


def verify_runtime(
    *,
    manifest_path: Path,
    allow_commit_mismatch: bool,
    skip_key_hashes: bool,
) -> None:
    """Verify the hydrated runtime against its manifest."""
    root = repo_root()
    if not manifest_path.exists():
        raise SystemExit(f"Missing runtime manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: list[str] = []

    expected_commit = str(manifest.get("git_commit", ""))
    actual_commit = _git("rev-parse", "HEAD")
    if (
        expected_commit
        and actual_commit != expected_commit
        and not allow_commit_mismatch
        and not _git_is_ancestor(expected_commit, actual_commit)
    ):
        failures.append(f"git commit mismatch: manifest={expected_commit} actual={actual_commit}")

    if not skip_key_hashes:
        _compare_key_hashes(root, manifest, failures)
    _compare_tabpfn(manifest, failures)
    _compare_data_stats(root, manifest, failures)

    if failures:
        raise SystemExit("Runtime verification failed:\n- " + "\n- ".join(failures))
    print(
        json.dumps(
            {
                "status": "ok",
                "git_commit": actual_commit,
                "manifest": str(manifest_path),
                "data_rows": manifest.get("data_stats", {}).get("master_rows"),
            },
            sort_keys=True,
        )
    )


def _compare_key_hashes(root: Path, manifest: dict[str, Any], failures: list[str]) -> None:
    expected = manifest.get("repo_key_files", {})
    if not isinstance(expected, dict):
        failures.append("manifest repo_key_files is missing or invalid")
        return
    for rel_path, record in expected.items():
        path = root / str(rel_path)
        if not path.exists():
            failures.append(f"missing key file: {rel_path}")
            continue
        if not isinstance(record, dict):
            failures.append(f"invalid key-file record: {rel_path}")
            continue
        if str(rel_path) in REBASEABLE_PARQUET_FILES:
            continue
        expected_sha = str(record.get("sha256", ""))
        actual_sha = sha256_file(path)
        if expected_sha and actual_sha != expected_sha:
            failures.append(f"sha256 mismatch for {rel_path}")


def _compare_tabpfn(manifest: dict[str, Any], failures: list[str]) -> None:
    expected = manifest.get("tabpfn_checkpoint", {})
    checkpoint = _tabpfn_path("")
    if not checkpoint.exists():
        failures.append(f"missing TabPFN checkpoint: {checkpoint}")
        return
    if isinstance(expected, dict):
        expected_sha = str(expected.get("sha256", ""))
        if expected_sha and sha256_file(checkpoint) != expected_sha:
            failures.append("sha256 mismatch for TabPFN checkpoint")


def _compare_data_stats(root: Path, manifest: dict[str, Any], failures: list[str]) -> None:
    expected = manifest.get("data_stats", {})
    if not isinstance(expected, dict):
        failures.append("manifest data_stats is missing or invalid")
        return
    actual = _data_stats(root)
    for key in ("master_rows", "mri_rows", "brainiac_non_null", "brainiac_existing"):
        if expected.get(key) != actual.get(key):
            failures.append(
                f"data stat mismatch {key}: manifest={expected.get(key)} actual={actual.get(key)}"
            )
    expected_splits = expected.get("split_rows", {})
    actual_splits = actual.get("split_rows", {})
    if expected_splits != actual_splits:
        failures.append(f"split row mismatch: manifest={expected_splits} actual={actual_splits}")


def _data_stats(root: Path) -> dict[str, Any]:
    master_path = root / "data/master.parquet"
    if not master_path.exists():
        return {
            "master_rows": 0,
            "mri_rows": 0,
            "brainiac_non_null": 0,
            "brainiac_existing": 0,
            "cohort_counts": {},
            "split_rows": {},
        }
    df = pd.read_parquet(master_path)
    mri_rows = df[df["mri_t1_uri"].notna()] if "mri_t1_uri" in df else df.iloc[0:0]
    brainiac = df["mri_brainiac_uri"].dropna() if "mri_brainiac_uri" in df else pd.Series([])
    split_rows = {
        path.stem: len(pd.read_parquet(path))
        for path in sorted((root / "data/splits").glob("*.parquet"))
    }
    return {
        "master_rows": len(df),
        "mri_rows": len(mri_rows),
        "brainiac_non_null": len(brainiac),
        "brainiac_existing": int(sum(Path(str(path)).exists() for path in brainiac)),
        "cohort_counts": {str(k): int(v) for k, v in df["cohort"].value_counts().items()},
        "split_rows": split_rows,
    }


def _key_file_hashes(root: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for rel_path in KEY_REPO_FILES:
        path = root / rel_path
        if path.exists():
            records[rel_path] = _file_record(path)
    return records


def _directory_record(path: Path) -> dict[str, int]:
    files = [item for item in path.rglob("*") if item.is_file()] if path.exists() else []
    return {"files": len(files), "bytes": sum(item.stat().st_size for item in files)}


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else 0,
        "sha256": sha256_file(path) if path.exists() else "",
    }


def _rebase_value(value: object, root: Path) -> object:
    if value is None or bool(pd.isna(value)):
        return value
    text = str(value)
    for marker in REBASE_MARKERS:
        idx = text.find(marker)
        if idx >= 0:
            return str(root) + text[idx:]
    if text.startswith("data/derivatives/") or text.startswith("models/"):
        return str(root / text)
    return value


def _tabpfn_path(explicit: str) -> Path:
    if explicit:
        return Path(explicit).expanduser()
    return Path.home() / ".cache" / "tabpfn" / TABPFN_CHECKPOINT


def _repo_path(path: Path) -> Path:
    return path if path.is_absolute() else repo_root() / path


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo_root(), text=True).strip()


def _git_is_ancestor(ancestor: str, descendant: str) -> bool:
    return (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, descendant],
            cwd=repo_root(),
            check=False,
        ).returncode
        == 0
    )


def sha256_file(path: Path) -> str:
    """Return the SHA256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
