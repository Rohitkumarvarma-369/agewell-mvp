"""Run Phase 5 calibration, evaluation, and training on Modal GPUs.

The app keeps code in the Modal image and mutable runtime artifacts in the
``agewell-runtime`` Modal Volume. Runtime data should be uploaded to
``/artifacts`` and Phase 5 runs to ``/outputs/runs`` inside that Volume.
"""

from __future__ import annotations

import contextlib
import importlib
import io
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Literal, cast

import modal

APP_NAME = "agewell-phase5-eval"
GPU_TYPE = "A100-40GB"
TRAIN_GPU_TYPE = os.environ.get("AGEWELL_MODAL_TRAIN_GPU", "H200")
VOLUME_NAME = os.environ.get("AGEWELL_MODAL_VOLUME", "agewell-runtime")
APP_ROOT = Path("/app")
RUNTIME_ROOT = Path("/runtime")
DEFAULT_STUDENT_RUN = "/app/outputs/runs/phase5_student_from_4ep_b256"
DEFAULT_TEACHER_RUN = "/app/outputs/runs/phase5_teacher_4ep_b128"
DEFAULT_TIMEOUT_SECONDS = 6 * 60 * 60
DEFAULT_TRAIN_TIMEOUT_SECONDS = 24 * 60 * 60
DEFAULT_H200_TEACHER_RUN_NAME = "phase5_teacher_16ep_b64_h200"
DEFAULT_H200_STUDENT_RUN_NAME = "phase5_student_from_16ep_teacher_b64_h200"

app = modal.App(APP_NAME)
runtime_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "rsync", "zstd")
    .uv_sync(
        uv_project_dir=".",
        groups=["dev"],
        extras=["mvp"],
        frozen=True,
    )
    .workdir("/app")
    .env(
        {
            "PYTHONPATH": "/app/src",
            "PYTHONUNBUFFERED": "1",
            "TABPFN_ALLOW_CPU_LARGE_DATASET": "1",
            "UV_LINK_MODE": "copy",
        }
    )
    .add_local_file("pyproject.toml", remote_path="/app/pyproject.toml")
    .add_local_file("uv.lock", remote_path="/app/uv.lock")
    .add_local_dir("configs", remote_path="/app/configs")
    .add_local_dir("src", remote_path="/app/src")
    .add_local_dir(".git", remote_path="/app/.git")
)


@app.local_entrypoint()
def status(use_gpu: bool = False) -> None:
    """Check that the Modal image and mounted Volume are usable."""
    remote = gpu_status if use_gpu else runtime_status
    print(json.dumps(remote.remote(), sort_keys=True))


@app.local_entrypoint()
def h200_status() -> None:
    """Check that the Modal image, Volume, and H200 GPU request are usable."""
    print(json.dumps(train_gpu_status.remote(), sort_keys=True))


@app.local_entrypoint()
def upload_runtime(artifact: str) -> None:
    """Upload a runtime archive and optional checksum sidecar to the Modal Volume."""
    source = Path(artifact).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    with runtime_volume.batch_upload(force=True) as batch:
        batch.put_file(source, f"/artifacts/{source.name}")
        sidecar = source.with_name(f"{source.name}.sha256")
        if sidecar.exists():
            batch.put_file(sidecar, f"/artifacts/{sidecar.name}")
    print(json.dumps({"uploaded": str(source), "volume": VOLUME_NAME}, sort_keys=True))


@app.local_entrypoint()
def upload_run(run_dir: str) -> None:
    """Upload a local Phase 5 run directory to ``/outputs/runs`` in the Volume."""
    source = Path(run_dir).expanduser().resolve()
    if not source.is_dir():
        raise NotADirectoryError(source)
    remote_path = f"/outputs/runs/{source.name}"
    with runtime_volume.batch_upload(force=True) as batch:
        batch.put_directory(source, remote_path)
    print(json.dumps({"uploaded": str(source), "remote_path": remote_path}, sort_keys=True))


@app.local_entrypoint()
def upload_run_minimal(run_dir: str) -> None:
    """Upload only files required for remote calibration/evaluation."""
    source = Path(run_dir).expanduser().resolve()
    if not source.is_dir():
        raise NotADirectoryError(source)
    run_name = source.name
    checkpoint = source / "checkpoints" / "best.ckpt"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    remote_root = f"/outputs/runs/{run_name}"
    with runtime_volume.batch_upload(force=True) as batch:
        batch.put_file(checkpoint, f"{remote_root}/checkpoints/best.ckpt")
        for name in ("config.yaml", "environment.json", "summary.json"):
            path = source / name
            if path.exists():
                batch.put_file(path, f"{remote_root}/{name}")
    print(json.dumps({"uploaded": str(source), "remote_path": remote_root}, sort_keys=True))


@app.function(
    image=image,
    volumes={RUNTIME_ROOT.as_posix(): runtime_volume},
    timeout=10 * 60,
    cpu=2,
    memory=8192,
)
def runtime_status() -> dict[str, Any]:
    """Return a lightweight Modal runtime status without requesting a GPU."""
    _link_runtime_layout()
    return _status_payload()


@app.function(
    image=image,
    volumes={RUNTIME_ROOT.as_posix(): runtime_volume},
    gpu=GPU_TYPE,
    timeout=10 * 60,
    cpu=4,
    memory=16384,
)
def gpu_status() -> dict[str, Any]:
    """Return a lightweight Modal runtime status on A100-40GB."""
    _link_runtime_layout()
    return _status_payload(include_torch=True)


@app.function(
    image=image,
    volumes={RUNTIME_ROOT.as_posix(): runtime_volume},
    gpu=TRAIN_GPU_TYPE,
    timeout=10 * 60,
    cpu=8,
    memory=32768,
)
def train_gpu_status() -> dict[str, Any]:
    """Return a lightweight Modal runtime status on the training GPU."""
    _link_runtime_layout()
    return _status_payload(include_torch=True, gpu_type=TRAIN_GPU_TYPE)


@app.local_entrypoint()
def hydrate(artifact_name: str = "", force_extract: bool = False) -> None:
    """Hydrate uploaded runtime artifacts into the Modal Volume."""
    print(
        json.dumps(
            hydrate_runtime.remote(
                artifact_name=artifact_name,
                force_extract=force_extract,
            ),
            sort_keys=True,
        )
    )


@app.local_entrypoint()
def student(batch_size: int = 256, robustness: str = "all") -> None:
    """Run student calibration and evaluation on Modal A100-40GB."""
    print(
        json.dumps(
            run_student_end_to_end.remote(
                batch_size=batch_size, robustness=_robustness(robustness)
            ),
            sort_keys=True,
        )
    )


@app.local_entrypoint()
def teacher(batch_size: int = 128, robustness: str = "all") -> None:
    """Run teacher calibration and evaluation on Modal A100-40GB."""
    print(
        json.dumps(
            run_teacher_end_to_end.remote(
                batch_size=batch_size, robustness=_robustness(robustness)
            ),
            sort_keys=True,
        )
    )


@app.local_entrypoint()
def calibrate(run_dir: str, batch_size: int = 256, split: str = "calib") -> None:
    """Run calibration for an arbitrary uploaded run directory."""
    print(
        json.dumps(
            run_calibration.remote(
                checkpoint=f"{run_dir}/checkpoints/best.ckpt",
                output_dir=run_dir,
                batch_size=batch_size,
                split=split,
            ),
            sort_keys=True,
        )
    )


@app.local_entrypoint()
def evaluate(
    run_dir: str,
    batch_size: int = 256,
    split: str = "test",
    robustness: str = "all",
) -> None:
    """Run evaluation for an arbitrary uploaded run directory."""
    print(
        json.dumps(
            run_evaluation.remote(
                checkpoint=f"{run_dir}/checkpoints/best.ckpt",
                calibration=f"{run_dir}/calibration/diagnosis_temperature.json",
                output_dir=run_dir,
                batch_size=batch_size,
                split=split,
                robustness=_robustness(robustness),
            ),
            sort_keys=True,
        )
    )


@app.local_entrypoint()
def train_h200(
    teacher_run_name: str = DEFAULT_H200_TEACHER_RUN_NAME,
    student_run_name: str = DEFAULT_H200_STUDENT_RUN_NAME,
    batch_size: int = 64,
    max_epochs: int = 16,
    num_workers: int = 4,
    tabpfn_estimators: int = 8,
    precision: str = "bf16-mixed",
    log_every_n_steps: int = 10,
    warm_cache: bool = True,
    warm_batch_size: int = 256,
) -> None:
    """Train teacher then student sequentially on a single Modal H200.

    Speed is dominated by frozen TabPFN inference; this entrypoint installs an
    in-process row cache (``_runtime_patches``) so each ``(modality, row)`` is
    embedded at most once and the cache persists across the teacher→student
    handoff. After cache warm-up only the trainable transformer is on the hot
    path, which the H200 actually saturates.
    """
    print(
        json.dumps(
            run_teacher_student_training.remote(
                teacher_run_name=teacher_run_name,
                student_run_name=student_run_name,
                batch_size=batch_size,
                max_epochs=max_epochs,
                num_workers=num_workers,
                tabpfn_estimators=tabpfn_estimators,
                precision=precision,
                log_every_n_steps=log_every_n_steps,
                warm_cache=warm_cache,
                warm_batch_size=warm_batch_size,
            ),
            sort_keys=True,
        )
    )


@app.function(
    image=image,
    volumes={RUNTIME_ROOT.as_posix(): runtime_volume},
    timeout=60 * 60,
    cpu=4,
    memory=16384,
)
def hydrate_runtime(*, artifact_name: str = "", force_extract: bool = False) -> dict[str, Any]:
    """Extract a runtime archive into the mounted Volume and verify data stats."""
    _link_runtime_layout()
    archive = _resolve_archive(artifact_name)
    if force_extract or not (RUNTIME_ROOT / "data" / "master.parquet").exists():
        _check_archive_hash(archive)
        _extract_archive(archive)
    _normalize_manifest_location()
    _link_runtime_layout()
    _run_python(
        "-m",
        "agewell.scripts.runtime_artifacts",
        "rebase",
        "--repo-root",
        str(APP_ROOT),
    )
    _run_python(
        "-m",
        "agewell.scripts.runtime_artifacts",
        "verify",
        "--manifest",
        str(APP_ROOT / ".runtime" / "runtime_manifest.json"),
        "--allow-commit-mismatch",
        "--skip-key-hashes",
    )
    runtime_volume.commit()
    return {
        "artifact": str(archive),
        "manifest": str(RUNTIME_ROOT / ".runtime" / "runtime_manifest.json"),
        "volume": VOLUME_NAME,
    }


@app.function(
    image=image,
    volumes={RUNTIME_ROOT.as_posix(): runtime_volume},
    gpu=GPU_TYPE,
    timeout=DEFAULT_TIMEOUT_SECONDS,
    cpu=16,
    memory=49152,
)
def run_student_end_to_end(
    *,
    batch_size: int = 256,
    robustness: Literal["observed", "core", "all"] = "all",
) -> dict[str, Any]:
    """Calibrate and evaluate the default student checkpoint."""
    return _run_end_to_end(
        run_dir=DEFAULT_STUDENT_RUN,
        batch_size=batch_size,
        robustness=robustness,
    )


@app.function(
    image=image,
    volumes={RUNTIME_ROOT.as_posix(): runtime_volume},
    gpu=GPU_TYPE,
    timeout=DEFAULT_TIMEOUT_SECONDS,
    cpu=16,
    memory=49152,
)
def run_teacher_end_to_end(
    *,
    batch_size: int = 128,
    robustness: Literal["observed", "core", "all"] = "all",
) -> dict[str, Any]:
    """Calibrate and evaluate the default teacher checkpoint."""
    return _run_end_to_end(
        run_dir=DEFAULT_TEACHER_RUN,
        batch_size=batch_size,
        robustness=robustness,
    )


@app.function(
    image=image,
    volumes={RUNTIME_ROOT.as_posix(): runtime_volume},
    gpu=GPU_TYPE,
    timeout=DEFAULT_TIMEOUT_SECONDS,
    cpu=16,
    memory=49152,
)
def run_calibration(
    *,
    checkpoint: str,
    output_dir: str,
    batch_size: int = 256,
    split: str = "calib",
) -> dict[str, Any]:
    """Run the Phase 5 calibration CLI remotely."""
    _prepare_remote_runtime()
    result = _run_phase5_cli(
        "agewell.scripts.calibrate",
        "--checkpoint",
        checkpoint,
        "--output-dir",
        output_dir,
        "--split",
        split,
        "--batch-size",
        str(batch_size),
        "--accelerator",
        "gpu",
        "--tabpfn-device",
        "cuda",
    )
    runtime_volume.commit()
    return result


@app.function(
    image=image,
    volumes={RUNTIME_ROOT.as_posix(): runtime_volume},
    gpu=GPU_TYPE,
    timeout=DEFAULT_TIMEOUT_SECONDS,
    cpu=16,
    memory=49152,
)
def run_evaluation(
    *,
    checkpoint: str,
    calibration: str,
    output_dir: str,
    batch_size: int = 256,
    split: str = "test",
    robustness: Literal["observed", "core", "all"] = "all",
) -> dict[str, Any]:
    """Run the Phase 5 evaluation CLI remotely."""
    _prepare_remote_runtime()
    result = _run_phase5_cli(
        "agewell.scripts.evaluate",
        "--checkpoint",
        checkpoint,
        "--calibration",
        calibration,
        "--output-dir",
        output_dir,
        "--split",
        split,
        "--robustness",
        robustness,
        "--batch-size",
        str(batch_size),
        "--accelerator",
        "gpu",
        "--tabpfn-device",
        "cuda",
    )
    runtime_volume.commit()
    return result


@app.function(
    image=image,
    volumes={RUNTIME_ROOT.as_posix(): runtime_volume},
    gpu=TRAIN_GPU_TYPE,
    timeout=DEFAULT_TRAIN_TIMEOUT_SECONDS,
    cpu=32,
    memory=131072,
)
def run_teacher_student_training(
    *,
    teacher_run_name: str = DEFAULT_H200_TEACHER_RUN_NAME,
    student_run_name: str = DEFAULT_H200_STUDENT_RUN_NAME,
    batch_size: int = 64,
    max_epochs: int = 16,
    num_workers: int = 4,
    tabpfn_estimators: int = 8,
    precision: str = "bf16-mixed",
    log_every_n_steps: int = 10,
    warm_cache: bool = True,
    warm_batch_size: int = 256,
) -> dict[str, Any]:
    """Train Phase 5 teacher and student in-process on the training GPU.

    Both runs share one Python process so the TabPFN row cache populated during
    teacher training is reused by the student with no recompute. Subprocess
    streaming is intentionally avoided here because the monkey-patches in
    ``_runtime_patches`` must apply to the same interpreter that owns the
    encoder instances.
    """
    _prepare_remote_runtime()
    from agewell.modal_apps._runtime_patches import (
        cache_stats,
        clear_cache,
        install_all,
    )

    clear_cache()
    teacher_label = f"teacher:{teacher_run_name}"
    install_all(
        teacher_label,
        every_n_steps=log_every_n_steps,
        warm_cache=warm_cache,
        warm_batch_size=warm_batch_size,
    )
    teacher_started = time.monotonic()
    teacher = _run_train_in_process(
        "agewell.scripts.train_teacher",
        label=teacher_label,
        run_name=teacher_run_name,
        batch_size=batch_size,
        max_epochs=max_epochs,
        num_workers=num_workers,
        precision=precision,
        tabpfn_estimators=tabpfn_estimators,
        log_every_n_steps=log_every_n_steps,
    )
    teacher_elapsed = time.monotonic() - teacher_started
    teacher_checkpoint = str(
        teacher.get(
            "best_checkpoint",
            APP_ROOT / "outputs" / "runs" / teacher_run_name / "checkpoints" / "best.ckpt",
        )
    )
    if not Path(teacher_checkpoint).exists():
        raise FileNotFoundError(teacher_checkpoint)
    runtime_volume.commit()

    student_label = f"student:{student_run_name}"
    # Re-install with the new label; cache itself is preserved (clear_cache not called).
    install_all(
        student_label,
        every_n_steps=log_every_n_steps,
        warm_cache=warm_cache,
        warm_batch_size=warm_batch_size,
    )
    student_started = time.monotonic()
    student = _run_train_in_process(
        "agewell.scripts.train_student",
        "--teacher-checkpoint",
        teacher_checkpoint,
        label=student_label,
        run_name=student_run_name,
        batch_size=batch_size,
        max_epochs=max_epochs,
        num_workers=num_workers,
        precision=precision,
        tabpfn_estimators=tabpfn_estimators,
        log_every_n_steps=log_every_n_steps,
    )
    student_elapsed = time.monotonic() - student_started
    runtime_volume.commit()

    final_cache = cache_stats()
    print(
        json.dumps(
            {
                "summary": {
                    "teacher_elapsed_s": round(teacher_elapsed, 2),
                    "student_elapsed_s": round(student_elapsed, 2),
                    "total_elapsed_s": round(teacher_elapsed + student_elapsed, 2),
                    "tabpfn_cache": final_cache,
                }
            }
        ),
        flush=True,
    )
    return {
        "batch_size": batch_size,
        "gpu": TRAIN_GPU_TYPE,
        "max_epochs": max_epochs,
        "num_workers": num_workers,
        "precision": precision,
        "log_every_n_steps": log_every_n_steps,
        "warm_cache": warm_cache,
        "warm_batch_size": warm_batch_size,
        "student": student,
        "student_run_dir": str(APP_ROOT / "outputs" / "runs" / student_run_name),
        "student_elapsed_s": round(student_elapsed, 2),
        "tabpfn_estimators": tabpfn_estimators,
        "tabpfn_cache_final": final_cache,
        "teacher": teacher,
        "teacher_run_dir": str(APP_ROOT / "outputs" / "runs" / teacher_run_name),
        "teacher_elapsed_s": round(teacher_elapsed, 2),
        "total_elapsed_s": round(teacher_elapsed + student_elapsed, 2),
    }


def _run_end_to_end(
    *,
    run_dir: str,
    batch_size: int,
    robustness: Literal["observed", "core", "all"],
) -> dict[str, Any]:
    _prepare_remote_runtime()
    checkpoint = f"{run_dir}/checkpoints/best.ckpt"
    calibration = _run_phase5_cli(
        "agewell.scripts.calibrate",
        "--checkpoint",
        checkpoint,
        "--output-dir",
        run_dir,
        "--split",
        "calib",
        "--batch-size",
        str(batch_size),
        "--accelerator",
        "gpu",
        "--tabpfn-device",
        "cuda",
    )
    evaluation = _run_phase5_cli(
        "agewell.scripts.evaluate",
        "--checkpoint",
        checkpoint,
        "--calibration",
        f"{run_dir}/calibration/diagnosis_temperature.json",
        "--output-dir",
        run_dir,
        "--split",
        "test",
        "--robustness",
        robustness,
        "--batch-size",
        str(batch_size),
        "--accelerator",
        "gpu",
        "--tabpfn-device",
        "cuda",
    )
    runtime_volume.commit()
    return {"calibration": calibration, "evaluation": evaluation, "run_dir": run_dir}


def _prepare_remote_runtime() -> None:
    _link_runtime_layout()
    required = (
        APP_ROOT / "data" / "master.parquet",
        APP_ROOT / "models",
        APP_ROOT / "outputs" / "runs",
        Path.home() / ".cache" / "tabpfn" / "tabpfn-v2.5-classifier-v2.5_default.ckpt",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing Modal runtime paths. Run upload_runtime/upload_run/hydrate first: "
            + ", ".join(missing)
        )


def _status_payload(*, include_torch: bool = False, gpu_type: str = GPU_TYPE) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "app_root": str(APP_ROOT),
        "data_exists": (APP_ROOT / "data" / "master.parquet").exists(),
        "gpu": gpu_type if include_torch else "",
        "models_exists": (APP_ROOT / "models").exists(),
        "outputs_runs_exists": (APP_ROOT / "outputs" / "runs").exists(),
        "python": sys.version.split()[0],
        "volume": VOLUME_NAME,
    }
    if include_torch:
        import torch

        payload.update(
            {
                "cuda_available": torch.cuda.is_available(),
                "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
                "torch": torch.__version__,
            }
        )
    return payload


def _robustness(value: str) -> Literal["observed", "core", "all"]:
    if value not in {"observed", "core", "all"}:
        raise ValueError(f"Unsupported robustness value: {value}")
    return cast(Literal["observed", "core", "all"], value)


def _run_training_cli(
    module: str,
    *leading_args: str,
    run_name: str,
    batch_size: int,
    max_epochs: int,
    num_workers: int,
    precision: str,
    tabpfn_estimators: int,
) -> dict[str, Any]:
    return _run_phase5_cli_streaming(
        module,
        *leading_args,
        "--run-name",
        run_name,
        "--output-root",
        str(APP_ROOT / "outputs"),
        "--max-epochs",
        str(max_epochs),
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(num_workers),
        "--accelerator",
        "gpu",
        "--devices",
        "1",
        "--precision",
        precision,
        "--tabpfn-device",
        "cuda",
        "--tabpfn-estimators",
        str(tabpfn_estimators),
        "--overwrite",
        "training.pin_memory=true",
        "training.trainer.log_every_n_steps=1",
    )


def _run_train_in_process(
    module: str,
    *leading_args: str,
    label: str,
    run_name: str,
    batch_size: int,
    max_epochs: int,
    num_workers: int,
    precision: str,
    tabpfn_estimators: int,
    log_every_n_steps: int,
) -> dict[str, Any]:
    """Invoke a Phase 5 training entrypoint inside this Python process.

    Argparse-based ``main()`` is reused by swapping ``sys.argv``. Standard
    output is teed so live progress streams to Modal logs and the trailing
    JSON summary is still captured for the return value.
    """
    cli_args = [
        module,
        *leading_args,
        "--run-name",
        run_name,
        "--output-root",
        str(APP_ROOT / "outputs"),
        "--max-epochs",
        str(max_epochs),
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(num_workers),
        "--accelerator",
        "gpu",
        "--devices",
        "1",
        "--precision",
        precision,
        "--tabpfn-device",
        "cuda",
        "--tabpfn-estimators",
        str(tabpfn_estimators),
        "--overwrite",
        "training.pin_memory=true",
        f"training.trainer.log_every_n_steps={int(log_every_n_steps)}",
    ]
    print(f"[modal] in-process launch label={label} module={module}", flush=True)
    print(f"[modal] argv={cli_args[1:]}", flush=True)
    captured = io.StringIO()
    real_stdout = sys.stdout
    tee = _TeeStream(real_stdout, captured)
    old_argv = sys.argv
    try:
        sys.argv = list(cli_args)
        with contextlib.redirect_stdout(tee):
            mod = importlib.import_module(module)
            mod.main()
    finally:
        sys.argv = old_argv
    out = captured.getvalue()
    payload = _last_json_payload(out)
    if payload is None:
        return {"module": module, "stdout_tail": out[-2048:]}
    return payload


class _TeeStream:
    """Minimal tee for sys.stdout used by ``_run_train_in_process``."""

    def __init__(self, *streams: Any) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            with contextlib.suppress(Exception):
                stream.flush()

    def isatty(self) -> bool:
        return False


def _run_phase5_cli(module: str, *args: str) -> dict[str, Any]:
    completed = _run_python("-m", module, *args, capture=True)
    stdout = completed.stdout.strip() if completed.stdout else ""
    if not stdout:
        return {"module": module, "stdout": ""}
    payload = _last_json_payload(stdout)
    return payload if payload is not None else {"module": module, "stdout": stdout}


def _run_phase5_cli_streaming(module: str, *args: str) -> dict[str, Any]:
    command = [sys.executable, "-m", module, *args]
    env = os.environ.copy()
    env["PYTHONPATH"] = "/app/src"
    completed = subprocess.Popen(
        command,
        cwd=APP_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    stdout_lines: list[str] = []
    assert completed.stdout is not None
    for line in completed.stdout:
        stdout_lines.append(line)
        print(line, end="", flush=True)
    returncode = completed.wait()
    stdout = "".join(stdout_lines)
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, command, output=stdout)
    payload = _last_json_payload(stdout)
    return payload if payload is not None else {"module": module, "stdout": stdout}


def _last_json_payload(stdout: str) -> dict[str, Any] | None:
    for line in reversed(stdout.splitlines()):
        candidate = line.strip()
        if not candidate:
            continue
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        return payload if isinstance(payload, dict) else {"result": payload}
    return None


def _run_python(*args: str, capture: bool = False) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, *args]
    env = os.environ.copy()
    env["PYTHONPATH"] = "/app/src"
    if capture:
        completed = subprocess.run(
            command,
            cwd=APP_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if completed.stdout:
            print(completed.stdout, end="", flush=True)
    else:
        completed = subprocess.run(command, cwd=APP_ROOT, env=env, text=True, check=False)
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(
            completed.returncode,
            command,
            output=completed.stdout if capture else None,
        )
    return completed


def _link_runtime_layout() -> None:
    (RUNTIME_ROOT / "outputs" / "runs").mkdir(parents=True, exist_ok=True)
    (RUNTIME_ROOT / ".runtime").mkdir(parents=True, exist_ok=True)
    (RUNTIME_ROOT / "tabpfn_cache").mkdir(parents=True, exist_ok=True)
    _replace_symlink(APP_ROOT / "data", RUNTIME_ROOT / "data")
    _replace_symlink(APP_ROOT / "models", RUNTIME_ROOT / "models")
    _replace_symlink(APP_ROOT / "outputs", RUNTIME_ROOT / "outputs")
    _replace_symlink(APP_ROOT / ".runtime", RUNTIME_ROOT / ".runtime")
    cache_root = Path.home() / ".cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    _replace_symlink(cache_root / "tabpfn", RUNTIME_ROOT / "tabpfn_cache")


def _replace_symlink(link: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if link.resolve() == target.resolve():
            return
        link.unlink()
    elif link.exists():
        if link.is_dir():
            shutil.rmtree(link)
        else:
            link.unlink()
    link.symlink_to(target, target_is_directory=True)


def _resolve_archive(artifact_name: str) -> Path:
    artifacts_dir = RUNTIME_ROOT / "artifacts"
    if artifact_name:
        archive = artifacts_dir / artifact_name
        if not archive.exists():
            archive = RUNTIME_ROOT / artifact_name
        if not archive.exists():
            raise FileNotFoundError(archive)
        return archive
    matches = sorted(
        [
            *artifacts_dir.glob("agewell-runtime-*.tar.zst"),
            *artifacts_dir.glob("agewell-runtime-*.tar.gz"),
            *artifacts_dir.glob("agewell-runtime-*.tgz"),
        ],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise FileNotFoundError(f"No runtime archive found in {artifacts_dir}")
    return matches[0]


def _check_archive_hash(archive: Path) -> None:
    sidecar = archive.with_name(f"{archive.name}.sha256")
    if sidecar.exists():
        subprocess.run(["sha256sum", "-c", sidecar.name], cwd=archive.parent, check=True)


def _extract_archive(archive: Path) -> None:
    if archive.name.endswith((".tar.zst", ".tzst")):
        subprocess.run(
            ["tar", "-C", str(RUNTIME_ROOT), "-I", "zstd", "-xf", str(archive)], check=True
        )
    elif archive.name.endswith((".tar.gz", ".tgz")):
        subprocess.run(["tar", "-C", str(RUNTIME_ROOT), "-xzf", str(archive)], check=True)
    else:
        raise ValueError(f"Unsupported runtime archive extension: {archive}")


def _normalize_manifest_location() -> None:
    legacy = RUNTIME_ROOT / "runtime_manifest.json"
    target = RUNTIME_ROOT / ".runtime" / "runtime_manifest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    if legacy.exists():
        legacy.replace(target)
    if not target.exists():
        raise FileNotFoundError(target)
