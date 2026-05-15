"""Runtime-only patches for Modal-side training and evaluation.

These patches are intentionally confined to the Modal app layer. They do not
modify any model, encoder, datamodule, or training scaffolding on disk. They
monkey-patch behaviour after the relevant modules import, so the same code on
a developer laptop is unaffected unless ``install_all`` is invoked.

The two patches:

1. ``install_tabpfn_row_cache`` wraps ``TabPFNFrozenEncoder._embed`` so each
   ``(modality, feature_vector)`` pair is computed at most once per Python
   process. TabPFN is frozen and its fit context is deterministic, so cached
   embeddings are bit-equivalent to fresh ones.

2. ``install_logging_callbacks`` wraps ``_phase5_common.build_trainer`` to
   attach Lightning callbacks that print per-step throughput, per-epoch wall
   clock, and TabPFN cache statistics. ``TabPFNCacheWarmer`` optionally
   prefills the cache from ``on_fit_start`` so step 1 of training already runs
   at cache-hit speed.
"""

from __future__ import annotations

import hashlib
import os
import time
from typing import Any

import numpy as np
import torch
from lightning.pytorch.callbacks import Callback

_CACHE: dict[tuple[str, bytes], np.ndarray] = {}
_STATS: dict[str, int] = {"hits": 0, "misses": 0, "encoder_calls": 0}


def cache_stats() -> dict[str, Any]:
    """Return a snapshot of TabPFN cache counters."""
    hits = _STATS["hits"]
    misses = _STATS["misses"]
    total = hits + misses
    return {
        "hits": hits,
        "misses": misses,
        "rows_seen": total,
        "hit_ratio": (hits / total) if total else 0.0,
        "encoder_calls": _STATS["encoder_calls"],
        "cached_rows": len(_CACHE),
    }


def clear_cache() -> None:
    """Reset cache and counters. Safe to call between unrelated runs."""
    _CACHE.clear()
    _STATS.update({"hits": 0, "misses": 0, "encoder_calls": 0})


def _row_key(modality: str, row: np.ndarray) -> tuple[str, bytes]:
    digest = hashlib.blake2b(np.ascontiguousarray(row).tobytes(), digest_size=16).digest()
    return (modality, digest)


def install_tabpfn_row_cache() -> None:
    """Wrap ``TabPFNFrozenEncoder._embed`` with a per-row cache.

    Idempotent. Disabled by ``AGEWELL_TABPFN_CACHE=0``. Cache miss rows still
    flow through the original ``_embed``, so behaviour matches exactly.
    """
    if os.environ.get("AGEWELL_TABPFN_CACHE", "1") == "0":
        print("[tabpfn-cache] disabled by AGEWELL_TABPFN_CACHE=0", flush=True)
        return

    from agewell.ml.encoders.tabular_tabpfn import TabPFNFrozenEncoder

    orig_embed = TabPFNFrozenEncoder._embed
    if getattr(orig_embed, "_agewell_cached", False):
        return

    def cached_embed(self: Any, x: torch.Tensor) -> torch.Tensor:
        _STATS["encoder_calls"] += 1
        x_np = x.detach().cpu().numpy().astype(np.float32, copy=False)
        batch_size = x_np.shape[0]
        out = np.empty((batch_size, self.embedding_dim), dtype=np.float32)
        missing: list[tuple[int, tuple[str, bytes]]] = []
        for i in range(batch_size):
            key = _row_key(self.modality, x_np[i])
            cached = _CACHE.get(key)
            if cached is None:
                missing.append((i, key))
            else:
                out[i] = cached
                _STATS["hits"] += 1
        if missing:
            idxs = [i for i, _ in missing]
            miss_tensor = torch.from_numpy(np.ascontiguousarray(x_np[idxs]))
            miss_emb = orig_embed(self, miss_tensor)
            miss_np = miss_emb.detach().cpu().numpy()
            for j, (i, key) in enumerate(missing):
                out[i] = miss_np[j]
                _CACHE[key] = miss_np[j]
                _STATS["misses"] += 1
        return torch.from_numpy(out)

    cached_embed._agewell_cached = True  # type: ignore[attr-defined]
    TabPFNFrozenEncoder._embed = cached_embed  # type: ignore[method-assign]
    print("[tabpfn-cache] row cache installed on TabPFNFrozenEncoder._embed", flush=True)


class TabPFNCacheWarmer(Callback):
    """Eager prefill of the TabPFN row cache for all splits before training."""

    def __init__(self, run_label: str = "", batch_size: int = 256) -> None:
        super().__init__()
        self.label = run_label
        self.batch_size = max(int(batch_size), 1)
        self._done = False

    def on_fit_start(self, trainer: Any, pl_module: Any) -> None:
        if self._done:
            return
        self._done = True
        if os.environ.get("AGEWELL_TABPFN_CACHE", "1") == "0":
            return
        if os.environ.get("AGEWELL_TABPFN_WARM", "1") == "0":
            print(f"[{self.label}] warm disabled by AGEWELL_TABPFN_WARM=0", flush=True)
            return
        dm = getattr(trainer, "datamodule", None)
        if dm is None:
            print(f"[{self.label}] no datamodule attached, skipping warm", flush=True)
            return
        try:
            from agewell.ml.encoders.tabular_tabpfn import TabPFNFrozenEncoder
        except Exception as exc:
            print(f"[{self.label}] could not import TabPFN encoder: {exc}", flush=True)
            return

        import pandas as pd

        frames = []
        for split in ("train_df", "val_df", "test_df"):
            df = getattr(dm, split, None)
            if df is not None and len(df):
                frames.append(df)
        if not frames:
            return
        all_rows = pd.concat(frames, ignore_index=True)
        total = len(all_rows)
        device = next(pl_module.parameters()).device
        tabpfn_encoders = {
            name: enc
            for name, enc in pl_module.model.encoders.items()
            if isinstance(enc, TabPFNFrozenEncoder)
        }
        if not tabpfn_encoders:
            return
        print(
            f"[{self.label}] tabpfn-warm: {total} rows x {len(tabpfn_encoders)} modalities"
            f" ({list(tabpfn_encoders)}), chunk={self.batch_size}",
            flush=True,
        )
        started = time.monotonic()
        before = cache_stats()
        chunks = (total + self.batch_size - 1) // self.batch_size
        pl_module.eval()
        with torch.no_grad():
            for chunk_idx, start in enumerate(range(0, total, self.batch_size), start=1):
                chunk_df = all_rows.iloc[start : start + self.batch_size]
                batch = dm.collate_rows(chunk_df.to_dict(orient="records"))
                batch_dev = {
                    key: (value.to(device) if isinstance(value, torch.Tensor) else value)
                    for key, value in batch.items()
                }
                for enc in tabpfn_encoders.values():
                    enc(batch_dev)
                if chunk_idx % max(chunks // 10, 1) == 0 or chunk_idx == chunks:
                    elapsed = time.monotonic() - started
                    cs = cache_stats()
                    print(
                        f"[{self.label}] tabpfn-warm chunk={chunk_idx}/{chunks}"
                        f" elapsed={elapsed:.1f}s cached_rows={cs['cached_rows']}",
                        flush=True,
                    )
        pl_module.train()
        elapsed = time.monotonic() - started
        after = cache_stats()
        print(
            f"[{self.label}] tabpfn-warm done in {elapsed:.1f}s"
            f" cached_rows={after['cached_rows']}"
            f" added={after['cached_rows'] - before['cached_rows']}",
            flush=True,
        )


class StepHeartbeat(Callback):
    """Periodic step-rate, loss, GPU memory, and TabPFN hit-rate line."""

    def __init__(self, every_n_steps: int = 10, run_label: str = "") -> None:
        super().__init__()
        self.every = max(int(every_n_steps), 1)
        self.label = run_label
        self.t_start = 0.0
        self.t_last = 0.0
        self.last_step = 0

    def on_train_start(self, trainer: Any, pl_module: Any) -> None:
        now = time.monotonic()
        self.t_start = now
        self.t_last = now
        self.last_step = 0
        print(
            f"[{self.label}] train start"
            f" max_epochs={trainer.max_epochs} max_steps={trainer.max_steps}"
            f" precision={trainer.precision} devices={trainer.num_devices}",
            flush=True,
        )

    def on_train_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        step = int(trainer.global_step)
        if step == 0 or step % self.every != 0:
            return
        now = time.monotonic()
        dt = max(now - self.t_last, 1e-9)
        steps = max(step - self.last_step, 1)
        self.t_last = now
        self.last_step = step
        if torch.cuda.is_available():
            mem_alloc = torch.cuda.memory_allocated() / (1 << 30)
            mem_peak = torch.cuda.max_memory_allocated() / (1 << 30)
        else:
            mem_alloc = mem_peak = 0.0
        if isinstance(outputs, dict) and "loss" in outputs:
            loss_val = float(outputs["loss"])
        elif isinstance(outputs, torch.Tensor):
            loss_val = float(outputs)
        else:
            loss_val = float("nan")
        cs = cache_stats()
        print(
            f"[{self.label}] step={step:5d} epoch={trainer.current_epoch}"
            f" loss={loss_val:.4f} steps/s={steps / dt:5.2f}"
            f" gpu_mem={mem_alloc:5.1f}/{mem_peak:5.1f}GiB"
            f" tabpfn_hit={cs['hit_ratio']:.1%} cached={cs['cached_rows']}",
            flush=True,
        )

    def on_train_end(self, trainer: Any, pl_module: Any) -> None:
        elapsed = time.monotonic() - self.t_start
        peak = torch.cuda.max_memory_allocated() / (1 << 30) if torch.cuda.is_available() else 0.0
        cs = cache_stats()
        print(
            f"[{self.label}] train end elapsed={elapsed:.1f}s"
            f" peak_gpu_mem={peak:.1f}GiB"
            f" tabpfn_hits={cs['hits']} misses={cs['misses']}"
            f" hit_ratio={cs['hit_ratio']:.2%} cached_rows={cs['cached_rows']}",
            flush=True,
        )


class EpochClock(Callback):
    """Per-epoch wall-clock and validation metric summary."""

    def __init__(self, run_label: str = "") -> None:
        super().__init__()
        self.label = run_label
        self._t_epoch = 0.0

    def on_train_epoch_start(self, trainer: Any, pl_module: Any) -> None:
        self._t_epoch = time.monotonic()

    def on_train_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        elapsed = time.monotonic() - self._t_epoch
        cs = cache_stats()
        print(
            f"[{self.label}] epoch {trainer.current_epoch} train done"
            f" elapsed={elapsed:.1f}s cached_rows={cs['cached_rows']}"
            f" tabpfn_hit={cs['hit_ratio']:.2%}",
            flush=True,
        )

    def on_validation_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        metrics = trainer.callback_metrics
        parts = []
        for key in ("val/loss", "val/diag_macro_f1", "val/diag"):
            value = metrics.get(key)
            if value is None:
                continue
            try:
                parts.append(f"{key}={float(value):.4f}")
            except (TypeError, ValueError):
                continue
        if parts:
            print(
                f"[{self.label}] val epoch={trainer.current_epoch} " + " ".join(parts),
                flush=True,
            )


def install_logging_callbacks(
    run_label: str,
    *,
    every_n_steps: int = 10,
    warm_cache: bool = True,
    warm_batch_size: int = 256,
) -> None:
    """Wrap ``_phase5_common.build_trainer`` to append heartbeat + warmer callbacks."""
    import agewell.scripts._phase5_common as _common

    orig_build = _common.build_trainer
    if getattr(orig_build, "_agewell_patched", False):
        # Already wrapped; just update the closure label by re-wrapping the original.
        orig_build = getattr(orig_build, "_agewell_orig", orig_build)

    def patched_build(cfg: Any, args: Any, artifacts: Any) -> Any:
        trainer, ckpt = orig_build(cfg, args, artifacts)
        extras: list[Callback] = []
        if warm_cache:
            extras.append(TabPFNCacheWarmer(run_label=run_label, batch_size=warm_batch_size))
        extras.append(StepHeartbeat(every_n_steps=every_n_steps, run_label=run_label))
        extras.append(EpochClock(run_label=run_label))
        trainer.callbacks.extend(extras)  # type: ignore[attr-defined]
        return trainer, ckpt

    patched_build._agewell_patched = True  # type: ignore[attr-defined]
    patched_build._agewell_orig = orig_build  # type: ignore[attr-defined]
    _common.build_trainer = patched_build
    print(
        f"[runtime-patches] logging callbacks installed for label={run_label!r}"
        f" every_n_steps={every_n_steps} warm_cache={warm_cache}",
        flush=True,
    )


def install_all(
    run_label: str,
    *,
    every_n_steps: int = 10,
    warm_cache: bool = True,
    warm_batch_size: int = 256,
) -> None:
    """Install both TabPFN row cache and logging callbacks. Idempotent."""
    install_tabpfn_row_cache()
    install_logging_callbacks(
        run_label,
        every_n_steps=every_n_steps,
        warm_cache=warm_cache,
        warm_batch_size=warm_batch_size,
    )


__all__ = [
    "EpochClock",
    "StepHeartbeat",
    "TabPFNCacheWarmer",
    "cache_stats",
    "clear_cache",
    "install_all",
    "install_logging_callbacks",
    "install_tabpfn_row_cache",
]
