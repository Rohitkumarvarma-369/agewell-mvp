"""Structured modality dropout for Phase 4 fusion training."""

from __future__ import annotations

import torch
from torch import Tensor


def scheduled_modality_dropout_p(
    *,
    global_step: int,
    total_steps: int,
    warmup_steps: int,
    max_p: float,
) -> float:
    """Return linearly warmed modality-dropout probability."""
    if max_p <= 0.0:
        return 0.0
    if total_steps <= 0:
        return max(0.0, min(float(max_p), 1.0))
    if global_step < warmup_steps:
        return 0.0
    ramp_steps = max(total_steps - warmup_steps, 1)
    progress = min(max((global_step - warmup_steps) / ramp_steps, 0.0), 1.0)
    return max(0.0, min(float(max_p) * progress, 1.0))


def sample_modality_mask(
    presence: Tensor,
    *,
    p_drop: float,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Randomly hide present modalities while preserving at least one per sample."""
    if presence.ndim != 2:
        raise ValueError(f"presence must be 2D; got {tuple(presence.shape)}")
    observed = presence.bool()
    if p_drop <= 0.0:
        return observed.clone()
    random_keep = torch.rand(
        observed.shape,
        device=observed.device,
        generator=generator,
    ) >= min(float(p_drop), 1.0)
    kept = observed & random_keep
    rows_need_restore = observed.any(dim=1) & ~kept.any(dim=1)
    for raw_row_idx in rows_need_restore.nonzero(as_tuple=False).flatten().tolist():
        row_idx = int(raw_row_idx)
        available = observed[row_idx].nonzero(as_tuple=False).flatten()
        choice_idx = int(
            torch.randint(
                available.numel(),
                (1,),
                device=observed.device,
                generator=generator,
            ).item()
        )
        modality_idx = int(available[choice_idx].item())
        kept[row_idx, modality_idx] = True
    return kept
