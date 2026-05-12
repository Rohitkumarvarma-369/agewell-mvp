"""Calibration helpers for Phase 5 diagnosis probabilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class TemperatureCalibration:
    """Scalar temperature calibration result."""

    temperature: float
    n: int
    nll_before: float
    nll_after: float

    def to_dict(self) -> dict[str, float | int]:
        """Return a JSON-serializable representation."""
        return asdict(self)


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    max_iter: int = 100,
) -> TemperatureCalibration:
    """Fit a positive scalar temperature by minimizing cross-entropy."""
    logits_t = torch.as_tensor(logits, dtype=torch.float32)
    labels_t = torch.as_tensor(labels, dtype=torch.long)
    mask = labels_t >= 0
    logits_t = logits_t[mask]
    labels_t = labels_t[mask]
    if logits_t.numel() == 0:
        raise ValueError("Cannot fit temperature without valid diagnosis labels")

    log_temperature = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=0.1,
        max_iter=max_iter,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp_min(1.0e-6)
        loss = F.cross_entropy(logits_t / temperature, labels_t)
        loss.backward()
        return loss

    nll_before = float(F.cross_entropy(logits_t, labels_t).item())
    optimizer.step(closure)
    temperature = float(log_temperature.exp().clamp_min(1.0e-6).item())
    nll_after = float(F.cross_entropy(logits_t / temperature, labels_t).item())
    return TemperatureCalibration(
        temperature=temperature,
        n=int(labels_t.numel()),
        nll_before=nll_before,
        nll_after=nll_after,
    )
