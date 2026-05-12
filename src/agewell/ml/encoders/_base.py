"""Base encoder contract for Phase 3 modality token producers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import Tensor

D_TOKEN: int = 256


class BaseEncoder(torch.nn.Module, ABC):
    """Base class for encoders that emit one 256-d token per sample."""

    modality: str
    out_dim: int = D_TOKEN

    @abstractmethod
    def forward(self, batch: dict[str, Any]) -> Tensor:
        """Return a finite tensor of shape ``(B, D_TOKEN)``."""

    def is_present(self, sample: dict[str, Any]) -> bool:
        """Return the sample-level presence flag for this encoder's modality."""
        modalities = sample.get("available_modalities", [])
        return self.modality in set(modalities or [])


def require_2d_features(batch: dict[str, Any], key: str, n_features: int) -> Tensor:
    """Load and validate a ``(B, n_features)`` tensor from a batch dict."""
    value = batch[key]
    tensor = value if isinstance(value, Tensor) else torch.as_tensor(value)
    if tensor.ndim != 2 or tensor.shape[1] != n_features:
        raise ValueError(f"{key} must have shape (B, {n_features}); got {tuple(tensor.shape)}")
    return tensor.float()
