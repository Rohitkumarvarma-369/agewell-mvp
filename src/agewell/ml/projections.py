"""Shared projection blocks for Phase 3 encoders."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class LayerNormProjection(nn.Module):
    """LayerNorm followed by a linear projection."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Project ``x`` to ``out_dim`` with finite float32 output."""
        out = self.proj(self.norm(x.float()))
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
