"""Lipid modality encoder."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from agewell.ml.encoders._base import D_TOKEN, BaseEncoder, require_2d_features


class LipidEncoder(BaseEncoder):
    """Small MLP for the fixed-width lipid/CSF biomarker panel."""

    modality = "lipid"

    def __init__(
        self,
        n_features: int = 213,
        hidden: int = 256,
        out_dim: int = D_TOKEN,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.LayerNorm(n_features),
            nn.Linear(n_features, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, batch: dict[str, Any]) -> Tensor:
        """Encode ``batch['lipid_features']`` into ``(B, 256)`` tokens."""
        x = torch.nan_to_num(require_2d_features(batch, "lipid_features", self.n_features))
        return torch.nan_to_num(self.net(x), nan=0.0, posinf=0.0, neginf=0.0)
