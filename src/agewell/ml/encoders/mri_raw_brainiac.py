"""Cached BrainIAC MRI encoder."""

from __future__ import annotations

from typing import Any

from torch import Tensor

from agewell.ml.encoders._base import D_TOKEN, BaseEncoder, require_2d_features
from agewell.ml.projections import LayerNormProjection


class BrainIACCachedEncoder(BaseEncoder):
    """Project cached 768-d BrainIAC CLS features to the model token size."""

    modality = "mri_raw"

    def __init__(self, in_dim: int = 768, out_dim: int = D_TOKEN) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.projection = LayerNormProjection(in_dim, out_dim)

    def forward(self, batch: dict[str, Any]) -> Tensor:
        """Encode ``batch['mri_raw_features']`` into ``(B, 256)`` tokens."""
        return self.projection(require_2d_features(batch, "mri_raw_features", self.in_dim))
