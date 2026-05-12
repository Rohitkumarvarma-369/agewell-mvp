"""APOE genetic encoder."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from agewell.ml.encoders._base import D_TOKEN, BaseEncoder


class APOEEncoder(BaseEncoder):
    """Embed APOE4 allele count into a 256-d token."""

    modality = "genetic"

    def __init__(self, out_dim: int = D_TOKEN) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.embedding = nn.Embedding(num_embeddings=3, embedding_dim=out_dim)

    def forward(self, batch: dict[str, Any]) -> Tensor:
        """Encode ``batch['genetic_apoe4']`` values in ``{0,1,2}``."""
        values = batch["genetic_apoe4"]
        tensor = values if isinstance(values, Tensor) else torch.as_tensor(values)
        return self.embedding(tensor.long().clamp(0, 2))
