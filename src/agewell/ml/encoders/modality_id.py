"""Modality identity embeddings."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from agewell.ml.encoders._base import D_TOKEN


class ModalityIDEmbedding(nn.Module):
    """Trainable embedding added to every modality token."""

    def __init__(self, n_modalities: int, out_dim: int = D_TOKEN) -> None:
        super().__init__()
        self.embedding = nn.Embedding(n_modalities, out_dim)

    def forward(self, batch_size: int, device: torch.device | None = None) -> Tensor:
        """Return IDs broadcast to ``(B, n_modalities, D_TOKEN)``."""
        indices = torch.arange(self.embedding.num_embeddings, device=device)
        ids = self.embedding(indices)
        return ids.unsqueeze(0).expand(batch_size, -1, -1)
