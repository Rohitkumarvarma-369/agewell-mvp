"""BrainIAC encoder service module."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from agewell._common.paths import models_root
from agewell.ml.encoders._brainiac_external.model import ViTBackboneNet
from agewell.services.brainiac_svc.preprocess import to_tensor

DEFAULT_CHECKPOINT = models_root() / "brainiac-v1-simclr.pt"


@dataclass(frozen=True)
class BrainIACEncodeOutput:
    """BrainIAC embeddings plus online preprocessing QC statistics."""

    features: torch.Tensor
    projected: torch.Tensor
    normalized_mean: float
    normalized_std: float


class BrainIACEncoder(nn.Module):
    """Frozen BrainIAC backbone plus deterministic 256-d compatibility projection."""

    def __init__(self, checkpoint_path: str | Path = DEFAULT_CHECKPOINT) -> None:
        super().__init__()
        self.backbone = ViTBackboneNet(checkpoint_path)
        self.norm = nn.LayerNorm(768)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1337)
            self.proj = nn.Linear(768, 256)
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)

    @torch.inference_mode()
    def encode_file(self, nifti_uri: str | Path, device: torch.device) -> BrainIACEncodeOutput:
        """Encode a preprocessed NIfTI into 768-d raw and 256-d projected features."""
        x = to_tensor(nifti_uri).to(device)
        normalized_mean = float(x.mean().cpu())
        normalized_std = float(x.std(unbiased=False).cpu())
        raw = self.backbone(x)
        projected = self.proj(self.norm(raw))
        return BrainIACEncodeOutput(
            features=raw.squeeze(0).cpu(),
            projected=projected.squeeze(0).cpu(),
            normalized_mean=normalized_mean,
            normalized_std=normalized_std,
        )
