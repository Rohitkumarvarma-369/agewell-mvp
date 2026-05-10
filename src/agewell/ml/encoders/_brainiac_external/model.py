"""BrainIAC ViT backbone wrapper.

This intentionally mirrors BrainIAC/src/model.py. The important behavior is
strict checkpoint loading and CLS-token extraction via ``features[0][:, 0]``.
"""

from __future__ import annotations

from importlib.metadata import version
from pathlib import Path

import torch
from monai.networks.nets import ViT
from torch import nn

EXPECTED_MONAI_VERSION = "1.3.2"


class ViTBackboneNet(nn.Module):
    """BrainIAC 3D ViT backbone that returns the 768-d CLS token."""

    def __init__(
        self,
        simclr_ckpt_path: str | Path,
        *,
        require_monai_132: bool = True,
    ) -> None:
        super().__init__()
        if require_monai_132:
            _assert_monai_runtime()

        self.backbone = ViT(
            in_channels=1,
            img_size=(96, 96, 96),
            patch_size=(16, 16, 16),
            hidden_size=768,
            mlp_dim=3072,
            num_layers=12,
            num_heads=12,
            save_attn=True,
        )

        ckpt = torch.load(simclr_ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)
        backbone_state_dict = {
            key[9:]: value for key, value in state_dict.items() if key.startswith("backbone.")
        }
        if not backbone_state_dict:
            raise ValueError("BrainIAC checkpoint does not contain backbone.* weights")
        self.backbone.load_state_dict(backbone_state_dict, strict=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return BrainIAC's canonical CLS-token features, shape ``(B, 768)``."""
        features = self.backbone(x)
        return features[0][:, 0]


def _assert_monai_runtime() -> None:
    installed = version("monai")
    if installed != EXPECTED_MONAI_VERSION:
        raise RuntimeError(
            "BrainIAC strict-load requires monai=="
            f"{EXPECTED_MONAI_VERSION}; installed monai=={installed}. "
            "Pin this only inside the brainiac service image."
        )
