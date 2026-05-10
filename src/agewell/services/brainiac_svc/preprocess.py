"""Online BrainIAC preprocessing transform."""

from __future__ import annotations

from pathlib import Path

import torch
from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    LoadImage,
    NormalizeIntensity,
    Orientation,
    Resize,
    ToTensor,
)

IMAGE_SIZE = (96, 96, 96)


def validation_transform() -> Compose:
    """Return the BrainIAC validation transform with defensive RAS orientation."""
    return Compose(
        [
            LoadImage(image_only=True),
            EnsureChannelFirst(),
            Orientation(axcodes="RAS"),
            Resize(spatial_size=IMAGE_SIZE, mode="trilinear"),
            NormalizeIntensity(nonzero=True, channel_wise=True),
            ToTensor(),
        ]
    )


def to_tensor(nifti_uri: str | Path) -> torch.Tensor:
    """Load and transform a NIfTI path into ``(1, 1, 96, 96, 96)``."""
    image = validation_transform()(str(nifti_uri))
    return image.unsqueeze(0).float()
