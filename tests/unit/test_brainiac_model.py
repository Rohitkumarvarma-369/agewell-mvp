"""Tests for BrainIAC model wrapper behavior."""

from importlib.metadata import version

import pytest
import torch
from torch import nn

from agewell.ml.encoders._brainiac_external.model import EXPECTED_MONAI_VERSION, ViTBackboneNet


class _FakeBackbone(nn.Module):
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        batch = x.shape[0]
        tokens = torch.arange(batch * 217 * 768, dtype=torch.float32).reshape(batch, 217, 768)
        return tokens, []


def test_brainiac_forward_uses_cls_token_not_mean_pool() -> None:
    """BrainIAC returns features[0][:, 0], not an average over patch tokens."""
    model = ViTBackboneNet.__new__(ViTBackboneNet)
    nn.Module.__init__(model)
    model.backbone = _FakeBackbone()

    output = model(torch.zeros(2, 1, 96, 96, 96))
    tokens = model.backbone(torch.zeros(2, 1, 96, 96, 96))[0]

    assert torch.equal(output, tokens[:, 0])
    assert not torch.equal(output, tokens.mean(dim=1))


def test_brainiac_constructor_requires_service_monai_pin() -> None:
    """Local dev MONAI drift should fail before strict checkpoint loading."""
    if version("monai") == EXPECTED_MONAI_VERSION:
        pytest.skip("strict service MONAI version is installed")
    with pytest.raises(RuntimeError, match=r"monai==1\.3\.2"):
        ViTBackboneNet("models/brainiac-v1-simclr.pt")
