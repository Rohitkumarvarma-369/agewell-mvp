"""Phase 4 task and reconstruction heads."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from torch import Tensor, nn


class HeadStack(nn.Module):
    """Diagnosis, survival, cognitive, and modality-reconstruction heads."""

    def __init__(
        self,
        *,
        d_model: int = 256,
        n_classes: int = 5,
        n_surv_bins: int = 4,
        cdr_classes: int = 5,
        modality_names: Sequence[str],
        recon_hidden: int = 256,
    ) -> None:
        super().__init__()
        self.diag = nn.Linear(d_model, n_classes)
        self.surv = nn.Linear(d_model, n_surv_bins + 1)
        self.mmse = nn.Linear(d_model, 1)
        self.cdr = nn.Linear(d_model, cdr_classes)
        self.reconstruction_heads = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, recon_hidden),
                    nn.GELU(),
                    nn.Linear(recon_hidden, d_model),
                )
                for name in modality_names
            }
        )

    def forward(self, cls: Tensor, modality_tokens: Tensor) -> dict[str, Any]:
        """Return all task outputs from CLS and fused modality tokens."""
        return {
            "diag_logits": self.diag(cls),
            "surv_logits": self.surv(cls),
            "mmse_pred": self.mmse(cls).squeeze(-1),
            "cdr_logits": self.cdr(cls),
            "recon": {
                name: head(modality_tokens[:, idx])
                for idx, (name, head) in enumerate(self.reconstruction_heads.items())
            },
        }
