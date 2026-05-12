"""Frozen TabPFN embedding encoder for tabular modalities."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor, nn

from agewell.ml.encoders._base import D_TOKEN, BaseEncoder, require_2d_features
from agewell.ml.projections import LayerNormProjection

EmbeddingBackend = Callable[[np.ndarray], np.ndarray]


class TabPFNFrozenEncoder(BaseEncoder):
    """Use a fitted TabPFN V2.5 classifier as a frozen embedding extractor."""

    def __init__(
        self,
        *,
        modality: str,
        n_features: int,
        ctx_X: Tensor,
        ctx_y: Tensor,
        out_dim: int = D_TOKEN,
        embedding_dim: int = 192,
        n_estimators: int = 8,
        fit_mode: Literal["low_memory", "fit_preprocessors", "fit_with_cache"] = "fit_with_cache",
        embedding_backend: EmbeddingBackend | None = None,
        device: str | None = None,
    ) -> None:
        super().__init__()
        self.modality = modality
        self.n_features = n_features
        self.out_dim = out_dim
        self.embedding_dim = embedding_dim
        self.n_estimators = n_estimators
        self.fit_mode = fit_mode
        self.device = device
        self.ctx_X: Tensor
        self.ctx_y: Tensor
        self.register_buffer("ctx_X", ctx_X.float(), persistent=False)
        self.register_buffer("ctx_y", ctx_y.long(), persistent=False)
        self._embedding_backend = embedding_backend
        self._tabpfn_model: Any | None = None
        self.projection = LayerNormProjection(embedding_dim, out_dim)

    def forward(self, batch: dict[str, Any]) -> Tensor:
        """Embed ``{modality}_features`` and project them to ``(B, 256)``."""
        key = f"{self.modality}_features"
        x = require_2d_features(batch, key, self.n_features)
        with torch.no_grad():
            emb = self._embed(x)
        return self.projection(emb.to(device=x.device, dtype=x.dtype))

    def _embed(self, x: Tensor) -> Tensor:
        """Return averaged TabPFN embeddings as a ``(B, embedding_dim)`` tensor."""
        x_np = x.detach().cpu().numpy().astype(np.float32)
        if self._embedding_backend is not None:
            raw = self._embedding_backend(x_np)
        else:
            raw = self._get_or_fit_model().get_embeddings(x_np, data_source="test")
        arr = np.asarray(raw, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr.mean(axis=0)
        if arr.ndim != 2:
            raise ValueError(f"TabPFN embeddings must be 2D or 3D; got shape {arr.shape}")
        if arr.shape[1] != self.embedding_dim:
            raise ValueError(
                f"TabPFN embedding dim mismatch: expected {self.embedding_dim}, got {arr.shape[1]}"
            )
        return torch.from_numpy(arr)

    def _get_or_fit_model(self) -> Any:
        if self._tabpfn_model is not None:
            return self._tabpfn_model
        try:
            from tabpfn import TabPFNClassifier
            from tabpfn.constants import ModelVersion
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "TabPFNFrozenEncoder requires tabpfn>=7.1 with ModelVersion.V2_5"
            ) from exc

        kwargs: dict[str, Any] = {
            "fit_mode": self.fit_mode,
            "n_estimators": self.n_estimators,
        }
        if self.device is not None:
            kwargs["device"] = self.device
        model = TabPFNClassifier.create_default_for_version(ModelVersion.V2_5, **kwargs)
        model.fit(
            self.ctx_X.detach().cpu().numpy().astype(np.float32),
            self.ctx_y.detach().cpu().numpy(),
        )
        self._tabpfn_model = model
        return model


def freeze_module(module: nn.Module) -> nn.Module:
    """Set all existing module parameters to ``requires_grad=False``."""
    for param in module.parameters():
        param.requires_grad = False
    return module
