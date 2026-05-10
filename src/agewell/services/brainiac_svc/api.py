"""FastAPI service for BrainIAC feature extraction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pydantic import BaseModel

from agewell.services._common.api import build_app
from agewell.services._common.cache import derivative_root, imaging_cache_stem
from agewell.services.brainiac_svc.encoder import BrainIACEncoder

app = build_app("brainiac-svc")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_ENCODER: BrainIACEncoder | None = None


class EncodeRequest(BaseModel):
    """Request payload for BrainIAC encoding."""

    subject_id: str
    visit_idx: int = 0
    preprocessed_uri: str
    preprocess_version: str
    source_nifti_uri: str | None = None


class EncodeResponse(BaseModel):
    """Response payload for BrainIAC encoding."""

    features_uri: str
    projected_uri: str
    feature_dim: int = 768
    projected_dim: int = 256
    embedding_norm: float
    normalized_mean: float
    normalized_std: float
    cache_hit: bool = False


@app.post("/encode", response_model=EncodeResponse)
def encode(req: EncodeRequest) -> EncodeResponse:
    """Encode a preprocessed NIfTI into cached BrainIAC feature files."""
    cache_source = req.source_nifti_uri or req.preprocessed_uri
    stem = imaging_cache_stem(cache_source, req.preprocess_version)
    out_dir = derivative_root("brainiac")
    out_dir.mkdir(parents=True, exist_ok=True)
    features_path = out_dir / f"{stem}.npy"
    projected_path = out_dir / f"{stem}_proj.npy"
    metadata_path = out_dir / f"{stem}.json"

    if features_path.exists() and projected_path.exists() and metadata_path.exists():
        raw = np.load(features_path)
        metadata = _load_metadata(metadata_path)
        return _response(features_path, projected_path, raw, metadata, cache_hit=True)

    encoder = _get_encoder()
    output = encoder.encode_file(req.preprocessed_uri, DEVICE)
    raw = output.features.numpy().astype(np.float32)
    projected = output.projected.numpy().astype(np.float32)
    metadata = {
        "embedding_norm": float(np.linalg.norm(raw)),
        "normalized_mean": output.normalized_mean,
        "normalized_std": output.normalized_std,
        "preprocess_version": req.preprocess_version,
        "source_uri": cache_source,
    }
    np.save(features_path, raw)
    np.save(projected_path, projected)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return _response(features_path, projected_path, raw, metadata, cache_hit=False)


def _get_encoder() -> BrainIACEncoder:
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = BrainIACEncoder().to(DEVICE).eval()
    return _ENCODER


def _response(
    features_path: Path,
    projected_path: Path,
    raw: np.ndarray,
    metadata: dict[str, Any],
    *,
    cache_hit: bool,
) -> EncodeResponse:
    norm = float(metadata.get("embedding_norm", np.linalg.norm(raw)))
    return EncodeResponse(
        features_uri=str(features_path),
        projected_uri=str(projected_path),
        embedding_norm=norm,
        normalized_mean=float(metadata["normalized_mean"]),
        normalized_std=float(metadata["normalized_std"]),
        cache_hit=cache_hit,
    )


def _load_metadata(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "normalized_mean" not in data or "normalized_std" not in data:
        raise ValueError(f"BrainIAC metadata is missing normalized stats: {path}")
    return data
