"""Canonical filesystem paths for the repository."""

from pathlib import Path


def repo_root() -> Path:
    """Return the repository root."""
    return Path(__file__).resolve().parents[3]


def data_root() -> Path:
    """Return the canonical data directory."""
    return repo_root() / "data"


def models_root() -> Path:
    """Return the canonical models directory."""
    return repo_root() / "models"


def kaggle_downloads_root() -> Path:
    """Return the local Kaggle downloads directory used by the public baseline."""
    return Path("/home/rohit/kaggle-iisc/kaggle-cli/downloads")
