"""Centralized Hydra and settings loading."""

from pathlib import Path

import hydra
from omegaconf import DictConfig
from pydantic import BaseModel


class StoragePaths(BaseModel):
    """Canonical local storage roots used across scripts and services."""

    repo_root: Path
    data_root: Path
    model_root: Path

    @classmethod
    def from_env(cls) -> "StoragePaths":
        """Build storage roots relative to the repository."""
        repo = Path(__file__).resolve().parents[2]
        return cls(repo_root=repo, data_root=repo / "data", model_root=repo / "models")


def repo_root() -> Path:
    """Return the repository root for the installed source tree."""
    return Path(__file__).resolve().parents[2]


def load_cfg(overrides: list[str] | None = None, config_name: str = "default") -> DictConfig:
    """Compose the Hydra config from the repository configs directory."""
    config_dir = repo_root() / "configs"
    with hydra.initialize_config_dir(config_dir=str(config_dir), version_base=None):
        return hydra.compose(config_name=config_name, overrides=overrides or [])
