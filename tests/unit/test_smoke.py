"""Phase 0 smoke tests."""

from pathlib import Path

from omegaconf import OmegaConf

import agewell
from agewell._common import paths
from agewell.config import StoragePaths, load_cfg


def test_version_string() -> None:
    """The package exposes a semantic-ish version string."""
    assert agewell.__version__ == "0.1.0"


def test_repo_root_exists() -> None:
    """Path helpers resolve to the project root."""
    root = paths.repo_root()
    assert root.exists()
    assert (root / "pyproject.toml").exists()


def test_storage_paths_from_env() -> None:
    """Storage paths are rooted in the repository."""
    storage = StoragePaths.from_env()
    assert storage.repo_root == paths.repo_root()
    assert storage.data_root == paths.repo_root() / "data"
    assert storage.model_root == paths.repo_root() / "models"


def test_hydra_config_loads() -> None:
    """The default Hydra composition loads without executing phase code."""
    cfg = load_cfg()
    assert cfg.seed == 1337
    assert Path(OmegaConf.to_container(cfg.data, resolve=True)["kaggle_root"]).is_absolute()
