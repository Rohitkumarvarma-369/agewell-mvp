"""Lightning data module for Phase 4 local and cloud training."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import lightning as L
import pandas as pd
from torch.utils.data import DataLoader, Dataset

from agewell.config import repo_root
from agewell.data.splits import subject_disjoint_split
from agewell.ml.config_utils import cfg_select
from agewell.ml.feature_extractors import compose_batch, compute_imputation_stats


class MasterRowsDataset(Dataset[dict[str, Any]]):
    """Thin row-dict dataset over a master split dataframe."""

    def __init__(self, rows: pd.DataFrame) -> None:
        self._records = list(rows.reset_index(drop=True).to_dict(orient="records"))

    def __len__(self) -> int:
        """Return row count."""
        return len(self._records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one row as a mutable plain dict."""
        return dict(self._records[index])


class AgeWellDataModule(L.LightningDataModule):
    """Load subject-disjoint master splits and compose Phase 4 batches."""

    def __init__(
        self,
        *,
        master_path: str | Path = "data/master.parquet",
        splits_dir: str | Path = "data/splits",
        subset: str = "all",
        batch_size: int = 32,
        num_workers: int = 0,
        strict_mri_paths: bool = True,
        seed: int = 1337,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()
        self.master_path = _repo_path(master_path)
        self.splits_dir = _repo_path(splits_dir)
        self.subset = subset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.strict_mri_paths = strict_mri_paths
        self.seed = seed
        self.pin_memory = pin_memory
        self.train_df: pd.DataFrame | None = None
        self.val_df: pd.DataFrame | None = None
        self.test_df: pd.DataFrame | None = None
        self.imputation_stats: dict[str, dict[str, float]] | None = None

    @classmethod
    def from_cfg(cls, cfg: Any, *, strict_mri_paths: bool | None = None) -> AgeWellDataModule:
        """Build from the project Hydra config."""
        strict = bool(
            cfg_select(cfg, "training.strict_mri_paths", True)
            if strict_mri_paths is None
            else strict_mri_paths
        )
        return cls(
            master_path=str(cfg_select(cfg, "data.master_path", "data/master.parquet")),
            splits_dir=str(cfg_select(cfg, "data.splits_dir", "data/splits")),
            subset=str(cfg_select(cfg, "training.subset", "all")),
            batch_size=int(cfg_select(cfg, "training.batch_size", 32)),
            num_workers=int(cfg_select(cfg, "training.num_workers", 0)),
            strict_mri_paths=strict,
            seed=int(cfg_select(cfg, "seed", 1337)),
            pin_memory=bool(cfg_select(cfg, "training.pin_memory", False)),
        )

    @property
    def context_df(self) -> pd.DataFrame:
        """Training dataframe used for TabPFN context fitting."""
        if self.train_df is None:
            raise RuntimeError("setup('fit') must be called before accessing context_df")
        return self.train_df

    def setup(self, stage: str | None = None) -> None:
        """Load or synthesize train/calib/test splits."""
        if self.train_df is not None:
            return
        splits = self._load_splits()
        self.train_df = self._filter_subset(splits["train"])
        self.val_df = self._filter_subset(splits["calib"])
        self.test_df = self._filter_subset(splits["test"])
        self.imputation_stats = compute_imputation_stats(self.train_df)

    def train_dataloader(self) -> DataLoader[dict[str, Any]]:
        """Return training dataloader."""
        return self._loader(self._required_df(self.train_df, "train"))

    def val_dataloader(self) -> DataLoader[dict[str, Any]]:
        """Return calibration/validation dataloader."""
        return self._loader(self._required_df(self.val_df, "calib"), shuffle=False)

    def test_dataloader(self) -> DataLoader[dict[str, Any]]:
        """Return test dataloader."""
        return self._loader(self._required_df(self.test_df, "test"), shuffle=False)

    def collate_rows(self, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        """Compose row dicts into tensor batch dictionaries."""
        if self.imputation_stats is None:
            raise RuntimeError("setup() must run before collate_rows")
        return compose_batch(
            pd.DataFrame(list(rows)),
            imputation_stats=self.imputation_stats,
            strict_mri_paths=self.strict_mri_paths,
        )

    def _load_splits(self) -> dict[str, pd.DataFrame]:
        split_paths = {
            name: self.splits_dir / f"{name}.parquet" for name in ("train", "calib", "test")
        }
        if all(path.exists() for path in split_paths.values()):
            return {name: pd.read_parquet(path) for name, path in split_paths.items()}
        master = pd.read_parquet(self.master_path)
        return subject_disjoint_split(master, seed=self.seed)

    def _filter_subset(self, rows: pd.DataFrame) -> pd.DataFrame:
        if self.subset in {"all", ""}:
            return rows.reset_index(drop=True)
        if self.subset in {"rich", "rich_only"}:
            return rows[rows["is_rich"].fillna(False)].reset_index(drop=True)
        if self.subset.startswith("cohort:"):
            cohort = self.subset.split(":", maxsplit=1)[1]
            return rows[rows["cohort"].astype(str) == cohort].reset_index(drop=True)
        raise ValueError(f"Unsupported Phase 4 subset: {self.subset}")

    def _loader(
        self,
        rows: pd.DataFrame,
        *,
        shuffle: bool = True,
    ) -> DataLoader[dict[str, Any]]:
        return DataLoader(
            MasterRowsDataset(rows),
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            collate_fn=self.collate_rows,
        )

    @staticmethod
    def _required_df(rows: pd.DataFrame | None, name: str) -> pd.DataFrame:
        if rows is None:
            raise RuntimeError(f"setup() must run before requesting {name} dataloader")
        return rows


def _repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else repo_root() / candidate
