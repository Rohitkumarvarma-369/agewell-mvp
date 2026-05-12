"""Machine-learning modules for AgeWell."""

from agewell.ml.datamodule import AgeWellDataModule, MasterRowsDataset
from agewell.ml.lightning_module import AgeWellINLightning
from agewell.ml.modality_registry import (
    MOD_IDX,
    MODALITY_LIST,
    N_MODALITIES,
    ModalityConfig,
    build_default_registry,
    registry_by_name,
)
from agewell.ml.model import AgeWellINModel, count_trainable_parameters

__all__ = [
    "MODALITY_LIST",
    "MOD_IDX",
    "N_MODALITIES",
    "AgeWellDataModule",
    "AgeWellINLightning",
    "AgeWellINModel",
    "MasterRowsDataset",
    "ModalityConfig",
    "build_default_registry",
    "count_trainable_parameters",
    "registry_by_name",
]
