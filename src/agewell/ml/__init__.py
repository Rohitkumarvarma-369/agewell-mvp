"""Machine-learning modules for AgeWell."""

from agewell.ml.modality_registry import (
    MOD_IDX,
    MODALITY_LIST,
    N_MODALITIES,
    ModalityConfig,
    build_default_registry,
    registry_by_name,
)

__all__ = [
    "MODALITY_LIST",
    "MOD_IDX",
    "N_MODALITIES",
    "ModalityConfig",
    "build_default_registry",
    "registry_by_name",
]
