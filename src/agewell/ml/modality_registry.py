"""Canonical modality registry for Phase 4 model wiring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd
from torch import Tensor

from agewell.data.schema import MODALITIES
from agewell.ml.encoders._base import BaseEncoder
from agewell.ml.encoders.genetic_apoe import APOEEncoder
from agewell.ml.encoders.mri_raw_brainiac import BrainIACCachedEncoder
from agewell.ml.encoders.tabular_lipid import LipidEncoder
from agewell.ml.encoders.tabular_tabpfn import TabPFNFrozenEncoder
from agewell.ml.feature_extractors import FEATURE_NAMES, build_tabpfn_context

MODALITY_LIST: tuple[str, ...] = tuple(MODALITIES)
MOD_IDX: dict[str, int] = {name: idx for idx, name in enumerate(MODALITY_LIST)}
N_MODALITIES: int = len(MODALITY_LIST)

EncoderKind = Literal["tabpfn_frozen", "brainiac_cached", "apoe_embedding", "lipid_mlp"]


@dataclass(frozen=True)
class ModalityConfig:
    """Metadata needed to build one modality encoder."""

    name: str
    encoder_type: EncoderKind
    feature_key: str
    presence_key: str
    n_features: int | None
    is_rich_required: bool
    ctx_size: int | None = None
    encoder_kwargs: dict[str, Any] | None = None

    def build_encoder(
        self,
        *,
        context_df: pd.DataFrame | None = None,
        ctx_X: Tensor | None = None,
        ctx_y: Tensor | None = None,
        imputation_stats: dict[str, dict[str, float]] | None = None,
        seed: int = 1337,
        **overrides: Any,
    ) -> BaseEncoder:
        """Instantiate this modality's encoder."""
        kwargs = dict(self.encoder_kwargs or {})
        kwargs.update(overrides)
        if self.encoder_type == "tabpfn_frozen":
            if self.n_features is None or self.ctx_size is None:
                raise ValueError(f"TabPFN modality {self.name} is missing dimensions/context size")
            if ctx_X is None or ctx_y is None:
                if context_df is None:
                    raise ValueError("TabPFN encoder requires ctx_X/ctx_y or context_df")
                ctx_X, ctx_y = build_tabpfn_context(
                    context_df,
                    modality=self.name,
                    ctx_size=self.ctx_size,
                    seed=seed,
                    imputation_stats=imputation_stats,
                )
            return TabPFNFrozenEncoder(
                modality=self.name,
                n_features=self.n_features,
                ctx_X=ctx_X,
                ctx_y=ctx_y,
                **kwargs,
            )
        if self.encoder_type == "brainiac_cached":
            return BrainIACCachedEncoder(**kwargs)
        if self.encoder_type == "apoe_embedding":
            return APOEEncoder(**kwargs)
        if self.encoder_type == "lipid_mlp":
            return LipidEncoder(**kwargs)
        raise ValueError(f"Unhandled encoder type: {self.encoder_type}")


def build_default_registry() -> list[ModalityConfig]:
    """Return all Phase 3 modality configs in canonical order."""
    return [
        ModalityConfig(
            name="clinical_demo",
            encoder_type="tabpfn_frozen",
            feature_key="clinical_demo_features",
            presence_key="clinical_demo_presence",
            n_features=len(FEATURE_NAMES["clinical_demo"]),
            is_rich_required=True,
            ctx_size=1500,
        ),
        ModalityConfig(
            name="clinical_lifestyle",
            encoder_type="tabpfn_frozen",
            feature_key="clinical_lifestyle_features",
            presence_key="clinical_lifestyle_presence",
            n_features=len(FEATURE_NAMES["clinical_lifestyle"]),
            is_rich_required=False,
            ctx_size=1000,
        ),
        ModalityConfig(
            name="clinical_comorbid",
            encoder_type="tabpfn_frozen",
            feature_key="clinical_comorbid_features",
            presence_key="clinical_comorbid_presence",
            n_features=len(FEATURE_NAMES["clinical_comorbid"]),
            is_rich_required=False,
            ctx_size=1500,
        ),
        ModalityConfig(
            name="cognitive",
            encoder_type="tabpfn_frozen",
            feature_key="cognitive_features",
            presence_key="cognitive_presence",
            n_features=len(FEATURE_NAMES["cognitive"]),
            is_rich_required=True,
            ctx_size=1500,
        ),
        ModalityConfig(
            name="blood",
            encoder_type="tabpfn_frozen",
            feature_key="blood_features",
            presence_key="blood_presence",
            n_features=len(FEATURE_NAMES["blood"]),
            is_rich_required=False,
            ctx_size=1500,
        ),
        ModalityConfig(
            name="mri_vol",
            encoder_type="tabpfn_frozen",
            feature_key="mri_vol_features",
            presence_key="mri_vol_presence",
            n_features=len(FEATURE_NAMES["mri_vol"]),
            is_rich_required=True,
            ctx_size=1500,
        ),
        ModalityConfig(
            name="mri_raw",
            encoder_type="brainiac_cached",
            feature_key="mri_raw_features",
            presence_key="mri_raw_presence",
            n_features=768,
            is_rich_required=True,
            encoder_kwargs={"in_dim": 768},
        ),
        ModalityConfig(
            name="genetic",
            encoder_type="apoe_embedding",
            feature_key="genetic_apoe4",
            presence_key="genetic_presence",
            n_features=None,
            is_rich_required=False,
        ),
        ModalityConfig(
            name="lipid",
            encoder_type="lipid_mlp",
            feature_key="lipid_features",
            presence_key="lipid_presence",
            n_features=213,
            is_rich_required=False,
            encoder_kwargs={"n_features": 213, "hidden": 256, "dropout": 0.1},
        ),
    ]


def registry_by_name(
    registry: list[ModalityConfig] | None = None,
) -> dict[str, ModalityConfig]:
    """Return registry configs keyed by modality name."""
    return {config.name: config for config in (registry or build_default_registry())}
