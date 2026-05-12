"""Small config access helpers shared by Phase 4 modules."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from omegaconf import DictConfig, OmegaConf

T = TypeVar("T")
_MISSING = object()


def cfg_select(cfg: Any, key: str, default: T) -> T:
    """Read a dotted key from DictConfig, mappings, or attribute objects."""
    if cfg is None:
        return default
    if isinstance(cfg, DictConfig):
        value = OmegaConf.select(cfg, key, default=default)
        return cast(T, value)
    current: Any = cfg
    for part in key.split("."):
        if isinstance(current, Mapping):
            current = current.get(part, _MISSING)
        else:
            current = getattr(current, part, _MISSING)
        if current is _MISSING:
            return default
    return cast(T, current)


def cfg_mapping(cfg: Any, key: str, default: Mapping[str, float]) -> dict[str, float]:
    """Read a numeric mapping from config into a plain ``dict[str, float]``."""
    raw_value: Any = cfg_select(cfg, key, default)
    if isinstance(raw_value, DictConfig):
        raw_value = OmegaConf.to_container(raw_value, resolve=True)
    if not isinstance(raw_value, Mapping):
        return dict(default)
    return {str(name): float(raw) for name, raw in raw_value.items()}
