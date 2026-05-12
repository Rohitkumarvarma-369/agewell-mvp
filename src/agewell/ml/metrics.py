"""Offline metrics for Phase 5 training and evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)

from agewell.ml.losses import CDR_VALUES


def softmax_np(logits: np.ndarray, *, temperature: float = 1.0) -> np.ndarray:
    """Return row-wise softmax probabilities."""
    scaled = np.asarray(logits, dtype=np.float64) / max(float(temperature), 1.0e-8)
    shifted = scaled - scaled.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def compute_phase5_metrics(
    arrays: Mapping[str, np.ndarray],
    *,
    diag_temperature: float = 1.0,
    prefix: str = "",
) -> dict[str, float | int | None]:
    """Compute Phase 5 task metrics from collected prediction arrays."""
    metrics: dict[str, float | int | None] = {}
    name = f"{prefix}_" if prefix else ""

    diag_logits = arrays["diag_logits"]
    diag_labels = arrays["diag_label"].astype(np.int64)
    metrics.update(
        _classification_metrics(
            logits=diag_logits,
            labels=diag_labels,
            metric_prefix=f"{name}diag",
            temperature=diag_temperature,
        )
    )

    surv_mask = arrays["has_survival"].astype(bool)
    metrics.update(
        _classification_metrics(
            logits=arrays["surv_logits"][surv_mask],
            labels=arrays["surv_bin"][surv_mask].astype(np.int64),
            metric_prefix=f"{name}surv",
        )
    )

    cdr_mask = arrays["has_cdr"].astype(bool)
    cdr_labels = _cdr_to_class_np(arrays["cdr"])
    metrics.update(
        _classification_metrics(
            logits=arrays["cdr_logits"][cdr_mask],
            labels=cdr_labels[cdr_mask],
            metric_prefix=f"{name}cdr",
        )
    )

    mmse_mask = arrays["has_mmse"].astype(bool)
    metrics.update(
        _regression_metrics(
            prediction=arrays["mmse_pred"][mmse_mask],
            target=arrays["mmse"][mmse_mask],
            metric_prefix=f"{name}mmse",
        )
    )
    return metrics


def _classification_metrics(
    *,
    logits: np.ndarray,
    labels: np.ndarray,
    metric_prefix: str,
    temperature: float = 1.0,
) -> dict[str, float | int | None]:
    labels = np.asarray(labels, dtype=np.int64)
    logits = np.asarray(logits, dtype=np.float64)
    valid = labels >= 0
    labels = labels[valid]
    logits = logits[valid]
    out: dict[str, float | int | None] = {f"{metric_prefix}_n": len(labels)}
    if len(labels) == 0 or logits.size == 0:
        out.update(
            {
                f"{metric_prefix}_accuracy": None,
                f"{metric_prefix}_balanced_accuracy": None,
                f"{metric_prefix}_macro_f1": None,
                f"{metric_prefix}_macro_auroc": None,
            }
        )
        return out

    probs = softmax_np(logits, temperature=temperature)
    pred = probs.argmax(axis=1)
    class_labels = list(range(logits.shape[1]))
    out[f"{metric_prefix}_accuracy"] = float(accuracy_score(labels, pred))
    out[f"{metric_prefix}_balanced_accuracy"] = float(balanced_accuracy_score(labels, pred))
    out[f"{metric_prefix}_macro_f1"] = float(
        f1_score(labels, pred, labels=class_labels, average="macro", zero_division=0)
    )
    out[f"{metric_prefix}_macro_auroc"] = _safe_macro_auroc(labels, probs, class_labels)
    return out


def _regression_metrics(
    *,
    prediction: np.ndarray,
    target: np.ndarray,
    metric_prefix: str,
) -> dict[str, float | int | None]:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    finite = np.isfinite(prediction) & np.isfinite(target)
    prediction = prediction[finite]
    target = target[finite]
    out: dict[str, float | int | None] = {f"{metric_prefix}_n": len(target)}
    if len(target) == 0:
        out[f"{metric_prefix}_mae"] = None
        out[f"{metric_prefix}_rmse"] = None
        return out
    out[f"{metric_prefix}_mae"] = float(mean_absolute_error(target, prediction))
    out[f"{metric_prefix}_rmse"] = float(mean_squared_error(target, prediction) ** 0.5)
    return out


def _safe_macro_auroc(
    labels: np.ndarray,
    probabilities: np.ndarray,
    class_labels: list[int],
) -> float | None:
    observed = set(int(label) for label in np.unique(labels))
    if len(observed) < 2 or observed != set(class_labels):
        return None
    try:
        value = float(
            roc_auc_score(
                labels,
                probabilities,
                labels=class_labels,
                multi_class="ovr",
                average="macro",
            )
        )
        return value if np.isfinite(value) else None
    except ValueError:
        return None


def _cdr_to_class_np(values: np.ndarray) -> np.ndarray:
    labels = np.full(values.shape, -1, dtype=np.int64)
    for idx, value in enumerate(CDR_VALUES):
        labels = np.where(np.isclose(values.astype(float), float(value)), idx, labels)
    return labels


def json_sanitize(value: Any) -> Any:
    """Convert numpy scalars and non-finite floats into JSON-safe values."""
    if isinstance(value, dict):
        return {str(key): json_sanitize(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_sanitize(item) for item in value]
    if isinstance(value, np.generic):
        return json_sanitize(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
