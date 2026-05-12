"""Tests for Phase 5 metrics and calibration helpers."""

from __future__ import annotations

import numpy as np

from agewell.ml.calibration import fit_temperature
from agewell.ml.metrics import compute_phase5_metrics, softmax_np


def test_phase5_metrics_handle_partial_targets_and_missing_auroc() -> None:
    arrays = {
        "diag_logits": np.array([[3.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float32),
        "diag_label": np.array([0, 1]),
        "surv_logits": np.array([[2.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        "surv_bin": np.array([0, 1]),
        "has_survival": np.array([True, False]),
        "mmse_pred": np.array([28.0, 20.0], dtype=np.float32),
        "mmse": np.array([30.0, 0.0], dtype=np.float32),
        "has_mmse": np.array([True, False]),
        "cdr_logits": np.array([[2.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        "cdr": np.array([0.0, 0.5], dtype=np.float32),
        "has_cdr": np.array([True, False]),
    }

    metrics = compute_phase5_metrics(arrays, prefix="observed")

    assert metrics["observed_diag_accuracy"] == 1.0
    assert metrics["observed_diag_macro_f1"] == 2.0 / 3.0
    assert metrics["observed_surv_n"] == 1
    assert metrics["observed_surv_macro_auroc"] is None
    assert metrics["observed_mmse_mae"] == 2.0


def test_temperature_calibration_returns_positive_temperature() -> None:
    logits = np.array([[4.0, 0.0], [0.0, 3.0], [2.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    labels = np.array([0, 1, 0, 1], dtype=np.int64)

    result = fit_temperature(logits, labels, max_iter=20)

    assert result.temperature > 0
    assert result.n == 4
    assert np.isfinite(result.nll_after)
    probs = softmax_np(logits, temperature=result.temperature)
    assert probs.shape == logits.shape
