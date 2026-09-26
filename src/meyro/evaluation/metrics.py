"""Shared detection metrics (Phases 11/18).

One implementation used by the benchmark, ablation and robustness suites means
every reported number is computed the same way. Threshold-dependent metrics are
reported at a fixed *operating point* — the threshold achieving 85% sensitivity
— so models are compared at equal recall rather than at whatever threshold
flatters each one.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

TARGET_SENSITIVITY = 0.85


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    """Trapezoidal integration compatible with NumPy 1.x and 2.x."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))  # type: ignore[attr-defined]


def detection_metrics(y_true: np.ndarray, y_scores: np.ndarray) -> dict[str, float]:
    """Computes threshold-free and threshold-dependent detection metrics.

    Returns:
        auroc, auprc, prevalence, and precision/recall/f1/specificity measured at
        the threshold that yields at least ``TARGET_SENSITIVITY`` sensitivity.
    """
    y_true = np.asarray(y_true).astype(int)
    y_scores = np.asarray(y_scores, dtype=np.float64)

    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return {
            "auroc": 0.5,
            "auprc": 0.0,
            "prevalence": float(y_true.mean()) if len(y_true) else 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "fpr_at_target_sensitivity": 1.0,
            "specificity": 0.0,
            "threshold": float("inf"),
            "n_samples": float(len(y_true)),
        }

    auroc = float(roc_auc_score(y_true, y_scores))
    auprc = float(average_precision_score(y_true, y_scores))

    # Operating point: smallest threshold reaching the target sensitivity
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    candidates = np.where(tpr >= TARGET_SENSITIVITY)[0]
    if len(candidates) > 0:
        op_index = candidates[0]
        op_threshold = float(thresholds[op_index])
        fpr_at_target = float(fpr[op_index])
        recall = float(tpr[op_index])
    else:
        op_threshold, fpr_at_target, recall = float("inf"), 1.0, 1.0

    predicted = y_scores >= op_threshold
    true_positives = int(np.sum(predicted & (y_true == 1)))
    false_positives = int(np.sum(predicted & (y_true == 0)))
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "auroc": round(auroc, 4),
        "auprc": round(auprc, 4),
        "prevalence": round(float(y_true.mean()), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "fpr_at_target_sensitivity": round(fpr_at_target, 4),
        "specificity": round(1.0 - fpr_at_target, 4),
        "threshold": round(op_threshold, 6) if np.isfinite(op_threshold) else float("inf"),
        "n_samples": float(len(y_true)),
    }


def default_rates(y_true: np.ndarray, y_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns ROC curve coordinates (for figure generation)."""
    fpr, tpr, _ = roc_curve(np.asarray(y_true).astype(int), y_scores)
    return fpr, tpr


def precision_recall_points(y_true: np.ndarray, y_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (precision, recall) curve coordinates (for figure generation)."""
    precision, recall, _ = precision_recall_curve(np.asarray(y_true).astype(int), y_scores)
    return precision, recall
