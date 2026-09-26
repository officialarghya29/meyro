"""Statistical inference utilities (Phase 27 / fairness-aware reporting).

The project's central claim is a *comparison* — personalized versus population
baselines. A single aggregate AUROC cannot support that claim: it hides both
between-subject variance and the fact that individual subjects are the natural
unit of analysis. These helpers provide:

* **per-subject AUROC**, so performance can be reported as a distribution across
  individuals rather than one pooled number;
* **bootstrap confidence intervals**, so a difference can be read against its
  own uncertainty;
* **paired Wilcoxon signed-rank tests**, the correct test for the same subjects
  measured under two conditions.

Deliberately no correction for multiple comparisons is applied automatically;
callers must state how many comparisons were made.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score


def per_group_auroc(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    groups: np.ndarray,
) -> dict[str, float]:
    """Computes AUROC within each group (typically a subject).

    Groups containing only one class are skipped: AUROC is undefined for them,
    and silently scoring them 0.5 would bias the pooled result. The number of
    scored and skipped groups is returned so the exclusion is visible.
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores, dtype=np.float64)
    groups = np.asarray(groups)

    values: dict[str, float] = {}
    skipped = 0
    for group in np.unique(groups):
        mask = groups == group
        labels = y_true[mask]
        if len(np.unique(labels)) < 2:
            skipped += 1
            continue
        values[str(group)] = float(roc_auc_score(labels, y_scores[mask]))

    values["_n_scored_groups"] = float(len(values))
    values["_n_skipped_single_class_groups"] = float(skipped)
    return values


def bootstrap_ci(
    values: np.ndarray | list[float],
    *,
    n_bootstrap: int = 5000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, float]:
    """Percentile bootstrap confidence interval for the mean of ``values``."""
    sample = np.asarray(values, dtype=np.float64)
    if sample.size == 0:
        return {"mean": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan"), "n": 0}

    if sample.size == 1:
        return {
            "mean": float(sample[0]),
            "ci_lower": float(sample[0]),
            "ci_upper": float(sample[0]),
            "n": 1,
        }

    rng = np.random.default_rng(seed)
    resamples = rng.choice(sample, size=(n_bootstrap, sample.size), replace=True).mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(resamples, [alpha, 1.0 - alpha])
    return {
        "mean": float(sample.mean()),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "n": int(sample.size),
    }


def paired_wilcoxon(
    a_values: dict[str, float] | np.ndarray,
    b_values: dict[str, float] | np.ndarray,
) -> dict[str, float]:
    """Paired Wilcoxon signed-rank test between two conditions on matched units.

    ``a_values`` and ``b_values`` must be matched per unit (subject). The
    alternative hypothesis is two-sided; the effect size reported is the median
    paired difference ``a - b``.
    """
    if isinstance(a_values, dict):
        common = sorted(set(a_values) & set(b_values))
        common = [key for key in common if not key.startswith("_")]
        a = np.array([a_values[key] for key in common], dtype=np.float64)
        b = np.array([b_values[key] for key in common], dtype=np.float64)
    else:
        a = np.asarray(a_values, dtype=np.float64)
        b = np.asarray(b_values, dtype=np.float64)

    if a.size != b.size:
        raise ValueError(f"Paired test requires equal lengths, got {a.size} and {b.size}")
    if a.size == 0:
        return {"n_pairs": 0, "statistic": float("nan"), "p_value": float("nan"), "median_difference": float("nan")}

    differences = a - b
    # Wilcoxon cannot run when every difference is exactly zero.
    if np.allclose(differences, 0.0):
        return {
            "n_pairs": int(a.size),
            "statistic": 0.0,
            "p_value": 1.0,
            "median_difference": 0.0,
            "median_a": float(np.median(a)),
            "median_b": float(np.median(b)),
            "win_rate_a": 0.0,
        }

    result = stats.wilcoxon(a, b)
    return {
        "n_pairs": int(a.size),
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "median_difference": float(np.median(differences)),
        "median_a": float(np.median(a)),
        "median_b": float(np.median(b)),
        "win_rate_a": float(np.mean(differences > 0)),
    }


def aggregate_seeds(per_seed: list[dict[str, float]]) -> dict[str, float]:
    """Mean, standard deviation and range of a metric across seeds."""
    if not per_seed:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan"), "n_seeds": 0}

    values = np.array([entry["value"] for entry in per_seed], dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "min": float(values.min()),
        "max": float(values.max()),
        "n_seeds": int(values.size),
    }
