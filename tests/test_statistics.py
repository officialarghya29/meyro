"""Tests for statistical inference helpers and the significance study."""

import numpy as np
import pytest

from meyro.evaluation.significance import PersonalizationSignificanceStudy
from meyro.evaluation.statistics import (
    aggregate_seeds,
    bootstrap_ci,
    paired_wilcoxon,
    per_group_auroc,
)


def test_per_group_auroc_skips_single_class_groups():
    y_true = np.array([0, 0, 1, 1, 0, 0])
    y_scores = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.4])
    groups = np.array(["A", "A", "A", "A", "B", "B"])  # B has only class 0

    result = per_group_auroc(y_true, y_scores, groups)

    assert "A" in result
    assert "B" not in result
    # A single-class group must be excluded, not silently scored 0.5.
    assert result["_n_scored_groups"] == 1.0
    assert result["_n_skipped_single_class_groups"] == 1.0
    assert result["A"] == pytest.approx(1.0)


def test_bootstrap_ci_brackets_the_mean():
    values = [0.80, 0.85, 0.90, 0.95, 0.88, 0.92, 0.86]
    result = bootstrap_ci(values, n_bootstrap=2000, seed=0)

    assert result["n"] == 7
    assert result["ci_lower"] <= result["mean"] <= result["ci_upper"]
    assert 0.8 <= result["mean"] <= 0.95


def test_bootstrap_ci_is_deterministic_for_a_fixed_seed():
    values = [0.1, 0.5, 0.9, 0.4]
    first = bootstrap_ci(values, n_bootstrap=500, seed=123)
    second = bootstrap_ci(values, n_bootstrap=500, seed=123)
    assert first == second


def test_paired_wilcoxon_detects_consistent_improvement():
    # Method A beats B on every subject by a similar margin.
    b = {"s1": 0.80, "s2": 0.82, "s3": 0.78, "s4": 0.85, "s5": 0.81, "s6": 0.83, "s7": 0.79, "s8": 0.84}
    a = {key: value + 0.05 for key, value in b.items()}

    result = paired_wilcoxon(a, b)

    assert result["n_pairs"] == 8
    assert result["p_value"] < 0.05
    assert result["median_difference"] == pytest.approx(0.05, abs=1e-9)
    assert result["win_rate_a"] == 1.0


def test_paired_wilcoxon_handles_all_ties():
    values = {"s1": 0.9, "s2": 0.8}
    result = paired_wilcoxon(values, dict(values))
    assert result["p_value"] == 1.0
    assert result["median_difference"] == 0.0


def test_paired_wilcoxon_uses_intersection_of_subject_keys():
    # Dict inputs are matched on shared keys: a subject missing from one side is
    # dropped rather than silently mis-paired.
    result = paired_wilcoxon({"s1": 0.9, "s2": 0.8}, {"s1": 0.8, "s3": 0.7})
    assert result["n_pairs"] == 1


def test_paired_wilcoxon_rejects_unmatched_array_lengths():
    with pytest.raises(ValueError, match="equal lengths"):
        paired_wilcoxon(np.array([0.9, 0.8]), np.array([0.8]))


def test_aggregate_seeds_summarises_repeated_runs():
    result = aggregate_seeds([{"value": 0.9}, {"value": 0.8}, {"value": 1.0}])
    assert result["n_seeds"] == 3
    assert result["mean"] == pytest.approx(0.9)
    assert result["min"] == 0.8
    assert result["max"] == 1.0


def test_significance_study_is_pairable_across_seeds():
    study = PersonalizationSignificanceStudy(seeds=(42, 7), n_subjects=8, n_days=60)
    results = study.run()

    # Subjects repeat across seeds, so pairing keys must be seed-qualified.
    static = results["per_method"]["Personal Baseline (Static)"]["per_subject_auroc"]
    population = results["per_method"]["Population Baseline"]["per_subject_auroc"]
    assert static["n"] == population["n"] > 0

    test = results["tests"]["personalized_static_vs_population"]
    assert 0.0 <= test["p_value"] <= 1.0
    assert 0.0 <= test["win_rate_a"] <= 1.0

    # Personalization must not be worse than the population threshold on average.
    assert static["mean"] >= population["mean"] - 0.02
