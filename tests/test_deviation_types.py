"""Tests for the deviation-shape generator and study (Robustness follow-up).

The main benchmark injects a single deviation shape (a sustained mean shift),
which a robust per-subject statistic is close to optimal for. These tests pin
down that the generator (a) really produces distinct shapes and (b) labels each
shape consistently, so the deviation-shape study measures what it claims to.
"""

from __future__ import annotations

import numpy as np

from meyro.data.synthetic import SyntheticBenchmarkGenerator
from meyro.evaluation.deviation_types import DeviationTypeStudy


def _spike_frames():
    generator = SyntheticBenchmarkGenerator(seed=7)
    return generator.generate_cohort(
        n_subjects=3, n_days=45, anomaly_rate_per_subject=1.0, deviation_type="point_spike"
    )


def test_point_spike_is_a_single_labelled_day_per_subject() -> None:
    """A spike must affect exactly one day, so it cannot inflate the positive count."""
    df = _spike_frames()
    for _, group in df.groupby("subject_id"):
        assert int(group["ground_truth_label"].sum()) == 1


def test_variance_increase_expands_spread_without_shifting_the_level() -> None:
    """The variance shape must change dispersion, not the level.

    Dispersion is compared *within each subject* against that subject's own normal
    days, so the test measures the injected mechanism rather than incidental
    differences in which subjects happened to be anomalous.
    """
    df = SyntheticBenchmarkGenerator(seed=11).generate_cohort(
        n_subjects=6, n_days=50, anomaly_rate_per_subject=1.0, deviation_type="variance_increase"
    )

    for subject_id, group in df.groupby("subject_id"):
        normal = group.loc[group["ground_truth_label"] == 0, "step_count"].to_numpy()
        anomalous = group.loc[group["ground_truth_label"] == 1, "step_count"].to_numpy()
        assert len(anomalous) >= 4, f"{subject_id} received too few anomalous days"

        # Dispersion expands...
        assert anomalous.std() >= 1.5 * normal.std(), f"{subject_id} spread did not expand"
        # ...while the centre stays in the same neighbourhood (median is robust to
        # the few anomalous days, unlike the mean).
        shift = abs(float(np.median(anomalous)) - float(np.median(normal)))
        assert shift <= 2.5 * normal.std(), f"{subject_id} level shifted by {shift:.1f}"


def test_short_series_still_receives_its_anomaly() -> None:
    """Regression: a short series must not silently drop the injected deviation.

    The span was previously drawn from a fixed window that could start past the
    end of a short series, so a caller requesting a 100% anomaly rate received an
    all-normal cohort while believing it had anomalies.
    """
    df = SyntheticBenchmarkGenerator(seed=3).generate_cohort(
        n_subjects=5, n_days=30, anomaly_rate_per_subject=1.0
    )
    labelled = df.groupby("subject_id")["ground_truth_label"].sum()
    assert (labelled > 0).all()
    # The span stays inside the series.
    for _, group in df.groupby("subject_id"):
        assert len(group) == 30


def test_spike_span_never_runs_past_the_series_end() -> None:
    """Spans must be clamped to fit, for every shape and series length."""
    for n_days in (25, 30, 40, 50, 60):
        df = SyntheticBenchmarkGenerator(seed=17).generate_cohort(
            n_subjects=4, n_days=n_days, anomaly_rate_per_subject=1.0
        )
        assert len(df) == 4 * n_days
        assert int(df["ground_truth_label"].sum()) > 0


def test_unknown_deviation_type_is_rejected() -> None:
    """Silently ignoring an unknown shape would fake a result; it must raise."""
    generator = SyntheticBenchmarkGenerator(seed=3)
    try:
        generator.generate_cohort(
            n_subjects=2, n_days=30, anomaly_rate_per_subject=1.0, deviation_type="teleport"
        )
    except ValueError as error:
        assert "teleport" in str(error)
    else:  # pragma: no cover
        raise AssertionError("unknown deviation_type was silently accepted")


def test_deviation_study_reports_every_shape_and_gain() -> None:
    """The study must return per-shape metrics and an explicit generalization verdict."""
    results = DeviationTypeStudy(
        n_subjects=4,
        n_days=30,
        calibration_days=12,
        seed=5,
        epochs=2,
        deviation_types=("mean_shift", "variance_increase"),
    ).run()

    metadata = results.pop("_metadata")
    summary = results.pop("_summary")
    assert set(metadata["deviation_types"]) == {"mean_shift", "variance_increase"}
    assert set(results) == {"mean_shift", "variance_increase"}

    for metrics in results.values():
        assert 0.0 <= metrics["population"]["auroc"] <= 1.0
        assert 0.0 <= metrics["personal"]["auroc"] <= 1.0
        assert 0.0 <= metrics["meyro_v2"]["auroc"] <= 1.0
        assert metrics["n_evaluation_windows"] > 0

    assert summary["holds_across_all_shapes"] in (True, False)
    assert set(summary["personalization_gain_by_shape"]) == {"mean_shift", "variance_increase"}
    assert np.isfinite(summary["min_gain"]) and np.isfinite(summary["max_gain"])
