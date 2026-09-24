"""Tests for Data Architecture & Synthetic Benchmark Generator."""

import pandas as pd

from meyro.data.synthetic import ObservationRecord, SyntheticBenchmarkGenerator


def test_observation_record_validation():
    rec = ObservationRecord(
        subject_id="SUBJ_001",
        timestamp="2025-01-01T12:00:00Z",
        modality="physiology",
        features={"resting_hr": 65.0, "hrv_rmssd": 45.0},
        quality_score=0.95,
        ground_truth_label=0,
    )
    assert rec.subject_id == "SUBJ_001"
    assert rec.quality_score == 0.95
    assert rec.timestamp.tzinfo is not None


def test_synthetic_benchmark_generation():
    gen = SyntheticBenchmarkGenerator(seed=123)
    df = gen.generate_cohort(n_subjects=3, n_days=40, anomaly_rate_per_subject=1.0)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3 * 40
    assert set(df.columns) == {
        "subject_id",
        "timestamp",
        "step_count",
        "resting_hr",
        "sleep_duration_min",
        "quality_score",
        "ground_truth_label",
    }
    # Verify temporal ordering per subject
    for _, group in df.groupby("subject_id"):
        assert group["timestamp"].is_monotonic_increasing
    # Verify ground truth anomalies were injected
    assert (df["ground_truth_label"] == 1).sum() > 0
