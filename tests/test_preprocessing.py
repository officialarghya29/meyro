"""Tests for Preprocessing Pipeline and Leakage Prevention."""

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from meyro.preprocessing.pipeline import PreprocessingPipeline


def test_clean_and_impute():
    data = {
        "subject_id": ["S1", "S1", "S1", "S2", "S2"],
        "timestamp": [
            datetime(2025, 1, 1, tzinfo=UTC),
            datetime(2025, 1, 2, tzinfo=UTC),
            datetime(2025, 1, 3, tzinfo=UTC),
            datetime(2025, 1, 1, tzinfo=UTC),
            datetime(2025, 1, 2, tzinfo=UTC),
        ],
        "resting_hr": [70.0, np.nan, 72.0, np.nan, 60.0],
    }
    df = pd.DataFrame(data)
    pipe = PreprocessingPipeline(feature_cols=["resting_hr"])
    cleaned = pipe.clean_and_impute(df)
    assert not cleaned["resting_hr"].isna().any()
    # S1 day 2 should be forward-filled with 70.0
    s1_vals = cleaned[cleaned["subject_id"] == "S1"]["resting_hr"].tolist()
    assert s1_vals == [70.0, 70.0, 72.0]


def test_create_subject_temporal_splits_no_leakage():
    dates = pd.date_range("2025-01-01", periods=30, freq="D", tz="UTC")
    df = pd.DataFrame({
        "subject_id": ["S1"] * 30,
        "timestamp": dates,
        "resting_hr": np.random.normal(70, 2, 30),
    })
    calib, eval_df = PreprocessingPipeline.create_subject_temporal_splits(df, calibration_days=10)
    assert len(calib) == 10
    assert len(eval_df) == 20
    # Causal ordering check: max calibration timestamp < min evaluation timestamp
    assert calib["timestamp"].max() < eval_df["timestamp"].min()


def test_sliding_windows():
    series = np.arange(10).reshape(-1, 1)
    windows = PreprocessingPipeline.create_sliding_windows(series, window_size=3)
    assert windows.shape == (8, 3, 1)
    assert np.array_equal(windows[0].flatten(), [0, 1, 2])
    assert np.array_equal(windows[-1].flatten(), [7, 8, 9])
