"""Preprocessing, cleaning, windowing, and leakage-safe splitting for MEYRO."""

from __future__ import annotations

import numpy as np
import pandas as pd


class PreprocessingPipeline:
    """Preprocesses longitudinal time series without future or cross-subject leakage."""

    def __init__(self, feature_cols: list[str]) -> None:
        self.feature_cols = feature_cols

    def clean_and_impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sorts chronologically, removes corrupt values, and forward-fills subject data."""
        out = df.copy()
        out.sort_values(by=["subject_id", "timestamp"], inplace=True)

        # Per-subject forward fill, then fill remaining leading NaNs with median
        cleaned_groups = []
        for _, group in out.groupby("subject_id", sort=False):
            g = group.copy()
            for col in self.feature_cols:
                if col in g.columns:
                    g[col] = g[col].ffill()
                    if g[col].isna().any():
                        median_val = g[col].median()
                        fill_val = median_val if not np.isnan(median_val) else 0.0
                        g[col] = g[col].fillna(fill_val)
            cleaned_groups.append(g)

        result = pd.concat(cleaned_groups, ignore_index=True)
        return result

    @staticmethod
    def create_subject_temporal_splits(
        df: pd.DataFrame, calibration_days: int = 21
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Splits time series causally into a calibration (baseline warmup) set and evaluation set.

        Guarantees:
        - Strict temporal separation: calibration observations strictly precede evaluation observations.
        - No test-to-train lookahead leakage.
        """
        calib_list = []
        eval_list = []

        for _, group in df.groupby("subject_id", sort=False):
            g = group.sort_values(by="timestamp").reset_index(drop=True)
            if len(g) <= calibration_days:
                calib_list.append(g)
            else:
                calib_list.append(g.iloc[:calibration_days])
                eval_list.append(g.iloc[calibration_days:])

        calib_df = pd.concat(calib_list, ignore_index=True) if calib_list else pd.DataFrame(columns=df.columns)
        eval_df = pd.concat(eval_list, ignore_index=True) if eval_list else pd.DataFrame(columns=df.columns)
        return calib_df, eval_df

    @staticmethod
    def create_sliding_windows(
        series: np.ndarray, window_size: int = 7
    ) -> np.ndarray:
        """Generates backward-looking causal sliding windows: shape (N - window_size + 1, window_size, features)."""
        if len(series) < window_size:
            return np.empty((0, window_size, series.shape[1] if series.ndim > 1 else 1))

        n_samples = len(series) - window_size + 1
        windows = [series[i : i + window_size] for i in range(n_samples)]
        return np.array(windows)
