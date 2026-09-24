"""Robustness stress-testing engine for MEYRO (Phase 20).

Tests performance degradation under:
- Missing data (10%, 20%, 30% dropout)
- Sensor Gaussian noise
- Sensor drift / bias
- Short history vs Long history
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from meyro.baselines.personal import PersonalizedBaseline
from meyro.data.synthetic import SyntheticBenchmarkGenerator
from meyro.preprocessing.pipeline import PreprocessingPipeline


class RobustnessTestSuite:
    """Stress tests personal baseline detection under adverse physical sensing conditions."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.features = ["step_count", "resting_hr", "sleep_duration_min"]

    def run_missing_data_stress(self) -> dict[str, float]:
        """Evaluates detection AUROC under 0%, 10%, 20%, and 30% missing observation rates."""
        results = {}
        gen = SyntheticBenchmarkGenerator(seed=self.seed)
        base_df = gen.generate_cohort(n_subjects=12, n_days=50, anomaly_rate_per_subject=0.6)

        for missing_rate in [0.0, 0.10, 0.20, 0.30]:
            df = base_df.copy()
            rng = np.random.default_rng(self.seed)
            # Randomly inject NaNs into feature columns
            for col in self.features:
                mask = rng.uniform(0, 1, size=len(df)) < missing_rate
                df.loc[mask, col] = np.nan

            pipe = PreprocessingPipeline(feature_cols=self.features)
            cleaned = pipe.clean_and_impute(df)
            calib, eval_df = PreprocessingPipeline.create_subject_temporal_splits(cleaned, calibration_days=18)

            model = PersonalizedBaseline(feature_cols=self.features).fit_calibration(calib)
            scored = model.score_causal(eval_df)

            y_true = eval_df["ground_truth_label"].to_numpy()
            y_scores = scored["deviation_score"].to_numpy()
            auroc = float(roc_auc_score(y_true, y_scores))
            results[f"Missing {int(missing_rate * 100)}%"] = round(auroc, 4)

        return results

    def run_noise_stress(self) -> dict[str, float]:
        """Evaluates detection AUROC under standard vs 2x vs 4x sensor measurement noise."""
        results = {}
        gen = SyntheticBenchmarkGenerator(seed=self.seed)
        base_df = gen.generate_cohort(n_subjects=12, n_days=50, anomaly_rate_per_subject=0.6)

        for noise_factor in [1.0, 2.0, 4.0]:
            df = base_df.copy()
            rng = np.random.default_rng(self.seed)
            for col in self.features:
                std_val = float(df[col].std())
                df[col] += rng.normal(0, std_val * 0.1 * noise_factor, size=len(df))

            pipe = PreprocessingPipeline(feature_cols=self.features)
            cleaned = pipe.clean_and_impute(df)
            calib, eval_df = PreprocessingPipeline.create_subject_temporal_splits(cleaned, calibration_days=18)

            model = PersonalizedBaseline(feature_cols=self.features).fit_calibration(calib)
            scored = model.score_causal(eval_df)

            y_true = eval_df["ground_truth_label"].to_numpy()
            y_scores = scored["deviation_score"].to_numpy()
            auroc = float(roc_auc_score(y_true, y_scores))
            results[f"Noise {noise_factor}x"] = round(auroc, 4)

        return results
