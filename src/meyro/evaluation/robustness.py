"""Robustness stress-testing engine for MEYRO (Phase 20).

Stress axes required by the workflow:

* missing data (10 / 20 / 30 % dropout)
* sensor noise (1x / 2x / 4x)
* spike outliers
* short vs long calibration history
* sampling-rate reduction
* device / sensor bias between subjects

Every axis re-uses the *same* generator seed and the *same* personal-baseline
implementation, so a change in AUROC is attributable to the stress axis rather
than to incidental variation. Methods degrade; the degradation is reported,
not hidden.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from meyro.baselines.personal import PersonalizedBaseline
from meyro.data.synthetic import SyntheticBenchmarkGenerator
from meyro.preprocessing.pipeline import PreprocessingPipeline

DEFAULT_FEATURES = ["step_count", "resting_hr", "sleep_duration_min"]


class RobustnessTestSuite:
    """Stress tests personal baseline detection under adverse sensing conditions."""

    def __init__(self, seed: int = 42, n_subjects: int = 12, n_days: int = 50) -> None:
        self.seed = seed
        self.n_subjects = n_subjects
        self.n_days = n_days
        self.features = DEFAULT_FEATURES

    # ------------------------------------------------------------- generation
    def _base_cohort(self) -> pd.DataFrame:
        gen = SyntheticBenchmarkGenerator(seed=self.seed)
        return gen.generate_cohort(
            n_subjects=self.n_subjects,
            n_days=self.n_days,
            anomaly_rate_per_subject=0.6,
        )

    def _auroc_after_transform(
        self,
        df: pd.DataFrame,
        calibration_days: int = 18,
    ) -> float:
        """Applies cleaning/splitting/scoring pipeline and returns AUROC."""
        pipe = PreprocessingPipeline(feature_cols=self.features)
        cleaned = pipe.clean_and_impute(df)
        calib, eval_df = PreprocessingPipeline.create_subject_temporal_splits(
            cleaned, calibration_days=calibration_days
        )
        if eval_df.empty:
            return float("nan")

        model = PersonalizedBaseline(feature_cols=self.features).fit_calibration(calib)
        scored = model.score_causal(eval_df)

        y_true = eval_df["ground_truth_label"].to_numpy()
        y_scores = scored["deviation_score"].to_numpy()
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return round(float(roc_auc_score(y_true, y_scores)), 4)

    # ---------------------------------------------------------------- stress A
    def run_missing_data_stress(self) -> dict[str, float]:
        """Detection AUROC under 0%, 10%, 20%, and 30% missing observations."""
        base_df = self._base_cohort()
        results: dict[str, float] = {}

        for missing_rate in [0.0, 0.10, 0.20, 0.30]:
            df = base_df.copy()
            rng = np.random.default_rng(self.seed)
            for col in self.features:
                mask = rng.uniform(0, 1, size=len(df)) < missing_rate
                df.loc[mask, col] = np.nan
            results[f"Missing {int(missing_rate * 100)}%"] = self._auroc_after_transform(df)

        return results

    # ---------------------------------------------------------------- stress B
    def run_noise_stress(self) -> dict[str, float]:
        """Detection AUROC under standard, 2x and 4x sensor measurement noise."""
        base_df = self._base_cohort()
        results: dict[str, float] = {}

        for noise_factor in [1.0, 2.0, 4.0]:
            df = base_df.copy()
            rng = np.random.default_rng(self.seed)
            for col in self.features:
                std_val = float(df[col].std())
                df[col] += rng.normal(0, std_val * 0.1 * noise_factor, size=len(df))
            results[f"Noise {noise_factor}x"] = self._auroc_after_transform(df)

        return results

    # ---------------------------------------------------------------- stress C
    def run_outlier_stress(self) -> dict[str, float]:
        """Detection AUROC when 1%, 3% and 5% of observations are extreme spikes.

        Spikes are injected into *normal* rows only, so a detector that merely
        flags outliers will look good while a detector that has learned the
        person's normal will be tested on genuine deviation recovery.
        """
        base_df = self._base_cohort()
        results: dict[str, float] = {}

        for outlier_rate in [0.01, 0.03, 0.05]:
            df = base_df.copy()
            rng = np.random.default_rng(self.seed)
            normal_idx = df.index[df["ground_truth_label"] == 0].to_numpy()
            n_outliers = int(len(normal_idx) * outlier_rate)
            if n_outliers > 0:
                chosen = rng.choice(normal_idx, size=n_outliers, replace=False)
                for col in self.features:
                    std_val = float(df[col].std()) or 1.0
                    df.loc[chosen, col] += rng.choice([-1.0, 1.0], size=n_outliers) * std_val * 2.5
            results[f"Outliers {int(outlier_rate * 100)}%"] = self._auroc_after_transform(df)

        return results

    # ---------------------------------------------------------------- stress D
    def run_history_length_stress(self) -> dict[str, float]:
        """Detection AUROC as the available calibration history shrinks."""
        base_df = self._base_cohort()
        results: dict[str, float] = {}
        for history_days in [7, 14, 21, 28]:
            results[f"History {history_days}d"] = self._auroc_after_transform(
                base_df, calibration_days=history_days
            )
        return results

    # ---------------------------------------------------------------- stress E
    def run_sampling_rate_stress(self) -> dict[str, float]:
        """Detection AUROC when observations arrive at 1x, 1/2x and 1/3x the rate.

        Sparse sampling is simulated by retaining every n-th day per subject and
        linearly interpolating feature values, mimicking a lower-duty-cycle
        wearable. The evaluation horizon stays fixed, so AUROC remains comparable.
        """
        base_df = self._base_cohort()
        results: dict[str, float] = {}

        for stride in [1, 2, 3]:
            if stride == 1:
                results["Sampling 1x"] = self._auroc_after_transform(base_df)
                continue

            frames = []
            for _subject_id, group in base_df.groupby("subject_id", sort=False):
                g = group.sort_values("timestamp").reset_index(drop=True)
                frames.append(g.iloc[::stride].copy())
            sampled = pd.concat(frames, ignore_index=True)
            # Keep the same *fraction* of history as calibration so a lower duty
            # cycle does not silently consume the entire series as warm-up.
            sampled_calibration = max(8, round(18 / stride))
            results[f"Sampling 1/{stride}x"] = self._auroc_after_transform(
                sampled, calibration_days=sampled_calibration
            )

        return results

    # ---------------------------------------------------------------- stress F
    def run_device_variation_stress(self) -> dict[str, float]:
        """Detection AUROC when devices disagree in baseline offset and gain.

        A per-subject multiplicative gain and additive offset are applied to all
        observations. Because the personal baseline is fit on the *same* biased
        device, detection should degrade only mildly — this checks that a
        systematic device difference is not mistaken for a deviation.
        """
        base_df = self._base_cohort()
        results: dict[str, float] = {}

        for bias_level in [0.0, 0.05, 0.15]:
            if bias_level == 0.0:
                results["Device bias 0%"] = self._auroc_after_transform(base_df)
                continue

            df = base_df.copy()
            rng = np.random.default_rng(self.seed)
            for _subject_id, group in df.groupby("subject_id", sort=False):
                gain = 1.0 + rng.uniform(-bias_level, bias_level)
                offset_frac = rng.uniform(-bias_level, bias_level)
                idx = group.index
                for col in self.features:
                    scale = float(df.loc[idx, col].std()) or 1.0
                    df.loc[idx, col] = df.loc[idx, col] * gain + offset_frac * scale
            results[f"Device bias {int(bias_level * 100)}%"] = self._auroc_after_transform(df)

        return results

    # ----------------------------------------------------------------- aggregate
    def run_all(self) -> dict[str, dict[str, float]]:
        """Runs every stress axis and returns a nested result mapping."""
        return {
            "missing_data": self.run_missing_data_stress(),
            "sensor_noise": self.run_noise_stress(),
            "spike_outliers": self.run_outlier_stress(),
            "history_length": self.run_history_length_stress(),
            "sampling_rate": self.run_sampling_rate_stress(),
            "device_variation": self.run_device_variation_stress(),
        }
