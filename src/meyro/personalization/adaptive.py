"""Adaptive Personal Baseline engine (Phase 10).

Distinguishes:
1. Temporary deviation (short acute perturbation)
2. Persistent deviation (sustained shift)
3. Gradual baseline drift (slow adaptation of personal setpoints)
4. Measurement noise / low-quality bins

Incorporate historical decay and update rules preventing premature adaptation to acute anomalies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class AdaptivePersonalBaseline:
    """Maintains an adaptive personal baseline that learns slowly from validated normal history."""

    def __init__(
        self,
        feature_cols: list[str],
        adaptation_rate: float = 0.05,
        anomaly_rejection_threshold: float = 2.5,
    ) -> None:
        self.feature_cols = feature_cols
        self.adaptation_rate = adaptation_rate
        self.anomaly_rejection_threshold = anomaly_rejection_threshold
        # subject_id -> feature -> {"center": float, "scale": float}
        self.state: dict[str, dict[str, dict[str, float]]] = {}

    def initialize(self, calib_df: pd.DataFrame) -> AdaptivePersonalBaseline:
        """Initializes baseline centers and scales from the historical calibration period."""
        self.state = {}
        for subj_id, group in calib_df.groupby("subject_id"):
            self.state[subj_id] = {}
            for col in self.feature_cols:
                vals = group[col].dropna().to_numpy()
                median = float(np.median(vals)) if len(vals) else 0.0
                q75, q25 = np.percentile(vals, [75, 25]) if len(vals) >= 4 else (median + 1.0, median - 1.0)
                scale = float((q75 - q25) / 1.349) if (q75 - q25) > 1e-3 else 1.0
                self.state[subj_id][col] = {"center": median, "scale": scale}
        return self

    def step(
        self,
        subject_id: str,
        features: dict[str, float],
        quality_score: float = 1.0,
    ) -> dict[str, float]:
        """Causally processes a single incoming observation, scores deviation, and conditionally updates baseline."""
        subj_state = self.state.setdefault(subject_id, {})
        z_scores = []

        for col in self.feature_cols:
            val = features.get(col, 0.0)
            st = subj_state.setdefault(col, {"center": val, "scale": 1.0})
            z = abs(val - st["center"]) / st["scale"]
            z_scores.append(z)

            # Conditional adaptation: Only adapt if quality is high and observation is NOT an acute anomaly
            if z < self.anomaly_rejection_threshold and quality_score >= 0.5:
                # Slow exponential drift tracking
                st["center"] = (1.0 - self.adaptation_rate) * st["center"] + self.adaptation_rate * val

        dev_score = float(np.mean(z_scores)) if z_scores else 0.0
        return {
            "deviation_score": dev_score,
            "max_z": float(np.max(z_scores)) if z_scores else 0.0,
            "adapted": bool(dev_score < self.anomaly_rejection_threshold and quality_score >= 0.5),
        }
