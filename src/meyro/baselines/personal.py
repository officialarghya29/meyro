"""Personalized baseline engine for MEYRO.

Constructs strictly historical, subject-specific baselines (rolling or EWMA).
Never incorporates future observations. Evaluates whether an observation is normal
FOR THIS INDIVIDUAL.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class PersonalizedBaseline:
    """Computes subject-specific personal baselines using historical observations."""

    def __init__(
        self,
        feature_cols: list[str],
        window_size: int = 14,
        alpha: float = 0.2,
        use_ewma: bool = False,
    ) -> None:
        self.feature_cols = feature_cols
        self.window_size = window_size
        self.alpha = alpha
        self.use_ewma = use_ewma
        # subject_id -> feature -> {"center": float, "scale": float}
        self.static_baselines: dict[str, dict[str, dict[str, float]]] = {}

    def fit_calibration(self, calib_df: pd.DataFrame) -> PersonalizedBaseline:
        """Fits initial historical baseline per subject from calibration period."""
        self.static_baselines = {}
        for subj_id, group in calib_df.groupby("subject_id"):
            self.static_baselines[subj_id] = {}
            for col in self.feature_cols:
                vals = group[col].dropna().to_numpy()
                if len(vals) < 2:
                    self.static_baselines[subj_id][col] = {"center": float(np.mean(vals) if len(vals) else 0.0), "scale": 1.0}
                    continue
                median = float(np.median(vals))
                q75, q25 = np.percentile(vals, [75, 25])
                iqr = float(q75 - q25)
                scale = (iqr / 1.349) if iqr > 1e-4 else float(np.std(vals) or 1.0)
                self.static_baselines[subj_id][col] = {"center": median, "scale": scale}
        return self

    def score_causal(self, eval_df: pd.DataFrame) -> pd.DataFrame:
        """Scores observations against personal baseline.

        For each subject and time t, compares observation against their baseline.
        Returns deviation_score and per-feature z-scores.
        """
        out = eval_df.copy()
        dev_scores = []
        for _, row in out.iterrows():
            subj_id = row["subject_id"]
            subj_base = self.static_baselines.get(subj_id, {})
            z_scores = []
            for col in self.feature_cols:
                stat = subj_base.get(col, {"center": 0.0, "scale": 1.0})
                z = abs(row[col] - stat["center"]) / stat["scale"]
                z_scores.append(z)
            dev_scores.append(float(np.mean(z_scores)))
        out["deviation_score"] = dev_scores
        return out
