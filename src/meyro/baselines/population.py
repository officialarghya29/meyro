"""Population-level baseline model (Control Condition for MEYRO).

Implements population mean, standard deviation, and robust Z-score deviation scores.
Never uses individual-specific tuning; represents standard clinical/population thresholds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class PopulationBaseline:
    """Computes fixed population-level statistics across a cohort calibration set."""

    def __init__(self, feature_cols: list[str], robust: bool = True) -> None:
        self.feature_cols = feature_cols
        self.robust = robust
        self.stats: dict[str, dict[str, float]] = {}

    def fit(self, df: pd.DataFrame) -> PopulationBaseline:
        """Fits population parameters across all subjects in the calibration set."""
        self.stats = {}
        for col in self.feature_cols:
            vals = df[col].dropna().to_numpy()
            if len(vals) == 0:
                self.stats[col] = {"center": 0.0, "scale": 1.0}
                continue

            if self.robust:
                median = float(np.median(vals))
                q75, q25 = np.percentile(vals, [75, 25])
                iqr = float(q75 - q25)
                scale = (iqr / 1.349) if iqr > 1e-6 else float(np.std(vals) or 1.0)
                self.stats[col] = {"center": median, "scale": scale}
            else:
                mean = float(np.mean(vals))
                std = float(np.std(vals))
                self.stats[col] = {"center": mean, "scale": std if std > 1e-6 else 1.0}
        return self

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Computes deviation score: max or aggregate absolute population Z-score across features."""
        out = df.copy()
        dev_scores = []
        for _, row in out.iterrows():
            z_scores = []
            for col in self.feature_cols:
                stat = self.stats.get(col, {"center": 0.0, "scale": 1.0})
                z = abs(row[col] - stat["center"]) / stat["scale"]
                z_scores.append(z)
            dev_scores.append(float(np.mean(z_scores)))
        out["deviation_score"] = dev_scores
        return out
