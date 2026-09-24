"""Classical anomaly detection models configured for personalized and cohort settings."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM


class ClassicalAnomalyEngine:
    """Wraps scikit-learn anomaly detectors with a standardized MEYRO scoring interface."""

    def __init__(
        self,
        algorithm: str = "isolation_forest",
        contamination: float = 0.05,
        random_state: int = 42,
    ) -> None:
        self.algorithm = algorithm
        self.contamination = contamination
        self.random_state = random_state
        self.models: dict[str, object] = {}

    def fit_per_subject(self, calib_df: pd.DataFrame, feature_cols: list[str]) -> ClassicalAnomalyEngine:
        """Trains an independent anomaly detector per subject using their historical calibration data."""
        self.models = {}
        for subj_id, group in calib_df.groupby("subject_id"):
            x_mat = group[feature_cols].dropna().to_numpy()
            if len(x_mat) < 5:
                continue

            if self.algorithm == "isolation_forest":
                model = IsolationForest(
                    contamination=self.contamination,
                    random_state=self.random_state,
                )
            elif self.algorithm == "one_class_svm":
                model = OneClassSVM(nu=self.contamination, kernel="rbf", gamma="scale")
            elif self.algorithm == "lof":
                n_neighbors = min(10, len(x_mat) - 1)
                model = LocalOutlierFactor(
                    n_neighbors=n_neighbors,
                    novelty=True,
                    contamination=self.contamination,
                )
            else:
                raise ValueError(f"Unknown algorithm: {self.algorithm}")

            model.fit(x_mat)
            self.models[subj_id] = model
        return self

    def score_per_subject(self, eval_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
        """Computes anomaly score per subject (higher score = more anomalous)."""
        out = eval_df.copy()
        scores = []
        for _, row in out.iterrows():
            subj_id = row["subject_id"]
            model = self.models.get(subj_id)
            if model is None:
                scores.append(0.0)
                continue
            x = np.array([[row[col] for col in feature_cols]])
            # In scikit-learn, decision_function returns negative values for outliers
            # Convert so that higher = higher anomaly score
            dec = model.decision_function(x)[0]
            scores.append(float(-dec))
        out["anomaly_score"] = scores
        return out
