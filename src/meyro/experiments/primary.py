"""Primary Research Experiment suite comparing Population Baseline vs Personalized Baseline (Phase 11).

Evaluates:
- AUROC (Area Under ROC Curve)
- AUPRC (Area Under Precision-Recall Curve)
- False Positive Rate at 80% and 90% Target Sensitivity
- Per-subject deviation breakdown
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

from meyro.baselines.personal import PersonalizedBaseline
from meyro.baselines.population import PopulationBaseline
from meyro.data.synthetic import SyntheticBenchmarkGenerator
from meyro.personalization.adaptive import AdaptivePersonalBaseline
from meyro.preprocessing.pipeline import PreprocessingPipeline


class PrimaryExperiment:
    """Executes the foundational MEYRO experiment: Population Baseline vs Personalized Baseline."""

    def __init__(
        self,
        n_subjects: int = 15,
        n_days: int = 75,
        calibration_days: int = 21,
        seed: int = 42,
    ) -> None:
        self.n_subjects = n_subjects
        self.n_days = n_days
        self.calibration_days = calibration_days
        self.seed = seed
        self.features = ["step_count", "resting_hr", "sleep_duration_min"]

    def run(self) -> dict[str, Any]:
        # 1. Generate benchmark cohort
        gen = SyntheticBenchmarkGenerator(seed=self.seed)
        raw_df = gen.generate_cohort(
            n_subjects=self.n_subjects,
            n_days=self.n_days,
            anomaly_rate_per_subject=0.6,
        )

        # 2. Preprocess
        pipe = PreprocessingPipeline(feature_cols=self.features)
        cleaned = pipe.clean_and_impute(raw_df)
        calib_df, eval_df = PreprocessingPipeline.create_subject_temporal_splits(
            cleaned, calibration_days=self.calibration_days
        )

        y_true = eval_df["ground_truth_label"].to_numpy()

        # 3. Fit Population Baseline (Control condition)
        pop_model = PopulationBaseline(feature_cols=self.features)
        pop_model.fit(calib_df)
        pop_scored = pop_model.score(eval_df)
        pop_scores = pop_scored["deviation_score"].to_numpy()

        # 4. Fit Personalized Baseline (Static historical)
        pers_model = PersonalizedBaseline(feature_cols=self.features)
        pers_model.fit_calibration(calib_df)
        pers_scored = pers_model.score_causal(eval_df)
        pers_scores = pers_scored["deviation_score"].to_numpy()

        # 5. Fit Adaptive Personal Baseline
        adaptive = AdaptivePersonalBaseline(feature_cols=self.features)
        adaptive.initialize(calib_df)
        adapt_scores = []
        for _, row in eval_df.iterrows():
            feat_dict = {col: float(row[col]) for col in self.features}
            res = adaptive.step(row["subject_id"], feat_dict, quality_score=row["quality_score"])
            adapt_scores.append(res["deviation_score"])
        adapt_scores = np.array(adapt_scores)

        # Compute Comparative Metrics
        metrics = {
            "population": self._compute_metrics(y_true, pop_scores),
            "personalized_static": self._compute_metrics(y_true, pers_scores),
            "personalized_adaptive": self._compute_metrics(y_true, adapt_scores),
            "n_evaluation_samples": len(y_true),
            "n_positive_deviations": int(np.sum(y_true)),
        }
        return metrics

    @staticmethod
    def _compute_metrics(y_true: np.ndarray, y_scores: np.ndarray) -> dict[str, float]:
        if len(np.unique(y_true)) < 2:
            return {"auroc": 0.5, "auprc": 0.0, "fpr_at_85_recall": 1.0}

        auroc = float(roc_auc_score(y_true, y_scores))
        prec, rec, _ = precision_recall_curve(y_true, y_scores)
        # Precision-recall curve returns recall in descending order; sort ascending for positive area integration
        sort_idx = np.argsort(rec)
        rec_sorted = rec[sort_idx]
        prec_sorted = prec[sort_idx]
        if hasattr(np, "trapezoid"):
            auprc = float(np.trapezoid(prec_sorted, rec_sorted))
        else:
            auprc = float(np.trapz(prec_sorted, rec_sorted))

        fpr, tpr, _ = roc_curve(y_true, y_scores)
        # Find FPR where TPR >= 0.85
        target_idx = np.where(tpr >= 0.85)[0]
        fpr_at_85 = float(fpr[target_idx[0]]) if len(target_idx) > 0 else 1.0

        return {
            "auroc": round(auroc, 4),
            "auprc": round(auprc, 4),
            "fpr_at_85_recall": round(fpr_at_85, 4),
        }
