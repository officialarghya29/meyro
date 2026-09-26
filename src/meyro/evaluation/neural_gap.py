"""Diagnosis of the neural/statistical performance gap.

The master benchmark shows the learned MEYRO models underperforming a robust
statistical personal baseline. Rather than assume a cause, this study isolates
candidate explanations under a controlled protocol:

* **Memory adaptation during sustained deviations.** MEYRO's memory updates at
  every step. If its own score is low early in a deviation, the gate permits
  adaptation, so the baseline drifts towards the anomalous state and detection
  decays. Comparing streaming (adaptive) against frozen (non-adapting) scoring
  isolates this effect.
* **Under-training.** The same configuration is re-run with substantially more
  epochs; if performance is flat, training time is not the binding constraint.
* **Architecture revision.** V1 and V2 are compared so a V2 regression cannot be
  attributed to "the neural approach" as a whole.

Each configuration is scored on identical windows with identical standardisation.
The statistical personal baseline is included as the control.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from meyro.baselines.personal import PersonalizedBaseline
from meyro.data.synthetic import SyntheticBenchmarkGenerator
from meyro.data.windows import FeatureStandardizer, align_row_scores, build_windows
from meyro.evaluation.metrics import detection_metrics
from meyro.experiments.streaming import ScoringOptions, stream_scores_v1, stream_scores_v2
from meyro.models.meyro import MEYROModel, MEYROModelV2
from meyro.preprocessing.pipeline import PreprocessingPipeline
from meyro.training.trainer import MEYROTrainer, MEYROV2Trainer
from meyro.utils.io import set_global_seed

DEFAULT_FEATURES = ["step_count", "resting_hr", "sleep_duration_min"]


class NeuralGapDiagnosis:
    """Isolates why learned models lose to the statistical personal baseline."""

    window_size = 7

    def __init__(
        self,
        n_subjects: int = 15,
        n_days: int = 60,
        calibration_days: int = 21,
        seed: int = 42,
        epochs: int = 30,
        epoch_sweep: tuple[int, ...] = (30, 60, 120),
    ) -> None:
        self.n_subjects = n_subjects
        self.n_days = n_days
        self.calibration_days = calibration_days
        self.seed = seed
        self.epochs = epochs
        self.epoch_sweep = epoch_sweep
        self.features = DEFAULT_FEATURES

    def _prepare(self):
        set_global_seed(self.seed)
        generator = SyntheticBenchmarkGenerator(seed=self.seed)
        cohort = generator.generate_cohort(
            n_subjects=self.n_subjects, n_days=self.n_days, anomaly_rate_per_subject=0.6
        )
        pipeline = PreprocessingPipeline(feature_cols=self.features)
        cleaned = pipeline.clean_and_impute(cohort)
        calib_df, eval_df = pipeline.create_subject_temporal_splits(
            cleaned, calibration_days=self.calibration_days
        )

        calib_windows = build_windows(calib_df, self.features, window_size=self.window_size)
        eval_windows = build_windows(eval_df, self.features, window_size=self.window_size)

        standardizer = FeatureStandardizer().fit(calib_windows)
        return (
            calib_df,
            eval_df,
            calib_windows,
            eval_windows,
            standardizer.transform(calib_windows),
            standardizer.transform(eval_windows),
        )

    def run(self) -> dict[str, Any]:
        (
            calib_df,
            eval_df,
            _calib_windows,
            eval_windows,
            calib_std,
            eval_std,
        ) = self._prepare()

        y_true = eval_windows.y
        results: dict[str, Any] = {}

        # ---------------- statistical control ----------------
        personal = PersonalizedBaseline(feature_cols=self.features).fit_calibration(calib_df)
        population_rows = personal.score_causal(eval_df)["deviation_score"].to_numpy()
        y_personal, scores_personal, _groups = align_row_scores(eval_df, population_rows, self.window_size)
        results["Control: Statistical Personal Baseline"] = self._summarise(y_personal, scores_personal)

        # ---------------- MEYRO-V1 ----------------
        set_global_seed(self.seed)
        v1 = MEYROModel(
            input_dim=len(self.features),
            context_dim=calib_std.context_dim,
            quality_dim=len(self.features),
            hidden_dim=16,
        )
        v1_trainer = MEYROTrainer(v1, epochs=self.epochs, seed=self.seed)
        v1_memories = v1_trainer.fit_windows(calib_std)

        results["MEYRO-V1 streaming (adaptive)"] = self._summarise(
            y_true, stream_scores_v1(v1, eval_std, v1_memories, ScoringOptions())
        )
        results["MEYRO-V1 frozen memory"] = self._summarise(
            y_true, stream_scores_v1(v1, eval_std, v1_memories, ScoringOptions(freeze_memory=True))
        )

        # ---------------- MEYRO-V2 ----------------
        set_global_seed(self.seed)
        v2 = MEYROModelV2(
            input_dim=len(self.features),
            context_dim=calib_std.context_dim,
            quality_dim=len(self.features),
            hidden_dim=16,
            num_layers=2,
        )
        v2_trainer = MEYROV2Trainer(v2, epochs=self.epochs, seed=self.seed)
        fast, slow = v2_trainer.fit_windows(calib_std)

        v2_streaming, _persistence_streaming = stream_scores_v2(
            v2, eval_std, fast, slow, ScoringOptions()
        )
        results["MEYRO-V2 streaming (adaptive)"] = self._summarise(y_true, v2_streaming)

        v2_frozen, _persistence_frozen = stream_scores_v2(
            v2, eval_std, fast, slow, ScoringOptions(freeze_memory=True)
        )
        results["MEYRO-V2 frozen memory"] = self._summarise(y_true, v2_frozen)

        # ------- under-training check: identical data, larger epoch budgets -------
        budget_aurocs: dict[int, float] = {}
        for budget in self.epoch_sweep:
            if budget == self.epochs:
                budget_aurocs[budget] = results["MEYRO-V2 streaming (adaptive)"]["auroc"]
                continue
            set_global_seed(self.seed)
            model = MEYROModelV2(
                input_dim=len(self.features),
                context_dim=calib_std.context_dim,
                quality_dim=len(self.features),
                hidden_dim=16,
                num_layers=2,
            )
            trainer = MEYROV2Trainer(model, epochs=budget, seed=self.seed)
            fast_b, slow_b = trainer.fit_windows(calib_std)
            scores_b, _p = stream_scores_v2(model, eval_std, fast_b, slow_b, ScoringOptions())
            results[f"MEYRO-V2 streaming, {budget} epochs"] = self._summarise(y_true, scores_b)
            budget_aurocs[budget] = results[f"MEYRO-V2 streaming, {budget} epochs"]["auroc"]

        # ------------------------------------------------------------- diagnosis
        streaming_auroc = results["MEYRO-V2 streaming (adaptive)"]["auroc"]
        frozen_auroc = results["MEYRO-V2 frozen memory"]["auroc"]
        control_auroc = results["Control: Statistical Personal Baseline"]["auroc"]
        best_budget = max(budget_aurocs, key=lambda key: budget_aurocs[key])
        long_auroc = budget_aurocs[best_budget]

        results["_diagnosis"] = {
            "adaptation_cost_auroc": round(frozen_auroc - streaming_auroc, 4),
            "adaptation_cost_interpretation": (
                "Positive means the memory adapting during deviations costs detection accuracy. "
                "Near zero clears adaptation as the cause of the gap."
            ),
            "epoch_budget_aurocs": {str(key): value for key, value in sorted(budget_aurocs.items())},
            "best_epoch_budget": best_budget,
            "training_gain_at_best_budget": round(long_auroc - streaming_auroc, 4),
            "training_interpretation": (
                "A large positive value means the shorter budget under-trained the model; "
                "a near-zero value clears training as the binding constraint."
            ),
            "remaining_gap_to_control": round(control_auroc - max(streaming_auroc, frozen_auroc, long_auroc), 4),
        }
        results["_metadata"] = {
            "n_subjects": int(calib_df["subject_id"].nunique()),
            "n_evaluation_windows": len(eval_windows),
            "evaluation_prevalence": round(float(y_true.mean()), 4),
            "epochs": self.epochs,
            "epoch_sweep": list(self.epoch_sweep),
            "seed": self.seed,
        }
        return results

    @staticmethod
    def _summarise(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
        """Detection metrics plus the normal/anomalous score separation."""
        metrics = detection_metrics(y_true, scores)
        normal = scores[y_true == 0]
        anomalous = scores[y_true == 1]
        metrics["mean_score_normal"] = round(float(normal.mean()), 4) if normal.size else 0.0
        metrics["mean_score_anomalous"] = round(float(anomalous.mean()), 4) if anomalous.size else 0.0
        metrics["score_separation"] = round(
            metrics["mean_score_anomalous"] - metrics["mean_score_normal"], 4
        )
        return metrics
