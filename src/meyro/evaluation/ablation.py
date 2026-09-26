"""Ablation study suite for MEYRO (Phase 19).

All ablations run on a **trained** MEYRO-V2. Ablating an untrained network
measures nothing, so every row below is a genuine test of one mechanism.

Ablated mechanisms (each is checked against the full model on identical
evaluation windows):

* personal memory removed — both memories set to zero
* single-timescale memory — fast memory forced equal to slow memory
* context encoder removed — context zeroed
* quality conditioning removed — quality set to a constant 1.0
* relational deviation module removed — replaced by naive Euclidean distance
* persistence removed — persistence score held at 1.0 (no gating)
* untrained model — transparency check on the value of training itself
"""

from __future__ import annotations

from typing import Any

from meyro.data.synthetic import SyntheticBenchmarkGenerator
from meyro.data.windows import FeatureStandardizer, build_windows
from meyro.evaluation.metrics import detection_metrics
from meyro.experiments.streaming import ScoringOptions, stream_scores_v2
from meyro.models.meyro import MEYROModelV2
from meyro.preprocessing.pipeline import PreprocessingPipeline
from meyro.training.trainer import MEYROV2Trainer
from meyro.utils.io import set_global_seed

DEFAULT_FEATURES = ["step_count", "resting_hr", "sleep_duration_min"]


class AblationStudy:
    """Executes controlled ablations on the trained MEYRO-V2 architecture."""

    window_size = 7

    def __init__(
        self,
        n_subjects: int = 12,
        n_days: int = 50,
        calibration_days: int = 21,
        seed: int = 42,
        epochs: int = 25,
    ) -> None:
        self.n_subjects = n_subjects
        self.n_days = n_days
        self.calibration_days = calibration_days
        self.seed = seed
        self.epochs = epochs
        self.features = DEFAULT_FEATURES

    def run(self) -> dict[str, dict[str, float]]:
        set_global_seed(self.seed)
        gen = SyntheticBenchmarkGenerator(seed=self.seed)
        df = gen.generate_cohort(
            n_subjects=self.n_subjects,
            n_days=self.n_days,
            anomaly_rate_per_subject=0.6,
        )

        pipe = PreprocessingPipeline(feature_cols=self.features)
        cleaned = pipe.clean_and_impute(df)
        calib_df, eval_df = PreprocessingPipeline.create_subject_temporal_splits(
            cleaned, calibration_days=self.calibration_days
        )

        calib_windows = build_windows(calib_df, self.features, window_size=self.window_size)
        eval_windows = build_windows(eval_df, self.features, window_size=self.window_size)

        # Standardisation fitted on calibration windows only.
        standardizer = FeatureStandardizer().fit(calib_windows)
        calib_std = standardizer.transform(calib_windows)
        eval_std = standardizer.transform(eval_windows)

        # Seed immediately before construction for reproducible initialisation.
        set_global_seed(self.seed)
        model = MEYROModelV2(
            input_dim=len(self.features),
            context_dim=calib_windows.context_dim,
            quality_dim=len(self.features),
            hidden_dim=16,
            num_layers=2,
        )
        trainer = MEYROV2Trainer(model, epochs=self.epochs, seed=self.seed)
        trainer.standardizer = standardizer
        fast_memories, slow_memories = trainer.fit_windows(calib_std)

        y_true = eval_windows.y
        results: dict[str, dict[str, float]] = {}

        def stream(options: ScoringOptions, scoring_model: MEYROModelV2 | None = None):
            anomalies, persistences = stream_scores_v2(
                scoring_model or model, eval_std, fast_memories, slow_memories, options
            )
            return anomalies, persistences

        # All ablations are streamed, because adaptation is only observable online:
        # at a single static step the two memories are identical by construction.

        # 1. Full model
        full_scores, _full_persistence = stream(ScoringOptions())
        results["MEYRO-V2 Full (trained)"] = detection_metrics(y_true, full_scores)

        # 2. Without personal memory (memory pinned at zero, never adapts)
        results["Ablation: w/o Personal Memory"] = detection_metrics(
            y_true, stream(ScoringOptions(zero_memory=True))[0]
        )

        # 3. Single-timescale memory (fast memory forced to equal slow memory)
        results["Ablation: Single-Timescale Memory"] = detection_metrics(
            y_true, stream(ScoringOptions(single_timescale=True))[0]
        )

        # 4. Without context
        results["Ablation: w/o Context Encoder"] = detection_metrics(
            y_true, stream(ScoringOptions(zero_context=True))[0]
        )

        # 5. Without quality conditioning
        results["Ablation: w/o Quality Conditioning"] = detection_metrics(
            y_true, stream(ScoringOptions(unit_quality=True))[0]
        )

        # 6. Naive Euclidean deviation instead of the learned relational module
        results["Ablation: Naive Euclidean Deviation"] = detection_metrics(
            y_true, stream(ScoringOptions(euclidean_deviation=True))[0]
        )

        # 7. Persistence gating applied to the anomaly score
        results["Ablation: w/o Persistence Gating"] = detection_metrics(y_true, full_scores)
        results["MEYRO-V2 + Persistence Gating"] = detection_metrics(
            y_true, stream(ScoringOptions(persistence_gating=True))[0]
        )

        # 8. Untrained model — what training is worth
        set_global_seed(self.seed)
        untrained = MEYROModelV2(
            input_dim=len(self.features),
            context_dim=calib_windows.context_dim,
            quality_dim=len(self.features),
            hidden_dim=16,
            num_layers=2,
        )
        results["Reference: MEYRO-V2 Untrained"] = detection_metrics(
            y_true, stream(ScoringOptions(), scoring_model=untrained)[0]
        )

        results["_metadata"] = {
            "n_subjects": int(calib_df["subject_id"].nunique()),
            "n_evaluation_windows": len(eval_windows),
            "evaluation_prevalence": round(float(y_true.mean()), 4),
            "training_windows": int(getattr(trainer, "training_windows", 0)),
            "seed": self.seed,
        }  # type: ignore[assignment]
        return results

    def run_full(self) -> dict[str, Any]:
        """Alias returning the same mapping, typed loosely for JSON export."""
        return self.run()
