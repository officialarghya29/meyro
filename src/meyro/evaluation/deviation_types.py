"""Deviation-shape study (tests the mean-shift hypothesis).

The master benchmark uses a single deviation shape: a sustained shift in the
mean of several features. That choice may be doing a lot of work. A robust
per-subject statistic (median/IQR) is close to optimal for exactly that kind of
deviation, so the benchmark may be measuring the *generator* rather than the
methods.

This study re-runs the comparison across four deviation shapes:

* ``mean_shift``        — sustained level change (the original benchmark case)
* ``gradual_ramp``      — linear onset, so the change is only clear late
* ``variance_increase`` — mean preserved, dispersion expanded
* ``point_spike``       — a single sharp day

For each shape it reports the population baseline, the personal baseline, and
MEYRO-V2, all on identical splits. If personalization's advantage holds across
shapes, the finding generalizes; if it is confined to mean shifts, that is the
honest scope of the claim.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from meyro.baselines.personal import PersonalizedBaseline
from meyro.baselines.population import PopulationBaseline
from meyro.data.synthetic import SyntheticBenchmarkGenerator
from meyro.data.windows import (
    FeatureStandardizer,
    align_row_scores,
    build_windows,
)
from meyro.evaluation.metrics import detection_metrics
from meyro.experiments.streaming import stream_scores_v2
from meyro.models.meyro import MEYROModelV2
from meyro.preprocessing.pipeline import PreprocessingPipeline
from meyro.training.trainer import MEYROV2Trainer
from meyro.utils.io import set_global_seed

DEFAULT_FEATURES = ["step_count", "resting_hr", "sleep_duration_min"]
DEVIATION_TYPES = ("mean_shift", "gradual_ramp", "variance_increase", "point_spike")


class DeviationTypeStudy:
    """Compares baseline paradigms across deviation shapes on identical splits."""

    window_size = 7

    def __init__(
        self,
        n_subjects: int = 15,
        n_days: int = 60,
        calibration_days: int = 21,
        seed: int = 42,
        epochs: int = 80,
        deviation_types: tuple[str, ...] = DEVIATION_TYPES,
    ) -> None:
        self.n_subjects = n_subjects
        self.n_days = n_days
        self.calibration_days = calibration_days
        self.seed = seed
        self.epochs = epochs
        self.deviation_types = deviation_types
        self.features = DEFAULT_FEATURES

    def _cohort(self, deviation_type: str, seed: int):
        set_global_seed(seed)
        generator = SyntheticBenchmarkGenerator(seed=seed)
        cohort = generator.generate_cohort(
            n_subjects=self.n_subjects,
            n_days=self.n_days,
            anomaly_rate_per_subject=0.6,
            deviation_type=deviation_type,
        )
        pipeline = PreprocessingPipeline(feature_cols=self.features)
        cleaned = pipeline.clean_and_impute(cohort)
        return pipeline.create_subject_temporal_splits(
            cleaned, calibration_days=self.calibration_days
        )

    def run(self) -> dict[str, Any]:
        results: dict[str, Any] = {}

        for deviation_type in self.deviation_types:
            calib_df, eval_df = self._cohort(deviation_type, self.seed)

            calib_windows = build_windows(calib_df, self.features, window_size=self.window_size)
            eval_windows = build_windows(eval_df, self.features, window_size=self.window_size)
            standardizer = FeatureStandardizer().fit(calib_windows)
            calib_std = standardizer.transform(calib_windows)
            eval_std = standardizer.transform(eval_windows)

            population = PopulationBaseline(feature_cols=self.features).fit(calib_df)
            personal = PersonalizedBaseline(feature_cols=self.features).fit_calibration(calib_df)

            population_y, population_scores, _ = align_row_scores(
                eval_df, population.score(eval_df)["deviation_score"].to_numpy(), self.window_size
            )
            personal_y, personal_scores, _ = align_row_scores(
                eval_df, personal.score_causal(eval_df)["deviation_score"].to_numpy(), self.window_size
            )

            set_global_seed(self.seed)
            model = MEYROModelV2(
                input_dim=len(self.features),
                context_dim=calib_std.context_dim,
                quality_dim=len(self.features),
                hidden_dim=16,
                num_layers=2,
            )
            trainer = MEYROV2Trainer(model, epochs=self.epochs, seed=self.seed)
            trainer.standardizer = standardizer
            fast_memories, slow_memories = trainer.fit_windows(calib_std)

            eval_scores, _persistence = stream_scores_v2(
                model, eval_std, fast_memories, slow_memories
            )

            entry: dict[str, Any] = {
                "prevalence": round(float(eval_windows.y.mean()), 4),
                "n_evaluation_windows": len(eval_windows),
                "population": detection_metrics(population_y, population_scores),
                "personal": detection_metrics(personal_y, personal_scores),
                "meyro_v2": detection_metrics(eval_windows.y, eval_scores),
            }
            entry["personalization_gain_auroc"] = round(
                entry["personal"]["auroc"] - entry["population"]["auroc"], 4
            )
            entry["meyro_gain_over_population_auroc"] = round(
                entry["meyro_v2"]["auroc"] - entry["population"]["auroc"], 4
            )
            results[deviation_type] = entry

        gains = [results[key]["personalization_gain_auroc"] for key in results]
        results["_summary"] = {
            "personalization_gain_by_shape": {
                key: results[key]["personalization_gain_auroc"] for key in self.deviation_types
            },
            "min_gain": round(float(np.min(gains)), 4),
            "max_gain": round(float(np.max(gains)), 4),
            "holds_across_all_shapes": bool(np.all(np.array(gains) > 0)),
            "interpretation": (
                "A positive gain in every shape means the personalization finding is not an "
                "artefact of the mean-shift generator."
            ),
        }
        results["_metadata"] = {
            "n_subjects": self.n_subjects,
            "n_days_per_subject": self.n_days,
            "calibration_days": self.calibration_days,
            "epochs": self.epochs,
            "seed": self.seed,
            "deviation_types": list(self.deviation_types),
        }
        return results
