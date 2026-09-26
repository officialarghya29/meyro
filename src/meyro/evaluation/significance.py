"""Significance study for the primary hypothesis (Phase 27).

Primary hypothesis: *a personalized baseline detects meaningful deviations
with fewer false positives than a population-level baseline.*

A single pooled AUROC can neither support nor refute this, because the
individual is the natural unit of analysis and a handful of easy subjects can
carry an aggregate number. This study therefore:

1. evaluates several generator seeds (not one);
2. computes **per-subject** AUROC, so results are a distribution across people;
3. reports **bootstrap confidence intervals** for the mean;
4. runs a **paired Wilcoxon signed-rank test** on the same subjects under both
   conditions, plus the share of subjects where personalization wins.

No hyperparameter is tuned against the evaluation data, and no per-seed result
is discarded.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from meyro.baselines.personal import PersonalizedBaseline
from meyro.baselines.population import PopulationBaseline
from meyro.data.synthetic import SyntheticBenchmarkGenerator
from meyro.data.windows import align_row_scores
from meyro.evaluation.metrics import detection_metrics
from meyro.evaluation.statistics import bootstrap_ci, paired_wilcoxon, per_group_auroc
from meyro.personalization.adaptive import AdaptivePersonalBaseline
from meyro.preprocessing.pipeline import PreprocessingPipeline
from meyro.utils.io import set_global_seed

DEFAULT_FEATURES = ["step_count", "resting_hr", "sleep_duration_min"]
WINDOW_SIZE = 7


class PersonalizationSignificanceStudy:
    """Multi-seed, per-subject statistical comparison of baseline paradigms."""

    def __init__(
        self,
        seeds: tuple[int, ...] = (42, 7, 2024, 1, 99),
        n_subjects: int = 15,
        n_days: int = 60,
        calibration_days: int = 21,
    ) -> None:
        self.seeds = seeds
        self.n_subjects = n_subjects
        self.n_days = n_days
        self.calibration_days = calibration_days
        self.features = DEFAULT_FEATURES

    # ------------------------------------------------------------------ helpers
    def _method_scores(
        self,
        calib_df: pd.DataFrame,
        eval_df: pd.DataFrame,
    ) -> dict[str, np.ndarray]:
        """Row-level deviation scores for each statistical baseline."""
        population = PopulationBaseline(feature_cols=self.features).fit(calib_df)
        population_scores = population.score(eval_df)["deviation_score"].to_numpy()

        personal = PersonalizedBaseline(feature_cols=self.features).fit_calibration(calib_df)
        personal_scores = personal.score_causal(eval_df)["deviation_score"].to_numpy()

        adaptive = AdaptivePersonalBaseline(feature_cols=self.features).initialize(calib_df)
        adaptive_scores = np.array(
            [
                adaptive.step(
                    row["subject_id"],
                    {column: float(row[column]) for column in self.features},
                    float(row.get("quality_score", 1.0)),
                )["deviation_score"]
                for _, row in eval_df.iterrows()
            ]
        )

        return {
            "Population Baseline": population_scores,
            "Personal Baseline (Static)": personal_scores,
            "Personal Baseline (Adaptive)": adaptive_scores,
        }

    # -------------------------------------------------------------------- runner
    def run(self) -> dict[str, Any]:
        pooled_auroc: dict[str, dict[str, float]] = {}
        global_by_seed: dict[str, list[dict[str, float]]] = {}
        per_seed_summary: list[dict[str, Any]] = []

        for seed in self.seeds:
            set_global_seed(seed)
            generator = SyntheticBenchmarkGenerator(seed=seed)
            cohort = generator.generate_cohort(
                n_subjects=self.n_subjects,
                n_days=self.n_days,
                anomaly_rate_per_subject=0.6,
            )
            pipeline = PreprocessingPipeline(feature_cols=self.features)
            cleaned = pipeline.clean_and_impute(cohort)
            calib_df, eval_df = pipeline.create_subject_temporal_splits(
                cleaned, calibration_days=self.calibration_days
            )

            row_scores = self._method_scores(calib_df, eval_df)
            seed_summary: dict[str, Any] = {"seed": seed}

            for method, scores in row_scores.items():
                y_true, aligned_scores, groups = align_row_scores(eval_df, scores, WINDOW_SIZE)
                if len(y_true) == 0:
                    continue

                per_subject = per_group_auroc(y_true, aligned_scores, groups)
                # Subject identifiers repeat across seeds, so pair on (seed, subject).
                store = pooled_auroc.setdefault(method, {})
                for key, value in per_subject.items():
                    store[f"{seed}:{key}" if not key.startswith("_") else key] = value

                metrics = detection_metrics(y_true, aligned_scores)
                global_by_seed.setdefault(method, []).append(
                    {
                        "value": metrics["auroc"],
                        "fpr": metrics["fpr_at_target_sensitivity"],
                        "f1": metrics["f1"],
                    }
                )
                seed_summary[method] = {
                    "auroc": metrics["auroc"],
                    "fpr_at_85": metrics["fpr_at_target_sensitivity"],
                    "subjects_scored": int(per_subject["_n_scored_groups"]),
                }

            per_seed_summary.append(seed_summary)

        # ------------------------------------------------ aggregate per subject
        method_report: dict[str, Any] = {}
        per_subject_only: dict[str, dict[str, float]] = {}
        for method, values in pooled_auroc.items():
            clean = {key: value for key, value in values.items() if not key.startswith("_")}
            per_subject_only[method] = clean
            method_report[method] = {
                "per_subject_auroc": bootstrap_ci(list(clean.values()), seed=self.seeds[0]),
                "n_subjects_scored": len(clean),
                "n_single_class_subjects_excluded": int(
                    values.get("_n_skipped_single_class_groups", 0)
                ),
                "global_auroc_across_seeds": {
                    "mean": float(np.mean([entry["value"] for entry in global_by_seed[method]])),
                    "std": float(np.std([entry["value"] for entry in global_by_seed[method]], ddof=0)),
                },
                "global_fpr_at_85_across_seeds": {
                    "mean": float(np.mean([entry["fpr"] for entry in global_by_seed[method]])),
                    "std": float(np.std([entry["fpr"] for entry in global_by_seed[method]], ddof=0)),
                },
            }

        # ------------------------------------------------------- paired testing
        tests: dict[str, Any] = {}
        static = per_subject_only.get("Personal Baseline (Static)", {})
        adaptive = per_subject_only.get("Personal Baseline (Adaptive)", {})
        population = per_subject_only.get("Population Baseline", {})
        if population and static:
            tests["personalized_static_vs_population"] = paired_wilcoxon(static, population)
        if population and adaptive:
            tests["personalized_adaptive_vs_population"] = paired_wilcoxon(adaptive, population)
        if static and adaptive:
            tests["adaptive_vs_static_personalized"] = paired_wilcoxon(adaptive, static)

        return {
            "hypothesis": (
                "Personalized baselines detect deviations with fewer false positives "
                "than population-level baselines."
            ),
            "protocol": {
                "seeds": list(self.seeds),
                "n_subjects_per_seed": self.n_subjects,
                "n_days_per_subject": self.n_days,
                "calibration_days": self.calibration_days,
                "window_size": WINDOW_SIZE,
                "test": "two-sided paired Wilcoxon signed-rank on per-subject AUROC",
                "unit_of_analysis": "subject",
            },
            "per_method": method_report,
            "tests": tests,
            "per_seed_summary": per_seed_summary,
        }
