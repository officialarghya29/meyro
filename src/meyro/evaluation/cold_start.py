"""Few-shot personalization and cold-start convergence evaluation (Phases 21-23).

Measures: How much historical data does MEYRO require before its personal baseline
becomes statistically superior to population-level thresholds?
"""

from __future__ import annotations

from sklearn.metrics import roc_auc_score

from meyro.baselines.personal import PersonalizedBaseline
from meyro.baselines.population import PopulationBaseline
from meyro.data.synthetic import SyntheticBenchmarkGenerator
from meyro.preprocessing.pipeline import PreprocessingPipeline


class ColdStartAnalysis:
    """Evaluates the calibration history curve: 3, 7, 14, 21, and 28 days."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.features = ["step_count", "resting_hr", "sleep_duration_min"]

    def run_curve(self) -> dict[str, dict[str, float]]:
        gen = SyntheticBenchmarkGenerator(seed=self.seed)
        # 60 days total monitoring per subject
        df = gen.generate_cohort(n_subjects=12, n_days=60, anomaly_rate_per_subject=0.6)
        pipe = PreprocessingPipeline(feature_cols=self.features)
        cleaned = pipe.clean_and_impute(df)

        history_days_list = [3, 7, 14, 21, 28]
        results = {}

        for h_days in history_days_list:
            calib_df, eval_df = PreprocessingPipeline.create_subject_temporal_splits(
                cleaned, calibration_days=h_days
            )

            # Evaluate fixed population baseline
            pop = PopulationBaseline(feature_cols=self.features).fit(calib_df)
            pop_scores = pop.score(eval_df)["deviation_score"].to_numpy()

            # Evaluate personal baseline
            pers = PersonalizedBaseline(feature_cols=self.features).fit_calibration(calib_df)
            pers_scores = pers.score_causal(eval_df)["deviation_score"].to_numpy()

            y_true = eval_df["ground_truth_label"].to_numpy()

            pop_auroc = float(roc_auc_score(y_true, pop_scores))
            pers_auroc = float(roc_auc_score(y_true, pers_scores))

            results[f"{h_days}_days"] = {
                "population_auroc": round(pop_auroc, 4),
                "personalized_auroc": round(pers_auroc, 4),
                "delta_advantage": round(pers_auroc - pop_auroc, 4),
            }

        return results
