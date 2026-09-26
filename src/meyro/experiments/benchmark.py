"""Master model comparison (Phase 18) — corrected for a fair comparison.

An earlier version of this benchmark evaluated the neural models at random
initialisation. That is not a baseline: it measures the initialisation, not the
architecture, and it made the comparison meaningless. Every neural model here is
now **trained on exactly the calibration windows** the statistical baselines are
fit on, and scored on the identical evaluation windows.

Models compared:

* Population baseline (statistical control)
* Personal baseline, static (statistical personalised control)
* Personal baseline, adaptive
* Isolation Forest (per subject)
* One-Class SVM (per subject)
* GRU / LSTM / TCN / Transformer autoencoders (trained, reconstruction error)
* MEYRO-V1 (trained)
* MEYRO-V2 (trained) — dual-timescale memory + persistence
* MEYRO-V2 untrained (transparency check on the cost of skipping training)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from meyro.anomaly.classical import ClassicalAnomalyEngine
from meyro.baselines.personal import PersonalizedBaseline
from meyro.baselines.population import PopulationBaseline
from meyro.data.synthetic import SyntheticBenchmarkGenerator
from meyro.data.windows import FeatureStandardizer, build_windows
from meyro.evaluation.metrics import detection_metrics
from meyro.experiments.streaming import stream_scores_v1, stream_scores_v2
from meyro.models.meyro import MEYROModel, MEYROModelV2
from meyro.models.tcn import TCNAutoencoder
from meyro.models.temporal import TemporalAutoencoder
from meyro.models.transformer import TransformerAutoencoder
from meyro.personalization.adaptive import AdaptivePersonalBaseline
from meyro.preprocessing.pipeline import PreprocessingPipeline
from meyro.training.reconstruction import train_reconstruction
from meyro.training.trainer import MEYROTrainer, MEYROV2Trainer
from meyro.utils.io import set_global_seed

DEFAULT_FEATURES = ["step_count", "resting_hr", "sleep_duration_min"]


class MasterModelBenchmark:
    """Trains and evaluates every competing architecture on identical data."""

    window_size = 7

    def __init__(
        self,
        n_subjects: int = 15,
        n_days: int = 60,
        calibration_days: int = 21,
        seed: int = 42,
        epochs: int = 30,
    ) -> None:
        self.n_subjects = n_subjects
        self.n_days = n_days
        self.calibration_days = calibration_days
        self.seed = seed
        self.epochs = epochs
        self.features = DEFAULT_FEATURES

    # -------------------------------------------------------------- data setup
    def _prepare_data(self) -> tuple[Any, Any, Any, dict[str, Any]]:
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

        metadata = {
            "n_subjects": int(calib_df["subject_id"].nunique()),
            "n_days_per_subject": self.n_days,
            "calibration_days": self.calibration_days,
            "n_calibration_windows": len(calib_windows),
            "n_evaluation_windows": len(eval_windows),
            "evaluation_prevalence": round(float(eval_windows.y.mean()), 4),
            "window_size": self.window_size,
            "features": list(self.features),
            "seed": self.seed,
        }
        return calib_df, eval_df, (calib_windows, eval_windows), metadata

    # ------------------------------------------------------------------- runner
    def run(self) -> dict[str, Any]:
        calib_df, eval_df, windows, metadata = self._prepare_data()
        calib_windows, eval_windows = windows

        # Neural models receive per-subject standardised inputs fitted on
        # calibration windows only (never on evaluation data).
        standardizer = FeatureStandardizer().fit(calib_windows)
        calib_std = standardizer.transform(calib_windows)
        eval_std = standardizer.transform(eval_windows)

        y_true = eval_windows.y
        x_eval = torch.tensor(eval_std.x, dtype=torch.float32)
        results: dict[str, Any] = {}

        # ---------------- statistical controls ----------------
        pop = PopulationBaseline(feature_cols=self.features).fit(calib_df)
        results["Population Baseline"] = self._metrics_for_row_scores(
            eval_df, y_true, pop.score(eval_df)["deviation_score"].to_numpy()
        )

        pers = PersonalizedBaseline(feature_cols=self.features).fit_calibration(calib_df)
        results["Personal Baseline (Static)"] = self._metrics_for_row_scores(
            eval_df, y_true, pers.score_causal(eval_df)["deviation_score"].to_numpy()
        )

        adapt = AdaptivePersonalBaseline(feature_cols=self.features).initialize(calib_df)
        adapt_scores = [
            adapt.step(
                row["subject_id"],
                {c: float(row[c]) for c in self.features},
                float(row.get("quality_score", 1.0)),
            )["deviation_score"]
            for _, row in eval_df.iterrows()
        ]
        # Adaptive engine scores individual rows; evaluation windows end at row
        # indices offset by (window_size - 1) per subject, so align accordingly.
        results["Personal Baseline (Adaptive)"] = self._metrics_for_row_scores(
            eval_df, y_true, np.array(adapt_scores)
        )

        # ---------------- classical anomaly detectors ----------------
        for label, algorithm in [
            ("Isolation Forest", "isolation_forest"),
            ("One-Class SVM", "one_class_svm"),
        ]:
            engine = ClassicalAnomalyEngine(algorithm=algorithm, random_state=self.seed)
            engine.fit_per_subject(calib_df, self.features)
            row_scores = engine.score_per_subject(eval_df, self.features)["anomaly_score"].to_numpy()
            results[label] = self._metrics_for_row_scores(eval_df, y_true, row_scores)

        # ---------------- deep autoencoder baselines (trained) ----------------
        x_calib, _c_calib, _q_calib = calib_std.tensors()

        gru = train_reconstruction(
            TemporalAutoencoder(input_dim=len(self.features), hidden_dim=16, cell_type="gru"),
            x_calib,
            epochs=self.epochs,
            seed=self.seed,
        )
        results["GRU Autoencoder (trained)"] = detection_metrics(
            y_true, gru.compute_deviation_score(x_eval).numpy()
        )

        lstm = train_reconstruction(
            TemporalAutoencoder(input_dim=len(self.features), hidden_dim=16, cell_type="lstm"),
            x_calib,
            epochs=self.epochs,
            seed=self.seed,
        )
        results["LSTM Autoencoder (trained)"] = detection_metrics(
            y_true, lstm.compute_deviation_score(x_eval).numpy()
        )

        tcn = train_reconstruction(
            TCNAutoencoder(input_dim=len(self.features), hidden_dim=16, num_layers=2),
            x_calib,
            epochs=self.epochs,
            seed=self.seed,
        )
        results["TCN Autoencoder (trained)"] = detection_metrics(
            y_true, tcn.compute_deviation_score(x_eval).numpy()
        )

        transformer = train_reconstruction(
            TransformerAutoencoder(input_dim=len(self.features), d_model=16, nhead=2, num_layers=1),
            x_calib,
            epochs=self.epochs,
            seed=self.seed,
        )
        results["Transformer Autoencoder (trained)"] = detection_metrics(
            y_true, transformer.compute_deviation_score(x_eval).numpy()
        )

        # ---------------- MEYRO-V1 (trained) ----------------
        # Seed immediately before construction so the init is reproducible and
        # independent of how much RNG earlier models consumed.
        set_global_seed(self.seed)
        meyro_v1 = MEYROModel(
            input_dim=len(self.features),
            context_dim=calib_windows.context_dim,
            quality_dim=len(self.features),
            hidden_dim=16,
        )
        v1_trainer = MEYROTrainer(meyro_v1, epochs=self.epochs, seed=self.seed)
        v1_trainer.standardizer = standardizer
        v1_memories = v1_trainer.fit_windows(calib_std)
        # Streamed causally: each window is scored against history already seen.
        scores_v1 = stream_scores_v1(meyro_v1, eval_std, v1_memories)
        results["MEYRO-V1 (ours, trained)"] = detection_metrics(y_true, scores_v1)

        # ---------------- MEYRO-V2 (trained) ----------------
        set_global_seed(self.seed)
        meyro_v2 = MEYROModelV2(
            input_dim=len(self.features),
            context_dim=calib_windows.context_dim,
            quality_dim=len(self.features),
            hidden_dim=16,
            num_layers=2,
        )
        v2_trainer = MEYROV2Trainer(meyro_v2, epochs=self.epochs, seed=self.seed)
        v2_trainer.standardizer = standardizer
        fast_memories, slow_memories = v2_trainer.fit_windows(calib_std)
        scores_v2, persistence_v2 = stream_scores_v2(
            meyro_v2, eval_std, fast_memories, slow_memories
        )
        results["MEYRO-V2 (ours, trained)"] = detection_metrics(y_true, scores_v2)

        # Persistence as a secondary signal (reported, not used to pick thresholds)
        results["MEYRO-V2 (ours, trained)"]["mean_persistence_score"] = round(
            float(np.mean(persistence_v2)), 4
        )

        # ---------------- untrained transparency check ----------------
        set_global_seed(self.seed)
        untrained = MEYROModelV2(
            input_dim=len(self.features),
            context_dim=calib_windows.context_dim,
            quality_dim=len(self.features),
            hidden_dim=16,
            num_layers=2,
        )
        untrained_scores, _untrained_persistence = stream_scores_v2(
            untrained, eval_std, fast_memories, slow_memories
        )
        results["MEYRO-V2 (untrained reference)"] = detection_metrics(y_true, untrained_scores)

        results["_metadata"] = metadata
        return results

    # ------------------------------------------------------------------ helpers
    def _metrics_for_row_scores(
        self,
        eval_df: Any,
        y_true: np.ndarray,
        row_scores: np.ndarray,
    ) -> dict[str, float]:
        """Aligns row-level scores to window-level labels.

        Window ``i`` of a subject ends at that subject's row ``window_size - 1``,
        so the matching row score is ``row_index`` offset accordingly. Using the
        *ending* row keeps the comparison causal and consistent with the
        window-based models.
        """
        scored = eval_df.copy()
        scored["_row_score"] = row_scores
        aligned: list[float] = []
        offset = self.window_size - 1
        for _subject_id, group in scored.groupby("subject_id", sort=False):
            ordered = group.sort_values("timestamp")
            aligned.extend(ordered["_row_score"].to_numpy()[offset:].tolist())
        return detection_metrics(y_true, np.array(aligned))
