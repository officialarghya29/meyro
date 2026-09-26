"""Cross-subject (held-out person) evaluation (Phase 21 / fairness-aware reporting).

A within-subject benchmark answers "can the system track deviations for people it
was trained on?" A deployed system must answer a harder question: "does it work
for a person it has **never** seen?"

Protocol:

1. Split the cohort into **disjoint subject sets** (no subject appears in both).
2. Fit the population standardizer and train every neural model on the training
   subjects' calibration windows only.
3. For each held-out subject, build their personal baseline from **their own
   calibration window** (the legitimate cold-start path) and evaluate on their
   evaluation windows.
4. Report AUROC on held-out subjects only, alongside the within-subject result
   from the same run so the generalization gap is explicit.

The statistical personal baseline is fitted per subject, so it needs no
cross-subject training; it is the control in both columns.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from meyro.baselines.personal import PersonalizedBaseline
from meyro.data.synthetic import SyntheticBenchmarkGenerator
from meyro.data.windows import (
    FeatureStandardizer,
    WindowBatch,
    align_row_scores,
    build_windows,
    subject_mean_baseline,
)
from meyro.evaluation.metrics import detection_metrics
from meyro.experiments.streaming import stream_scores_v2
from meyro.models.meyro import MEYROModelV2
from meyro.preprocessing.pipeline import PreprocessingPipeline
from meyro.training.trainer import MEYROV2Trainer
from meyro.utils.io import set_global_seed

DEFAULT_FEATURES = ["step_count", "resting_hr", "sleep_duration_min"]


def _subset_windows(batch: WindowBatch, subjects: set[str]) -> WindowBatch:
    """Selects the windows belonging to a set of subjects."""
    mask = np.array([str(subject) in subjects for subject in batch.subject_ids])
    return WindowBatch(
        x=batch.x[mask],
        y=batch.y[mask],
        context=batch.context[mask],
        quality=batch.quality[mask],
        subject_ids=batch.subject_ids[mask],
    )


class CrossSubjectStudy:
    """Held-out-subject evaluation of the learned model versus the control."""

    window_size = 7

    def __init__(
        self,
        n_subjects: int = 20,
        n_days: int = 60,
        calibration_days: int = 21,
        train_fraction: float = 0.7,
        seed: int = 42,
        epochs: int = 80,
    ) -> None:
        self.n_subjects = n_subjects
        self.n_days = n_days
        self.calibration_days = calibration_days
        self.train_fraction = train_fraction
        self.seed = seed
        self.epochs = epochs
        self.features = DEFAULT_FEATURES

    def run(self) -> dict[str, Any]:
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

        subjects = sorted(cleaned["subject_id"].unique())
        n_train = max(1, round(len(subjects) * self.train_fraction))
        train_subjects = set(subjects[:n_train])
        test_subjects = set(subjects[n_train:])
        if not test_subjects:  # pragma: no cover - guarded for tiny cohorts
            raise ValueError("train_fraction leaves no held-out subjects")

        calib_windows_all = build_windows(calib_df, self.features, window_size=self.window_size)
        eval_windows_all = build_windows(eval_df, self.features, window_size=self.window_size)

        # Standardizer is fitted on TRAINING subjects' calibration windows only.
        train_calib = _subset_windows(calib_windows_all, train_subjects)
        test_calib = _subset_windows(calib_windows_all, test_subjects)
        test_eval = _subset_windows(eval_windows_all, test_subjects)

        standardizer = FeatureStandardizer().fit(train_calib)
        train_calib_std = standardizer.transform(train_calib)
        test_calib_std = standardizer.transform(test_calib)
        test_eval_std = standardizer.transform(test_eval)

        # ---------------- train on seen subjects ----------------
        set_global_seed(self.seed)
        model = MEYROModelV2(
            input_dim=len(self.features),
            context_dim=train_calib_std.context_dim,
            quality_dim=len(self.features),
            hidden_dim=16,
            num_layers=2,
        )
        trainer = MEYROV2Trainer(model, epochs=self.epochs, seed=self.seed)
        trainer.fit_windows(train_calib_std)

        # ---------------- cold-start the held-out subjects ----------------
        # Memories only: the model is NOT re-trained here. A held-out subject's
        # baseline comes from their own calibration window, which is legitimate
        # historical data available before evaluation.
        model.eval()
        unseen_slow = subject_mean_baseline(model.encode_observation, test_calib_std, model.hidden_dim)
        unseen_fast = {key: value.clone() for key, value in unseen_slow.items()}

        # ---------------- within-subject reference ----------------
        all_calib_std = standardizer.transform(calib_windows_all)
        all_eval_std = standardizer.transform(eval_windows_all)
        all_slow = subject_mean_baseline(model.encode_observation, all_calib_std, model.hidden_dim)
        all_fast = {key: value.clone() for key, value in all_slow.items()}
        within_scores, _p = stream_scores_v2(model, all_eval_std, all_fast, all_slow)

        # ---------------- held-out scores ----------------
        unseen_scores, _p2 = stream_scores_v2(model, test_eval_std, unseen_fast, unseen_slow)

        # ---------------- statistical control ----------------
        personal = PersonalizedBaseline(feature_cols=self.features).fit_calibration(calib_df)
        control_rows = personal.score_causal(eval_df)["deviation_score"].to_numpy()
        control_y, control_scores, control_subjects = align_row_scores(
            eval_df, control_rows, self.window_size
        )
        control_mask = np.array([str(subject) in test_subjects for subject in control_subjects])

        results: dict[str, Any] = {
            "within_subject": {
                "MEYRO-V2 streaming": detection_metrics(eval_windows_all.y, within_scores),
            },
            "cross_subject": {
                "MEYRO-V2 streaming (unseen subjects)": detection_metrics(
                    test_eval.y, unseen_scores
                ),
                "Statistical Personal Baseline (unseen subjects)": detection_metrics(
                    control_y[control_mask], control_scores[control_mask]
                ),
            },
            "protocol": {
                "n_subjects_total": len(subjects),
                "n_train_subjects": len(train_subjects),
                "n_held_out_subjects": len(test_subjects),
                "train_fraction": self.train_fraction,
                "epochs": self.epochs,
                "seed": self.seed,
                "note": (
                    "Held-out subjects never contribute windows to training or to the "
                    "standardizer; their personal baseline is built from their own "
                    "calibration window only (legitimate cold start)."
                ),
            },
        }

        gap = (
            results["within_subject"]["MEYRO-V2 streaming"]["auroc"]
            - results["cross_subject"]["MEYRO-V2 streaming (unseen subjects)"]["auroc"]
        )
        results["generalization_gap_auroc"] = round(float(gap), 4)
        return results
