"""Baseline-drift experiment (Phase 24).

Central research question addressed here:

    Can personalization distinguish a *sustained lifestyle change* (which should
    be absorbed into the baseline) from a *transient anomalous excursion* (which
    should not)?

Two failure modes are measured at once:

* **Persistent false alarm** — a static baseline keeps flagging the post-drift
  steady state forever, because its reference has gone stale.
* **Premature adaptation** — an over-eager adaptive baseline absorbs a single
  acute outlier within one observation, erasing a real deviation.

Both are evaluated on the *same* generator so the comparison is like-for-like.
Thresholds are self-calibrating: the alert threshold for each method is
``pre-drift mean + 3 * pre-drift std``, computed from the pre-drift period only,
so no post-drift information influences the decision rule.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from meyro.baselines.personal import PersonalizedBaseline
from meyro.data.synthetic import SyntheticBenchmarkGenerator
from meyro.personalization.adaptive import AdaptivePersonalBaseline
from meyro.preprocessing.pipeline import PreprocessingPipeline
from meyro.utils.io import set_global_seed

# Synthetic series always start here (see SyntheticBenchmarkGenerator).
SERIES_START = datetime(2025, 1, 1, 0, 0, tzinfo=UTC)
# Evaluation must contain some pre-drift days so each method can self-calibrate.
PRE_DRIFT_EVAL_DAYS = 7


class BaselineDriftAnalysis:
    """Compares static vs. adaptive personal baselines under gradual drift."""

    def __init__(
        self,
        seed: int = 42,
        n_subjects: int = 10,
        n_days: int = 70,
        drift_start_day: int = 28,
        drift_duration_days: int = 12,
    ) -> None:
        self.seed = seed
        self.n_subjects = n_subjects
        self.n_days = n_days
        self.drift_start_day = drift_start_day
        self.drift_duration_days = drift_duration_days
        self.features = ["step_count", "resting_hr", "sleep_duration_min"]
        # Calibration must end before drift onset, leaving pre-drift evaluation rows.
        self.calibration_days = max(7, drift_start_day - PRE_DRIFT_EVAL_DAYS)

    def _cohort(self) -> pd.DataFrame:
        gen = SyntheticBenchmarkGenerator(seed=self.seed)
        df = gen.generate_drift_cohort(
            n_subjects=self.n_subjects,
            n_days=self.n_days,
            drift_start_day=self.drift_start_day,
            drift_duration_days=self.drift_duration_days,
        )
        return PreprocessingPipeline(feature_cols=self.features).clean_and_impute(df)

    def run(self, include_meyro: bool = True) -> dict[str, dict[str, float | None]]:
        tuned = self._cohort()

        static_scores = self._score_static(tuned)
        adaptive_scores = self._score_adaptive(tuned)

        results: dict[str, dict[str, float | None]] = {
            "Personal Baseline (Static)": self._summarise(static_scores),
            "Personal Baseline (Adaptive)": self._summarise(adaptive_scores),
        }

        if include_meyro:
            results["MEYRO-V2 (streamed, trained)"] = self._summarise(self._score_meyro(tuned))
        results["interpretation"] = {
            "persistent_false_alarm_rate": "fraction of post-drift steady-state days still alerted (lower is better)",
            "adaptation_lag_days": "days from drift onset to the LAST alert; equals remaining horizon if it never adapts (lower is better)",
            "baseline_shift_std_total": "total baseline centre movement over the evaluation period, in pre-drift std units (context, not a score)",
            "acute_outlier_probe": "baseline displacement after one 3x observation; measures premature adaptation (lower is better)",
        }
        return results

    # ------------------------------------------------------------------ scoring
    def _score_static(self, df: pd.DataFrame) -> pd.DataFrame:
        calib, eval_df = PreprocessingPipeline.create_subject_temporal_splits(
            df, calibration_days=self.calibration_days
        )
        model = PersonalizedBaseline(feature_cols=self.features).fit_calibration(calib)
        scored = model.score_causal(eval_df)

        # Premature adaptation cannot occur: the static baseline never updates.
        scored["baseline_shift_std"] = 0.0
        return scored

    def _score_adaptive(self, df: pd.DataFrame) -> pd.DataFrame:
        calib, eval_df = PreprocessingPipeline.create_subject_temporal_splits(
            df, calibration_days=self.calibration_days
        )
        engine = AdaptivePersonalBaseline(feature_cols=self.features).initialize(calib)

        # Snapshot centres before streaming so adaptation magnitude can be measured.
        initial_centres = {
            subj: {col: state["center"] for col, state in cols.items()}
            for subj, cols in engine.state.items()
        }

        rows = []
        for _, row in eval_df.sort_values(by=["subject_id", "timestamp"]).iterrows():
            result = engine.step(
                row["subject_id"],
                {c: float(row[c]) for c in self.features},
                float(row.get("quality_score", 1.0)),
            )
            rows.append({**row.to_dict(), "deviation_score": result["deviation_score"]})

        scored = pd.DataFrame(rows)

        # How far did the baseline centre move, in pre-drift std units?
        shifts = []
        for _, row in scored.iterrows():
            subj = row["subject_id"]
            total = []
            for col in self.features:
                base = engine.state.get(subj, {}).get(col)
                if base is None or base["scale"] <= 1e-6:
                    continue
                initial = initial_centres.get(subj, {}).get(col, base["center"])
                total.append(abs(base["center"] - initial) / base["scale"])
            shifts.append(float(np.mean(total)) if total else 0.0)
        scored["baseline_shift_std"] = shifts
        return scored

    # ---------------------------------------------------------------- summarise
    def _summarise(self, scored: pd.DataFrame) -> dict[str, float | None]:
        if scored.empty:
            return {
                "pre_drift_alert_rate": 0.0,
                "persistent_false_alarm_rate": 0.0,
                "adaptation_lag_days": None,
                "premature_adaptation_index": 0.0,
            }

        # Drift onset is measured from the series origin, not from eval start.
        drift_onset = pd.Timestamp(SERIES_START) + pd.Timedelta(days=self.drift_start_day)
        drift_end = drift_onset + pd.Timedelta(days=self.drift_duration_days)

        pre = scored[scored["timestamp"] < drift_onset]
        post_steady = scored[scored["timestamp"] >= drift_end + pd.Timedelta(days=3)]

        # Self-calibrating threshold from the pre-drift period only
        if len(pre) >= 3:
            pre_mean = float(pre["deviation_score"].mean())
            pre_std = float(pre["deviation_score"].std(ddof=0))
        else:
            pre_mean, pre_std = 0.0, 1.0
        threshold = pre_mean + 3.0 * pre_std

        pre_alert_rate = float((pre["deviation_score"] > threshold).mean()) if len(pre) else 0.0
        persistent_rate = (
            float((post_steady["deviation_score"] > threshold).mean()) if len(post_steady) else 0.0
        )

        adaptation_lag = self._adaptation_lag(scored, drift_onset, threshold)

        return {
            "pre_drift_alert_rate": round(pre_alert_rate, 4),
            "alert_threshold": round(threshold, 4),
            "persistent_false_alarm_rate": round(persistent_rate, 4),
            "adaptation_lag_days": adaptation_lag,
            "baseline_shift_std_total": round(float(scored["baseline_shift_std"].iloc[-1]), 4),
        }

    @staticmethod
    def _adaptation_lag(scored: pd.DataFrame, drift_onset: pd.Timestamp, threshold: float) -> float | None:
        """Days from drift onset to the final alert.

        A baseline that never adapts keeps alerting until the end of the
        horizon, so this returns the full remaining duration; a baseline that
        adapts returns a small value. This is a permanent-cessation measure,
        unlike a first-quiet-stretch measure which is easily satisfied by noise.
        """
        post = scored[scored["timestamp"] >= drift_onset].sort_values(by="timestamp").reset_index(drop=True)
        if post.empty:
            return None

        alerted = post["deviation_score"].to_numpy() > threshold
        if not alerted.any():
            return 0.0

        last_alert_index = int(np.max(np.where(alerted)[0]))
        return round(
            float((post["timestamp"].iloc[last_alert_index] - drift_onset).total_seconds() / 86400.0), 2
        )

    def _score_meyro(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trains MEYRO-V2 on pre-drift history and streams it across the drift.

        This is the only setting in which adaptation can actually be observed:
        each window is scored against memories built exclusively from earlier
        windows, and the memories then update.
        """
        from meyro.data.windows import FeatureStandardizer, build_windows
        from meyro.experiments.streaming import stream_scores_v2
        from meyro.models.meyro import MEYROModelV2
        from meyro.training.trainer import MEYROV2Trainer

        calib_df, eval_df = PreprocessingPipeline.create_subject_temporal_splits(
            df, calibration_days=self.calibration_days
        )
        window = 7
        calib_windows = build_windows(calib_df, self.features, window_size=window)
        eval_windows = build_windows(eval_df, self.features, window_size=window)

        standardizer = FeatureStandardizer().fit(calib_windows)
        calib_std = standardizer.transform(calib_windows)
        eval_std = standardizer.transform(eval_windows)

        set_global_seed(self.seed)
        model = MEYROModelV2(
            input_dim=len(self.features),
            context_dim=calib_windows.context_dim,
            quality_dim=len(self.features),
            hidden_dim=16,
            num_layers=2,
        )
        trainer = MEYROV2Trainer(model, epochs=25, seed=self.seed)
        trainer.standardizer = standardizer
        fast_memories, slow_memories = trainer.fit_windows(calib_std)

        scores, _persistence = stream_scores_v2(model, eval_std, fast_memories, slow_memories)

        rows = []
        offset = window - 1
        for subject_id, group in eval_df.groupby("subject_id", sort=False):
            ordered = group.sort_values("timestamp").reset_index(drop=True)
            subject_scores = scores[eval_std.subject_ids == subject_id]
            for timestamp, value in zip(ordered["timestamp"].to_numpy()[offset:], subject_scores, strict=True):
                rows.append(
                    {
                        "subject_id": subject_id,
                        "timestamp": timestamp,
                        "deviation_score": float(value),
                        # MEYRO's baseline is a latent vector, not a scalar centre.
                        "baseline_shift_std": 0.0,
                    }
                )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------ outlier probe
    def acute_outlier_probe(self) -> dict[str, float]:
        """Injects a single extreme observation and measures baseline displacement.

        A well-behaved adaptive baseline should barely move (rule: a single
        unusual measurement must not redefine the person's normal).
        """
        gen = SyntheticBenchmarkGenerator(seed=self.seed)
        df = gen.generate_cohort(n_subjects=6, n_days=45, anomaly_rate_per_subject=0.0)
        cleaned = PreprocessingPipeline(feature_cols=self.features).clean_and_impute(df)
        calib, eval_df = PreprocessingPipeline.create_subject_temporal_splits(cleaned, calibration_days=21)

        engine = AdaptivePersonalBaseline(feature_cols=self.features).initialize(calib)
        before = {s: {c: v["center"] for c, v in cols.items()} for s, cols in engine.state.items()}

        ordered = eval_df.sort_values(by=["subject_id", "timestamp"])
        for i, (_, row) in enumerate(ordered.iterrows()):
            features = {c: float(row[c]) for c in self.features}
            if i == 0:  # extreme single-observation spike
                features = {c: v * 3.0 for c, v in features.items()}
            engine.step(row["subject_id"], features, float(row.get("quality_score", 1.0)))

        displacements = []
        for subj, cols in engine.state.items():
            for col, state in cols.items():
                scale = state["scale"] if state["scale"] > 1e-6 else 1.0
                displacements.append(abs(state["center"] - before[subj][col]) / scale)

        return {
            "mean_baseline_displacement_std": round(float(np.mean(displacements)), 4),
            "max_baseline_displacement_std": round(float(np.max(displacements)), 4),
        }
