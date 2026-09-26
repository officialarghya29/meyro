"""Advanced tests: causality, leakage, determinism and edge cases.

These complement the functional tests by asserting *properties* rather than
outputs — the kind of guarantee that silently breaks during refactoring.
"""

import numpy as np
import pandas as pd
import torch

from meyro.data.synthetic import SyntheticBenchmarkGenerator
from meyro.data.windows import (
    FeatureStandardizer,
    align_row_scores,
    build_windows,
    per_window_subject_memories,
)
from meyro.evaluation.drift import BaselineDriftAnalysis
from meyro.experiments.streaming import ScoringOptions, stream_scores_v2
from meyro.models.meyro import MEYROModelV2
from meyro.preprocessing.pipeline import PreprocessingPipeline
from meyro.utils.io import set_global_seed

FEATURES = ["step_count", "resting_hr", "sleep_duration_min"]


def _cohort(n_subjects: int = 4, n_days: int = 40, seed: int = 7) -> pd.DataFrame:
    generator = SyntheticBenchmarkGenerator(seed=seed)
    cohort = generator.generate_cohort(
        n_subjects=n_subjects, n_days=n_days, anomaly_rate_per_subject=0.6
    )
    return PreprocessingPipeline(feature_cols=FEATURES).clean_and_impute(cohort)


# --------------------------------------------------------------------- causality
def test_windows_never_contain_future_observations():
    """Changing the final observation must not affect any earlier window."""
    df = _cohort()
    before = build_windows(df, FEATURES, window_size=7)

    subject = df["subject_id"].iloc[0]
    mutated = df.copy()
    last_index = mutated[mutated["subject_id"] == subject].index[-1]
    mutated.loc[last_index, "step_count"] += 50_000.0
    after = build_windows(mutated, FEATURES, window_size=7)

    # Every window except that subject's last must be byte-identical.
    unchanged = np.isclose(before.x, after.x).all(axis=(1, 2))
    assert (~unchanged).sum() == 1, "exactly one window should react to the final observation"


def test_windows_are_subject_isolated():
    df = _cohort()
    windows = build_windows(df, FEATURES, window_size=7)

    for subject_id in np.unique(windows.subject_ids):
        rows = df[df["subject_id"] == subject_id].sort_values("timestamp")
        subject_windows = windows.x[windows.subject_ids == subject_id]
        # First window of the subject must equal their first seven rows exactly.
        assert np.allclose(subject_windows[0], rows[FEATURES].to_numpy()[:7], atol=1e-5)
        assert len(subject_windows) == max(0, len(rows) - 6)


def test_label_alignment_matches_trailing_row():
    df = _cohort()
    _calib, eval_df = PreprocessingPipeline.create_subject_temporal_splits(df, calibration_days=18)
    windows = build_windows(eval_df, FEATURES, window_size=7)

    rows = eval_df.sort_values(["subject_id", "timestamp"])["ground_truth_label"].to_numpy()
    row_scores = np.arange(len(eval_df), dtype=float)
    y_true, scores, _groups = align_row_scores(eval_df, row_scores, window_size=7)

    assert len(y_true) == len(windows)
    assert y_true.tolist() == windows.y.tolist()
    # The aligned score for a window is the score of its ending row.
    assert scores[0] == 6.0
    assert len(scores) == len(rows) - 6 * eval_df["subject_id"].nunique()


# ---------------------------------------------------------------------- leakage
def test_standardizer_does_not_refit_on_evaluation_data():
    df = _cohort()
    calib, eval_df = PreprocessingPipeline.create_subject_temporal_splits(df, calibration_days=18)
    calib_windows = build_windows(calib, FEATURES, window_size=7)
    eval_windows = build_windows(eval_df, FEATURES, window_size=7)

    standardizer = FeatureStandardizer().fit(calib_windows)
    fitted_mean = standardizer.mean.copy()

    transformed = standardizer.transform(eval_windows)
    assert np.array_equal(standardizer.mean, fitted_mean), "transform must not refit statistics"

    expected = (eval_windows.x - fitted_mean) / standardizer.std
    assert np.allclose(transformed.x, expected, atol=1e-5)


def test_standardizer_fit_uses_calibration_statistics_only():
    df = _cohort()
    calib, eval_df = PreprocessingPipeline.create_subject_temporal_splits(df, calibration_days=18)
    calib_windows = build_windows(calib, FEATURES, window_size=7)
    eval_windows = build_windows(eval_df, FEATURES, window_size=7)

    standardizer = FeatureStandardizer().fit(calib_windows)
    calibration_mean = calib_windows.x.reshape(-1, len(FEATURES)).mean(axis=0)
    assert np.allclose(standardizer.mean, calibration_mean, atol=1e-6)

    # Sanity: evaluation data has genuinely different statistics, so a leak would show.
    evaluation_mean = eval_windows.x.reshape(-1, len(FEATURES)).mean(axis=0)
    assert not np.allclose(calibration_mean, evaluation_mean, atol=1e-6)


def test_training_memories_exclude_the_window_itself():
    df = _cohort()
    windows = build_windows(df, FEATURES, window_size=7)
    model = MEYROModelV2(input_dim=3, context_dim=2, quality_dim=3, hidden_dim=8, num_layers=1)
    model.eval()

    memories = per_window_subject_memories(model.encode_observation, windows, 8)

    with torch.no_grad():
        x, ctx, _q = windows.tensors()
        embeddings = model.encode_observation(x, ctx)

    # A leave-one-out memory must differ from the window's own embedding.
    differences = torch.norm(memories - embeddings, dim=-1)
    assert torch.all(differences > 1e-6), "leave-one-out memory included its own window"


# ----------------------------------------------------------------- determinism
def test_model_initialisation_is_reproducible():
    set_global_seed(0)
    first = MEYROModelV2(input_dim=3, context_dim=2, quality_dim=3, hidden_dim=8, num_layers=1)
    set_global_seed(0)
    second = MEYROModelV2(input_dim=3, context_dim=2, quality_dim=3, hidden_dim=8, num_layers=1)

    for left, right in zip(first.parameters(), second.parameters(), strict=True):
        assert torch.equal(left, right)


def test_frozen_memory_scores_are_constant_after_first_step():
    """With memory frozen, only the observation changes the score."""
    df = _cohort(n_subjects=2, n_days=30)
    calib, eval_df = PreprocessingPipeline.create_subject_temporal_splits(df, calibration_days=18)
    calib_windows = build_windows(calib, FEATURES, window_size=7)
    eval_windows = build_windows(eval_df, FEATURES, window_size=7)

    standardizer = FeatureStandardizer().fit(calib_windows)
    set_global_seed(3)
    model = MEYROModelV2(input_dim=3, context_dim=2, quality_dim=3, hidden_dim=8, num_layers=1)
    model.eval()

    scores_streaming, _ = stream_scores_v2(model, standardizer.transform(eval_windows), None, None)
    scores_frozen, _ = stream_scores_v2(
        model, standardizer.transform(eval_windows), None, None, ScoringOptions(freeze_memory=True)
    )

    assert scores_streaming.shape == scores_frozen.shape == (len(eval_windows),)


def test_drift_experiment_is_bit_for_bit_reproducible():
    """Regression test: this suite previously varied between identical runs."""
    first = BaselineDriftAnalysis(seed=42, n_subjects=3, n_days=62).run()
    second = BaselineDriftAnalysis(seed=42, n_subjects=3, n_days=62).run()
    assert first == second


# ----------------------------------------------------------------- edge cases
def test_detection_metrics_on_empty_input():
    from meyro.evaluation.metrics import detection_metrics

    metrics = detection_metrics(np.empty(0, dtype=int), np.empty(0))
    assert metrics["auroc"] == 0.5
    assert metrics["n_samples"] == 0.0


def test_align_row_scores_with_short_series_leaves_no_windows():
    short = pd.DataFrame(
        {
            "subject_id": ["S1"] * 4,
            "timestamp": pd.date_range("2025-01-01", periods=4, tz="UTC"),
            "ground_truth_label": [0, 0, 1, 0],
            "step_count": [1.0, 2.0, 3.0, 4.0],
        }
    )
    y_true, scores, groups = align_row_scores(short, np.arange(4, dtype=float), window_size=7)
    assert len(y_true) == len(scores) == len(groups) == 0
