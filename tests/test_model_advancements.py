"""Tests for the V2 architecture, window builder, metrics and checkpoint registry."""

from pathlib import Path

import numpy as np
import torch

from meyro.data.synthetic import SyntheticBenchmarkGenerator
from meyro.data.windows import build_windows, expand_subject_memories, subject_mean_baseline
from meyro.evaluation.metrics import detection_metrics
from meyro.models.meyro import MEYROModelV2
from meyro.models.registry import ModelCardMetadata, load_checkpoint, save_checkpoint
from meyro.preprocessing.pipeline import PreprocessingPipeline

FEATURES = ["step_count", "resting_hr", "sleep_duration_min"]


def _cohort(n_subjects: int = 4, n_days: int = 40):
    gen = SyntheticBenchmarkGenerator(seed=7)
    df = gen.generate_cohort(n_subjects=n_subjects, n_days=n_days, anomaly_rate_per_subject=0.6)
    return PreprocessingPipeline(feature_cols=FEATURES).clean_and_impute(df)


def test_window_builder_is_causal_and_subject_isolated():
    df = _cohort()
    windows = build_windows(df, FEATURES, window_size=7)

    assert windows.x.ndim == 3
    assert windows.x.shape[1] == 7
    assert windows.x.shape[2] == len(FEATURES)
    assert len(windows) == len(windows.y) == len(windows.context) == len(windows.subject_ids)

    # Subject ordering must not be interleaved: every window belongs to one subject.
    assert set(np.unique(windows.subject_ids)) == set(df["subject_id"].unique())


def test_window_builder_rejects_tiny_windows():
    df = _cohort()
    try:
        build_windows(df, FEATURES, window_size=1)
    except ValueError as exc:
        assert "window_size" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected ValueError for window_size < 2")


def test_meyro_v2_forward_shapes_and_memory_gating():
    model = MEYROModelV2(input_dim=3, context_dim=2, quality_dim=3, hidden_dim=8, num_layers=1)
    x = torch.randn(5, 7, 3)
    ctx = torch.randn(5, 2)
    slow = torch.randn(5, 8)
    fast = torch.randn(5, 8)
    qual = torch.ones(5, 3)

    anomaly, uncertainty, persistence, d_t, fast_next, slow_next = model(x, ctx, slow, fast, qual)

    assert anomaly.shape == (5, 1)
    assert uncertainty.shape == (5, 1)
    assert persistence.shape == (5, 1)
    assert d_t.shape == (5, 8)
    assert fast_next.shape == (5, 8)
    assert slow_next.shape == (5, 8)
    assert torch.all((anomaly >= 0) & (anomaly <= 1))
    assert torch.all((persistence >= 0) & (persistence <= 1))


def test_v2_default_slow_gate_is_the_documented_score_gate():
    """The measured-ineffective persistence gate must not be the default.

    V2.1's persistence-confirmed gate was implemented to repair the drift failure
    and measured to be indistinguishable from the score gate. Defaulting to an
    unvalidated mechanism would silently change what "MEYRO-V2" means, so the
    default stays `score` and V2.1 is opt-in.
    """
    model = MEYROModelV2(input_dim=3, context_dim=2, quality_dim=3, hidden_dim=8, num_layers=1)
    assert model.slow_gate_mode == "score"
    assert model.anomaly_confirm_threshold is None


def test_v2_rejects_unknown_slow_gate_mode():
    try:
        MEYROModelV2(input_dim=3, hidden_dim=8, slow_gate_mode="telepathy")
    except ValueError as exc:
        assert "slow_gate_mode" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected ValueError for an unknown slow_gate_mode")


def test_forward_adaptive_requires_a_calibrated_threshold():
    """Streaming without calibration would silently compare against None."""
    model = MEYROModelV2(input_dim=3, context_dim=2, quality_dim=3, hidden_dim=8, num_layers=1)
    x = torch.randn(2, 7, 3)
    zeros = torch.zeros(2, 8)
    try:
        model.forward_adaptive(x, torch.zeros(2, 2), zeros, zeros, torch.ones(2, 3), torch.zeros(2, 1))
    except ValueError as exc:
        assert "confirm_threshold" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected ValueError when no threshold is calibrated")


def test_forward_adaptive_holds_the_baseline_until_patience_is_reached():
    """The established baseline must not migrate before `patience` windows."""
    torch.manual_seed(3)
    model = MEYROModelV2(input_dim=3, context_dim=2, quality_dim=3, hidden_dim=8, num_layers=1)
    model.eval()

    x = torch.randn(1, 7, 3) * 6.0  # a large excursion
    ctx = torch.zeros(1, 2)
    slow = torch.zeros(1, 8)
    fast = torch.zeros(1, 8)
    qual = torch.ones(1, 3)
    streak = torch.zeros(1, 1)

    with torch.no_grad():
        for _ in range(2):  # fewer than patience
            _a, _u, _r, _f, slow_out, streak = model.forward_adaptive(
                x, ctx, slow, fast, qual, streak, patience=4, confirm_threshold=-1.0
            )
            assert torch.allclose(slow_out, slow), "baseline migrated before confirmation"
    assert float(streak.squeeze()) == 2.0


def test_v2_rejects_invalid_timescale_ordering():
    try:
        MEYROModelV2(input_dim=3, hidden_dim=8, fast_rate=0.01, slow_rate=0.5)
    except ValueError as exc:
        assert "timescale" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected ValueError when slow_rate >= fast_rate")


def test_v2_memory_moves_less_for_anomalous_observations():
    """A high anomaly score must suppress memory adaptation."""
    torch.manual_seed(0)
    model = MEYROModelV2(
        input_dim=3, context_dim=2, quality_dim=3, hidden_dim=8, num_layers=1, anomaly_gate_scale=20.0
    )
    model.eval()

    x = torch.randn(2, 7, 3)
    ctx = torch.zeros(2, 2)
    slow = torch.zeros(2, 8)
    fast = torch.zeros(2, 8)
    qual = torch.ones(2, 3)

    with torch.no_grad():
        anomaly, _u, _r, _d, _fast_next, slow_next = model(x, ctx, slow, fast, qual)
        # The gate exponent is negative, so a high anomaly score suppresses movement.
        movement = torch.norm(slow_next - slow, dim=-1)

    assert torch.all(movement >= 0)
    assert torch.all(anomaly >= 0)


def test_adapt_sequence_returns_one_score_per_window():
    torch.manual_seed(1)
    model = MEYROModelV2(input_dim=3, context_dim=2, quality_dim=3, hidden_dim=8, num_layers=1)
    model.eval()

    x = torch.randn(9, 7, 3)
    ctx = torch.zeros(9, 2)
    qual = torch.ones(9, 3)
    init = torch.zeros(1, 8)

    anomalies, persistences, slow_final = model.adapt_sequence(x, ctx, qual, init, init)
    assert anomalies.shape == (9, 1)
    assert persistences.shape == (9, 1)
    assert slow_final.shape == (1, 8)


def test_detection_metrics_bounds():
    rng = np.random.default_rng(0)
    y_true = np.array([0] * 50 + [1] * 20)
    y_scores = np.concatenate([rng.normal(0.2, 0.1, 50), rng.normal(0.7, 0.1, 20)])

    metrics = detection_metrics(y_true, y_scores)
    assert 0.0 <= metrics["auroc"] <= 1.0
    assert metrics["auroc"] > 0.9
    assert 0.0 <= metrics["fpr_at_target_sensitivity"] <= 1.0
    assert metrics["n_samples"] == 70


def test_detection_metrics_handles_single_class():
    metrics = detection_metrics(np.zeros(10), np.random.default_rng(0).random(10))
    assert metrics["auroc"] == 0.5
    assert metrics["auprc"] == 0.0


def test_subject_memory_expansion_and_fallback():
    df = _cohort()
    windows = build_windows(df, FEATURES, window_size=7)

    model = MEYROModelV2(input_dim=3, context_dim=2, quality_dim=3, hidden_dim=8, num_layers=1)
    model.eval()
    memories = subject_mean_baseline(model.encode_observation, windows, model.hidden_dim)
    assert set(memories) == set(np.unique(windows.subject_ids))

    expanded = expand_subject_memories(memories, windows, model.hidden_dim, default="cohort_mean")
    assert expanded.shape == (len(windows), model.hidden_dim)


def test_checkpoint_roundtrip(tmp_path: Path):
    model = MEYROModelV2(input_dim=3, context_dim=2, quality_dim=3, hidden_dim=8, num_layers=1)
    metadata = ModelCardMetadata(
        model_name="MEYRO",
        version="TEST-V1",
        architecture="MEYROModelV2",
        seed=11,
        training_epochs=1,
        training_windows=2,
    )
    path = save_checkpoint(model, metadata, directory=tmp_path)
    assert path.exists()
    assert (tmp_path / "TEST-V1.json").exists()

    fresh = MEYROModelV2(input_dim=3, context_dim=2, quality_dim=3, hidden_dim=8, num_layers=1)
    loaded = load_checkpoint("TEST-V1", fresh, directory=tmp_path)
    assert loaded.version == "TEST-V1"
    assert loaded.n_parameters > 0

    for original, restored in zip(model.parameters(), fresh.parameters(), strict=True):
        assert torch.allclose(original, restored)
