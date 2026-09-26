"""Tests for Ablation, Robustness, Cold Start, Drift, Efficiency, Explainability."""

import numpy as np
import torch

from meyro.anomaly.explanation import DeviationExplainer
from meyro.evaluation.ablation import AblationStudy
from meyro.evaluation.cold_start import ColdStartAnalysis
from meyro.evaluation.drift import BaselineDriftAnalysis
from meyro.evaluation.efficiency import measure_efficiency
from meyro.evaluation.robustness import RobustnessTestSuite
from meyro.models.meyro import MEYROModelV2
from meyro.uncertainty.calibration import TemperatureScaler


def test_ablation_study_execution():
    ablation = AblationStudy(n_subjects=4, n_days=45, calibration_days=18, seed=42, epochs=4)
    results = ablation.run()

    # The trained model and its ablated variants must all be present.
    assert "MEYRO-V2 Full (trained)" in results
    assert "Ablation: w/o Personal Memory" in results
    assert "Ablation: Single-Timescale Memory" in results
    assert "Ablation: w/o Context Encoder" in results
    assert "Ablation: w/o Quality Conditioning" in results
    assert "Ablation: Naive Euclidean Deviation" in results
    assert "Reference: MEYRO-V2 Untrained" in results

    for key, metrics in results.items():
        if key == "_metadata":
            continue
        assert 0.0 <= metrics["auroc"] <= 1.0
    assert results["_metadata"]["n_evaluation_windows"] > 0


def test_robustness_stress_testing():
    suite = RobustnessTestSuite(seed=42, n_subjects=6, n_days=45)
    missing_res = suite.run_missing_data_stress()
    assert "Missing 0%" in missing_res
    assert "Missing 30%" in missing_res
    # Personal baseline is highly robust to randomly dropped observations
    assert missing_res["Missing 30%"] >= 0.85

    noise_res = suite.run_noise_stress()
    assert noise_res["Noise 1.0x"] >= 0.9
    assert noise_res["Noise 4.0x"] >= 0.85


def test_robustness_extended_axes():
    suite = RobustnessTestSuite(seed=42, n_subjects=6, n_days=45)

    outliers = suite.run_outlier_stress()
    assert set(outliers) == {"Outliers 1%", "Outliers 3%", "Outliers 5%"}

    history = suite.run_history_length_stress()
    assert "History 7d" in history and "History 28d" in history

    sampling = suite.run_sampling_rate_stress()
    # Sparse sampling must still produce a usable evaluation set, never NaN.
    assert all(not np.isnan(v) for v in sampling.values())

    device = suite.run_device_variation_stress()
    assert "Device bias 0%" in device and "Device bias 15%" in device

    everything = suite.run_all()
    assert set(everything) == {
        "missing_data",
        "sensor_noise",
        "spike_outliers",
        "history_length",
        "sampling_rate",
        "device_variation",
    }


def test_cold_start_curve():
    cold = ColdStartAnalysis(seed=42)
    curve = cold.run_curve()
    assert "3_days" in curve
    assert "28_days" in curve
    # Longer personal history must not hurt the personalization advantage.
    assert curve["28_days"]["delta_advantage"] >= curve["3_days"]["delta_advantage"]


def test_baseline_drift_adaptation():
    analysis = BaselineDriftAnalysis(seed=42, n_subjects=8, n_days=70)
    results = analysis.run()

    static = results["Personal Baseline (Static)"]
    adaptive = results["Personal Baseline (Adaptive)"]

    # A static baseline cannot re-baseline a sustained lifestyle change.
    assert static["persistent_false_alarm_rate"] > adaptive["persistent_false_alarm_rate"]
    assert adaptive["adaptation_lag_days"] <= static["adaptation_lag_days"]

    probe = analysis.acute_outlier_probe()
    # One extreme observation must not redefine the personal baseline.
    assert probe["max_baseline_displacement_std"] < 1.0


def test_efficiency_measurement():
    model = MEYROModelV2(input_dim=3, context_dim=2, quality_dim=3, hidden_dim=8, num_layers=1)
    x = torch.randn(16, 7, 3)
    ctx = torch.randn(16, 2)
    zeros = torch.zeros(16, 8)
    qual = torch.ones(16, 3)

    metrics = measure_efficiency(
        model, lambda: model(x, ctx, zeros, zeros, qual), n_warmup=1, n_runs=3
    )
    assert metrics["parameters"] > 0
    assert metrics["mean_latency_ms"] >= 0.0
    assert metrics["model_size_kb"] > 0.0


def test_deviation_explainer():
    features = {"resting_hr": 84.0, "step_count": 2100.0}
    baseline = {
        "resting_hr": {"center": 68.0, "scale": 4.0},
        "step_count": {"center": 8500.0, "scale": 1200.0},
    }
    exp = DeviationExplainer.explain(
        subject_id="SUBJ_001",
        current_features=features,
        baseline_stats=baseline,
        anomaly_score=0.88,
        uncertainty=0.12,
        persistence_count=3,
        data_quality=0.95,
    )
    assert exp["is_meaningful_deviation"] is True
    assert exp["deviation_magnitude"] == "high"
    assert exp["confidence_level"] == "high"
    assert "safety_disclaimer" in exp
    assert "not a clinical or medical diagnosis" in exp["safety_disclaimer"]


def test_temperature_scaler():
    scaler = TemperatureScaler()
    logits = torch.randn(10, 1)
    scaled = scaler(logits)
    assert scaled.shape == (10, 1)
    probs = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    ece = scaler.compute_expected_calibration_error(probs, labels)
    assert 0.0 <= ece <= 1.0
