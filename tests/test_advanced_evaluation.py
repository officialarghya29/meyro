"""Tests for Ablation, Robustness, Cold Start, and Explainability."""

import numpy as np
import torch

from meyro.anomaly.explanation import DeviationExplainer
from meyro.evaluation.ablation import AblationStudy
from meyro.evaluation.cold_start import ColdStartAnalysis
from meyro.evaluation.robustness import RobustnessTestSuite
from meyro.uncertainty.calibration import TemperatureScaler


def test_ablation_study_execution():
    ablation = AblationStudy(n_subjects=4, n_days=35, seed=42)
    results = ablation.run()
    assert "MEYRO Full Model" in results
    assert "Ablation: w/o Personal Memory (B_t=0)" in results
    assert "Ablation: w/o Context (C_t=0)" in results
    assert results["MEYRO Full Model"]["auroc"] >= 0.0


def test_robustness_stress_testing():
    suite = RobustnessTestSuite(seed=42)
    missing_res = suite.run_missing_data_stress()
    assert "Missing 0%" in missing_res
    assert "Missing 30%" in missing_res
    # Performance with 0% missing should be high
    assert missing_res["Missing 0%"] >= 0.85

    noise_res = suite.run_noise_stress()
    assert "Noise 1.0x" in noise_res
    assert "Noise 4.0x" in noise_res


def test_cold_start_curve():
    cold = ColdStartAnalysis(seed=42)
    curve = cold.run_curve()
    assert "3_days" in curve
    assert "28_days" in curve
    # 28-day history should provide a stronger personalization advantage than 3-day history
    assert curve["28_days"]["delta_advantage"] > 0


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
