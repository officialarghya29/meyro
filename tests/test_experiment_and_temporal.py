"""Tests for Primary Research Experiment, Temporal Models, and Adaptive Personal Baselines."""

import torch

from meyro.experiments.primary import PrimaryExperiment
from meyro.models.temporal import TemporalAutoencoder


def test_temporal_autoencoder_gru():
    model = TemporalAutoencoder(input_dim=3, hidden_dim=16, cell_type="gru")
    x = torch.randn(4, 7, 3)  # [batch=4, seq_len=7, features=3]
    recon, context = model(x)
    assert recon.shape == (4, 7, 3)
    assert context.shape == (4, 16)
    scores = model.compute_deviation_score(x)
    assert scores.shape == (4,)
    assert (scores >= 0).all()


def test_primary_experiment_run():
    exp = PrimaryExperiment(n_subjects=6, n_days=45, calibration_days=15, seed=42)
    results = exp.run()

    assert "population" in results
    assert "personalized_static" in results
    assert "personalized_adaptive" in results

    pop_res = results["population"]
    pers_res = results["personalized_static"]

    # In synthetic data with diverse personal baselines, personalized AUROC should be strong
    assert pers_res["auroc"] >= 0.70
    # FPR of personalized baseline at high recall should be competitive or lower than population
    assert pers_res["fpr_at_85_recall"] <= pop_res["fpr_at_85_recall"] + 0.15
