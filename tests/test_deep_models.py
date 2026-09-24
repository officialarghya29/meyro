"""Tests for MEYRO Architecture and Deep Baseline Models."""

import torch

from meyro.models.meyro import MEYROModel
from meyro.models.tcn import TCNAutoencoder
from meyro.models.transformer import TransformerAutoencoder


def test_tcn_autoencoder():
    model = TCNAutoencoder(input_dim=3, hidden_dim=16, num_layers=2)
    x = torch.randn(4, 14, 3)  # [batch=4, seq_len=14, channels=3]
    recon = model(x)
    assert recon.shape == (4, 14, 3)
    scores = model.compute_deviation_score(x)
    assert scores.shape == (4,)
    assert (scores >= 0).all()


def test_transformer_autoencoder():
    model = TransformerAutoencoder(input_dim=3, d_model=16, nhead=2, num_layers=1)
    x = torch.randn(4, 10, 3)
    recon = model(x)
    assert recon.shape == (4, 10, 3)
    scores = model.compute_deviation_score(x)
    assert scores.shape == (4,)


def test_meyro_model_forward():
    model = MEYROModel(input_dim=3, context_dim=2, quality_dim=3, hidden_dim=16)
    x_seq = torch.randn(4, 7, 3)
    context = torch.randn(4, 2)
    baseline_mem = torch.randn(4, 16)
    quality = torch.ones(4, 3)

    anomaly, uncertainty, d_t, updated_mem = model(x_seq, context, baseline_mem, quality)

    assert anomaly.shape == (4, 1)
    assert (anomaly >= 0.0).all() and (anomaly <= 1.0).all()
    assert uncertainty.shape == (4, 1)
    assert (uncertainty >= 0.0).all() and (uncertainty <= 1.0).all()
    assert d_t.shape == (4, 16)
    assert updated_mem.shape == (4, 16)
