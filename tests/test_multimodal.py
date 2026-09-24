"""Tests for Multimodal Fusion and Missing-Modality Robustness (Phases 15-16)."""

import torch

from meyro.fusion.multimodal import ModalityEncoder, MultimodalGatedFusion


def test_modality_encoder():
    enc = ModalityEncoder(input_dim=2, hidden_dim=16)
    x = torch.randn(4, 7, 2)  # [batch=4, seq_len=7, dims=2]
    out = enc(x)
    assert out.shape == (4, 16)


def test_multimodal_gated_fusion_all_present():
    fusion = MultimodalGatedFusion(activity_dim=2, physiology_dim=2, sleep_dim=2, hidden_dim=16)
    fusion.eval()

    act = torch.randn(4, 7, 2)
    phys = torch.randn(4, 7, 2)
    slp = torch.randn(4, 7, 2)

    fused, weights = fusion(act, phys, slp)
    assert fused.shape == (4, 16)
    assert weights.shape == (4, 3)
    # Weights sum to 1.0 across the 3 modalities
    assert torch.allclose(weights.sum(dim=-1), torch.ones(4), atol=1e-5)


def test_multimodal_missing_modality_graceful_handling():
    fusion = MultimodalGatedFusion(activity_dim=2, physiology_dim=2, sleep_dim=2, hidden_dim=16)
    fusion.eval()

    act = torch.randn(2, 7, 2)
    phys = torch.randn(2, 7, 2)
    slp = torch.randn(2, 7, 2)

    # Simulate sleep modality completely missing (e.g. user didn't wear tracker to bed)
    # Mask: [Activity=1, Physiology=1, Sleep=0]
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]])

    fused, weights = fusion(act, phys, slp, availability_mask=mask)
    assert fused.shape == (2, 16)
    # For sample 0, sleep weight must be strictly zero
    assert weights[0, 2].item() == 0.0
    # For sample 1, physiology weight must be strictly zero
    assert weights[1, 1].item() == 0.0
    # Active weights must re-normalize to 1.0
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2), atol=1e-5)
