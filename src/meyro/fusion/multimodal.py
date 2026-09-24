"""Multimodal Fusion and Missing-Modality Robustness Engine for MEYRO (Phases 15-16).

Architecture:
- Modality-specific encoders (Activity, Physiology, Sleep)
- Missing-modality dropout masking during training
- Modality-aware cross-attention and gated fusion
- Graceful degradation when one or more signals drop out (e.g. sleep missing on day shifts)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ModalityEncoder(nn.Module):
    """Causal encoder for an individual physiological/behavioral modality."""

    def __init__(self, input_dim: int, hidden_dim: int = 16) -> None:
        super().__init__()
        self.encoder = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, input_dim] -> [batch, hidden_dim]
        _, h_n = self.encoder(x)
        return self.norm(h_n[-1])


class MultimodalGatedFusion(nn.Module):
    """Fuses multiple asynchronous modalities with dynamic quality & availability gating.

    Operates seamlessly under:
    - All modalities present: Activity + Physiology + Sleep
    - Partial availability: e.g. Activity present, Sleep missing (masked with quality = 0)
    """

    def __init__(
        self,
        activity_dim: int = 2,
        physiology_dim: int = 2,
        sleep_dim: int = 2,
        hidden_dim: int = 16,
        modality_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.modality_dropout = modality_dropout

        # Individual modality encoders
        self.enc_activity = ModalityEncoder(activity_dim, hidden_dim)
        self.enc_physiology = ModalityEncoder(physiology_dim, hidden_dim)
        self.enc_sleep = ModalityEncoder(sleep_dim, hidden_dim)

        # Dynamic gating projection: computes importance weight per modality
        self.gate_projector = nn.Sequential(
            nn.Linear(hidden_dim * 3 + 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
            nn.Softmax(dim=-1),
        )

        # Final unified multimodal representation
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(
        self,
        activity: torch.Tensor,
        physiology: torch.Tensor,
        sleep: torch.Tensor,
        availability_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            activity: [batch, seq_len, 2]
            physiology: [batch, seq_len, 2]
            sleep: [batch, seq_len, 2]
            availability_mask: [batch, 3] binary indicators for (activity, physiology, sleep)

        Returns:
            fused_representation: [batch, hidden_dim]
            modality_weights: [batch, 3]
        """
        batch_size = activity.size(0)
        device = activity.device

        if availability_mask is None:
            availability_mask = torch.ones(batch_size, 3, device=device)

        # Apply training-time modality dropout to enforce missing-modality robustness
        if self.training and self.modality_dropout > 0:
            rand_mask = (torch.rand(batch_size, 3, device=device) > self.modality_dropout).float()
            # Ensure at least one modality remains active per sample
            all_zeros = (rand_mask.sum(dim=-1, keepdim=True) == 0)
            rand_mask = torch.where(all_zeros, torch.tensor([1.0, 0.0, 0.0], device=device), rand_mask)
            availability_mask = availability_mask * rand_mask

        # Encode each stream
        h_act = self.enc_activity(activity) * availability_mask[:, 0:1]
        h_phys = self.enc_physiology(physiology) * availability_mask[:, 1:2]
        h_slp = self.enc_sleep(sleep) * availability_mask[:, 2:3]

        # Compute dynamic gating conditioned on latent representations and explicit availability
        concat_all = torch.cat([h_act, h_phys, h_slp, availability_mask], dim=-1)
        raw_weights = self.gate_projector(concat_all)

        # Zero out weights of completely unavailable modalities and re-normalize
        effective_weights = raw_weights * availability_mask
        sum_w = effective_weights.sum(dim=-1, keepdim=True) + 1e-6
        norm_weights = effective_weights / sum_w

        # Weighted combination
        stacked = torch.stack([h_act, h_phys, h_slp], dim=1)  # [batch, 3, hidden_dim]
        fused = torch.bmm(norm_weights.unsqueeze(1), stacked).squeeze(1)  # [batch, hidden_dim]

        out = self.fusion_layer(fused)
        return out, norm_weights
