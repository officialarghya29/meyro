"""MEYRO Architecture: Unified Model for Personalized Longitudinal Health Baseline Modeling."""

from __future__ import annotations

import torch
import torch.nn as nn


class MEYROModel(nn.Module):
    """The MEYRO Neural Architecture.

    Incorporates:
    - Context-conditioned causal temporal encoder
    - Latent personal baseline memory
    - Relational deviation module
    - Anomaly detection head
    - Quality-conditioned uncertainty estimation head
    - Gated adaptive baseline memory evolution
    """

    def __init__(
        self,
        input_dim: int = 3,
        context_dim: int = 2,
        quality_dim: int = 3,
        hidden_dim: int = 32,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        # 1. Context Encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 2. Temporal Encoder (Causal GRU)
        self.temporal_encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.temporal_norm = nn.LayerNorm(hidden_dim)

        # 3. Quality Encoder
        self.quality_encoder = nn.Sequential(
            nn.Linear(quality_dim, hidden_dim),
            nn.GELU(),
        )

        # 4. Relational Deviation Module: takes [E_t, B_t, E_t - B_t, E_t * B_t, Q_t]
        self.deviation_net = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # 5. Anomaly Scoring Head
        self.anomaly_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # 6. Uncertainty Estimation Head
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # 7. Decoder / Reconstruction auxiliary head for self-supervision
        self.recon_head = nn.Linear(hidden_dim, input_dim)

    def encode_observation(
        self, x_seq: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        """x_seq: [batch, seq_len, input_dim], context: [batch, context_dim] -> E_t: [batch, hidden_dim]"""
        _, h_n = self.temporal_encoder(x_seq)
        temporal_emb = h_n[-1]
        context_emb = self.context_encoder(context)
        return self.temporal_norm(temporal_emb + context_emb)

    def forward(
        self,
        x_seq: torch.Tensor,
        context: torch.Tensor,
        baseline_memory: torch.Tensor,
        quality: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns:
            anomaly_score: [batch, 1]
            uncertainty: [batch, 1]
            deviation_emb: [batch, hidden_dim]
            updated_baseline: [batch, hidden_dim]
        """
        e_t = self.encode_observation(x_seq, context)
        q_emb = self.quality_encoder(quality)

        # Deviation representation
        diff = e_t - baseline_memory
        prod = e_t * baseline_memory
        concat_feat = torch.cat([e_t, baseline_memory, diff, prod, q_emb], dim=-1)
        d_t = self.deviation_net(concat_feat)

        # Predictions
        anomaly_score = self.anomaly_head(d_t)
        u_feat = torch.cat([d_t, q_emb], dim=-1)
        uncertainty = self.uncertainty_head(u_feat)

        # Gated adaptive baseline update: alpha = 0.05 * exp(-3 * anomaly_score) * mean_quality
        q_mean = torch.mean(quality, dim=-1, keepdim=True)
        alpha = 0.05 * torch.exp(-3.0 * anomaly_score) * q_mean
        updated_baseline = (1.0 - alpha) * baseline_memory + alpha * e_t

        return anomaly_score, uncertainty, d_t, updated_baseline
