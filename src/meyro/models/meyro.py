"""MEYRO architecture family.

MEYRO-V1
    Context-conditioned causal temporal encoder + single latent personal
    baseline memory + relational deviation module + anomaly/uncertainty heads.

MEYRO-V2 (current)
    Adds two mechanisms identified as gaps in ``docs/model_gap_analysis.md``:

    1. **Dual-timescale personal memory.** A *fast* memory tracks recent personal
       state; a *slow* memory represents the established baseline. Single-memory
       models cannot separate a transient excursion from a genuine baseline
       shift, because one time constant must serve both roles.
    2. **Persistence module.** A causal recurrent head over the within-window
       trajectory of deviation embeddings distinguishes a one-off excursion from
       a sustained change, which no per-timestep scorer can express.

    Both memories adapt with an anomaly- and quality-gated rule, so the baseline
    cannot be redefined by a single outlier observation.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MEYROModel(nn.Module):
    """MEYRO-V1: single-memory personalized baseline model.

    Retained as the permanent V1 reference so V2's contribution can be measured
    rather than asserted.
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

    def encode_observation(self, x_seq: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """x_seq: [batch, seq_len, input_dim], context: [batch, context_dim] -> [batch, hidden_dim]."""
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


class MEYROModelV2(nn.Module):
    """MEYRO-V2: dual-timescale personal memory with explicit persistence modelling.

    Forward returns ``(anomaly, uncertainty, persistence, deviation_emb, fast_memory, slow_memory)``.
    The two memories have different time constants (``fast_rate`` >> ``slow_rate``)
    and both adapt only in proportion to ``exp(-k * anomaly) * quality``, so an
    outlying observation leaves the baseline essentially unchanged.
    """

    def __init__(
        self,
        input_dim: int = 3,
        context_dim: int = 2,
        quality_dim: int = 3,
        hidden_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.1,
        fast_rate: float = 0.35,
        slow_rate: float = 0.03,
        persistence_hidden: int = 16,
        anomaly_gate_scale: float = 4.0,
    ) -> None:
        super().__init__()
        if not 0.0 < slow_rate < fast_rate <= 1.0:
            raise ValueError(
                "Require 0 < slow_rate < fast_rate <= 1 for distinct timescales; "
                f"received slow_rate={slow_rate}, fast_rate={fast_rate}"
            )

        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.fast_rate = fast_rate
        self.slow_rate = slow_rate
        self.anomaly_gate_scale = anomaly_gate_scale

        # Context encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Causal temporal encoder producing a per-timestep latent trajectory
        self.temporal_encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.temporal_norm = nn.LayerNorm(hidden_dim)

        # Quality encoder
        self.quality_encoder = nn.Sequential(
            nn.Linear(quality_dim, hidden_dim),
            nn.GELU(),
        )

        # Relational deviation over both memories:
        # [E_t, B_slow, B_fast, E_t - B_slow, E_t * B_fast, Q_t]
        self.deviation_net = nn.Sequential(
            nn.Linear(hidden_dim * 6, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.anomaly_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Persistence: causal recurrence over the deviation trajectory
        self.persistence_encoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=persistence_hidden,
            batch_first=True,
        )
        self.persistence_head = nn.Sequential(
            nn.Linear(persistence_hidden, persistence_hidden // 2),
            nn.GELU(),
            nn.Linear(persistence_hidden // 2, 1),
            nn.Sigmoid(),
        )

        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        self.recon_head = nn.Linear(hidden_dim, input_dim)

    # ------------------------------------------------------------------ encoders
    def encode_sequence(self, x_seq: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Returns the causal latent trajectory: [batch, seq_len, hidden_dim]."""
        out, _ = self.temporal_encoder(x_seq)
        ctx = self.context_encoder(context).unsqueeze(1)
        return self.temporal_norm(out + ctx)

    def encode_observation(self, x_seq: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Returns the latent vector for the window-ending observation."""
        return self.encode_sequence(x_seq, context)[:, -1]

    def _deviation(
        self,
        e_t: torch.Tensor,
        baseline_slow: torch.Tensor,
        baseline_fast: torch.Tensor,
        q_emb: torch.Tensor,
    ) -> torch.Tensor:
        diff = e_t - baseline_slow
        prod = e_t * baseline_fast
        features = torch.cat([e_t, baseline_slow, baseline_fast, diff, prod, q_emb], dim=-1)
        return self.deviation_net(features)

    # ------------------------------------------------------------------- forward
    def forward(
        self,
        x_seq: torch.Tensor,
        context: torch.Tensor,
        baseline_slow: torch.Tensor,
        baseline_fast: torch.Tensor,
        quality: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass over a batch of causal windows.

        Returns:
            anomaly: [batch, 1] deviation score for the window-ending observation
            uncertainty: [batch, 1]
            persistence: [batch, 1] confidence that the deviation is sustained
            deviation_emb: [batch, hidden_dim]
            fast_memory: [batch, hidden_dim] updated fast memory
            slow_memory: [batch, hidden_dim] updated slow memory
        """
        _batch_size, seq_len, _ = x_seq.shape

        e_seq = self.encode_sequence(x_seq, context)  # [batch, seq_len, hidden_dim]
        q_emb = self.quality_encoder(quality)  # [batch, hidden_dim]

        q_seq = q_emb.unsqueeze(1).expand(-1, seq_len, -1)
        slow_seq = baseline_slow.unsqueeze(1).expand(-1, seq_len, -1)
        fast_seq = baseline_fast.unsqueeze(1).expand(-1, seq_len, -1)

        # Deviation trajectory inside the (causal) window
        d_seq = self._deviation(e_seq, slow_seq, fast_seq, q_seq)  # [batch, seq_len, hidden_dim]
        anomaly_seq = self.anomaly_head(d_seq)  # [batch, seq_len, 1]

        anomaly = anomaly_seq[:, -1]
        deviation_emb = d_seq[:, -1]

        # Persistence over the deviation trajectory (causal GRU -> last state)
        persistence_out, _ = self.persistence_encoder(d_seq)
        persistence = self.persistence_head(persistence_out[:, -1])

        u_feat = torch.cat([deviation_emb, q_emb], dim=-1)
        uncertainty = self.uncertainty_head(u_feat)

        # Anomaly- and quality-gated dual-timescale memory update
        e_t = e_seq[:, -1]
        q_mean = torch.mean(quality, dim=-1, keepdim=True)
        gate = torch.exp(-self.anomaly_gate_scale * anomaly) * q_mean

        fast_memory = (1.0 - self.fast_rate * gate) * baseline_fast + (self.fast_rate * gate) * e_t
        slow_memory = (1.0 - self.slow_rate * gate) * baseline_slow + (self.slow_rate * gate) * e_t

        return anomaly, uncertainty, persistence, deviation_emb, fast_memory, slow_memory

    # --------------------------------------------------------------- adaptation
    def adapt_sequence(
        self,
        x_seq: torch.Tensor,
        context: torch.Tensor,
        quality: torch.Tensor,
        init_fast: torch.Tensor,
        init_slow: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Streams a subject's window through the memory update rule one step at a time.

        Used by the drift experiment: every window is scored against the memory
        state built *only* from preceding windows, then the memory is updated.
        Returns ``(anomalies, persistences, final_slow_memory)``.
        """
        anomalies: list[torch.Tensor] = []
        persistences: list[torch.Tensor] = []
        fast, slow = init_fast, init_slow

        for step in range(x_seq.shape[0]):
            window = x_seq[step : step + 1]
            ctx = context[step : step + 1]
            qual = quality[step : step + 1]
            anomaly, _unc, persistence, _d, fast, slow = self(window, ctx, slow, fast, qual)
            anomalies.append(anomaly)
            persistences.append(persistence)

        if not anomalies:
            empty = torch.empty(0)
            return empty, empty, init_slow
        return torch.cat(anomalies, dim=0), torch.cat(persistences, dim=0), slow
