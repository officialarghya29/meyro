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

import numpy as np
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
        slow_gate_mode: str = "score",
        persistence_threshold: float = 0.5,
        persistence_band: float = 0.05,
    ) -> None:
        super().__init__()
        if not 0.0 < slow_rate < fast_rate <= 1.0:
            raise ValueError(
                "Require 0 < slow_rate < fast_rate <= 1 for distinct timescales; "
                f"received slow_rate={slow_rate}, fast_rate={fast_rate}"
            )
        if slow_gate_mode not in {"score", "persistence"}:
            raise ValueError(
                f"slow_gate_mode must be 'score' or 'persistence', received {slow_gate_mode!r}"
            )
        if not 0.0 <= persistence_threshold < 1.0:
            raise ValueError(
                f"persistence_threshold must lie in [0, 1), received {persistence_threshold}"
            )
        if persistence_band <= 0.0:
            raise ValueError(f"persistence_band must be > 0, received {persistence_band}")

        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.fast_rate = fast_rate
        self.slow_rate = slow_rate
        self.anomaly_gate_scale = anomaly_gate_scale
        self.slow_gate_mode = slow_gate_mode
        self.persistence_threshold = persistence_threshold
        self.persistence_band = persistence_band
        # Set by calibrate_anomaly_threshold; used by forward_adaptive.
        self.anomaly_confirm_threshold: float | None = None

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

        # ------------------------------------------------------ memory update
        # The FAST memory tracks recent state and is always anomaly-gated, so an
        # acute excursion cannot drag it around.
        #
        # The SLOW memory represents the established baseline, and its gate is the
        # design decision that separates V2 from V2.1:
        #
        #   "score"       gate = exp(-k*A) * q        (V2)
        #       Adaptation is blocked exactly when a deviation is flagged, so a
        #       *sustained* change is alerted on forever and the baseline never
        #       re-establishes. This is a documented failure (see docs/benchmark_report.md).
        #
        #   "persistence" gate = confirm(R_t) * q     (V2.1, opt-in)
        #       Adaptation is blocked for a transient excursion but permitted once
        #       the deviation is *confirmed persistent* by the (causal) persistence
        #       head. This is the threshold-confirmed migration rule.
        #
        #       It is NOT the default and it is NOT a fix. Measured on the drift
        #       benchmark it is indistinguishable from the "score" gate (92.96% vs
        #       92.96% persistent false alarms, 0.0 baseline movement in both): the
        #       persistence head does not separate its normal and drift regimes
        #       (0.33 vs 0.37), so the calibrated threshold is never crossed and the
        #       mechanism stays inert. Documented as a negative result in
        #       docs/benchmark_report.md rather than presented as a repair.
        e_t = e_seq[:, -1]
        q_mean = torch.mean(quality, dim=-1, keepdim=True)
        fast_gate = torch.exp(-self.anomaly_gate_scale * anomaly) * q_mean

        # A hard threshold on a head whose outputs cluster in a narrow band is
        # useless: nothing ever crosses it. Instead the gate ramps across a
        # defined band above a threshold that is *calibrated* on the subject's
        # own normal history (see calibrate_persistence_threshold).
        confirmed = torch.clamp(
            (persistence - self.persistence_threshold) / self.persistence_band, 0.0, 1.0
        )
        slow_gate = (
            confirmed * q_mean
            if self.slow_gate_mode == "persistence"
            else fast_gate
        )

        fast_memory = (1.0 - self.fast_rate * fast_gate) * baseline_fast + (self.fast_rate * fast_gate) * e_t
        slow_memory = (1.0 - self.slow_rate * slow_gate) * baseline_slow + (self.slow_rate * slow_gate) * e_t

        return anomaly, uncertainty, persistence, deviation_emb, fast_memory, slow_memory

    def _calibration_baselines(
        self,
        batch,
        slow_memories: dict[str, torch.Tensor] | None,
        fast_memories: dict[str, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-window baselines for calibration, matching the deployment regime.

        Calibrating against zero memories while evaluating against subject-specific
        memories puts the threshold in a different regime from the scores it must
        gate, which makes the rule meaningless. Memories are therefore expanded to
        per-window tensors exactly as they are at inference.
        """
        from meyro.data.windows import expand_subject_memories

        if slow_memories is None or fast_memories is None:
            zeros = torch.zeros(len(batch), self.hidden_dim)
            return zeros, zeros
        slow = expand_subject_memories(slow_memories, batch, self.hidden_dim, default="cohort_mean")
        fast = expand_subject_memories(fast_memories, batch, self.hidden_dim, default="cohort_mean")
        return slow, fast

    def calibrate_persistence_threshold(
        self,
        batch,
        slow_memories: dict[str, torch.Tensor] | None = None,
        fast_memories: dict[str, torch.Tensor] | None = None,
        *,
        quantile: float = 0.99,
    ) -> float:
        """Sets the slow-gate threshold from the subject's own normal history.

        Uses only calibration windows, so no evaluation information enters the
        decision rule. The threshold is the ``quantile`` of persistence observed
        on normal data: a deviation must be more persistent than the subject's
        normal fluctuation before the baseline may migrate. This mirrors the
        statistical baselines' self-calibrating ``mean + 3 sigma`` rule.

        Returns the threshold that was set.
        """
        if not 0.0 < quantile < 1.0:
            raise ValueError(f"quantile must lie in (0, 1), received {quantile}")

        x_t, ctx_t, qual_t = batch.tensors()
        slow, fast = self._calibration_baselines(batch, slow_memories, fast_memories)
        self.eval()
        persistences: list[float] = []
        with torch.no_grad():
            for start in range(0, len(x_t), 64):
                window = x_t[start : start + 64]
                context = ctx_t[start : start + 64]
                quality = qual_t[start : start + 64]
                _a, _u, persistence, _d, _f, _s = self.forward(
                    window,
                    context,
                    slow[start : start + 64],
                    fast[start : start + 64],
                    quality,
                )
                persistences.extend(persistence.squeeze(-1).tolist())

        threshold = float(np.quantile(np.asarray(persistences), quantile))
        self.persistence_threshold = threshold
        return threshold

    def calibrate_anomaly_threshold(
        self,
        batch,
        slow_memories: dict[str, torch.Tensor] | None = None,
        fast_memories: dict[str, torch.Tensor] | None = None,
        *,
        quantile: float = 0.995,
    ) -> float:
        """Sets the streak-activation threshold for confirmed slow-memory migration.

        Derived from the subject's own calibration windows, so it is a personal
        decision rule and cannot leak evaluation information.
        """
        if not 0.0 < quantile < 1.0:
            raise ValueError(f"quantile must lie in (0, 1), received {quantile}")

        x_t, ctx_t, qual_t = batch.tensors()
        slow, fast = self._calibration_baselines(batch, slow_memories, fast_memories)
        self.eval()
        anomalies: list[float] = []
        with torch.no_grad():
            for start in range(0, len(x_t), 64):
                window = x_t[start : start + 64]
                context = ctx_t[start : start + 64]
                quality = qual_t[start : start + 64]
                anomaly, _u, _p, _d, _f, _s = self.forward(
                    window,
                    context,
                    slow[start : start + 64],
                    fast[start : start + 64],
                    quality,
                )
                anomalies.extend(anomaly.squeeze(-1).tolist())

        threshold = float(np.quantile(np.asarray(anomalies), quantile))
        self.anomaly_confirm_threshold = threshold
        return threshold

    def forward_adaptive(
        self,
        x_seq: torch.Tensor,
        context: torch.Tensor,
        baseline_slow: torch.Tensor,
        baseline_fast: torch.Tensor,
        quality: torch.Tensor,
        streak: torch.Tensor,
        *,
        patience: int = 4,
        confirm_threshold: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with streak-confirmed slow-memory migration.

        The established baseline may only migrate once the deviation has been
        continuously present for ``patience`` consecutive windows. A single
        extreme observation cannot reach that count, so it cannot redefine the
        baseline; a sustained lifestyle change eventually does. The fast memory
        and the anomaly score are unaffected, so detection sensitivity is
        preserved during the confirmation period.

        Returns ``(anomaly, uncertainty, persistence, fast_memory, slow_memory, new_streak)``.
        """
        if patience < 1:
            raise ValueError(f"patience must be >= 1, received {patience}")

        threshold = (
            self.anomaly_confirm_threshold if confirm_threshold is None else confirm_threshold
        )
        if threshold is None:
            raise ValueError(
                "confirm_threshold is unset; call calibrate_anomaly_threshold first "
                "or pass confirm_threshold explicitly"
            )

        anomaly, uncertainty, persistence, _d, fast_next, slow_next = self.forward(
            x_seq, context, baseline_slow, baseline_fast, quality
        )

        elevated = anomaly >= threshold
        new_streak = torch.where(
            elevated,
            streak + 1.0,
            torch.zeros_like(streak),
        )
        confirmed = new_streak >= float(patience)
        migrated_slow = torch.where(confirmed, slow_next, baseline_slow)

        return anomaly, uncertainty, persistence, fast_next, migrated_slow, new_streak

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
