"""Training pipelines for the MEYRO model family (Phases 11/15/30).

Two trainers are provided:

``MEYROTrainer``
    Trains MEYRO-V1 (single memory). Kept so V1 remains a genuine, trained
    reference rather than a randomly initialised straw man.

``MEYROV2Trainer``
    Trains MEYRO-V2 (dual-timescale memory + persistence).

Shared objective (strictly self-supervised):

* ``L_normal``      — calibration windows are normal, so the anomaly head is
  pushed towards 0.
* ``L_deviation``   — windows perturbed by *calibration-derived* magnitudes are
  labelled anomalous, teaching the head what a personal deviation looks like.
* ``L_persistence`` — strongly perturbed windows are persistent, mildly
  perturbed windows are transient.
* ``L_memory``      — memories must not move far on normal observations.

Two properties matter for correctness and are easy to get wrong:

1. **Per-subject memories during training.** A cohort-level baseline at training
   time, replaced by a subject-specific baseline at inference, is a
   train/inference mismatch that lets the deviation module collapse to a
   constant score. Memories are therefore computed per subject, leave-one-window-out.
2. **Standardised features.** Raw physiological scales saturate recurrent
   encoders, so windows are standardised with statistics fitted on calibration
   data only (see ``SubjectStandardizer``).

Nothing here reads ``ground_truth_label``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from meyro.data.windows import (
    FeatureStandardizer,
    WindowBatch,
    build_windows,
    per_window_subject_memories,
    subject_mean_baseline,
)
from meyro.models.meyro import MEYROModel, MEYROModelV2
from meyro.utils.io import set_global_seed


def _augment_deviations(
    batch: WindowBatch,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Creates perturbed windows from calibration (already standardised) data.

    Returns ``(x_perturbed, severity)`` with severity in ``[0, 1]``, used as the
    persistence target. Perturbation magnitude is expressed relative to the
    subject's own standardised spread, so it is a personal deviation.
    """
    x = batch.x.copy()
    severity = rng.uniform(0.35, 1.0, size=len(batch)).astype(np.float32)

    for i in range(len(batch)):
        feature_index = int(rng.integers(0, x.shape[-1]))
        direction = 1.0 if rng.uniform() < 0.5 else -1.0
        x[i, :, feature_index] += direction * severity[i] * 1.5

    return x, severity


class _BaseTrainer:
    """Shared optimisation loop for the MEYRO family."""

    def __init__(
        self,
        learning_rate: float = 1e-3,
        epochs: int = 25,
        window_size: int = 7,
        seed: int = 42,
        memory_weight: float = 0.1,
    ) -> None:
        self.lr = learning_rate
        self.epochs = epochs
        self.window_size = window_size
        self.seed = seed
        self.memory_weight = memory_weight
        self.training_windows = 0
        self.standardizer: FeatureStandardizer | None = None

    def prepare_batch(self, calib_df: pd.DataFrame, feature_cols: list[str]) -> WindowBatch:
        """Builds calibration windows and fits/scales with a personal standardizer."""
        batch = build_windows(calib_df, feature_cols, window_size=self.window_size)
        self.standardizer = FeatureStandardizer().fit(batch)
        return self.standardizer.transform(batch)


class MEYROTrainer(_BaseTrainer):
    """Trains MEYRO-V1 on a subject's calibration history."""

    def __init__(self, model: MEYROModel, **kwargs) -> None:
        super().__init__(**kwargs)
        self.model = model

    def fit_windows(self, batch: WindowBatch) -> dict[str, torch.Tensor]:
        """Trains on already-standardised calibration windows; returns per-subject memories."""
        set_global_seed(self.seed)
        self.model.train()

        x_train, c_train, q_train = batch.tensors()
        n_windows = len(x_train)
        self.training_windows = n_windows

        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        bce = nn.BCELoss()
        mse = nn.MSELoss()

        rng = np.random.default_rng(self.seed)
        target_normal = torch.zeros(n_windows, 1)

        for _epoch in range(self.epochs):
            # Personal, leave-one-out memories recomputed as the encoder changes.
            memories = per_window_subject_memories(
                self.model.encode_observation, batch, self.model.hidden_dim
            )
            optimizer.zero_grad()

            a_normal, _u, _d, b_next = self.model(x_train, c_train, memories, q_train)
            loss_normal = bce(a_normal, target_normal)
            loss_memory = mse(b_next, memories)

            x_pert, severity = _augment_deviations(batch, rng)
            a_pert, _u2, _d2, _ = self.model(
                torch.tensor(x_pert, dtype=torch.float32),
                c_train,
                memories,
                q_train,
            )
            loss_deviation = bce(a_pert, torch.tensor(severity, dtype=torch.float32).unsqueeze(1))

            total = loss_normal + loss_deviation + self.memory_weight * loss_memory
            total.backward()
            optimizer.step()

        self.model.eval()
        return subject_mean_baseline(self.model.encode_observation, batch, self.model.hidden_dim)

    def fit_calibration_cohort(
        self,
        calib_df: pd.DataFrame,
        feature_cols: list[str],
    ) -> dict[str, torch.Tensor]:
        """Convenience wrapper: builds, standardises and trains in one call."""
        return self.fit_windows(self.prepare_batch(calib_df, feature_cols))


class MEYROV2Trainer(_BaseTrainer):
    """Trains MEYRO-V2 (dual memory + persistence) on calibration history."""

    def __init__(self, model: MEYROModelV2, **kwargs) -> None:
        super().__init__(**kwargs)
        self.model = model

    def fit_windows(self, batch: WindowBatch) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Trains on standardised calibration windows; returns ``(fast, slow)`` memories."""
        set_global_seed(self.seed)
        self.model.train()

        x_train, c_train, q_train = batch.tensors()
        n_windows = len(x_train)
        self.training_windows = n_windows

        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        bce = nn.BCELoss()
        mse = nn.MSELoss()

        rng = np.random.default_rng(self.seed)
        target_normal = torch.zeros(n_windows, 1)

        for _epoch in range(self.epochs):
            personal = per_window_subject_memories(
                self.model.encode_observation, batch, self.model.hidden_dim
            )
            fast = personal.clone()
            slow = personal.clone()
            optimizer.zero_grad()

            a_normal, _u, r_normal, _d, fast_next, slow_next = self.model(
                x_train, c_train, slow, fast, q_train
            )
            loss_normal = bce(a_normal, target_normal)
            loss_persistence_normal = bce(r_normal, target_normal)
            loss_memory = mse(fast_next, fast) + mse(slow_next, slow)

            x_pert, severity = _augment_deviations(batch, rng)
            severity_t = torch.tensor(severity, dtype=torch.float32).unsqueeze(1)
            a_pert, _u2, r_pert, _d2, _f, _s = self.model(
                torch.tensor(x_pert, dtype=torch.float32), c_train, slow, fast, q_train
            )
            loss_deviation = bce(a_pert, severity_t)
            loss_persistence = bce(r_pert, severity_t)

            total = (
                loss_normal
                + loss_deviation
                + loss_persistence
                + loss_persistence_normal
                + self.memory_weight * loss_memory
            )
            total.backward()
            optimizer.step()

        self.model.eval()
        slow_memories = subject_mean_baseline(self.model.encode_observation, batch, self.model.hidden_dim)
        # Cold start: the fast memory begins where the slow memory does.
        fast_memories = {k: v.clone() for k, v in slow_memories.items()}
        return fast_memories, slow_memories

    def fit_calibration_cohort(
        self,
        calib_df: pd.DataFrame,
        feature_cols: list[str],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Convenience wrapper: builds, standardises and trains in one call."""
        return self.fit_windows(self.prepare_batch(calib_df, feature_cols))
