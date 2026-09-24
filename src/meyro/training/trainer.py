"""Lightweight trainer for MEYRO model using calibration history."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from meyro.models.meyro import MEYROModel
from meyro.preprocessing.pipeline import PreprocessingPipeline


class MEYROTrainer:
    """Trains MEYRO on calibration normal history using self-supervised contrastive reconstruction."""

    def __init__(
        self,
        model: MEYROModel,
        learning_rate: float = 1e-3,
        epochs: int = 15,
        window_size: int = 7,
    ) -> None:
        self.model = model
        self.lr = learning_rate
        self.epochs = epochs
        self.window_size = window_size
        self.optimizer = optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-4)

    def fit_calibration_cohort(
        self,
        calib_df: pd.DataFrame,
        feature_cols: list[str],
    ) -> dict[str, torch.Tensor]:
        """Fits baseline memories and trains deviation representations on normal calibration history."""
        self.model.train()

        # Build training sequences from calibration (all label = 0 normal)
        windows = []
        contexts = []
        qualities = []

        for _, grp in calib_df.groupby("subject_id", sort=False):
            g = grp.sort_values(by="timestamp").reset_index(drop=True)
            feats = g[feature_cols].to_numpy()
            quals = np.stack([g["quality_score"].to_numpy()] * len(feature_cols), axis=-1)
            dows = np.array([ts.weekday() for ts in g["timestamp"]])
            ctx = np.stack([np.sin(2 * np.pi * dows / 7.0), np.cos(2 * np.pi * dows / 7.0)], axis=-1)

            w = PreprocessingPipeline.create_sliding_windows(feats, window_size=self.window_size)
            if len(w) > 0:
                windows.append(w)
                contexts.append(ctx[self.window_size - 1 :])
                qualities.append(quals[self.window_size - 1 :])

        if not windows:
            return {}

        x_train = torch.tensor(np.concatenate(windows, axis=0), dtype=torch.float32)
        c_train = torch.tensor(np.concatenate(contexts, axis=0), dtype=torch.float32)
        q_train = torch.tensor(np.concatenate(qualities, axis=0), dtype=torch.float32)

        # Baseline memory initialization per sample (starts from mean observation encoding)
        with torch.no_grad():
            e_init = self.model.encode_observation(x_train, c_train)
            mean_base = torch.mean(e_init, dim=0, keepdim=True)
            b_train = mean_base.repeat(len(x_train), 1)

        bce_loss_fn = nn.BCELoss()
        mse_loss_fn = nn.MSELoss()

        target_normal = torch.zeros(len(x_train), 1)

        for _epoch in range(self.epochs):
            self.optimizer.zero_grad()
            a_score, _u_score, _d_t, b_next = self.model(x_train, c_train, b_train, q_train)

            # Normal samples in calibration should have near-zero anomaly score
            loss_anomaly = bce_loss_fn(a_score, target_normal)
            # Memory should be stable across consecutive normal windows
            loss_memory = mse_loss_fn(b_next, b_train)

            total_loss = loss_anomaly + 0.1 * loss_memory
            total_loss.backward()
            self.optimizer.step()
            b_train = b_next.detach()

        # Compute final subject-specific baseline memories
        subj_memories = {}
        self.model.eval()
        with torch.no_grad():
            for subj_id, grp in calib_df.groupby("subject_id", sort=False):
                g = grp.sort_values(by="timestamp").reset_index(drop=True)
                feats = g[feature_cols].to_numpy()
                dows = np.array([ts.weekday() for ts in g["timestamp"]])
                ctx = np.stack([np.sin(2 * np.pi * dows / 7.0), np.cos(2 * np.pi * dows / 7.0)], axis=-1)
                w = PreprocessingPipeline.create_sliding_windows(feats, window_size=self.window_size)
                if len(w) > 0:
                    tw = torch.tensor(w, dtype=torch.float32)
                    tc = torch.tensor(ctx[self.window_size - 1 :], dtype=torch.float32)
                    e_sub = self.model.encode_observation(tw, tc)
                    subj_memories[subj_id] = torch.mean(e_sub, dim=0, keepdim=True)

        return subj_memories
