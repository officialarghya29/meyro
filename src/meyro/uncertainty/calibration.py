"""Uncertainty calibration and temperature scaling module for MEYRO (Phase 25)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class TemperatureScaler(nn.Module):
    """Calibrates model anomaly probabilities using Platt scaling / temperature scaling."""

    def __init__(self) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def fit_calibration(self, val_logits: torch.Tensor, val_labels: torch.Tensor) -> TemperatureScaler:
        """Optimizes temperature T on validation set using negative log likelihood."""
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        criterion = nn.BCEWithLogitsLoss()

        def eval_step():
            optimizer.zero_grad()
            loss = criterion(self.forward(val_logits), val_labels)
            loss.backward()
            return loss

        optimizer.step(eval_step)
        return self

    def compute_expected_calibration_error(
        self, probs: np.ndarray, labels: np.ndarray, n_bins: int = 10
    ) -> float:
        """Computes Expected Calibration Error (ECE)."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
            in_bin = (probs >= bin_lower) & (probs < bin_upper)
            prop_in_bin = np.mean(in_bin)

            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(labels[in_bin])
                avg_confidence_in_bin = np.mean(probs[in_bin])
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return float(ece)
