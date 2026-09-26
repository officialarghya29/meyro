"""Reconstruction trainer for deep autoencoder baselines.

Without this, the deep baselines in the master benchmark would be evaluated at
random initialisation — which would make any comparison against them
meaningless. Every autoencoder in the benchmark is trained here on exactly the
same calibration windows used to fit the statistical baselines.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from meyro.utils.io import set_global_seed


def train_reconstruction(
    model: nn.Module,
    x_train: torch.Tensor,
    *,
    epochs: int = 40,
    learning_rate: float = 5e-3,
    seed: int = 42,
    batch_size: int = 128,
) -> nn.Module:
    """Trains ``model`` to reconstruct its input windows (self-supervised).

    Args:
        model: an autoencoder whose forward maps ``[B, T, F] -> [B, T, F]``.
        x_train: calibration windows, shape ``[N, T, F]``.
        epochs: full passes over the calibration windows.
        learning_rate: AdamW learning rate.
        seed: RNG seed for reproducibility.
        batch_size: mini-batch size.

    Returns:
        The trained model in eval mode.
    """
    set_global_seed(seed)
    model.train()

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.MSELoss()

    n_samples = x_train.shape[0]
    generator = torch.Generator().manual_seed(seed)

    for _epoch in range(epochs):
        perm = torch.randperm(n_samples, generator=generator)
        for start in range(0, n_samples, batch_size):
            idx = perm[start : start + batch_size]
            batch = x_train[idx]
            optimizer.zero_grad()
            output = model(batch)
            # Some autoencoders return (reconstruction, context); take the reconstruction.
            reconstruction = output[0] if isinstance(output, tuple) else output
            loss = criterion(reconstruction, batch)
            loss.backward()
            optimizer.step()

    model.eval()
    return model
