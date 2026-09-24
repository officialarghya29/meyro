"""Temporal Convolutional Network (TCN) baseline model for time series anomaly detection."""

from __future__ import annotations

import torch
import torch.nn as nn


class TemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation
        )
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation
        )
        self.relu2 = nn.ReLU()
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Causal trim of right-side padding
        res = x if self.downsample is None else self.downsample(x)
        out = self.conv1(x)
        out = out[:, :, : -self.padding] if self.padding > 0 else out
        out = self.relu1(out)
        out = self.conv2(out)
        out = out[:, :, : -self.padding] if self.padding > 0 else out
        out = self.relu2(out)
        return out + res


class TCNAutoencoder(nn.Module):
    """Causal dilated TCN autoencoder for temporal sequence modeling."""

    def __init__(self, input_dim: int, hidden_dim: int = 32, num_layers: int = 2) -> None:
        super().__init__()
        layers = []
        channels = [input_dim] + [hidden_dim] * num_layers
        for i in range(num_layers):
            layers.append(
                TemporalBlock(
                    channels[i], channels[i + 1], kernel_size=3, dilation=2**i
                )
            )
        self.tcn = nn.Sequential(*layers)
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, input_dim] -> conv requires [batch, input_dim, seq_len]
        x_trans = x.transpose(1, 2)
        feat = self.tcn(x_trans).transpose(1, 2)
        recon = self.decoder(feat)
        return recon

    def compute_deviation_score(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            recon = self.forward(x)
            error = torch.mean((x - recon) ** 2, dim=(1, 2))
        return error
