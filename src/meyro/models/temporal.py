"""PyTorch temporal sequence models (GRU/LSTM/TCN) for personalized deviation estimation.

Architecture:
  Input Window [batch, seq_len, features]
       ↓
  Temporal Encoder (GRU / LSTM / 1D-CNN)
       ↓
  Latent Personal Context Vector [batch, hidden_dim]
       ↓
  Reconstruction / Deviation Head
       ↓
  Squared Error / Personal Deviation Score
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TemporalAutoencoder(nn.Module):
    """Sequence-to-sequence autoencoder for learning personal physiological dynamics.

    Trained on normal calibration history. High reconstruction error during evaluation
    signals a temporal deviation from personal dynamics.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        num_layers: int = 1,
        cell_type: str = "gru",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.cell_type = cell_type.lower()

        if self.cell_type == "lstm":
            self.encoder = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
            )
            self.decoder = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
            )
        else:
            self.encoder = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
            )
            self.decoder = nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
            )

        self.output_projection = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: [batch, seq_len, input_dim] -> returns (reconstruction, context_embedding)."""
        seq_len = x.size(1)
        if self.cell_type == "lstm":
            _enc_out, (h_n, _) = self.encoder(x)
            context = h_n[-1]  # [batch, hidden_dim]
        else:
            _enc_out, h_n = self.encoder(x)
            context = h_n[-1]

        # Repeat context across time steps to decode
        decoder_input = context.unsqueeze(1).repeat(1, seq_len, 1)
        dec_out, _ = self.decoder(decoder_input)
        reconstruction = self.output_projection(dec_out)
        return reconstruction, context

    def compute_deviation_score(self, x: torch.Tensor) -> torch.Tensor:
        """Computes mean squared reconstruction error per window."""
        self.eval()
        with torch.no_grad():
            recon, _ = self.forward(x)
            # MSE per sample across [seq_len, features]
            error = torch.mean((x - recon) ** 2, dim=(1, 2))
        return error
