"""Proprioceptive MLP encoder."""

import torch
import torch.nn as nn

from vista.encoders.base import SensorEncoder


class ProprioMLP(SensorEncoder):
    """Flatten low-dim proprio over time into tokens."""

    def __init__(self, input_dim: int, d_embed: int = 256, hidden_dim: int = 512):
        super().__init__(d_embed=d_embed)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_embed),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B, T, D -> B, T, d_embed
        return self.net(x)
