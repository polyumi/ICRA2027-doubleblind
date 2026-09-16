"""MLP regression head."""

from typing import Optional

import torch
import torch.nn as nn

from vista.heads.base import PolicyHead
from vista.models.condition import Condition


class MLPHead(PolicyHead):
    """Direct action regression from pooled condition vector."""

    def __init__(
        self,
        action_dim: int,
        action_horizon: int,
        d_embed: int = 256,
        hidden_dim: int = 512,
    ):
        super().__init__(
            action_dim=action_dim,
            action_horizon=action_horizon,
            d_embed=d_embed,
            cond_mode="vector",
        )
        out_dim = action_dim * action_horizon
        self.net = nn.Sequential(
            nn.Linear(d_embed, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(
        self,
        condition: Condition,
        noisy_action: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _, vector = self._select_cond(condition)
        out = self.net(vector)
        return out.reshape(vector.shape[0], self.action_horizon, self.action_dim)
