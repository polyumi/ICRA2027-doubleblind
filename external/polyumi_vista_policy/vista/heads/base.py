"""Policy head base class."""

from abc import ABC, abstractmethod
from typing import Literal, Optional

import torch
import torch.nn as nn

from vista.models.condition import Condition

CondMode = Literal["tokens", "vector", "both"]


class PolicyHead(nn.Module, ABC):
    """Predict action trajectories from fused condition."""

    def __init__(
        self,
        action_dim: int,
        action_horizon: int,
        d_embed: int = 256,
        cond_mode: CondMode = "vector",
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.d_embed = d_embed
        self.cond_mode = cond_mode

    def _select_cond(self, condition: Condition) -> tuple:
        tokens = condition.tokens if self.cond_mode in ("tokens", "both") else None
        vector = condition.vector if self.cond_mode in ("vector", "both") else None
        if self.cond_mode == "tokens" and tokens is None:
            raise ValueError("Head expects tokens but condition has none.")
        if self.cond_mode == "vector" and vector is None:
            vector = condition.vector
        return tokens, vector

    @abstractmethod
    def forward(
        self,
        condition: Condition,
        noisy_action: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return model output (noise/sample/velocity depending on objective)."""
