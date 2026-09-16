"""Conditional 1D UNet policy head."""

from typing import Optional

import torch
import torch.nn as nn

from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from vista.heads.base import PolicyHead
from vista.models.condition import Condition


class Unet1DHead(PolicyHead):
    """Wrap ConditionalUnet1D; uses global vector condition."""

    def __init__(
        self,
        action_dim: int,
        action_horizon: int,
        d_embed: int = 256,
        down_dims=(256, 512, 1024),
        **unet_kwargs,
    ):
        super().__init__(
            action_dim=action_dim,
            action_horizon=action_horizon,
            d_embed=d_embed,
            cond_mode="vector",
        )
        self.unet = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=d_embed,
            diffusion_step_embed_dim=d_embed,
            down_dims=list(down_dims),
            **unet_kwargs,
        )

    def forward(
        self,
        condition: Condition,
        noisy_action: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _, vector = self._select_cond(condition)
        assert noisy_action is not None and timestep is not None
        # ConditionalUnet1D expects (B, T, Da) and rearranges to (B, Da, T) internally.
        return self.unet(noisy_action, timestep, global_cond=vector)
