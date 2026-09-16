"""Transformer action decoder head (Sparsh-X IL attachment)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from vista.heads.base import PolicyHead
from vista.models.condition import Condition


class TransformerActionDecoder(PolicyHead):
    """
    Action queries cross-attend fused observation tokens → (B, H, Da).

    Used by SparshXPolicy as a behavior-cloning head (no diffusion).
    """

    def __init__(
        self,
        action_dim: int,
        action_horizon: int,
        d_embed: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__(
            action_dim=action_dim,
            action_horizon=action_horizon,
            d_embed=d_embed,
            cond_mode="tokens",
        )
        self.action_queries = nn.Parameter(torch.randn(1, action_horizon, d_embed) * 0.02)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_embed,
            nhead=n_heads,
            dim_feedforward=int(d_embed * mlp_ratio),
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.out = nn.Linear(d_embed, action_dim)

    def forward(
        self,
        condition: Condition,
        noisy_action: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del noisy_action, timestep
        tokens, _ = self._select_cond(condition)
        assert tokens is not None
        b = tokens.shape[0]
        queries = self.action_queries.expand(b, -1, -1)
        decoded = self.decoder(queries, tokens)
        return self.out(decoded)
