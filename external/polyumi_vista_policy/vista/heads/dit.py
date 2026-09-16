"""Transformer denoiser head (mitas port)."""

from typing import Optional

import torch
import torch.nn as nn

from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb
from vista.heads.base import PolicyHead
from vista.models.condition import Condition


class AdaLNZeroBlock(nn.Module):
    """Adaptive layer norm with zero-init modulation."""

    def __init__(self, d_embed: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_embed, elementwise_affine=False)
        self.mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * d_embed),
        )
        nn.init.zeros_(self.mod[-1].weight)
        nn.init.zeros_(self.mod[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.mod(cond).chunk(2, dim=-1)
        while scale.ndim < x.ndim:
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)
        return self.norm(x) * (1 + scale) + shift


class TransformerDenoiser(PolicyHead):
    """Decoder-only transformer denoising actions with optional AdaLN-Zero."""

    def __init__(
        self,
        action_dim: int,
        action_horizon: int,
        d_embed: int = 256,
        n_layer: int = 4,
        n_head: int = 4,
        mlp_ratio: int = 4,
        max_cond_tokens: int = 1024,
        adaln_zero: bool = False,
        cond_mode: str = "both",
    ):
        super().__init__(
            action_dim=action_dim,
            action_horizon=action_horizon,
            d_embed=d_embed,
            cond_mode=cond_mode,
        )
        self.adaln_zero = adaln_zero
        self.input_emb = nn.Linear(action_dim, d_embed)
        self.pos_emb = nn.Parameter(torch.zeros(1, action_horizon, d_embed))
        self.time_emb = SinusoidalPosEmb(d_embed)
        self.cond_pos_emb = nn.Parameter(torch.zeros(1, max_cond_tokens, d_embed))
        self.time_proj = nn.Linear(d_embed, d_embed)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_embed,
            nhead=n_head,
            dim_feedforward=d_embed * mlp_ratio,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layer)
        self.adaln = AdaLNZeroBlock(d_embed, d_embed) if adaln_zero else None
        self.ln_f = nn.LayerNorm(d_embed)
        self.out = nn.Linear(d_embed, action_dim)

    def _condition_pos_emb(self, n_tokens: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Slice learned cond positions, zero-padding when fusion emits more tokens."""
        pos = self.cond_pos_emb
        if n_tokens <= pos.shape[1]:
            return pos[:, :n_tokens]
        pad = torch.zeros(1, n_tokens - pos.shape[1], pos.shape[2], device=device, dtype=dtype)
        return torch.cat([pos, pad], dim=1)

    def forward(
        self,
        condition: Condition,
        noisy_action: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        tokens, vector = self._select_cond(condition)
        assert noisy_action is not None and timestep is not None
        b = noisy_action.shape[0]
        x = self.input_emb(noisy_action) + self.pos_emb[:, : noisy_action.shape[1]]
        t_emb = self.time_proj(self.time_emb(timestep))
        if tokens is not None:
            mem = tokens + self._condition_pos_emb(tokens.shape[1], tokens.device, tokens.dtype)
        else:
            mem = vector.unsqueeze(1)
        if self.adaln is not None:
            x = self.adaln(x, t_emb)
        tgt = x
        out = self.decoder(tgt, mem)
        out = self.ln_f(out)
        return self.out(out)
