"""Attentive pooling: one learnable query over a token sequence → vector."""

from __future__ import annotations

import torch
import torch.nn as nn


class AttentivePooling(nn.Module):
    """
    Aggregate ``(B, N, D)`` tokens into a single ``(B, D)`` condition vector.

    One learnable query token cross-attends to all input tokens, then a linear
    projects the attended query.
    """

    def __init__(self, d_embed: int, num_heads: int = 8):
        super().__init__()
        if d_embed % num_heads != 0:
            raise ValueError(
                f"d_embed ({d_embed}) must be divisible by num_heads ({num_heads})"
            )
        self.query = nn.Parameter(torch.zeros(1, 1, d_embed))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.attn = nn.MultiheadAttention(
            d_embed, num_heads, batch_first=True, dropout=0.0
        )
        self.proj = nn.Linear(d_embed, d_embed)

    def forward(
        self,
        tokens: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            tokens: ``(B, N, D)`` fused modality tokens.
            key_padding_mask: optional ``(B, N)`` True = ignore (PyTorch MHA convention).
        Returns:
            ``(B, D)`` pooled condition vector.
        """
        if tokens.ndim != 3:
            raise ValueError(f"Expected tokens (B, N, D), got {tuple(tokens.shape)}")
        b = tokens.shape[0]
        query = self.query.expand(b, -1, -1)
        attended, _ = self.attn(
            query,
            tokens,
            tokens,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return self.proj(attended.squeeze(1))
