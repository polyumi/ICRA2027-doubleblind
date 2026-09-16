"""Q-Former fusion: learnable queries with self-attn, multi cross-attn, and MLP."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from vista.fusion.base import Fusion


class QFormerLayer(nn.Module):
    """
    One Q-Former block.

    Self-attn on queries, then ``n_cross_attn`` cross-attns to shared context,
    then an MLP with ``mlp_ratio`` (queries only). Prenorm residuals.
    """

    def __init__(
        self,
        d_embed: int,
        n_heads: int,
        n_cross_attn: int = 3,
        mlp_ratio: int = 2,
    ):
        super().__init__()
        if n_cross_attn < 1:
            raise ValueError(f"n_cross_attn must be >= 1, got {n_cross_attn}")
        self.self_norm = nn.LayerNorm(d_embed)
        self.self_attn = nn.MultiheadAttention(d_embed, n_heads, batch_first=True)
        self.cross_norms = nn.ModuleList(
            [nn.LayerNorm(d_embed) for _ in range(n_cross_attn)]
        )
        self.cross_attns = nn.ModuleList(
            [
                nn.MultiheadAttention(d_embed, n_heads, batch_first=True)
                for _ in range(n_cross_attn)
            ]
        )
        hidden = d_embed * mlp_ratio
        self.mlp_norm = nn.LayerNorm(d_embed)
        self.mlp = nn.Sequential(
            nn.Linear(d_embed, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_embed),
        )

    def forward(self, queries: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        q = self.self_norm(queries)
        attn_out, _ = self.self_attn(q, q, q)
        queries = queries + attn_out
        for norm, cross in zip(self.cross_norms, self.cross_attns):
            q = norm(queries)
            attn_out, _ = cross(q, context, context)
            queries = queries + attn_out
        queries = queries + self.mlp(self.mlp_norm(queries))
        return queries


class QFormerEncoder(Fusion):
    """
    Learnable query tokens that read a variable-length multimodal context.

    Dropped sensors simply omit keys from ``tokens_by_key``; context length
    shrinks and nothing else in the stack changes.
    """

    def __init__(
        self,
        d_embed: int = 256,
        n_queries: int = 128,
        n_latents: Optional[int] = None,
        n_layers: int = 6,
        n_heads: int = 8,
        n_cross_attn: int = 3,
        mlp_ratio: int = 2,
        mode: Optional[str] = None,
        block_mode: Optional[str] = None,
        **kwargs,
    ):
        del kwargs
        if mode is not None or block_mode is not None:
            raise ValueError(
                "QFormerEncoder no longer supports mode/block_mode; "
                "use stacked self-attn + cross-attn + MLP layers."
            )
        if n_latents is not None:
            n_queries = n_latents
        super().__init__(d_embed=d_embed)
        self.n_queries = n_queries
        self.queries = nn.Parameter(torch.randn(1, n_queries, d_embed) * 0.02)
        self.layers = nn.ModuleList(
            [
                QFormerLayer(
                    d_embed=d_embed,
                    n_heads=n_heads,
                    n_cross_attn=n_cross_attn,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(
        self,
        tokens_by_key: Dict[str, torch.Tensor],
        pad_masks: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        del pad_masks
        if not tokens_by_key:
            raise ValueError("QFormerEncoder requires at least one context stream")
        keys = sorted(tokens_by_key.keys())
        context = torch.cat([tokens_by_key[k] for k in keys], dim=1)
        queries = self.queries.expand(context.shape[0], -1, -1)
        for layer in self.layers:
            queries = layer(queries, context)
        return queries, None
