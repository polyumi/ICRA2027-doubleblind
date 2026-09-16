"""
PolyTouch modality combiner.

Implements the combiner described in arxiv:2504.19341 §IV-A / Fig. 5:
6-block 12-head cross-attention between T3 (tactile) and CLIP (vision) tokens,
then concatenate three CLS tokens (CLIP, T3, AST) and project; proprio is
concatenated after projection for the diffusion U-Net condition.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CrossAttentionBlock(nn.Module):
    """One bidirectional cross-attention block between vision and tactile."""

    def __init__(self, dim: int, n_heads: int = 12, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm_v = nn.LayerNorm(dim)
        self.norm_t = nn.LayerNorm(dim)
        self.attn_t2v = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.attn_v2t = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        hidden = int(dim * mlp_ratio)
        self.mlp_v = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        self.mlp_t = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, vision: torch.Tensor, tactile: torch.Tensor):
        # tactile queries vision (and vice versa)
        t_n = self.norm_t(tactile)
        v_n = self.norm_v(vision)
        t_out, _ = self.attn_t2v(t_n, v_n, v_n)
        v_out, _ = self.attn_v2t(v_n, t_n, t_n)
        tactile = tactile + t_out
        vision = vision + v_out
        tactile = tactile + self.mlp_t(tactile)
        vision = vision + self.mlp_v(vision)
        return vision, tactile


class PolyTouchCombiner(nn.Module):
    """
    Cross-attend CLIP ↔ T3 for ``n_blocks`` layers, then
    concat(CLIP_CLS, T3_CLS, AST_CLS) → Linear → concat proprio.
    """

    def __init__(
        self,
        d_model: int = 768,
        n_blocks: int = 6,
        n_heads: int = 12,
        proprio_dim: int = 32,
        out_dim: int = 256,
        proprio_out_dim: int = 64,
    ):
        super().__init__()
        self.d_model = d_model
        self.blocks = nn.ModuleList(
            [CrossAttentionBlock(d_model, n_heads=n_heads) for _ in range(n_blocks)]
        )
        self.cls_proj = nn.Sequential(
            nn.Linear(d_model * 3, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )
        # Proprio is low-dim (~16·H); keep a small side channel vs fused vision CLS.
        self.proprio_proj = nn.Sequential(
            nn.Linear(proprio_dim, proprio_out_dim),
            nn.GELU(),
            nn.Linear(proprio_out_dim, proprio_out_dim),
        )
        self.out_dim = out_dim + proprio_out_dim

    def forward(
        self,
        clip_tokens: torch.Tensor,
        t3_tokens: torch.Tensor,
        ast_cls: torch.Tensor,
        proprio: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            clip_tokens: (B, 1+P, D) — CLS first
            t3_tokens: (B, 1+P, D) — CLS first
            ast_cls: (B, D)
            proprio: (B, Dp)
        Returns:
            (B, out_dim + proprio_out_dim) global condition vector
        """
        vision, tactile = clip_tokens, t3_tokens
        for block in self.blocks:
            vision, tactile = block(vision, tactile)
        clip_cls = vision[:, 0]
        t3_cls = tactile[:, 0]
        fused = torch.cat([clip_cls, t3_cls, ast_cls], dim=-1)
        fused = self.cls_proj(fused)
        prop = self.proprio_proj(proprio)
        return torch.cat([fused, prop], dim=-1)
