"""
Multimodal Bottleneck Transformer (MBT) fusion.

Reimplementation of the bottleneck fusion strategy from Nagrani et al. 2021
("Attention bottlenecks for multimodal fusion"), as used by Sparsh-X.
Do NOT copy facebookresearch/sparsh-multisensory-touch (CC-BY-NC-4.0).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn


class MBTFusion(nn.Module):
    """
    Per-modality self-attention for ``fusion_layer`` blocks, then bottleneck
    fusion for the remaining depth: each modality attends with shared bottleneck
    tokens; bottlenecks are averaged across modalities after each block.
    """

    def __init__(
        self,
        modals: List[str],
        embed_dim: int = 256,
        depth: int = 8,
        fusion_layer: int = 4,
        num_heads: int = 8,
        num_bottlenecks: int = 4,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        assert 0 <= fusion_layer <= depth
        self.modals = list(modals)
        self.embed_dim = embed_dim
        self.fusion_layer = fusion_layer
        self.num_bottlenecks = num_bottlenecks

        def _block() -> nn.TransformerEncoderLayer:
            return nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=0.0,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )

        self.uni_blocks = nn.ModuleList(
            [
                nn.ModuleDict({m: _block() for m in self.modals})
                for _ in range(fusion_layer)
            ]
        )
        self.fusion_blocks = nn.ModuleList(
            [
                nn.ModuleDict({m: _block() for m in self.modals})
                for _ in range(depth - fusion_layer)
            ]
        )
        self.bottleneck = nn.Parameter(
            torch.zeros(1, num_bottlenecks, embed_dim)
        )
        nn.init.trunc_normal_(self.bottleneck, std=0.02)
        self.norm = nn.ModuleDict(
            {m: nn.LayerNorm(embed_dim) for m in self.modals}
        )

    def forward(
        self,
        tokens_by_modal: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            tokens_by_modal: modal → (B, N_m, D)
        Returns:
            modal → normalized tokens (B, N_m, D)
        """
        xs = {m: tokens_by_modal[m] for m in self.modals}
        for blocks in self.uni_blocks:
            xs = {m: blocks[m](x) for m, x in xs.items()}

        bsz = next(iter(xs.values())).shape[0]
        bottleneck = self.bottleneck.expand(bsz, -1, -1)
        for blocks in self.fusion_blocks:
            updated = {}
            bottle_outs = []
            for m, x in xs.items():
                cat = torch.cat([bottleneck, x], dim=1)
                out = blocks[m](cat)
                bottle_outs.append(out[:, : self.num_bottlenecks])
                updated[m] = out[:, self.num_bottlenecks :]
            bottleneck = torch.stack(bottle_outs, dim=0).mean(dim=0)
            xs = updated

        return {m: self.norm[m](x) for m, x in xs.items()}
