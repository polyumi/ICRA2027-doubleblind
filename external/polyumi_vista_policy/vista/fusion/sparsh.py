"""Sparsh-style tactile-centric fusion (Vista port)."""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from vista.fusion.base import Fusion


class SparshFusionEncoder(Fusion):
    """
    Tactile query attends vision/audio keys then self-attn refines.

    Expects ``tactile`` or ``finger_rgb`` as primary key when present.
    """

    def __init__(self, d_embed: int = 256, n_heads: int = 4, n_layers: int = 2):
        super().__init__(d_embed=d_embed)
        self.cross = nn.MultiheadAttention(d_embed, n_heads, batch_first=True)
        layer = nn.TransformerEncoderLayer(
            d_model=d_embed,
            nhead=n_heads,
            dim_feedforward=d_embed * 4,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.self_attn = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_embed)

    def _primary_key(self, keys):
        for candidate in ("tactile", "finger_rgb", "vision", "rgb"):
            if candidate in keys:
                return candidate
        return keys[0]

    def forward(
        self,
        tokens_by_key: Dict[str, torch.Tensor],
        pad_masks: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        keys = sorted(tokens_by_key.keys())
        primary = self._primary_key(keys)
        query = tokens_by_key[primary]
        others = torch.cat(
            [tokens_by_key[k] for k in keys if k != primary],
            dim=1,
        )
        if others.numel() == 0:
            fused = query
        else:
            attn_out, _ = self.cross(query, others, others)
            fused = self.norm(query + attn_out)
        fused = self.self_attn(fused)
        mask = pad_masks.get(primary) if pad_masks else None
        return fused, mask
