"""Bidirectional cross-attention fusion rounds."""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from vista.fusion.base import Fusion


class CrossAttentionFusion(Fusion):
    """Alternate cross-attention between modality pairs for ``n_rounds``."""

    def __init__(self, d_embed: int = 256, n_heads: int = 4, n_rounds: int = 2):
        super().__init__(d_embed=d_embed)
        self.n_rounds = n_rounds
        self.layers = nn.ModuleList(
            [
                nn.MultiheadAttention(d_embed, n_heads, batch_first=True)
                for _ in range(n_rounds)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(d_embed) for _ in range(n_rounds)])

    def forward(
        self,
        tokens_by_key: Dict[str, torch.Tensor],
        pad_masks: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        keys = sorted(tokens_by_key.keys())
        streams = [tokens_by_key[k] for k in keys]
        for layer, norm in zip(self.layers, self.norms):
            updated = []
            for i, query in enumerate(streams):
                others = [streams[j] for j in range(len(streams)) if j != i]
                kv = torch.cat(others, dim=1) if others else query
                attn_out, _ = layer(query, kv, kv)
                updated.append(norm(query + attn_out))
            streams = updated
        fused = torch.cat(streams, dim=1)
        mask = None
        if pad_masks is not None:
            mask = torch.cat([pad_masks[k] for k in keys if k in pad_masks], dim=1)
        return fused, mask
