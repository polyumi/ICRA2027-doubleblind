"""Self-attention over concatenated tokens."""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from vista.fusion.base import Fusion


class TransformerFusion(Fusion):
    """Self-attention transformer over concat modality tokens."""

    def __init__(
        self,
        d_embed: int = 256,
        n_layers: int = 2,
        n_heads: int = 4,
        mlp_ratio: int = 4,
    ):
        super().__init__(d_embed=d_embed)
        layer = nn.TransformerEncoderLayer(
            d_model=d_embed,
            nhead=n_heads,
            dim_feedforward=d_embed * mlp_ratio,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(
        self,
        tokens_by_key: Dict[str, torch.Tensor],
        pad_masks: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        keys = sorted(tokens_by_key.keys())
        tokens = torch.cat([tokens_by_key[k] for k in keys], dim=1)
        mask = None
        if pad_masks is not None:
            parts = [pad_masks[k] for k in keys if k in pad_masks]
            if parts:
                mask = torch.cat(parts, dim=1)
        if mask is not None:
            attn_mask = mask
            tokens = self.encoder(tokens, src_key_padding_mask=attn_mask)
        else:
            tokens = self.encoder(tokens)
        return tokens, mask
