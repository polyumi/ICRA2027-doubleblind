"""Concatenate modality tokens and project."""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from vista.fusion.base import Fusion


class ConcatProjFusion(Fusion):
    """Concat tokens along sequence dim then linear project."""

    def __init__(self, d_embed: int = 256):
        super().__init__(d_embed=d_embed)
        self.proj = nn.Linear(d_embed, d_embed)

    def forward(
        self,
        tokens_by_key: Dict[str, torch.Tensor],
        pad_masks: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        keys = sorted(tokens_by_key.keys())
        tokens = torch.cat([tokens_by_key[k] for k in keys], dim=1)
        tokens = self.proj(tokens)
        mask = None
        if pad_masks is not None:
            masks = [pad_masks[k] for k in keys if k in pad_masks]
            if masks:
                mask = torch.cat(masks, dim=1)
        return tokens, mask
