"""Fusion module base class."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class Fusion(nn.Module, ABC):
    """Fuse per-modality token dict into a single token sequence."""

    def __init__(self, d_embed: int = 256):
        super().__init__()
        self.d_embed = d_embed

    @abstractmethod
    def forward(
        self,
        tokens_by_key: Dict[str, torch.Tensor],
        pad_masks: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return fused tokens ``(B, N, D)`` and optional pad mask."""

    def pool(
        self,
        tokens: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if pad_mask is None:
            return tokens.mean(dim=1)
        weights = (~pad_mask).float().unsqueeze(-1)
        return (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
