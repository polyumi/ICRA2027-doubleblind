"""Base sensor encoder interface."""

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn


class SensorEncoder(nn.Module, ABC):
    """Encode a single modality into token sequences."""

    def __init__(self, d_embed: int = 256):
        super().__init__()
        self.d_embed = d_embed

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return tokens ``(B, N, D)``."""

    def pooled(
        self,
        tokens: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Mean-pool tokens or take CLS (first token)."""
        if tokens.shape[1] == 1:
            return tokens[:, 0]
        if pad_mask is None:
            return tokens.mean(dim=1)
        weights = (~pad_mask).float().unsqueeze(-1)
        return (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
