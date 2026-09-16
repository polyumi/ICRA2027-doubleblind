"""Reconstruction heads for masked pretraining (VisTA port)."""

from typing import Dict, Optional

import torch
import torch.nn as nn


class ReconHead(nn.Module):
    """Decode fused tokens back to pixel / spectrogram targets."""

    def __init__(
        self,
        d_embed: int = 256,
        out_channels: int = 3,
        spatial_size: int = 224,
    ):
        super().__init__()
        self.spatial_size = spatial_size
        self.decoder = nn.Sequential(
            nn.Linear(d_embed, d_embed * 4),
            nn.GELU(),
            nn.Linear(d_embed * 4, out_channels * spatial_size * spatial_size),
        )
        self.out_channels = out_channels

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: B, N, D — pool then decode
        pooled = tokens.mean(dim=1)
        out = self.decoder(pooled)
        h = w = self.spatial_size
        return out.reshape(pooled.shape[0], self.out_channels, h, w)


class MultiKeyReconHead(nn.Module):
    """Per-key lightweight decoders."""

    def __init__(self, key_specs: Dict[str, dict], d_embed: int = 256):
        super().__init__()
        self.heads = nn.ModuleDict()
        for key, spec in key_specs.items():
            self.heads[key] = ReconHead(
                d_embed=d_embed,
                out_channels=spec.get("channels", 3),
                spatial_size=spec.get("size", 224),
            )

    def forward(
        self,
        tokens_by_key: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        return {k: self.heads[k](tokens_by_key[k]) for k in tokens_by_key if k in self.heads}
