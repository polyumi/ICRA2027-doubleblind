"""Patch embedding stem for fixed-resolution vision."""

import torch
import torch.nn as nn

from vista.encoders.base import SensorEncoder


class PatchStem(SensorEncoder):
    """Conv patchify stem for 224x224 inputs."""

    def __init__(
        self,
        in_channels: int = 3,
        patch_size: int = 16,
        img_size: int = 224,
        d_embed: int = 256,
    ):
        super().__init__(d_embed=d_embed)
        self.patch_size = patch_size
        self.grid = img_size // patch_size
        self.proj = nn.Conv2d(
            in_channels,
            d_embed,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.pos = nn.Parameter(torch.zeros(1, self.grid * self.grid, d_embed))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 5:
            b, t, c, h, w = x.shape
            x = x.reshape(b * t, c, h, w)
            tokens = self.proj(x).flatten(2).transpose(1, 2) + self.pos
            return tokens.reshape(b, t * tokens.shape[1], -1)
        tokens = self.proj(x).flatten(2).transpose(1, 2) + self.pos
        return tokens
