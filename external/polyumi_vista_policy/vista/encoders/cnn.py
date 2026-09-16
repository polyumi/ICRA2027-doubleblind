"""CNN stems for vision, tactile, and audio mel."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from vista.encoders.base import SensorEncoder


def _conv_bn_relu(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


def _tokens_from_feat(
    feat: torch.Tensor, batch_t: Optional[Tuple[int, int]] = None
) -> torch.Tensor:
    """Flatten Conv feature map ``(BT, C, H, W)`` to tokens; optional ``(B, T)`` reshape."""
    tokens = feat.flatten(2).transpose(1, 2)
    if batch_t is None:
        return tokens
    b, t = batch_t
    return tokens.reshape(b, t * tokens.shape[1], -1)


class VisionCNNStem(SensorEncoder):
    """
    CNN stem for 224×224 RGB → 7×7 tokens (49 / frame).

    Five stride-2 layers (total /32) so H=2 history yields 98 tokens, aligned
    with the audio stem (~96 tokens on 128×48 mel).
    """

    def __init__(self, in_channels: int = 3, d_embed: int = 256):
        super().__init__(d_embed=d_embed)
        self.stem = nn.Sequential(
            _conv_bn_relu(in_channels, 64, stride=2),
            _conv_bn_relu(64, 128, stride=2),
            _conv_bn_relu(128, 256, stride=2),
            _conv_bn_relu(256, d_embed, stride=2),
            _conv_bn_relu(d_embed, d_embed, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 5:
            b, t, c, h, w = x.shape
            feat = self.stem(x.reshape(b * t, c, h, w))
            return _tokens_from_feat(feat, (b, t))
        return _tokens_from_feat(self.stem(x))


class TactileCNNStem(SensorEncoder):
    """
    CNN stem for 224×224 tactile images → 7×7 tokens (49 / frame).

    Extra stride-1 first layer, then five stride-2 layers (same /32 as vision).
    """

    def __init__(self, in_channels: int = 3, d_embed: int = 256):
        super().__init__(d_embed=d_embed)
        self.stem = nn.Sequential(
            _conv_bn_relu(in_channels, 32, stride=1),
            _conv_bn_relu(32, 64, stride=2),
            _conv_bn_relu(64, 128, stride=2),
            _conv_bn_relu(128, 256, stride=2),
            _conv_bn_relu(256, d_embed, stride=2),
            _conv_bn_relu(d_embed, d_embed, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return VisionCNNStem.forward(self, x)


class AudioCNNStem(SensorEncoder):
    """
    CNN stem for shared log-mel ``(B, 1, 128, 48)`` → 16×6 = 96 tokens.

    Three stride-2 layers (not four) so audio token count matches H=2 vision (~98).
    """

    def __init__(self, in_channels: int = 1, d_embed: int = 256):
        super().__init__(d_embed=d_embed)
        mid = max(64, d_embed // 2)
        self.stem = nn.Sequential(
            _conv_bn_relu(in_channels, 64, stride=2),
            _conv_bn_relu(64, 128, stride=2),
            _conv_bn_relu(128, d_embed, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"AudioCNNStem expects log-mel (B, C, n_mels, time), got {tuple(x.shape)}"
            )
        return _tokens_from_feat(self.stem(x))
