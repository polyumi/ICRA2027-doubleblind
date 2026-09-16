#!/usr/bin/env python3
"""
Water-level classifier: per-modality CNN encoders -> concatenate -> MLP head.

Mirrors the shape of the Vista stems (vista/encoders/cnn.py) rather than inventing a new one:
stride-2 conv/BN/ReLU blocks down to a small spatial map, global-pooled to one embedding per
modality. The four experimental variants are the SAME class with different encoders attached, so
a difference between them is about the sensor and not about the architecture.

* vision / tactile -- ``(B, K, 3, IMG, IMG)``. The K sampled frames are folded into the batch,
  encoded independently, then mean-pooled back. Pooling rather than concatenating over K keeps
  the head's input size independent of how many frames were sampled, and makes the encoder
  see one frame at a time, which is all a water level needs.
* audio -- ``(B, 1, N_MELS, MEL_T)``, a single log-mel image per trial, encoded as a 2-D image.
  This is the standard way to hand a spectrogram to a CNN and matches AudioCNNStem's treatment.

CAPACITY WARNING, stated once here rather than in every caller: this has far more parameters than
the ~60 trials it is trained on, so it reaches 100% training accuracy on any modality set
including pure noise. That is expected and is why train accuracy is not reported as a result.
Only held-out accuracy distinguishes the variants.
"""

from __future__ import annotations

import torch
from torch import nn

MODALITIES = ('vision', 'tactile', 'audio')


def _conv_bn_relu(cin: int, cout: int, stride: int = 2) -> nn.Sequential:
    """One downsampling block, the same shape the Vista stems use."""
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class ImageEncoder(nn.Module):
    """K frames -> one embedding. Frames fold into the batch and mean-pool back out."""

    def __init__(self, d_embed: int = 128, in_channels: int = 3, width: int = 32):
        """Four stride-2 blocks (total /16), then global average pool to `d_embed`."""
        super().__init__()
        self.stem = nn.Sequential(
            _conv_bn_relu(in_channels, width),
            _conv_bn_relu(width, width * 2),
            _conv_bn_relu(width * 2, width * 4),
            _conv_bn_relu(width * 4, d_embed),
        )
        self.d_embed = d_embed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, K, C, H, W) -> (B, d_embed)."""
        b, k = x.shape[:2]
        z = self.stem(x.reshape(b * k, *x.shape[2:]))
        z = z.mean(dim=(-2, -1))  # global average pool
        return z.reshape(b, k, -1).mean(1)  # pool over the K sampled frames


class AudioEncoder(nn.Module):
    """One log-mel image per trial -> one embedding."""

    def __init__(self, d_embed: int = 128, width: int = 32):
        """Four stride-2 blocks over the (mel x time) plane, then global average pool."""
        super().__init__()
        self.stem = nn.Sequential(
            _conv_bn_relu(1, width),
            _conv_bn_relu(width, width * 2),
            _conv_bn_relu(width * 2, width * 4),
            _conv_bn_relu(width * 4, d_embed),
        )
        self.d_embed = d_embed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 1, N_MELS, MEL_T) -> (B, d_embed)."""
        return self.stem(x).mean(dim=(-2, -1))


class WaterClassifier(nn.Module):
    """Encoders for the chosen modalities, concatenated into an MLP head."""

    def __init__(
        self,
        modalities: tuple[str, ...],
        n_classes: int = 3,
        d_embed: int = 128,
        hidden: int = 256,
        dropout: float = 0.5,
    ):
        """Attach one encoder per named modality; the head sizes itself from how many there are."""
        super().__init__()
        unknown = set(modalities) - set(MODALITIES)
        if unknown:
            raise ValueError(f'unknown modalities: {sorted(unknown)}')
        self.modalities = tuple(modalities)
        self.encoders = nn.ModuleDict(
            {m: (AudioEncoder(d_embed) if m == 'audio' else ImageEncoder(d_embed)) for m in self.modalities}
        )
        # Dropout is the only thing standing between this head and memorising 60 trials, so it is
        # deliberately heavy; the encoders are small for the same reason.
        self.head = nn.Sequential(
            nn.Linear(d_embed * len(self.modalities), hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode each attached modality, concatenate the embeddings, classify."""
        z = [self.encoders[m](batch[m]) for m in self.modalities]
        return self.head(torch.cat(z, dim=-1))

    def n_parameters(self) -> int:
        """Total trainable parameters, for the capacity note in the training report."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
