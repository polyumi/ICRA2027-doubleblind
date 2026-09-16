"""timm ViT encoder returning patch tokens."""

from typing import Optional

import torch
import torch.nn as nn

from vista.encoders.base import SensorEncoder


class TimmViTEncoder(SensorEncoder):
    """Wrap a timm ViT and return patch tokens (no classification head)."""

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        d_embed: int = 256,
        freeze: bool = False,
    ):
        super().__init__(d_embed=d_embed)
        try:
            import timm
        except ImportError as exc:
            raise ImportError("timm required for TimmViTEncoder") from exc
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
        )
        feat_dim = self.backbone.num_features
        self.proj = nn.Linear(feat_dim, d_embed) if feat_dim != d_embed else nn.Identity()
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B, C, H, W or B, T, C, H, W
        if x.ndim == 5:
            b, t, c, h, w = x.shape
            x = x.reshape(b * t, c, h, w)
            tokens = self.backbone.forward_features(x)
            if tokens.ndim == 3:
                out = self.proj(tokens)
            else:
                out = self.proj(tokens.unsqueeze(1))
            return out.reshape(b, t * out.shape[1], -1)
        tokens = self.backbone.forward_features(x)
        if tokens.ndim == 2:
            tokens = tokens.unsqueeze(1)
        return self.proj(tokens)
