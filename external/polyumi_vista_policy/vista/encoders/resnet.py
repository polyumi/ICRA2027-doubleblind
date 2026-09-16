"""timm ResNet encoder with spatial tokens."""

import torch
import torch.nn as nn

from vista.encoders.base import SensorEncoder


class ResNetEncoder(SensorEncoder):
    """ResNet backbone that outputs spatial feature tokens."""

    def __init__(
        self,
        model_name: str = "resnet18",
        pretrained: bool = True,
        d_embed: int = 256,
    ):
        super().__init__(d_embed=d_embed)
        try:
            import timm
        except ImportError as exc:
            raise ImportError("timm required for ResNetEncoder") from exc
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(-1,),
        )
        feat_dim = self.backbone.feature_info.channels()[-1]
        self.proj = nn.Conv2d(feat_dim, d_embed, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 5:
            b, t, c, h, w = x.shape
            x = x.reshape(b * t, c, h, w)
            feat = self.backbone(x)[0]
            feat = self.proj(feat)
            tokens = feat.flatten(2).transpose(1, 2)
            return tokens.reshape(b, t * tokens.shape[1], -1)
        feat = self.backbone(x)[0]
        feat = self.proj(feat)
        return feat.flatten(2).transpose(1, 2)
