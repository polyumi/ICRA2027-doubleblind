"""
CoordConv ResNet encoders adapted from See, Hear, and Feel.

Source: JunzheJosephZhu/see_hear_feel src/models/encoders.py (MIT).
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    from torchvision.models import resnet18, resnet50, resnet101
except ImportError as exc:  # pragma: no cover
    raise ImportError("torchvision required for SHF ResNet encoders") from exc


from vista.preproc.log_mel import mic_rows_to_waveform, shared_log_mel

# The third element was an FX node name for create_feature_extractor; it is gone because the trunk
# is now truncated directly (see _ResNetEncoder). Both named the final ReLU of the last block --
# resnet18's "layer4.1.relu_1" and resnet50/101's "layer4.2.relu_2" -- which IS layer4's output,
# so dropping the last two children (avgpool, fc) gives the identical tensor. Verified bit-identical
# for resnet18 and resnet50.
_RESNET_BUILDERS = {
    "resnet18": (resnet18, 512),
    "resnet50": (resnet50, 2048),
    "resnet101": (resnet101, 2048),
}


def _make_resnet(name: str = "resnet18"):
    if name not in _RESNET_BUILDERS:
        raise ValueError(
            f"Unknown SHF backbone '{name}'. Choose from {list(_RESNET_BUILDERS)}"
        )
    ctor, feat_dim = _RESNET_BUILDERS[name]
    try:
        backbone = ctor(weights=None)
    except TypeError:
        backbone = ctor(pretrained=False)
    return backbone, feat_dim


class CoordConv(nn.Module):
    """Add coordinates in [-1, 1] to an image (CoordConv)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4
        h, w = x.shape[2:]
        type_dev = dict(dtype=x.dtype, device=x.device)
        lin_h = torch.linspace(-1, 1, h, **type_dev)[:, None]
        lin_w = torch.linspace(-1, 1, w, **type_dev)[None, :]
        ones_h = x.new_ones((h, 1))
        ones_w = x.new_ones((1, w))
        new_maps = torch.stack((lin_h * ones_w, lin_w * ones_h), dim=0)[None]
        new_maps = new_maps.repeat(x.size(0), 1, 1, 1)
        return torch.cat((x, new_maps), dim=1)


class _ResNetEncoder(nn.Module):
    """ResNet trunk with CoordConv input and AdaptiveAvgPool → vector."""

    def __init__(self, in_channels: int, out_dim: int, backbone: str = "resnet18"):
        super().__init__()
        net, feat_dim = _make_resnet(backbone)
        net.conv1 = nn.Conv2d(
            in_channels + 2,
            64,
            kernel_size=7,
            stride=1,
            padding=3,
            bias=False,
        )
        # Plain nn.Sequential rather than torchvision's create_feature_extractor. That returns a
        # DualGraphModule, and copy.deepcopy of one drops `eval_graph` on torchvision < 0.21
        # (pytorch/vision#8634, fixed by #8708) -- which EMAModel does at construction, so every
        # SHF run died with "'GraphModule' object has no attribute 'eval_graph'". Dropping the last
        # two children (avgpool, fc) yields exactly the tensor the FX node did, with no FX involved.
        self.feature_extractor = nn.Sequential(*list(net.children())[:-2])
        self.coord_conv = CoordConv()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(feat_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.coord_conv(x)
        x = self.feature_extractor(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class SHFVisionEncoder(nn.Module):
    """SeeHearFeel vision / tactile ResNet encoder (RGB → vector)."""

    def __init__(self, out_dim: int = 128, backbone: str = "resnet18"):
        super().__init__()
        self.encoder = _ResNetEncoder(
            in_channels=3, out_dim=out_dim, backbone=backbone
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, 3, H, W) or (B, 3, H, W)
        if x.ndim == 5:
            b, n, c, h, w = x.shape
            x = x.reshape(b * n, c, h, w)
            out = self.encoder(x)
            return out.reshape(b, n * self.out_dim)
        return self.encoder(x)


class SHFAudioEncoder(nn.Module):
    """
    SeeHearFeel audio: contiguous mic rows → waveform → log-mel → CoordConv ResNet.

    Input ``(B, audio_horizon, 536)`` (or flat ``(B, T)``). One clip → one vector
    of size ``out_dim`` (typically ``d_embed * n_obs_history`` to match vision stack).
    Uses the shared 128-mel 25/10 ms frontend; ResNet adaptive-pools native time.
    """

    def __init__(
        self,
        out_dim: int = 128,
        sample_rate: int = 16000,
        backbone: str = "resnet18",
    ):
        super().__init__()
        self.out_dim = out_dim
        self.sample_rate = sample_rate
        # Native time length — AdaptiveAvgPool handles variable T.
        self.log_mel = shared_log_mel(sample_rate=sample_rate, time_frames=None)
        # Spec_Encoder: 1-channel mel + CoordConv → 3 channels into ResNet.
        self.encoder = _ResNetEncoder(
            in_channels=1, out_dim=out_dim, backbone=backbone
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        wav = mic_rows_to_waveform(x)
        mel = self.log_mel(wav)  # (B, 1, 128, T)
        return self.encoder(mel)


class SHFProprioEncoder(nn.Module):
    """MLP over flattened proprio history → vector of size out_dim * n_stack."""

    def __init__(self, input_dim: int, out_dim: int = 128, n_obs_steps: int = 2):
        super().__init__()
        self.out_dim = out_dim
        self.n_obs_steps = n_obs_steps
        self.net = nn.Sequential(
            nn.Linear(input_dim * n_obs_steps, out_dim * n_obs_steps),
            nn.GELU(),
            nn.Linear(out_dim * n_obs_steps, out_dim * n_obs_steps),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D)
        b = x.shape[0]
        return self.net(x.reshape(b, -1))
