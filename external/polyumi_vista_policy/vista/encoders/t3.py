"""
Transferable Tactile Transformer (T3) tactile encoder.

Architecture follows alanzjl/t3 (Zhao et al., FoTa / T3):
sensor-specific ViT stem (``L_enc`` blocks) + shared Transformer trunk
(``L_tru`` blocks). Pretrained weights live on the FoTa HuggingFace dataset:

  https://huggingface.co/datasets/alanz-mit/FoundationTactile
  models/t3_{tiny|small|medium|large}/encoders/{sensor}.pth
  models/t3_{tiny|small|medium|large}/trunk.pth

Paper sizes (Appendix Table 1): tiny 192/3/3/9, small 384/6/3/9,
medium 768/12/3/9, large 1024/8/3/9.
"""

from __future__ import annotations

from functools import partial
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from vista.encoders.base import SensorEncoder

# (embed_dim, num_heads, encoder_depth, trunk_depth)
T3_SIZE_PRESETS: Dict[str, Tuple[int, int, int, int]] = {
    "tiny": (192, 3, 3, 9),
    "small": (384, 6, 3, 9),
    "medium": (768, 12, 3, 9),
    "large": (1024, 8, 3, 9),
}

DEFAULT_HF_DATASET = "alanz-mit/FoundationTactile"


def _t3_norm_layer():
    return partial(nn.LayerNorm, eps=1e-6)


class _T3ViTStem(nn.Module):
    """Sensor-specific ViT stem (patch embed + L_enc blocks, no final norm)."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        depth: int,
        img_size: int = 224,
        patch_size: int = 16,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        try:
            import timm.models.vision_transformer as timm_vit
        except ImportError as exc:
            raise ImportError("timm required for T3Encoder") from exc

        norm_layer = _t3_norm_layer()
        self.patch_embed = timm_vit.PatchEmbed(img_size, patch_size, 3, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(0.0)
        self.blocks = nn.ModuleList(
            [
                timm_vit.Block(
                    embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        return x


class _T3TransformerTrunk(nn.Module):
    """Shared trunk: L_tru ViT blocks + LayerNorm (pooling_type=none)."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        depth: int,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        try:
            import timm.models.vision_transformer as timm_vit
        except ImportError as exc:
            raise ImportError("timm required for T3Encoder") from exc

        norm_layer = _t3_norm_layer()
        self.blocks = nn.ModuleList(
            [
                timm_vit.Block(
                    embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer
                )
                for _ in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


class T3Encoder(SensorEncoder):
    """
    FoTa T3 tactile backbone: sensor stem → shared trunk → project to ``d_embed``.

    Returns patch tokens ``(B, N, d_embed)`` or ``(B, T*N, d_embed)`` for history.
    """

    def __init__(
        self,
        d_embed: int = 768,
        pretrained: bool = True,
        t3_size: str = "small",
        t3_sensor: str = "svelte",
        hf_repo: str = DEFAULT_HF_DATASET,
        hf_repo_type: str = "dataset",
        t3_ckpt: Optional[str] = None,
        trunk_ckpt: Optional[str] = None,
        freeze: bool = False,
        # Deprecated aliases kept for older YAML / call sites.
        model_name: str = "",
        hf_filename: str = "",
    ):
        del model_name, hf_filename
        super().__init__(d_embed=d_embed)
        if t3_size not in T3_SIZE_PRESETS:
            raise ValueError(
                f"Unknown t3_size '{t3_size}'. Choose from {list(T3_SIZE_PRESETS)}"
            )
        embed_dim, num_heads, enc_depth, tru_depth = T3_SIZE_PRESETS[t3_size]
        self.t3_size = t3_size
        self.t3_sensor = t3_sensor
        self.embed_dim = embed_dim

        self.stem = _T3ViTStem(
            embed_dim=embed_dim, num_heads=num_heads, depth=enc_depth
        )
        self.trunk = _T3TransformerTrunk(
            embed_dim=embed_dim, num_heads=num_heads, depth=tru_depth
        )
        self.proj = (
            nn.Linear(embed_dim, d_embed) if embed_dim != d_embed else nn.Identity()
        )

        if t3_ckpt or trunk_ckpt:
            self._load_local(t3_ckpt, trunk_ckpt)
        elif pretrained and hf_repo:
            self._load_hf_fota(
                repo=hf_repo,
                repo_type=hf_repo_type,
                size=t3_size,
                sensor=t3_sensor,
            )

        if freeze:
            for p in self.parameters():
                p.requires_grad = False

    def _load_state(self, module: nn.Module, path: str, label: str) -> None:
        state = torch.load(path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        missing, unexpected = module.load_state_dict(state, strict=True)
        if missing or unexpected:
            raise RuntimeError(
                f"T3 {label} load mismatch from {path}: "
                f"missing={missing} unexpected={unexpected}"
            )

    def _load_local(
        self, encoder_ckpt: Optional[str], trunk_ckpt: Optional[str]
    ) -> None:
        if encoder_ckpt:
            self._load_state(self.stem, encoder_ckpt, "encoder")
        if trunk_ckpt:
            self._load_state(self.trunk, trunk_ckpt, "trunk")

    def _load_hf_fota(
        self, repo: str, repo_type: str, size: str, sensor: str
    ) -> None:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError(
                "huggingface_hub required to download FoTa T3 weights"
            ) from exc

        enc_name = f"models/t3_{size}/encoders/{sensor}.pth"
        tru_name = f"models/t3_{size}/trunk.pth"
        enc_path = hf_hub_download(
            repo_id=repo, repo_type=repo_type, filename=enc_name
        )
        tru_path = hf_hub_download(
            repo_id=repo, repo_type=repo_type, filename=tru_name
        )
        self._load_state(self.stem, enc_path, f"encoder({sensor})")
        self._load_state(self.trunk, tru_path, "trunk")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B, C, H, W or B, T, C, H, W
        if x.ndim == 5:
            b, t, c, h, w = x.shape
            x = x.reshape(b * t, c, h, w)
            tokens = self.proj(self.trunk(self.stem(x)))
            return tokens.reshape(b, t * tokens.shape[1], -1)
        return self.proj(self.trunk(self.stem(x)))
