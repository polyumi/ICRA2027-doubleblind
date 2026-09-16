"""Per-key image normalization front-end."""

from typing import Dict, Optional

import torch
import torch.nn as nn


class ImageFrontEnd(nn.Module):
    """
    Apply per-key image normalization before encoders.

    Modes: ``pretrained`` (timm resolve_data_config), ``minmax`` (x*2-1), ``none``.
    """

    def __init__(
        self,
        key_configs: Dict[str, dict],
        default_mode: str = "pretrained",
    ):
        super().__init__()
        self.key_configs = key_configs
        self.default_mode = default_mode
        self._timm_stats: Dict[str, dict] = {}

    def _resolve_pretrained(self, key: str, model_name: str) -> dict:
        if key in self._timm_stats:
            return self._timm_stats[key]
        try:
            import timm
        except ImportError as exc:
            raise ImportError("timm required for pretrained image_norm") from exc
        dummy = timm.create_model(model_name, pretrained=False, num_classes=0)
        cfg = timm.data.resolve_data_config({}, model=dummy)
        self._timm_stats[key] = cfg
        return cfg

    def _normalize_key(self, x: torch.Tensor, key: str) -> torch.Tensor:
        cfg = self.key_configs.get(key, {})
        mode = cfg.get("image_norm", self.default_mode)
        if mode == "none":
            return x
        if mode == "minmax":
            return x * 2.0 - 1.0
        if mode == "pretrained":
            timm_cfg = self._resolve_pretrained(key, cfg.get("timm_model", "resnet18"))
            mean = torch.tensor(timm_cfg["mean"], device=x.device, dtype=x.dtype)
            std = torch.tensor(timm_cfg["std"], device=x.device, dtype=x.dtype)
            if x.ndim == 5:
                # B, T, C, H, W
                view = (1, 1, 3, 1, 1)
            elif x.ndim == 4:
                # B, C, H, W
                view = (1, 3, 1, 1)
            else:
                raise ValueError(f"Expected 4D or 5D image, got {x.ndim}D")
            mean = mean.view(*view)
            std = std.view(*view)
            return (x - mean) / std
        raise ValueError(f"Unknown image_norm mode: {mode}")

    def forward(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = dict(obs)
        for key in self.key_configs:
            if key not in obs:
                continue
            out[key] = self._normalize_key(obs[key], key)
        return out
