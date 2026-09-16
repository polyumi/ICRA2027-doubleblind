"""Independent TransformerEncoder per modality, then concat (no cross-sensor attn)."""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn

from vista.fusion.base import Fusion


class PerSensorTransformerFusion(Fusion):
    """Run a separate TransformerEncoder on each modality, then concat.

    Same concat key order as ``TransformerFusion`` (sorted keys). Isolates
    cross-modal mixing while preserving DiT conditioning length.
    """

    def __init__(
        self,
        sensor_keys: Iterable[str],
        d_embed: int = 256,
        n_layers: int = 2,
        n_heads: int = 4,
        mlp_ratio: int = 4,
    ):
        super().__init__(d_embed=d_embed)
        keys = tuple(sensor_keys)
        if not keys:
            raise ValueError("PerSensorTransformerFusion requires at least one sensor key")
        if len(set(keys)) != len(keys):
            raise ValueError(f"Duplicate sensor keys: {keys}")

        encoders = {}
        for key in keys:
            layer = nn.TransformerEncoderLayer(
                d_model=d_embed,
                nhead=n_heads,
                dim_feedforward=d_embed * mlp_ratio,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            encoders[key] = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.encoders = nn.ModuleDict(encoders)

    def forward(
        self,
        tokens_by_key: Dict[str, torch.Tensor],
        pad_masks: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        keys = sorted(tokens_by_key.keys())
        missing = [k for k in keys if k not in self.encoders]
        if missing:
            raise KeyError(
                f"No per-sensor encoder for {missing}. "
                f"Built for {sorted(self.encoders.keys())}"
            )
        unused = [k for k in self.encoders.keys() if k not in tokens_by_key]
        if unused:
            raise KeyError(
                f"Per-sensor encoders unused at forward: {unused}. "
                f"Got tokens for {keys}"
            )

        outs = []
        mask_parts = []
        for key in keys:
            tok = tokens_by_key[key]
            enc = self.encoders[key]
            if pad_masks is not None and key in pad_masks:
                m = pad_masks[key]
                outs.append(enc(tok, src_key_padding_mask=m))
                mask_parts.append(m)
            else:
                outs.append(enc(tok))

        fused = torch.cat(outs, dim=1)
        mask = torch.cat(mask_parts, dim=1) if mask_parts else None
        return fused, mask
