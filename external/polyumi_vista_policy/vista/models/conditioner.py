"""Observation conditioner: encode and fuse multimodal observations."""

from typing import Dict, Optional

import torch
import torch.nn as nn

from vista.encoders.registry import build_encoder
from vista.fusion.registry import build_fusion
from vista.models.condition import Condition
from vista.preproc.audio_features import AudioFrontEnd
from vista.preproc.image_features import ImageFrontEnd


class ObservationConditioner(nn.Module):
    """Per-key preprocessing, encoding, and fusion into ``Condition``."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.d_embed = cfg.get("d_embed", 256)
        sensor_cfg = cfg.get("sensors", {})
        self.sensor_keys = list(sensor_cfg.keys())

        image_keys = {
            k: v for k, v in sensor_cfg.items() if v.get("modality") in ("rgb", "vision")
        }
        self.image_front = (
            ImageFrontEnd(image_keys, default_mode=cfg.get("image_norm", "pretrained"))
            if image_keys
            else None
        )
        audio_keys = {k: v for k, v in sensor_cfg.items() if v.get("modality") == "audio"}
        self.audio_front = None
        if audio_keys:
            try:
                self.audio_front = AudioFrontEnd(**cfg.get("audio_front", {}))
            except ImportError as exc:
                import warnings

                warnings.warn(
                    f"AudioFrontEnd unavailable ({exc}); mic_0 will pass through raw until torchaudio is installed."
                )

        self.encoders = nn.ModuleDict()
        for key, scfg in sensor_cfg.items():
            enc_name = scfg.get("encoder", "vit")
            enc_kw = dict(scfg.get("encoder_kwargs", {}))
            enc_kw.setdefault("d_embed", self.d_embed)
            if scfg.get("modality") == "proprio":
                enc_kw.setdefault("input_dim", scfg.get("dim", 10))
            if scfg.get("modality") in ("tactile",) and enc_name in ("tactile_cnn", "vision_cnn", "patch_stem"):
                enc_kw.setdefault("in_channels", scfg.get("channels", 3))
            self.encoders[key] = build_encoder(enc_name, **enc_kw)

        fusion_name = cfg.get("fusion", "concat_proj")
        fusion_kw = dict(cfg.get("fusion_kwargs", {}))
        fusion_kw.setdefault("d_embed", self.d_embed)
        self.fusion = build_fusion(fusion_name, **fusion_kw)

    def encode_obs(self, obs: Dict[str, torch.Tensor]) -> Condition:
        proc = dict(obs)
        if self.image_front is not None:
            proc = self.image_front(proc)
        if self.audio_front is not None:
            proc = self.audio_front(proc)

        tokens_by_key = {}
        for key in self.sensor_keys:
            if key not in proc:
                continue
            x = proc[key]
            tokens = self.encoders[key](x)
            tokens_by_key[key] = tokens

        fused, pad_mask = self.fusion(tokens_by_key)
        return Condition(
            tokens=fused,
            pad_mask=pad_mask,
            _pool_fn=self.fusion.pool,
        )

    def forward(self, obs: Dict[str, torch.Tensor]) -> Condition:
        return self.encode_obs(obs)
