"""
QformerPolicy — CNN stems → Q-Former → DiT + flow matching.

Vision / tactile / audio use CNN tokenizers (shared log-mel for audio).
Learnable queries fuse a variable-length multimodal context. Sensor ablation
drops encoders and shrinks Q-Former context only; proprio is always kept.

Wrist and finger tokens get per-camera spatial embeddings and a shared
temporal embedding (one vector per history step).
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from vista.encoders.cnn import AudioCNNStem, TactileCNNStem, VisionCNNStem
from vista.fusion.qformer import QFormerEncoder
from vista.heads.dit import TransformerDenoiser
from vista.models.condition import Condition
from vista.models.sensor_ablation import SENSOR_GROUPS
from vista.objectives.flow_matching import FlowMatchingObjective
from vista.policy.base import PROPRIO_KEYS, BaseVistaPolicy
from vista.preproc.log_mel import mic_rows_to_waveform, shared_log_mel


class _ProprioToken(nn.Module):
    """Encode proprio into fusion tokens.

    With ``mean_over_time=True`` (Qformer): mean over H → 1 token.
    With ``mean_over_time=False`` (VisTA): encode each timestep → H tokens.
    """

    def __init__(
        self, in_dim: int, d_embed: int = 384, mean_over_time: bool = True
    ):
        super().__init__()
        self.mean_over_time = mean_over_time
        self.net = nn.Sequential(
            nn.Linear(in_dim, d_embed),
            nn.GELU(),
            nn.Linear(d_embed, d_embed),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean_over_time:
            # (B, T, D) → mean over T → (B, 1, D)
            if x.ndim == 3:
                x = x.mean(dim=1)
            return self.net(x).unsqueeze(1)
        # (B, T, D) → (B, T, d_embed); Linear acts on the last dim.
        if x.ndim != 3:
            raise ValueError(
                f"_ProprioToken(mean_over_time=False) expects (B, T, D), got {tuple(x.shape)}"
            )
        return self.net(x)


class _RgbPosEmbed(nn.Module):
    """
    Per-camera spatial table and optional shared temporal table.

    Holds Parameters so they appear in ``encoder_parameters()`` via module registration.
    """

    def __init__(
        self,
        d_embed: int,
        n_spatial: Optional[int] = None,
        n_obs_steps: Optional[int] = None,
        with_temporal: bool = False,
    ):
        super().__init__()
        self.spatial: Optional[nn.Parameter] = None
        self.temporal: Optional[nn.Parameter] = None
        if n_spatial is not None:
            self.spatial = nn.Parameter(torch.zeros(1, n_spatial, d_embed))
            nn.init.trunc_normal_(self.spatial, std=0.02)
        if with_temporal and n_obs_steps is not None:
            self.temporal = nn.Parameter(torch.zeros(1, n_obs_steps, 1, d_embed))
            nn.init.trunc_normal_(self.temporal, std=0.02)


def _image_size_from_shape_meta(shape_meta: dict, key: str) -> int:
    """Spatial side length from ``shape_meta['obs'][key]['shape']`` = [C, H, W]."""
    shape = shape_meta["obs"][key]["shape"]
    return int(shape[-1])


class QformerPolicy(BaseVistaPolicy):
    """Wrist / finger / audio / proprio → Q-Former → DiT + flow matching."""

    # Vision/tactile CNN total stride (5× stride-2).
    RGB_STRIDE = 32

    def __init__(
        self,
        shape_meta: dict,
        n_obs_steps: int = 2,
        sensor_group: str = "vta",
        d_embed: int = 384,
        n_queries: int = 128,
        qformer_layers: int = 6,
        n_cross_attn: int = 3,
        n_heads: int = 8,
        mlp_ratio: int = 2,
        dit_layers: int = 4,
        dit_mlp_ratio: int = 4,
        adaln_zero: bool = True,
        n_inference_steps: int = 10,
        sample_rate: int = 16000,
        mel_time_frames: int = 48,
    ):
        super().__init__(shape_meta=shape_meta, n_obs_steps=n_obs_steps)
        if sensor_group not in SENSOR_GROUPS:
            raise ValueError(
                f"Unknown sensor_group '{sensor_group}'. "
                f"Choose from {list(SENSOR_GROUPS)}"
            )
        self.sensor_group = sensor_group
        self.active_sensors = set(SENSOR_GROUPS[sensor_group])
        self.d_embed = d_embed
        self.n_obs_steps = n_obs_steps

        proprio_dim = sum(
            int(shape_meta["obs"][k]["shape"][0])
            for k in PROPRIO_KEYS
            if k in shape_meta["obs"]
        )

        encoders = []
        self.vision_tok: Optional[VisionCNNStem] = None
        self.tactile_tok: Optional[TactileCNNStem] = None
        self.audio_tok: Optional[AudioCNNStem] = None
        self.log_mel = None
        self.pos_camera: Optional[_RgbPosEmbed] = None
        self.pos_finger: Optional[_RgbPosEmbed] = None
        self.pos_temporal: Optional[_RgbPosEmbed] = None

        if "camera0_rgb" in self.active_sensors:
            self.vision_tok = VisionCNNStem(in_channels=3, d_embed=d_embed)
            encoders.append(self.vision_tok)
            n_spatial = self._spatial_tokens(
                _image_size_from_shape_meta(shape_meta, "camera0_rgb")
            )
            self.pos_camera = _RgbPosEmbed(d_embed, n_spatial=n_spatial)
            encoders.append(self.pos_camera)
        if "finger_rgb" in self.active_sensors:
            self.tactile_tok = TactileCNNStem(in_channels=3, d_embed=d_embed)
            encoders.append(self.tactile_tok)
            n_spatial = self._spatial_tokens(
                _image_size_from_shape_meta(shape_meta, "finger_rgb")
            )
            self.pos_finger = _RgbPosEmbed(d_embed, n_spatial=n_spatial)
            encoders.append(self.pos_finger)
        if "mic_0" in self.active_sensors:
            self.log_mel = shared_log_mel(
                sample_rate=sample_rate, time_frames=mel_time_frames
            )
            self.audio_tok = AudioCNNStem(in_channels=1, d_embed=d_embed)
            encoders.append(self.audio_tok)

        # Shared temporal embedding for wrist and finger (t=0 older, t=H-1 now).
        if self.vision_tok is not None or self.tactile_tok is not None:
            self.pos_temporal = _RgbPosEmbed(
                d_embed, n_obs_steps=n_obs_steps, with_temporal=True
            )
            encoders.append(self.pos_temporal)

        self.proprio_tok = _ProprioToken(proprio_dim, d_embed)
        encoders.append(self.proprio_tok)

        self.qformer = QFormerEncoder(
            d_embed=d_embed,
            n_queries=n_queries,
            n_layers=qformer_layers,
            n_heads=n_heads,
            n_cross_attn=n_cross_attn,
            mlp_ratio=mlp_ratio,
        )
        encoders.append(self.qformer)

        self.head = TransformerDenoiser(
            action_dim=self.action_dim,
            action_horizon=self.action_horizon,
            d_embed=d_embed,
            n_layer=dit_layers,
            n_head=n_heads,
            mlp_ratio=dit_mlp_ratio,
            adaln_zero=adaln_zero,
            cond_mode="tokens",
            max_cond_tokens=max(n_queries, 128),
        )
        self.objective = FlowMatchingObjective(n_inference_steps=n_inference_steps)

        self._register_encoder_modules(*encoders)
        self._register_head_modules(self.head)

    @classmethod
    def _spatial_tokens(cls, img_size: int) -> int:
        grid = img_size // cls.RGB_STRIDE
        if grid < 1:
            raise ValueError(
                f"Image size {img_size} too small for stride {cls.RGB_STRIDE}"
            )
        return grid * grid

    def _add_rgb_pos(
        self,
        tokens: torch.Tensor,
        spatial: nn.Parameter,
        n_obs: int,
    ) -> torch.Tensor:
        """
        Add per-frame spatial and shared temporal embeddings.

        ``tokens`` is ``(B, T * N, D)`` from the CNN stem; ``spatial`` is ``(1, N, D)``.
        """
        b, tn, d = tokens.shape
        n = spatial.shape[1]
        if tn != n_obs * n:
            raise ValueError(
                f"Token length {tn} != n_obs ({n_obs}) * spatial ({n})"
            )
        tok = tokens.view(b, n_obs, n, d)
        tok = tok + spatial.unsqueeze(1)  # (1, 1, N, D) broadcast
        assert self.pos_temporal is not None and self.pos_temporal.temporal is not None
        tok = tok + self.pos_temporal.temporal[:, :n_obs]
        return tok.view(b, tn, d)

    def _encode_audio(self, mic: torch.Tensor) -> torch.Tensor:
        assert self.log_mel is not None and self.audio_tok is not None
        wav = mic_rows_to_waveform(mic)
        mel = self.log_mel(wav)
        return self.audio_tok(mel)

    def encode_condition(self, obs: Dict[str, torch.Tensor]) -> Condition:
        tokens: Dict[str, torch.Tensor] = {}
        if self.vision_tok is not None:
            assert self.pos_camera is not None and self.pos_camera.spatial is not None
            raw = self.vision_tok(obs["camera0_rgb"])
            n_obs = obs["camera0_rgb"].shape[1]
            tokens["camera0_rgb"] = self._add_rgb_pos(
                raw, self.pos_camera.spatial, n_obs
            )
        if self.tactile_tok is not None:
            assert self.pos_finger is not None and self.pos_finger.spatial is not None
            raw = self.tactile_tok(obs["finger_rgb"])
            n_obs = obs["finger_rgb"].shape[1]
            tokens["finger_rgb"] = self._add_rgb_pos(
                raw, self.pos_finger.spatial, n_obs
            )
        if self.audio_tok is not None:
            tokens["mic_0"] = self._encode_audio(obs["mic_0"])
        tokens["proprio"] = self.proprio_tok(self._concat_proprio(obs))
        queries, _ = self.qformer(tokens)
        return Condition(tokens=queries, pad_mask=None)

    def compute_loss(self, batch: dict) -> Dict[str, torch.Tensor]:
        nobs = self._normalize_obs(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])
        cond = self.encode_condition(nobs)
        loss = self.objective.compute_loss(self.head, cond, nactions)
        return {"loss": loss}

    def predict_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
        fixed_action_prefix: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        del fixed_action_prefix
        nobs = self._normalize_obs(obs_dict)
        cond = self.encode_condition(nobs)
        b = next(iter(nobs.values())).shape[0]
        nsample = self.objective.predict(
            self.head,
            cond,
            (b, self.action_horizon, self.action_dim),
            device=self.device,
            dtype=self.dtype,
        )
        action = self.normalizer["action"].unnormalize(nsample)
        return {"action": action, "action_pred": action}
