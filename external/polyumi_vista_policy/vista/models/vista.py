"""
VisTAPolicy — CNN stems → Transformer fusion → DiT + flow matching / DDPM.

Same tokenization as QformerPolicy (CNN stems, spatial/temporal embeds, shared
log-mel), but fuses the full high-token context with a TransformerEncoder
instead of a Q-Former bottleneck. Proprio is encoded per history step (H
tokens, no mean-pool). Sensor ablation drops encoders and shrinks context;
proprio is always kept.

Ablations (Hydra-overridable):
- ``fusion_mode``: ``joint`` (default) | ``per_sensor``
- ``objective_type``: ``flow_matching`` (default) | ``diffusion``
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from vista.encoders.cnn import AudioCNNStem, TactileCNNStem, VisionCNNStem
from vista.fusion.per_sensor_transformer import PerSensorTransformerFusion
from vista.fusion.transformer import TransformerFusion
from vista.heads.dit import TransformerDenoiser
from vista.models.condition import Condition
from vista.models.qformer import (
    _ProprioToken,
    _RgbPosEmbed,
    _image_size_from_shape_meta,
)
from vista.models.sensor_ablation import SENSOR_GROUPS
from vista.objectives.diffusion import DiffusionObjective
from vista.objectives.flow_matching import FlowMatchingObjective
from vista.policy.base import PROPRIO_KEYS, BaseVistaPolicy
from vista.preproc.log_mel import mic_rows_to_waveform, shared_log_mel


class VisTAPolicy(BaseVistaPolicy):
    """Wrist / finger / audio / proprio → Transformer fusion → DiT + FM/DDPM."""

    RGB_STRIDE = 32
    MAX_COND_TOKENS = 320

    def __init__(
        self,
        shape_meta: dict,
        n_obs_steps: int = 2,
        sensor_group: str = "vta",
        d_embed: int = 256,
        fusion_layers: int = 6,
        n_heads: int = 8,
        mlp_ratio: int = 4,
        dit_layers: int = 8,
        dit_mlp_ratio: int = 4,
        adaln_zero: bool = True,
        n_inference_steps: int = 10,
        sample_rate: int = 16000,
        mel_time_frames: int = 48,
        max_cond_tokens: int = MAX_COND_TOKENS,
        fusion_mode: str = "joint",
        objective_type: str = "flow_matching",
        num_train_timesteps: int = 100,
        input_perturb: float = 0.1,
    ):
        super().__init__(shape_meta=shape_meta, n_obs_steps=n_obs_steps)
        if sensor_group not in SENSOR_GROUPS:
            raise ValueError(
                f"Unknown sensor_group '{sensor_group}'. "
                f"Choose from {list(SENSOR_GROUPS)}"
            )
        if fusion_mode not in ("joint", "per_sensor"):
            raise ValueError(
                f"Unknown fusion_mode '{fusion_mode}'. "
                "Choose from ['joint', 'per_sensor']"
            )
        if objective_type not in ("flow_matching", "diffusion"):
            raise ValueError(
                f"Unknown objective_type '{objective_type}'. "
                "Choose from ['flow_matching', 'diffusion']"
            )
        self.sensor_group = sensor_group
        self.active_sensors = set(SENSOR_GROUPS[sensor_group])
        self.fusion_mode = fusion_mode
        self.objective_type = objective_type
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

        if self.vision_tok is not None or self.tactile_tok is not None:
            self.pos_temporal = _RgbPosEmbed(
                d_embed, n_obs_steps=n_obs_steps, with_temporal=True
            )
            encoders.append(self.pos_temporal)

        self.proprio_tok = _ProprioToken(
            proprio_dim, d_embed, mean_over_time=False
        )
        encoders.append(self.proprio_tok)

        fusion_keys = sorted(self.active_sensors | {"proprio"})
        if fusion_mode == "joint":
            self.fusion = TransformerFusion(
                d_embed=d_embed,
                n_layers=fusion_layers,
                n_heads=n_heads,
                mlp_ratio=mlp_ratio,
            )
        else:
            self.fusion = PerSensorTransformerFusion(
                sensor_keys=fusion_keys,
                d_embed=d_embed,
                n_layers=fusion_layers,
                n_heads=n_heads,
                mlp_ratio=mlp_ratio,
            )
        encoders.append(self.fusion)

        self.head = TransformerDenoiser(
            action_dim=self.action_dim,
            action_horizon=self.action_horizon,
            d_embed=d_embed,
            n_layer=dit_layers,
            n_head=n_heads,
            mlp_ratio=dit_mlp_ratio,
            adaln_zero=adaln_zero,
            cond_mode="tokens",
            max_cond_tokens=max_cond_tokens,
        )
        if objective_type == "flow_matching":
            self.objective = FlowMatchingObjective(
                n_inference_steps=n_inference_steps
            )
        else:
            noise_scheduler = DDPMScheduler(
                num_train_timesteps=num_train_timesteps,
                beta_schedule="squaredcos_cap_v2",
                prediction_type="epsilon",
            )
            self.objective = DiffusionObjective(
                noise_scheduler=noise_scheduler,
                num_inference_steps=n_inference_steps,
                input_perturb=input_perturb,
            )

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
        b, tn, d = tokens.shape
        n = spatial.shape[1]
        if tn != n_obs * n:
            raise ValueError(
                f"Token length {tn} != n_obs ({n_obs}) * spatial ({n})"
            )
        tok = tokens.view(b, n_obs, n, d)
        tok = tok + spatial.unsqueeze(1)
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
        fused, pad_mask = self.fusion(tokens)
        return Condition(tokens=fused, pad_mask=pad_mask)

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
