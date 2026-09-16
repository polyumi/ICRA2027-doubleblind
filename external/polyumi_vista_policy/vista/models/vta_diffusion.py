"""
VTADiffusionPolicy — ResNet V/T/A + proprio MLP → concat → diffusion U-Net.

Vision / tactile / audio use CoordConv-ResNet encoders (shared SHF trunks).
Proprioception is an MLP. Per-modality vectors are concatenated into a global
condition for ConditionalUnet1D (DDPM).
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from vista.encoders.shf_resnet import (
    SHFAudioEncoder,
    SHFVisionEncoder,
)
from vista.heads.unet1d import Unet1DHead
from vista.models.condition import Condition
from vista.objectives.diffusion import DiffusionObjective
from vista.policy.base import PROPRIO_KEYS, BaseVistaPolicy


class _ProprioMLP(nn.Module):
    """Flattened proprio history → single vector of size ``d_embed``."""

    def __init__(self, input_dim: int, d_embed: int, n_obs_steps: int):
        super().__init__()
        in_dim = input_dim * n_obs_steps
        self.net = nn.Sequential(
            nn.Linear(in_dim, d_embed),
            nn.GELU(),
            nn.Linear(d_embed, d_embed),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, H, D) → (B, d_embed)
        return self.net(x.reshape(x.shape[0], -1))


class VTADiffusionPolicy(BaseVistaPolicy):
    """
    Wrist ResNet + finger ResNet + mel ResNet + proprio MLP
    → concat → Unet1DHead + DDPM.
    """

    def __init__(
        self,
        shape_meta: dict,
        n_obs_steps: int = 2,
        d_embed: int = 384,
        backbone: str = "resnet50",
        down_dims: Sequence[int] = (256, 512, 1024),
        num_train_timesteps: int = 100,
        num_inference_steps: int = 16,
        input_perturb: float = 0.1,
        encoder_ckpt: Optional[str] = None,
    ):
        super().__init__(shape_meta=shape_meta, n_obs_steps=n_obs_steps)
        self.d_embed = int(d_embed)

        proprio_dim = sum(
            int(shape_meta["obs"][k]["shape"][0])
            for k in PROPRIO_KEYS
            if k in shape_meta["obs"]
        )

        # Per-frame ResNet → (B, H * d_embed); mean-pool history to (B, d_embed).
        self.vision_encoder = SHFVisionEncoder(out_dim=d_embed, backbone=backbone)
        self.tactile_encoder = SHFVisionEncoder(out_dim=d_embed, backbone=backbone)
        self.audio_encoder = SHFAudioEncoder(out_dim=d_embed, backbone=backbone)
        self.proprio_encoder = _ProprioMLP(
            input_dim=proprio_dim, d_embed=d_embed, n_obs_steps=n_obs_steps
        )

        cond_dim = 4 * d_embed  # vision | tactile | audio | proprio
        self.head = Unet1DHead(
            action_dim=self.action_dim,
            action_horizon=self.action_horizon,
            d_embed=cond_dim,
            down_dims=list(down_dims),
        )
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon",
        )
        self.objective = DiffusionObjective(
            noise_scheduler=noise_scheduler,
            num_inference_steps=num_inference_steps,
            input_perturb=input_perturb,
        )

        self._register_encoder_modules(
            self.vision_encoder,
            self.tactile_encoder,
            self.audio_encoder,
            self.proprio_encoder,
        )
        self._register_head_modules(self.head)

        if encoder_ckpt:
            state = torch.load(encoder_ckpt, map_location="cpu")
            self.load_state_dict(state, strict=False)

    def _pool_history(self, flat: torch.Tensor) -> torch.Tensor:
        """(B, H * d_embed) → (B, d_embed) by mean over history."""
        b = flat.shape[0]
        return flat.reshape(b, -1, self.d_embed).mean(dim=1)

    def encode_condition(self, obs: Dict[str, torch.Tensor]) -> Condition:
        v = self._pool_history(self.vision_encoder(obs["camera0_rgb"]))
        t = self._pool_history(self.tactile_encoder(obs["finger_rgb"]))
        a = self.audio_encoder(obs["mic_0"])
        p = self.proprio_encoder(self._concat_proprio(obs))
        vector = torch.cat([v, t, a, p], dim=-1)  # (B, 4 * d_embed)
        return Condition.from_vector(vector)

    def compute_loss(self, batch: dict) -> Dict[str, torch.Tensor]:
        nobs = self._normalize_obs(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])
        condition = self.encode_condition(nobs)
        loss = self.objective.compute_loss(self.head, condition, nactions)
        return {"loss": loss}

    def predict_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
        fixed_action_prefix: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        del fixed_action_prefix
        nobs = self._normalize_obs(obs_dict)
        condition = self.encode_condition(nobs)
        b = next(iter(nobs.values())).shape[0]
        nsample = self.objective.predict(
            self.head,
            condition,
            (b, self.action_horizon, self.action_dim),
            device=self.device,
            dtype=self.dtype,
        )
        action = self.normalizer["action"].unnormalize(nsample)
        return {"action": action, "action_pred": action}
