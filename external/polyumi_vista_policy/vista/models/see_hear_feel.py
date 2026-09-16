"""
SeeHearFeelPolicy — ResNet + MSA + MLP conditioner → diffusion U-Net.

Encoders / MHA adapted from JunzheJosephZhu/see_hear_feel:
  - src/models/encoders.py (CoordConv ResNet)
  - src/models/imi_models.py (Actor MHA + bottleneck)
MIT licensed. Original discrete 3^k classifier is replaced by an MLP that
emits a global condition vector for ConditionalUnet1D (DDPM).

Vision / tactile / proprio use ``n_obs_history`` downsampled frames.
Audio uses a contiguous ``audio_obs_horizon`` mic window → shared log-mel → ResNet.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from vista.encoders.shf_resnet import (
    SHFAudioEncoder,
    SHFProprioEncoder,
    SHFVisionEncoder,
)
from vista.heads.unet1d import Unet1DHead
from vista.models.condition import Condition
from vista.objectives.diffusion import DiffusionObjective
from vista.policy.base import PROPRIO_KEYS, BaseVistaPolicy


class SeeHearFeelPolicy(BaseVistaPolicy):
    """Wrist + finger + audio + proprio → MHA+MLP latent → Unet1DHead + DDPM."""

    def __init__(
        self,
        shape_meta: dict,
        n_obs_steps: int = 2,
        d_embed: int = 128,
        n_heads: int = 4,
        mlp_hidden: int = 1024,
        n_mha_layers: int = 1,
        n_mlp_layers: int = 2,
        cond_dim: Optional[int] = None,
        backbone: str = "resnet18",
        down_dims: Sequence[int] = (256, 512, 1024),
        num_train_timesteps: int = 100,
        num_inference_steps: int = 16,
        input_perturb: float = 0.1,
        encoder_ckpt: Optional[str] = None,
        image_size: int = 224,
    ):
        super().__init__(shape_meta=shape_meta, n_obs_steps=n_obs_steps)
        del image_size  # reserved for future resize hooks
        if n_mha_layers < 1:
            raise ValueError(f"n_mha_layers must be >= 1, got {n_mha_layers}")
        if n_mlp_layers < 1:
            raise ValueError(f"n_mlp_layers must be >= 1, got {n_mlp_layers}")
        self.d_embed = d_embed
        self.stack_dim = d_embed * n_obs_steps
        self.cond_dim = int(cond_dim) if cond_dim is not None else int(mlp_hidden)

        proprio_dim = 0
        for key in PROPRIO_KEYS:
            if key in shape_meta["obs"]:
                proprio_dim += int(shape_meta["obs"][key]["shape"][0])
        self.proprio_dim = proprio_dim

        self.vision_encoder = SHFVisionEncoder(out_dim=d_embed, backbone=backbone)
        self.tactile_encoder = SHFVisionEncoder(out_dim=d_embed, backbone=backbone)
        # One mel clip → stack_dim so MHA matches vision flatten length.
        self.audio_encoder = SHFAudioEncoder(
            out_dim=self.stack_dim, backbone=backbone
        )
        self.proprio_encoder = SHFProprioEncoder(
            input_dim=proprio_dim, out_dim=d_embed, n_obs_steps=n_obs_steps
        )

        self.layernorm = nn.LayerNorm(self.stack_dim)
        self.mha_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(self.stack_dim, n_heads, batch_first=False)
                for _ in range(n_mha_layers)
            ]
        )
        n_modals = 4  # vision, tactile, audio, proprio
        self.bottleneck = nn.Linear(self.stack_dim * n_modals, self.stack_dim)

        # MHA fused vector → MLP latent consumed by the diffusion U-Net.
        cond_layers: list = [nn.Linear(self.stack_dim, mlp_hidden), nn.ReLU()]
        for _ in range(max(n_mlp_layers - 1, 0)):
            cond_layers.extend([nn.Linear(mlp_hidden, mlp_hidden), nn.ReLU()])
        cond_layers.append(nn.Linear(mlp_hidden, self.cond_dim))
        self.cond_mlp = nn.Sequential(*cond_layers)

        self.head = Unet1DHead(
            action_dim=self.action_dim,
            action_horizon=self.action_horizon,
            d_embed=self.cond_dim,
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
            self.layernorm,
            self.mha_layers,
            self.bottleneck,
            self.cond_mlp,
        )
        self._register_head_modules(self.head)

        if encoder_ckpt:
            state = torch.load(encoder_ckpt, map_location="cpu")
            self.load_state_dict(state, strict=False)

    def encode_condition(self, obs: Dict[str, torch.Tensor]) -> Condition:
        v = self.vision_encoder(obs["camera0_rgb"])
        t = self.tactile_encoder(obs["finger_rgb"])
        a = self.audio_encoder(obs["mic_0"])
        p = self.proprio_encoder(self._concat_proprio(obs))
        embeds = torch.stack(
            [self.layernorm(v), self.layernorm(t), self.layernorm(a), self.layernorm(p)],
            dim=0,
        )  # (4, B, stack_dim)
        mha_out = embeds
        for mha in self.mha_layers:
            attn_out, _ = mha(mha_out, mha_out, mha_out)
            mha_out = mha_out + attn_out
        fused = torch.cat([mha_out[i] for i in range(mha_out.shape[0])], dim=-1)
        vector = self.cond_mlp(self.bottleneck(fused))
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
