"""
PolyTouchPolicy — CLIP + T3 + AST + cross-attn combiner + Diffusion U-Net.

Combiner follows arxiv:2504.19341 §IV-A / Fig. 5 (no public policy repo):
6-block 12-head cross-attention between CLIP (wrist) and T3 (finger) tokens;
three CLS tokens (CLIP, T3, AST) concatenated and projected; proprio MLP
concatenated; ConditionalUnet1D diffusion head.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from vista.encoders.ast import ASTEncoder
from vista.encoders.clip_vit import CLIPViTEncoder
from vista.encoders.cnn import AudioCNNStem
from vista.encoders.t3 import T3Encoder
from vista.fusion.polytouch_combiner import PolyTouchCombiner
from vista.heads.unet1d import Unet1DHead
from vista.models.condition import Condition
from vista.objectives.diffusion import DiffusionObjective
from vista.policy.base import PROPRIO_KEYS, BaseVistaPolicy
from vista.preproc.ast_log_mel import ASTLogMel
from vista.preproc.log_mel import mic_rows_to_waveform, shared_log_mel


class _MelCnnAudioEncoder(nn.Module):
    """Shared log-mel → AudioCNNStem → mean-pool CLS (AST-sized drop-in)."""

    def __init__(self, d_embed: int = 384, sample_rate: int = 16000, time_frames: int = 48):
        super().__init__()
        self.log_mel = shared_log_mel(sample_rate=sample_rate, time_frames=time_frames)
        self.stem = AudioCNNStem(in_channels=1, d_embed=d_embed)
        self.proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept (B, T_mel, n_mels) from PolyTouch audio path or raw mel.
        if x.ndim == 3 and x.shape[-1] != 128:
            # Feature rows (B, N, 536) — convert via mic waveform.
            wav = mic_rows_to_waveform(x)
            mel = self.log_mel(wav)
        elif x.ndim == 3:
            # (B, T, n_mels) → (B, 1, n_mels, T)
            mel = x.transpose(1, 2).unsqueeze(1)
        else:
            mel = x
        tokens = self.stem(mel)  # (B, N, D)
        return tokens


class _LiteImageEncoder(nn.Module):
    """Conv stub returning (B, N_tok, D) for tests."""

    def __init__(self, d_embed: int = 64):
        super().__init__()
        self.proj = nn.Conv2d(3, d_embed, kernel_size=16, stride=16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 5:
            b, t, c, h, w = x.shape
            x = x.reshape(b * t, c, h, w)
            tok = self.proj(x).flatten(2).transpose(1, 2)
            return tok.reshape(b, t * tok.shape[1], -1)
        return self.proj(x).flatten(2).transpose(1, 2)


class _LiteAudioEncoder(nn.Module):
    def __init__(self, d_embed: int = 64):
        super().__init__()
        self.net = nn.Linear(128, d_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 128)
        return self.net(x)


class PolyTouchPolicy(BaseVistaPolicy):
    """Pretrained CLIP / T3 / AST + PolyTouch combiner + diffusion U-Net."""

    def __init__(
        self,
        shape_meta: dict,
        n_obs_steps: int = 2,
        d_model: int = 768,
        fusion_dim: Optional[int] = None,
        n_blocks: int = 6,
        n_heads: int = 12,
        cond_dim: int = 256,
        proprio_out_dim: int = 64,
        pretrained: bool = True,
        t3_ckpt: Optional[str] = None,
        t3_trunk_ckpt: Optional[str] = None,
        t3_hf_repo: str = "alanz-mit/FoundationTactile",
        t3_size: str = "small",
        t3_sensor: str = "svelte",
        encoder_ckpt: Optional[str] = None,
        num_train_timesteps: int = 100,
        num_inference_steps: int = 16,
        down_dims=(256, 512, 1024),
        lite: bool = False,
        clip_model_name: str = "vit_base_patch16_clip_224.openai",
        audio_backend: str = "ast",
        # Deprecated: T3 is FoTa encoder+trunk, not a plain timm ViT name.
        t3_model_name: str = "",
    ):
        del t3_model_name
        super().__init__(shape_meta=shape_meta, n_obs_steps=n_obs_steps)

        proprio_dim = sum(
            int(shape_meta["obs"][k]["shape"][0]) * n_obs_steps
            for k in PROPRIO_KEYS
            if k in shape_meta["obs"]
        )

        self.lite = lite
        self.audio_backend = audio_backend
        self.ast_log_mel: Optional[nn.Module] = None
        # Cross-attn width; keep pretrained encoder width (d_model) separate so
        # FoTa/CLIP/AST stay full-size while fusion/UNet match ~160M peers.
        if fusion_dim is None:
            fusion_dim = d_model
        self.fusion_dim = int(fusion_dim)

        if lite:
            # Tiny stand-ins for unit tests (no timm / transformers download).
            self.clip = _LiteImageEncoder(d_model)
            self.t3 = _LiteImageEncoder(d_model)
            self.ast = _LiteAudioEncoder(d_model)
        else:
            self.clip = CLIPViTEncoder(
                d_embed=d_model,
                pretrained=pretrained,
                model_name=clip_model_name,
            )
            self.t3 = T3Encoder(
                d_embed=d_model,
                pretrained=pretrained and not (t3_ckpt or t3_trunk_ckpt),
                t3_size=t3_size,
                t3_sensor=t3_sensor,
                hf_repo=t3_hf_repo or "",
                t3_ckpt=t3_ckpt,
                trunk_ckpt=t3_trunk_ckpt,
            )
            if audio_backend == "ast":
                # Kaldi fbank + AudioSet norm → pretrained HF AST (1024×128).
                self.ast_log_mel = ASTLogMel()
                self.ast = ASTEncoder(d_embed=d_model)
            elif audio_backend == "mel_cnn":
                self.ast = _MelCnnAudioEncoder(d_embed=d_model)
            else:
                raise ValueError(
                    f"Unknown audio_backend '{audio_backend}'. "
                    "Choose from ['ast', 'mel_cnn']."
                )

        # Project encoder outputs into fusion width when they differ.
        if self.fusion_dim != d_model:
            self.clip_fuse = nn.Linear(d_model, self.fusion_dim)
            self.t3_fuse = nn.Linear(d_model, self.fusion_dim)
            self.ast_fuse = nn.Linear(d_model, self.fusion_dim)
        else:
            self.clip_fuse = nn.Identity()
            self.t3_fuse = nn.Identity()
            self.ast_fuse = nn.Identity()

        # Smaller combiner in lite mode for speed
        if lite:
            n_blocks = min(n_blocks, 2)
            n_heads = min(n_heads, 4)

        self.combiner = PolyTouchCombiner(
            d_model=self.fusion_dim,
            n_blocks=n_blocks,
            n_heads=n_heads,
            proprio_dim=proprio_dim,
            out_dim=cond_dim,
            proprio_out_dim=proprio_out_dim,
        )
        global_cond_dim = self.combiner.out_dim

        self.head = Unet1DHead(
            action_dim=self.action_dim,
            action_horizon=self.action_horizon,
            d_embed=global_cond_dim,
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
        )

        encoder_mods: list = [
            self.clip,
            self.t3,
            self.ast,
            self.clip_fuse,
            self.t3_fuse,
            self.ast_fuse,
            self.combiner,
        ]
        if self.ast_log_mel is not None:
            encoder_mods.append(self.ast_log_mel)
        self._register_encoder_modules(*encoder_mods)
        self._register_head_modules(self.head)

        if encoder_ckpt:
            state = torch.load(encoder_ckpt, map_location="cpu")
            self.load_state_dict(state, strict=False)

    def _encode_audio_cls(self, mic: torch.Tensor) -> torch.Tensor:
        """Return (B, D) audio CLS from mic (B, N, 536) feature rows."""
        b, n, _f = mic.shape
        if self.lite:
            mel = mic[..., :128] if mic.shape[-1] >= 128 else F.pad(mic, (0, 128 - mic.shape[-1]))
            mel = mel.unsqueeze(2).expand(b, n, 4, mel.shape[-1]).reshape(b, n * 4, -1)
            tokens = self.ast(mel)
            return tokens.mean(dim=1)
        if getattr(self, "audio_backend", "ast") == "mel_cnn":
            tokens = self.ast(mic)
            return tokens.mean(dim=1)
        # Pretrained AST: Kaldi log-mel (B, 1024, 128) → CLS token.
        assert self.ast_log_mel is not None
        mel = self.ast_log_mel(mic)
        tokens = self.ast(mel)
        return tokens[:, 0]

    def _tokens_with_cls(self, encoder: nn.Module, images: torch.Tensor) -> torch.Tensor:
        """
        Encode (B, N, 3, H, W) and return (B, 1+P_total, D) with a mean-pooled
        CLS placed first for each history step collapsed into one stream.
        """
        # Existing CLIP/T3 encoders return (B, N*tokens, D) for 5D input.
        tokens = encoder(images)
        # Use mean over tokens as a CLS-like vector, keep patches after.
        cls = tokens.mean(dim=1, keepdim=True)
        return torch.cat([cls, tokens], dim=1)

    def encode_condition(self, obs: Dict[str, torch.Tensor]) -> Condition:
        clip_tokens = self.clip_fuse(self._tokens_with_cls(self.clip, obs["camera0_rgb"]))
        t3_tokens = self.t3_fuse(self._tokens_with_cls(self.t3, obs["finger_rgb"]))
        ast_cls = self.ast_fuse(self._encode_audio_cls(obs["mic_0"]))
        proprio = self._concat_proprio(obs).reshape(obs["camera0_rgb"].shape[0], -1)
        vector = self.combiner(clip_tokens, t3_tokens, ast_cls, proprio)
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
