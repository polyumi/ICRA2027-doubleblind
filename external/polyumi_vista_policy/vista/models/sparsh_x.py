"""
SparshXPolicy — MBT fusion + attentive pool + DiT action denoiser (flow matching).

MBT fusion reimplemented from Nagrani et al. 2021 (not Meta Sparsh code;
facebookresearch/sparsh-multisensory-touch is CC-BY-NC-4.0). Architecture
inspired by Sparsh-X (Higuera et al., CoRL 2025): per-modality self-attn then
bottleneck fusion; fused tokens are aggregated with attentive pooling (one
learnable query) before conditioning the policy head. Head is a transformer
decoder DiT block trained with conditional flow matching.

Vision / tactile: H=2 frames channel-stacked (6ch) → patch-16 stem (196 tokens).
Audio: contiguous mic window → shared log-mel (1×128×48) → patch-8 stem (96 tokens).

Learnable per-modality positional embeddings match Meta Sparsh's
``MultimodalTransformer`` default (``pos_embed_fn="learned"``): after patch
tokenization, each stream gets ``x + pos`` before MBT. Pos tables are sized to
the patch grid (or proprio history length) and init with trunc_normal std=0.02.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional

import torch
import torch.nn as nn

from vista.fusion.attentive_pool import AttentivePooling
from vista.fusion.mbt import MBTFusion
from vista.heads.dit import TransformerDenoiser
from vista.models.condition import Condition
from vista.objectives.flow_matching import FlowMatchingObjective
from vista.policy.base import PROPRIO_KEYS, BaseVistaPolicy
from vista.preproc.log_mel import mic_rows_to_waveform, shared_log_mel


def _stack_history_channels(x: torch.Tensor) -> torch.Tensor:
    """
    Stack H=2 RGB frames on the channel axis: ``(B, 2, 3, H, W) → (B, 6, H, W)``.

    Sampler order is oldest→newest; past channels come first (``I_{t-1}`` then ``I_t``).
    """
    if x.ndim != 5:
        raise ValueError(f"Expected image history (B, T, C, H, W), got {tuple(x.shape)}")
    b, t, c, h, w = x.shape
    if t != 2:
        raise ValueError(f"Sparsh-X channel-stack requires H=2, got T={t}")
    if c != 3:
        raise ValueError(f"Expected 3 RGB channels per frame, got C={c}")
    return x.reshape(b, t * c, h, w)


class _PatchTokenizer(nn.Module):
    """Conv patch stem on a single image (optionally 6ch stacked history)."""

    def __init__(self, in_ch: int = 6, d_embed: int = 256, patch: int = 16):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, d_embed, kernel_size=patch, stride=patch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 5:
            x = _stack_history_channels(x)
        tokens = self.proj(x).flatten(2).transpose(1, 2)
        return tokens


class _AudioMelTokenizer(nn.Module):
    """
    Contiguous mic rows → shared log-mel → patch-8 tokens.

    Mel image is ``(B, 1, 128, 48)``. Patch size 8 → 16×6 = 96 tokens.
    """

    def __init__(
        self,
        d_embed: int = 256,
        patch: int = 8,
        sample_rate: int = 16000,
        time_frames: int = 48,
    ):
        super().__init__()
        self.log_mel = shared_log_mel(sample_rate=sample_rate, time_frames=time_frames)
        self.proj = nn.Conv2d(1, d_embed, kernel_size=patch, stride=patch)

    def forward(self, mic: torch.Tensor) -> torch.Tensor:
        wav = mic_rows_to_waveform(mic)
        mel = self.log_mel(wav)  # (B, 1, 128, 48)
        tokens = self.proj(mel).flatten(2).transpose(1, 2)  # (B, N, D)
        return tokens


class _ProprioTokenizer(nn.Module):
    def __init__(self, in_dim: int, d_embed: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, d_embed),
            nn.GELU(),
            nn.Linear(d_embed, d_embed),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, H, D_prop) → (B, H, D); Linear acts on the last dim.
        return self.net(x)


class _LearnedPosEmbeds(nn.Module):
    """
    Per-modality learned positional embeddings (Sparsh / Meta default).

    Mirrors ``MultimodalTransformer.init_pos_embed(pos_embed_fn="learned")``:
    one ``(1, N_m, D)`` table per modality, trunc_normal std=0.02, added after
    patch tokenization and before fusion. Not applied to register/bottleneck
    tokens (those are created inside MBT).
    """

    def __init__(self, token_counts: Mapping[str, int], d_embed: int):
        super().__init__()
        self.pos = nn.ParameterDict(
            {
                modal: nn.Parameter(torch.zeros(1, int(n), d_embed))
                for modal, n in token_counts.items()
            }
        )
        for p in self.pos.values():
            nn.init.trunc_normal_(p, std=0.02)

    def forward(
        self, tokens_by_modal: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for modal, x in tokens_by_modal.items():
            pos = self.pos[modal]
            if x.shape[1] != pos.shape[1]:
                raise ValueError(
                    f"Sparsh-X pos embed length mismatch for '{modal}': "
                    f"tokens={x.shape[1]}, pos={pos.shape[1]}"
                )
            out[modal] = x + pos
        return out


def _rgb_patch_tokens(img_hw: int, patch: int) -> int:
    if img_hw % patch != 0:
        raise ValueError(
            f"Image side {img_hw} must be divisible by patch size {patch}"
        )
    g = img_hw // patch
    return g * g


def _mel_patch_tokens(mel_h: int, mel_w: int, patch: int) -> int:
    if mel_h % patch != 0 or mel_w % patch != 0:
        raise ValueError(
            f"Mel size ({mel_h}, {mel_w}) must be divisible by patch {patch}"
        )
    return (mel_h // patch) * (mel_w // patch)


class SparshXPolicy(BaseVistaPolicy):
    """Wrist + finger + audio + proprio → MBT → attentive pool → DiT + flow matching."""

    MODALS: List[str] = ["vision", "tactile", "audio", "proprio"]
    MEL_FREQ_BINS: int = 128

    def __init__(
        self,
        shape_meta: dict,
        n_obs_steps: int = 2,
        d_embed: int = 256,
        depth: int = 8,
        fusion_layer: int = 4,
        num_heads: int = 8,
        num_bottlenecks: int = 4,
        dit_layers: int = 4,
        adaln_zero: bool = True,
        n_inference_steps: int = 10,
        encoder_ckpt: Optional[str] = None,
        patch_size: int = 16,
        audio_patch_size: int = 8,
        mel_time_frames: int = 48,
    ):
        super().__init__(shape_meta=shape_meta, n_obs_steps=n_obs_steps)
        if n_obs_steps != 2:
            raise ValueError(
                f"Sparsh-X channel-stacks H=2 frames; got n_obs_steps={n_obs_steps}"
            )
        self.d_embed = d_embed

        proprio_dim = sum(
            int(shape_meta["obs"][k]["shape"][0])
            for k in PROPRIO_KEYS
            if k in shape_meta["obs"]
        )

        cam_hw = int(shape_meta["obs"]["camera0_rgb"]["shape"][-1])
        finger_hw = int(shape_meta["obs"]["finger_rgb"]["shape"][-1])
        token_counts = {
            "vision": _rgb_patch_tokens(cam_hw, patch_size),
            "tactile": _rgb_patch_tokens(finger_hw, patch_size),
            "audio": _mel_patch_tokens(
                self.MEL_FREQ_BINS, mel_time_frames, audio_patch_size
            ),
            "proprio": int(n_obs_steps),
        }

        self.vision_tok = _PatchTokenizer(6, d_embed, patch_size)
        self.tactile_tok = _PatchTokenizer(6, d_embed, patch_size)
        self.audio_tok = _AudioMelTokenizer(
            d_embed, audio_patch_size, time_frames=mel_time_frames
        )
        self.proprio_tok = _ProprioTokenizer(proprio_dim, d_embed)
        self.pos_embeds = _LearnedPosEmbeds(token_counts, d_embed)
        self.mbt = MBTFusion(
            modals=self.MODALS,
            embed_dim=d_embed,
            depth=depth,
            fusion_layer=fusion_layer,
            num_heads=num_heads,
            num_bottlenecks=num_bottlenecks,
        )
        self.pool = AttentivePooling(d_embed=d_embed, num_heads=num_heads)
        self.head = TransformerDenoiser(
            action_dim=self.action_dim,
            action_horizon=self.action_horizon,
            d_embed=d_embed,
            n_layer=dit_layers,
            n_head=num_heads,
            mlp_ratio=4,
            adaln_zero=adaln_zero,
            cond_mode="vector",
        )
        self.objective = FlowMatchingObjective(n_inference_steps=n_inference_steps)

        self._register_encoder_modules(
            self.vision_tok,
            self.tactile_tok,
            self.audio_tok,
            self.proprio_tok,
            self.pos_embeds,
            self.mbt,
            self.pool,
        )
        self._register_head_modules(self.head)

        if encoder_ckpt:
            state = torch.load(encoder_ckpt, map_location="cpu")
            self.load_state_dict(state, strict=False)

    def encode_condition(self, obs: Dict[str, torch.Tensor]) -> Condition:
        tokens = {
            "vision": self.vision_tok(obs["camera0_rgb"]),
            "tactile": self.tactile_tok(obs["finger_rgb"]),
            "audio": self.audio_tok(obs["mic_0"]),
            "proprio": self.proprio_tok(self._concat_proprio(obs)),
        }
        tokens = self.pos_embeds(tokens)
        fused = self.mbt(tokens)
        memory = torch.cat([fused[m] for m in self.MODALS], dim=1)
        vector = self.pool(memory)
        return Condition.from_vector(vector)

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
