"""Parameter count helper: total + perception encoder vs policy-head split.

Vista policies use ``encoder_parameters()`` / ``head_parameters()`` (matches
optimizer groups). The base PolyUMI diffusion policy splits ``obs_encoder`` vs
``model`` (ConditionalUnet1D).

Examples:
  python -m vista.scripts.param_report --model diffusion_unet
  python -m vista.scripts.param_report --model qformer
  python -m vista.scripts.param_report --all --breakdown
  python -m vista.scripts.param_report --model mitas --sensor-group vt
"""

from __future__ import annotations

import argparse
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch.nn as nn

from vista.policy.base import BaseVistaPolicy


def count_params(module: nn.Module, trainable_only: bool = False) -> int:
    params = module.parameters()
    if trainable_only:
        return sum(p.numel() for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def _unique_count(params: Iterable[nn.Parameter], trainable_only: bool = False) -> int:
    seen = set()
    total = 0
    for p in params:
        pid = id(p)
        if pid in seen:
            continue
        seen.add(pid)
        if trainable_only and not p.requires_grad:
            continue
        total += p.numel()
    return total


def _perception_head_params(
    model: nn.Module,
) -> Tuple[Iterable[nn.Parameter], Iterable[nn.Parameter]]:
    """Return (perception params, policy-head params) for supported policies."""
    if isinstance(model, BaseVistaPolicy):
        return model.encoder_parameters(), model.head_parameters()
    # DiffusionUnetTimmPolicy (and siblings): TimmObsEncoder vs ConditionalUnet1D
    if hasattr(model, "obs_encoder") and hasattr(model, "model"):
        return model.obs_encoder.parameters(), model.model.parameters()
    raise TypeError(
        f"Unsupported policy type {type(model).__name__}; "
        "need BaseVistaPolicy or DiffusionUnetTimmPolicy-style obs_encoder/model."
    )


def param_split(
    model: nn.Module, trainable_only: bool = False
) -> Dict[str, int]:
    """Return total / perception / head / other parameter counts."""
    total = count_params(model, trainable_only=trainable_only)
    perc_params, head_params = _perception_head_params(model)
    perc_list = list(perc_params)
    head_list = list(head_params)
    perception = _unique_count(perc_list, trainable_only=trainable_only)
    head = _unique_count(head_list, trainable_only=trainable_only)
    accounted = _unique_count(perc_list + head_list, trainable_only=trainable_only)
    return {
        "total": total,
        "perception": perception,
        "head": head,
        "other": total - accounted,
    }


def _fmt(n: int) -> str:
    return f"{n:,}"


def _pct(part: int, total: int) -> str:
    if total <= 0:
        return "n/a"
    return f"{100.0 * part / total:5.1f}%"


def format_split_report(
    name: str,
    split: Dict[str, int],
    *,
    module_rows: Optional[Sequence[Tuple[str, str, int]]] = None,
) -> str:
    """Human-readable balance report for one model."""
    total = split["total"]
    perc = split["perception"]
    head = split["head"]
    other = split["other"]
    ratio = (perc / head) if head > 0 else float("inf")

    lines = [
        f"{name}",
        f"  total:       {_fmt(total):>12}",
        f"  perception:  {_fmt(perc):>12}  ({_pct(perc, total)})  "
        f"[encoders + fusion]",
        f"  policy-head: {_fmt(head):>12}  ({_pct(head, total)})",
    ]
    if other:
        lines.append(f"  other:       {_fmt(other):>12}  ({_pct(other, total)})")
    lines.append(f"  balance:     perception/head = {ratio:.2f}x  (target ~1)")
    if abs(ratio - 1.0) > 0.25 and head > 0:
        heavier = "perception" if ratio > 1 else "policy-head"
        lines.append(f"  note:        {heavier} is larger; consider rebalancing")

    if module_rows:
        lines.append("  modules:")
        for group, mod_name, n in module_rows:
            lines.append(f"    [{group:11}] {mod_name:<28} {_fmt(n):>12}")
    return "\n".join(lines)


def module_breakdown(model: nn.Module) -> List[Tuple[str, str, int]]:
    """Per perception / head submodule counts."""
    rows: List[Tuple[str, str, int]] = []
    if isinstance(model, BaseVistaPolicy):
        for i, mod in enumerate(model._encoder_modules):
            rows.append(("perception", f"{i}:{type(mod).__name__}", count_params(mod)))
        for i, mod in enumerate(model._head_modules):
            rows.append(("policy-head", f"{i}:{type(mod).__name__}", count_params(mod)))
        return rows
    if hasattr(model, "obs_encoder") and hasattr(model, "model"):
        rows.append(("perception", "obs_encoder", count_params(model.obs_encoder)))
        rows.append(("policy-head", type(model.model).__name__, count_params(model.model)))
        return rows
    raise TypeError(f"Unsupported policy type {type(model).__name__}")


def _vista_shape_meta(n_obs: int = 2, audio_h: int = 10, action_h: int = 16):
    return {
        "obs": {
            "camera0_rgb": {"shape": [3, 224, 224], "horizon": n_obs, "type": "rgb"},
            "finger_rgb": {"shape": [3, 224, 224], "horizon": n_obs, "type": "rgb"},
            "mic_0": {"shape": [536], "horizon": audio_h, "type": "low_dim"},
            "robot0_eef_pos": {"shape": [3], "horizon": n_obs, "type": "low_dim"},
            "robot0_eef_rot_axis_angle": {
                "shape": [6],
                "horizon": n_obs,
                "type": "low_dim",
            },
            "robot0_gripper_width": {
                "shape": [1],
                "horizon": n_obs,
                "type": "low_dim",
            },
            "robot0_eef_rot_axis_angle_wrt_start": {
                "shape": [6],
                "horizon": n_obs,
                "type": "low_dim",
            },
        },
        "action": {"shape": [10], "horizon": action_h},
    }


def _diffusion_shape_meta(n_obs: int = 2, action_h: int = 16):
    """PolyUMI / UMI vision+proprio contract (no finger/mic)."""
    return {
        "obs": {
            "camera0_rgb": {"shape": [3, 224, 224], "horizon": n_obs, "type": "rgb"},
            "robot0_eef_pos": {"shape": [3], "horizon": n_obs, "type": "low_dim"},
            "robot0_eef_rot_axis_angle": {
                "shape": [6],
                "horizon": n_obs,
                "type": "low_dim",
            },
            "robot0_gripper_width": {
                "shape": [1],
                "horizon": n_obs,
                "type": "low_dim",
            },
            "robot0_eef_rot_axis_angle_wrt_start": {
                "shape": [6],
                "horizon": n_obs,
                "type": "low_dim",
            },
        },
        "action": {"shape": [10], "horizon": action_h},
    }


def build_model(
    name: str,
    *,
    sensor_group: str = "vta",
    lite: bool = False,
) -> nn.Module:
    """Instantiate a policy with train-config defaults (~160M, balanced)."""
    if name == "diffusion_unet":
        from diffusers.schedulers.scheduling_ddim import DDIMScheduler

        from diffusion_policy.model.vision.timm_obs_encoder import TimmObsEncoder
        from diffusion_policy.policy.diffusion_unet_timm_policy import (
            DiffusionUnetTimmPolicy,
        )

        # Match train_diffusion_unet_timm_polyumi_workspace.yaml (pretrained
        # weights do not change architecture; False keeps the report offline).
        obs_encoder = TimmObsEncoder(
            shape_meta=_diffusion_shape_meta(),
            model_name="vit_base_patch16_clip_224.openai",
            pretrained=False,
            frozen=False,
            global_pool="",
            transforms=None,
            use_group_norm=True,
            share_rgb_model=False,
            imagenet_norm=True,
            feature_aggregation="attention_pool_2d",
            downsample_ratio=32,
            position_encording="sinusoidal",
        )
        noise_scheduler = DDIMScheduler(
            num_train_timesteps=50,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type="epsilon",
        )
        return DiffusionUnetTimmPolicy(
            shape_meta=_diffusion_shape_meta(),
            noise_scheduler=noise_scheduler,
            obs_encoder=obs_encoder,
            num_inference_steps=16,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=128,
            down_dims=[256, 512, 1024],
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            input_pertub=0.1,
            train_diffusion_n_samples=1,
        )

    shape = _vista_shape_meta()
    if name == "see_hear_feel":
        from vista.models.see_hear_feel import SeeHearFeelPolicy

        return SeeHearFeelPolicy(
            shape,
            n_obs_steps=2,
            d_embed=256,
            n_heads=8,
            mlp_hidden=2048,
            n_mha_layers=6,
            n_mlp_layers=5,
            cond_dim=512,
            backbone="resnet18",
            down_dims=(256, 512, 1024),
            num_train_timesteps=100,
            num_inference_steps=16,
        )
    if name == "sparsh_x":
        from vista.models.sparsh_x import SparshXPolicy

        return SparshXPolicy(
            shape,
            n_obs_steps=2,
            d_embed=512,
            depth=6,
            fusion_layer=3,
            num_heads=8,
            dit_layers=18,
        )
    if name == "polytouch":
        from vista.models.polytouch import PolyTouchPolicy

        audio_backend = "mel_cnn" if lite else "ast"
        model = PolyTouchPolicy(
            shape,
            n_obs_steps=2,
            pretrained=False,
            lite=lite,
            d_model=768,
            fusion_dim=384,
            n_blocks=6,
            n_heads=12,
            cond_dim=512,
            down_dims=(256, 512, 1024),
            clip_model_name="vit_base_patch16_clip_224.openai",
            t3_size="small",
            t3_sensor="svelte",
            t3_hf_repo="",
            audio_backend=audio_backend,
        )
        # Offline envs without transformers get an empty AST; fall back so counts
        # still include an audio tower (train YAML keeps audio_backend=ast).
        if (
            not lite
            and audio_backend == "ast"
            and count_params(model.ast) == 0
        ):
            model = PolyTouchPolicy(
                shape,
                n_obs_steps=2,
                pretrained=False,
                lite=False,
                d_model=768,
                fusion_dim=384,
                n_blocks=6,
                n_heads=12,
                cond_dim=512,
                down_dims=(256, 512, 1024),
                clip_model_name="vit_base_patch16_clip_224.openai",
                t3_size="small",
                t3_sensor="svelte",
                t3_hf_repo="",
                audio_backend="mel_cnn",
            )
            model._param_report_audio_note = "ast unavailable; counted mel_cnn"
        return model
    if name == "qformer":
        from vista.models.qformer import QformerPolicy

        return QformerPolicy(
            shape,
            n_obs_steps=2,
            sensor_group=sensor_group,
            d_embed=640,
            n_queries=128,
            qformer_layers=8,
            dit_layers=12,
        )
    if name == "mitas":
        from vista.models.mitas import MitasPolicy

        return MitasPolicy(
            shape,
            n_obs_steps=2,
            sensor_group=sensor_group,
            d_embed=624,
            fusion_layers=8,
            dit_layers=18,
        )
    if name == "vta_diffusion":
        from vista.models.vta_diffusion import VTADiffusionPolicy

        return VTADiffusionPolicy(
            shape,
            n_obs_steps=2,
            d_embed=384,
            backbone="resnet50",
            down_dims=(256, 512, 1024),
            num_train_timesteps=100,
            num_inference_steps=16,
        )
    raise ValueError(f"Unknown model '{name}'")


MODEL_CHOICES = (
    "diffusion_unet",
    "see_hear_feel",
    "sparsh_x",
    "polytouch",
    "qformer",
    "mitas",
    "vta_diffusion",
)


def report_model(
    name: str,
    *,
    sensor_group: str = "vta",
    lite: bool = False,
    breakdown: bool = False,
    trainable_only: bool = False,
) -> str:
    model = build_model(name, sensor_group=sensor_group, lite=lite)
    label = name
    if name == "diffusion_unet":
        label = "diffusion_unet (PolyUMI CLIP-ViT + Unet1D)"
    if name in ("qformer", "mitas"):
        label = f"{name} (sensor_group={sensor_group})"
    if name == "polytouch" and lite:
        label = f"{name} (lite)"
    if name == "polytouch" and hasattr(model, "_param_report_audio_note"):
        label = f"{name} ({model._param_report_audio_note})"
    split = param_split(model, trainable_only=trainable_only)
    rows = module_breakdown(model) if breakdown else None
    return format_split_report(label, split, module_rows=rows)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Count total / perception / policy-head parameters."
    )
    parser.add_argument(
        "--model",
        choices=MODEL_CHOICES,
        default="qformer",
        help="Policy to instantiate (ignored with --all)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Report all registered baselines",
    )
    parser.add_argument(
        "--lite",
        action="store_true",
        help="PolyTouch lite stubs (no pretrained backbones)",
    )
    parser.add_argument(
        "--sensor-group",
        default="vta",
        choices=["v", "vt", "va", "vta"],
        help="Qformer/Mitas sensor ablation group",
    )
    parser.add_argument(
        "--breakdown",
        action="store_true",
        help="Per registered submodule counts",
    )
    parser.add_argument(
        "--trainable-only",
        action="store_true",
        help="Count only parameters with requires_grad=True",
    )
    args = parser.parse_args(argv)

    names = list(MODEL_CHOICES) if args.all else [args.model]
    blocks = []
    for name in names:
        blocks.append(
            report_model(
                name,
                sensor_group=args.sensor_group,
                lite=args.lite,
                breakdown=args.breakdown,
                trainable_only=args.trainable_only,
            )
        )
    print("\n\n".join(blocks))


if __name__ == "__main__":
    main()
