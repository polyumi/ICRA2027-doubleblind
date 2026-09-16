"""Sensor ablation presets for the Vista training pipeline."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, Optional

SENSOR_GROUPS: Dict[str, tuple[str, ...]] = {
    "v": ("camera0_rgb",),
    "vt": ("camera0_rgb", "finger_rgb"),
    "va": ("camera0_rgb", "mic_0"),
    "vta": ("camera0_rgb", "finger_rgb", "mic_0"),
}

PROPRIO_KEYS: tuple[str, ...] = (
    "robot0_eef_pos",
    "robot0_eef_rot_axis_angle",
    "robot0_gripper_width",
    "robot0_eef_rot_axis_angle_wrt_start",
)

EXOGENOUS_KEYS: tuple[str, ...] = ("camera0_rgb", "finger_rgb", "mic_0")

# Injected when an architecture preset omits audio but the ablation group needs it.
AUDIO_SENSOR_BY_ARCHITECTURE: Dict[str, Dict[str, Any]] = {
    "qformer": {"modality": "audio", "encoder": "audio_cnn"},
}

STANDARD_DIT_HEAD_KWARGS: Dict[str, Any] = {
    "d_model": 256,
    "n_layers": 4,
    "n_heads": 8,
    "mlp_ratio": 4,
    "adaln_zero": True,
    "max_cond_tokens": 1024,
}

STANDARD_FLOW_MATCHING_KWARGS: Dict[str, Any] = {
    "n_inference_steps": 10,
}

FIXED_TOKEN_FUSIONS: frozenset[str] = frozenset({"qformer"})


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def apply_ablation_policy_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Apply standardized DiT + flow-matching overrides when ``sensor_group`` is set."""
    group = cfg.get("sensor_group")
    if group is None:
        return cfg
    if group not in SENSOR_GROUPS:
        raise ValueError(f"Unknown sensor_group '{group}'. Choose from {list(SENSOR_GROUPS)}")

    out = deepcopy(cfg)
    out["head"] = "dit"
    out["objective"] = "flow_matching"
    out["head_kwargs"] = _deep_merge(STANDARD_DIT_HEAD_KWARGS, out.get("head_kwargs", {}))
    out["objective_kwargs"] = _deep_merge(
        STANDARD_FLOW_MATCHING_KWARGS,
        out.get("objective_kwargs", {}),
    )
    return out


def _is_proprio(sensor_cfg: Dict[str, Any]) -> bool:
    return sensor_cfg.get("modality") == "proprio"


def _allowed_keys(group: str) -> set[str]:
    return set(SENSOR_GROUPS[group]) | set(PROPRIO_KEYS)


def _inject_missing_audio(
    sensors: Dict[str, Dict[str, Any]],
    architecture: str,
    allowed: Iterable[str],
) -> Dict[str, Dict[str, Any]]:
    if "mic_0" not in allowed or "mic_0" in sensors:
        return sensors
    if architecture not in AUDIO_SENSOR_BY_ARCHITECTURE:
        raise ValueError(
            f"Architecture '{architecture}' has no audio encoder for ablation groups "
            f"that include mic_0. Add mic_0 to the preset or extend "
            "AUDIO_SENSOR_BY_ARCHITECTURE."
        )
    out = deepcopy(sensors)
    audio_cfg = deepcopy(AUDIO_SENSOR_BY_ARCHITECTURE[architecture])
    d_embed = out.get("camera0_rgb", {}).get("encoder_kwargs", {}).get("d_embed")
    if d_embed is not None:
        audio_cfg.setdefault("encoder_kwargs", {})
        audio_cfg["encoder_kwargs"].setdefault("d_embed", d_embed)
    out["mic_0"] = audio_cfg
    return out


def apply_sensor_group(
    merged: Dict[str, Any],
    group: str,
    architecture: str,
) -> Dict[str, Any]:
    """Filter conditioner sensors to the ablation group plus proprio."""
    if group not in SENSOR_GROUPS:
        raise ValueError(f"Unknown sensor group '{group}'. Choose from {list(SENSOR_GROUPS)}")

    out = deepcopy(merged)
    allowed = _allowed_keys(group)
    sensors = deepcopy(out.get("sensors", {}))
    sensors = _inject_missing_audio(sensors, architecture, allowed)

    filtered: Dict[str, Dict[str, Any]] = {}
    for key, sensor_cfg in sensors.items():
        if _is_proprio(sensor_cfg):
            if key in PROPRIO_KEYS:
                filtered[key] = sensor_cfg
            continue
        if key in allowed:
            filtered[key] = sensor_cfg

    missing_exogenous = set(SENSOR_GROUPS[group]) - set(filtered)
    if missing_exogenous:
        raise ValueError(
            f"Architecture '{architecture}' is missing sensors {sorted(missing_exogenous)} "
            f"required for ablation group '{group}'."
        )

    out["sensors"] = filtered
    return out


def expected_condition_tokens(
    conditioner_cfg: Dict[str, Any],
    *,
    batch_size: int = 1,
) -> Optional[int]:
    """Return fixed fusion output length when known; None for variable-token fusion."""
    fusion = conditioner_cfg.get("fusion")
    if fusion not in FIXED_TOKEN_FUSIONS:
        return None
    fusion_kw = conditioner_cfg.get("fusion_kwargs", {})
    if "n_latents" in fusion_kw:
        return int(fusion_kw["n_latents"])
    if "n_queries" in fusion_kw:
        return int(fusion_kw["n_queries"])
    return 256
