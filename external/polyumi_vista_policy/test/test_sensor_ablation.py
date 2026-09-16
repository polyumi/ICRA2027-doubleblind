"""Sensor ablation for QformerPolicy / VisTAPolicy: dropped sensors are not encoded."""

import numpy as np
import torch

from diffusion_policy.common.normalize_util import (
    array_to_stats,
    get_identity_normalizer_from_stat,
    get_image_identity_normalizer,
)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from vista.models.sensor_ablation import SENSOR_GROUPS
from vista.models.vista import VisTAPolicy
from vista.models.qformer import QformerPolicy

N_OBS = 2
AUDIO_HORIZON = 10


def _identity_normalizer(shape_meta):
    n = LinearNormalizer()
    for key in shape_meta["obs"]:
        n[key] = get_image_identity_normalizer()
    stat = array_to_stats(np.zeros((1, shape_meta["action"]["shape"][0]), dtype=np.float32))
    n["action"] = get_identity_normalizer_from_stat(stat)
    for key in n.params_dict.keys():
        params = n.params_dict[key]
        for field in ("scale", "offset", "input_stats"):
            if field in params:
                if isinstance(params[field], dict):
                    for sk in params[field]:
                        if torch.is_tensor(params[field][sk]):
                            params[field][sk] = params[field][sk].float()
                elif torch.is_tensor(params[field]):
                    params[field] = params[field].float()
    return n


def _shape_meta(img: int = 64):
    return {
        "obs": {
            "camera0_rgb": {"shape": [3, img, img], "horizon": N_OBS},
            "finger_rgb": {"shape": [3, img, img], "horizon": N_OBS},
            "mic_0": {"shape": [536], "horizon": AUDIO_HORIZON},
            "robot0_eef_pos": {"shape": [3], "horizon": N_OBS},
            "robot0_eef_rot_axis_angle": {"shape": [6], "horizon": N_OBS},
            "robot0_gripper_width": {"shape": [1], "horizon": N_OBS},
            "robot0_eef_rot_axis_angle_wrt_start": {"shape": [6], "horizon": N_OBS},
        },
        "action": {"shape": [10], "horizon": 16},
    }


def _batch(img: int = 64, batch: int = 2):
    return {
        "obs": {
            "camera0_rgb": torch.rand(batch, N_OBS, 3, img, img),
            "finger_rgb": torch.rand(batch, N_OBS, 3, img, img),
            "mic_0": torch.randn(batch, AUDIO_HORIZON, 536),
            "robot0_eef_pos": torch.randn(batch, N_OBS, 3),
            "robot0_eef_rot_axis_angle": torch.randn(batch, N_OBS, 6),
            "robot0_gripper_width": torch.rand(batch, N_OBS, 1),
            "robot0_eef_rot_axis_angle_wrt_start": torch.randn(batch, N_OBS, 6),
        },
        "action": torch.randn(batch, 16, 10),
    }


def _lite_kwargs():
    return dict(
        d_embed=64,
        n_queries=8,
        qformer_layers=1,
        n_cross_attn=2,
        n_heads=4,
        mlp_ratio=2,
        dit_layers=1,
        dit_mlp_ratio=2,
        adaln_zero=True,
        n_inference_steps=2,
    )


def _vista_lite_kwargs():
    return dict(
        d_embed=64,
        fusion_layers=1,
        n_heads=4,
        mlp_ratio=2,
        dit_layers=1,
        dit_mlp_ratio=2,
        adaln_zero=True,
        n_inference_steps=2,
        max_cond_tokens=128,
    )


def test_ablation_builds_only_active_encoders():
    shape = _shape_meta()
    vt = QformerPolicy(shape, n_obs_steps=N_OBS, sensor_group="vt", **_lite_kwargs())
    assert vt.vision_tok is not None
    assert vt.tactile_tok is not None
    assert vt.audio_tok is None
    assert vt.log_mel is None
    assert set(SENSOR_GROUPS["vt"]) == vt.active_sensors

    v = QformerPolicy(shape, n_obs_steps=N_OBS, sensor_group="v", **_lite_kwargs())
    assert v.vision_tok is not None
    assert v.tactile_tok is None
    assert v.audio_tok is None


def test_ablation_qformer_out_fixed_and_predict():
    shape = _shape_meta()
    policy = QformerPolicy(shape, n_obs_steps=N_OBS, sensor_group="vt", **_lite_kwargs())
    policy.set_normalizer(_identity_normalizer(shape))
    batch = _batch()
    cond = policy.encode_condition(policy._normalize_obs(batch["obs"]))
    assert cond.tokens is not None
    assert cond.tokens.shape == (2, 8, 64)
    out = policy.compute_loss(batch)
    assert out["loss"].ndim == 0
    pred = policy.predict_action(batch["obs"])
    assert pred["action"].shape == (2, 16, 10)


def test_all_sensor_groups_forward():
    shape = _shape_meta()
    batch = _batch()
    for group in SENSOR_GROUPS:
        policy = QformerPolicy(
            shape, n_obs_steps=N_OBS, sensor_group=group, **_lite_kwargs()
        )
        policy.set_normalizer(_identity_normalizer(shape))
        pred = policy.predict_action(batch["obs"])
        assert pred["action"].shape == (2, 16, 10)


def test_vista_ablation_builds_only_active_encoders():
    shape = _shape_meta()
    vt = VisTAPolicy(shape, n_obs_steps=N_OBS, sensor_group="vt", **_vista_lite_kwargs())
    assert vt.vision_tok is not None
    assert vt.tactile_tok is not None
    assert vt.audio_tok is None
    assert vt.log_mel is None

    v = VisTAPolicy(shape, n_obs_steps=N_OBS, sensor_group="v", **_vista_lite_kwargs())
    assert v.vision_tok is not None
    assert v.tactile_tok is None
    assert v.audio_tok is None


def test_vista_ablation_context_shrinks_and_predict():
    shape = _shape_meta()
    policy = VisTAPolicy(
        shape, n_obs_steps=N_OBS, sensor_group="vt", **_vista_lite_kwargs()
    )
    policy.set_normalizer(_identity_normalizer(shape))
    batch = _batch()
    # 8 cam + 8 finger + 2 proprio (no audio)
    cond = policy.encode_condition(policy._normalize_obs(batch["obs"]))
    assert cond.tokens is not None
    assert cond.tokens.shape == (2, 18, 64)
    out = policy.compute_loss(batch)
    assert out["loss"].ndim == 0
    pred = policy.predict_action(batch["obs"])
    assert pred["action"].shape == (2, 16, 10)


def test_vista_all_sensor_groups_forward():
    shape = _shape_meta()
    batch = _batch()
    for group in SENSOR_GROUPS:
        policy = VisTAPolicy(
            shape, n_obs_steps=N_OBS, sensor_group=group, **_vista_lite_kwargs()
        )
        policy.set_normalizer(_identity_normalizer(shape))
        pred = policy.predict_action(batch["obs"])
        assert pred["action"].shape == (2, 16, 10)


def test_vista_per_sensor_ablation_builds_only_active_encoders():
    shape = _shape_meta()
    vt = VisTAPolicy(
        shape,
        n_obs_steps=N_OBS,
        sensor_group="vt",
        fusion_mode="per_sensor",
        **_vista_lite_kwargs(),
    )
    assert vt.vision_tok is not None
    assert vt.tactile_tok is not None
    assert vt.audio_tok is None
    assert set(vt.fusion.encoders.keys()) == {"camera0_rgb", "finger_rgb", "proprio"}
    assert "mic_0" not in vt.fusion.encoders

    policy = VisTAPolicy(
        shape,
        n_obs_steps=N_OBS,
        sensor_group="vt",
        fusion_mode="per_sensor",
        **_vista_lite_kwargs(),
    )
    policy.set_normalizer(_identity_normalizer(shape))
    batch = _batch()
    # Same cond length as joint vt: 8 cam + 8 finger + 2 proprio
    cond = policy.encode_condition(policy._normalize_obs(batch["obs"]))
    assert cond.tokens is not None
    assert cond.tokens.shape == (2, 18, 64)
    out = policy.compute_loss(batch)
    assert out["loss"].ndim == 0
    pred = policy.predict_action(batch["obs"])
    assert pred["action"].shape == (2, 16, 10)

