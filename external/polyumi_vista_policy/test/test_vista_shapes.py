"""Shape and forward-pass tests for hardcoded baseline policies."""

import numpy as np
import torch

from diffusion_policy.common.normalize_util import (
    array_to_stats,
    get_identity_normalizer_from_stat,
    get_image_identity_normalizer,
)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from vista.models.polytouch import PolyTouchPolicy
from vista.models.see_hear_feel import SeeHearFeelPolicy
from vista.models.sparsh_x import SparshXPolicy
from vista.models.qformer import QformerPolicy
from vista.models.mitas import MitasPolicy
from vista.models.vta_diffusion import VTADiffusionPolicy
from vista.encoders.cnn import AudioCNNStem, VisionCNNStem
from vista.preproc.log_mel import mic_rows_to_waveform, shared_log_mel

N_OBS = 2
AUDIO_HORIZON = 10


def _identity_normalizer(shape_meta):
    n = LinearNormalizer()
    for key in shape_meta["obs"]:
        n[key] = get_image_identity_normalizer()
    stat = array_to_stats(np.zeros((1, shape_meta["action"]["shape"][0]), dtype=np.float32))
    n["action"] = get_identity_normalizer_from_stat(stat)
    # Keep everything float32 (numpy defaults can promote to float64).
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


def _shape_meta(n_obs: int = N_OBS, audio_h: int = AUDIO_HORIZON, img: int = 64):
    return {
        "obs": {
            "camera0_rgb": {"shape": [3, img, img], "horizon": n_obs},
            "finger_rgb": {"shape": [3, img, img], "horizon": n_obs},
            "mic_0": {"shape": [536], "horizon": audio_h},
            "robot0_eef_pos": {"shape": [3], "horizon": n_obs},
            "robot0_eef_rot_axis_angle": {"shape": [6], "horizon": n_obs},
            "robot0_gripper_width": {"shape": [1], "horizon": n_obs},
            "robot0_eef_rot_axis_angle_wrt_start": {"shape": [6], "horizon": n_obs},
        },
        "action": {"shape": [10], "horizon": 16},
    }


def _synthetic_batch(batch=2, n_obs=N_OBS, audio_h=AUDIO_HORIZON, img=64):
    obs = {
        "camera0_rgb": torch.rand(batch, n_obs, 3, img, img),
        "finger_rgb": torch.rand(batch, n_obs, 3, img, img),
        "mic_0": torch.randn(batch, audio_h, 536),
        "robot0_eef_pos": torch.randn(batch, n_obs, 3),
        "robot0_eef_rot_axis_angle": torch.randn(batch, n_obs, 6),
        "robot0_gripper_width": torch.rand(batch, n_obs, 1),
        "robot0_eef_rot_axis_angle_wrt_start": torch.randn(batch, n_obs, 6),
    }
    action = torch.randn(batch, 16, 10)
    return {"obs": obs, "action": action}


def test_see_hear_feel_forward_and_loss():
    shape = _shape_meta()
    policy = SeeHearFeelPolicy(
        shape,
        n_obs_steps=N_OBS,
        d_embed=32,
        n_heads=4,
        mlp_hidden=64,
        cond_dim=32,
        down_dims=(64, 128),
        num_train_timesteps=4,
        num_inference_steps=2,
    )
    policy.set_normalizer(_identity_normalizer(shape))
    batch = _synthetic_batch()
    cond = policy.encode_condition(policy._normalize_obs(batch["obs"]))
    assert cond.vector is not None
    assert cond.vector.shape == (2, 32)
    out = policy.compute_loss(batch)
    assert out["loss"].ndim == 0
    pred = policy.predict_action(batch["obs"])
    assert pred["action"].shape == (2, 16, 10)


def test_sparsh_x_forward_and_loss():
    shape = _shape_meta(img=64)
    policy = SparshXPolicy(
        shape,
        n_obs_steps=N_OBS,
        d_embed=64,
        depth=2,
        fusion_layer=1,
        num_heads=4,
        num_bottlenecks=2,
        dit_layers=1,
        adaln_zero=True,
        n_inference_steps=2,
        patch_size=16,
        audio_patch_size=8,
    )
    policy.set_normalizer(_identity_normalizer(shape))
    batch = _synthetic_batch(img=64)
    # Channel-stack → patch-16 on 64² → 4×4 = 16 tokens/camera; audio patch-8 → 96.
    assert policy.vision_tok(batch["obs"]["camera0_rgb"]).shape == (2, 16, 64)
    assert policy.audio_tok(batch["obs"]["mic_0"]).shape == (2, 96, 64)
    # Learned pos tables (Meta Sparsh default) sized to each modality's token count.
    assert policy.pos_embeds.pos["vision"].shape == (1, 16, 64)
    assert policy.pos_embeds.pos["tactile"].shape == (1, 16, 64)
    assert policy.pos_embeds.pos["audio"].shape == (1, 96, 64)
    assert policy.pos_embeds.pos["proprio"].shape == (1, N_OBS, 64)
    cond = policy.encode_condition(policy._normalize_obs(batch["obs"]))
    assert cond.tokens is None
    assert cond.vector.shape == (2, 64)
    out = policy.compute_loss(batch)
    assert out["loss"].ndim == 0
    pred = policy.predict_action(batch["obs"])
    assert pred["action"].shape == (2, 16, 10)


def test_polytouch_lite_forward_and_loss():
    # PolyTouch left on shared contract but architecture unchanged; same batch shapes.
    shape = _shape_meta(img=64)
    policy = PolyTouchPolicy(
        shape,
        n_obs_steps=N_OBS,
        d_model=64,
        n_blocks=2,
        n_heads=4,
        cond_dim=32,
        pretrained=False,
        lite=True,
        num_train_timesteps=4,
        num_inference_steps=2,
        down_dims=(64, 128),
    )
    policy.set_normalizer(_identity_normalizer(shape))
    batch = _synthetic_batch(img=64)
    out = policy.compute_loss(batch)
    assert out["loss"].ndim == 0
    pred = policy.predict_action(batch["obs"])
    assert pred["action"].shape == (2, 16, 10)


def test_vta_diffusion_forward_and_loss():
    shape = _shape_meta(img=64)
    policy = VTADiffusionPolicy(
        shape,
        n_obs_steps=N_OBS,
        d_embed=32,
        backbone="resnet18",
        down_dims=(64, 128),
        num_train_timesteps=4,
        num_inference_steps=2,
    )
    policy.set_normalizer(_identity_normalizer(shape))
    batch = _synthetic_batch(img=64)
    cond = policy.encode_condition(policy._normalize_obs(batch["obs"]))
    assert cond.vector is not None
    assert cond.vector.shape == (2, 4 * 32)
    out = policy.compute_loss(batch)
    assert out["loss"].ndim == 0
    pred = policy.predict_action(batch["obs"])
    assert pred["action"].shape == (2, 16, 10)


def test_encoder_head_param_groups_disjoint():
    shape = _shape_meta()
    policy = SeeHearFeelPolicy(
        shape,
        n_obs_steps=N_OBS,
        d_embed=32,
        n_heads=4,
        mlp_hidden=64,
        cond_dim=32,
        down_dims=(64, 128),
        num_train_timesteps=4,
        num_inference_steps=2,
    )
    enc_ids = {id(p) for p in policy.encoder_parameters()}
    head_ids = {id(p) for p in policy.head_parameters()}
    assert enc_ids
    assert head_ids
    assert enc_ids.isdisjoint(head_ids)


def test_log_mel_shapes():
    from vista.preproc.log_mel import mic_rows_to_waveform, shf_log_mel, shared_log_mel

    mic = torch.randn(2, AUDIO_HORIZON, 536)
    wav = mic_rows_to_waveform(mic)
    assert wav.shape == (2, AUDIO_HORIZON * 536)
    shf = shf_log_mel()(wav)
    assert shf.shape[0] == 2 and shf.shape[1] == 1 and shf.shape[2] == 128
    sp = shared_log_mel()(wav)
    assert sp.shape == (2, 1, 128, 48)


def test_audio_cnn_stem_tokens():
    mic = torch.randn(2, AUDIO_HORIZON, 536)
    mel = shared_log_mel()(mic_rows_to_waveform(mic))
    tokens = AudioCNNStem(d_embed=64)(mel)
    assert tokens.shape == (2, 96, 64)


def test_vision_cnn_stem_tokens_h2():
    # 64² / 32 → 2×2 = 4 per frame; H=2 → 8 tokens.
    x = torch.rand(2, N_OBS, 3, 64, 64)
    tokens = VisionCNNStem(d_embed=64)(x)
    assert tokens.shape == (2, 8, 64)


def test_vista_forward_and_loss():
    shape = _shape_meta(img=64)
    policy = QformerPolicy(
        shape,
        n_obs_steps=N_OBS,
        sensor_group="vta",
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
    policy.set_normalizer(_identity_normalizer(shape))
    assert policy.pos_camera is not None and policy.pos_camera.spatial is not None
    assert policy.pos_finger is not None and policy.pos_finger.spatial is not None
    assert policy.pos_temporal is not None and policy.pos_temporal.temporal is not None
    assert policy.pos_camera.spatial.shape == (1, 4, 64)  # 64/32 → 2×2
    assert policy.pos_temporal.temporal.shape == (1, N_OBS, 1, 64)
    batch = _synthetic_batch(img=64)
    cond = policy.encode_condition(policy._normalize_obs(batch["obs"]))
    assert cond.tokens is not None
    assert cond.tokens.shape == (2, 8, 64)
    out = policy.compute_loss(batch)
    assert out["loss"].ndim == 0
    pred = policy.predict_action(batch["obs"])
    assert pred["action"].shape == (2, 16, 10)


def test_mitas_forward_and_loss():
    shape = _shape_meta(img=64)
    policy = MitasPolicy(
        shape,
        n_obs_steps=N_OBS,
        sensor_group="vta",
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
    policy.set_normalizer(_identity_normalizer(shape))
    assert policy.pos_camera is not None and policy.pos_camera.spatial is not None
    assert policy.pos_finger is not None and policy.pos_finger.spatial is not None
    assert policy.pos_temporal is not None and policy.pos_temporal.temporal is not None
    batch = _synthetic_batch(img=64)
    # 64²/32 → 4 tok/frame × H=2 → 8 cam + 8 finger + 96 audio + 2 proprio
    cond = policy.encode_condition(policy._normalize_obs(batch["obs"]))
    assert cond.tokens is not None
    assert cond.tokens.shape == (2, 114, 64)
    out = policy.compute_loss(batch)
    assert out["loss"].ndim == 0
    pred = policy.predict_action(batch["obs"])
    assert pred["action"].shape == (2, 16, 10)


def _mitas_lite_kwargs():
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


def test_mitas_diffusion_objective_forward_and_loss():
    shape = _shape_meta(img=64)
    policy = MitasPolicy(
        shape,
        n_obs_steps=N_OBS,
        sensor_group="vta",
        objective_type="diffusion",
        num_train_timesteps=4,
        input_perturb=0.1,
        **_mitas_lite_kwargs(),
    )
    from vista.objectives.diffusion import DiffusionObjective

    assert isinstance(policy.objective, DiffusionObjective)
    policy.set_normalizer(_identity_normalizer(shape))
    batch = _synthetic_batch(img=64)
    out = policy.compute_loss(batch)
    assert out["loss"].ndim == 0
    pred = policy.predict_action(batch["obs"])
    assert pred["action"].shape == (2, 16, 10)


def test_mitas_per_sensor_fusion_same_token_count():
    shape = _shape_meta(img=64)
    batch = _synthetic_batch(img=64)
    joint = MitasPolicy(
        shape, n_obs_steps=N_OBS, sensor_group="vta", fusion_mode="joint", **_mitas_lite_kwargs()
    )
    per = MitasPolicy(
        shape,
        n_obs_steps=N_OBS,
        sensor_group="vta",
        fusion_mode="per_sensor",
        **_mitas_lite_kwargs(),
    )
    from vista.fusion.per_sensor_transformer import PerSensorTransformerFusion
    from vista.fusion.transformer import TransformerFusion

    assert isinstance(joint.fusion, TransformerFusion)
    assert isinstance(per.fusion, PerSensorTransformerFusion)
    assert set(per.fusion.encoders.keys()) == {
        "camera0_rgb",
        "finger_rgb",
        "mic_0",
        "proprio",
    }

    joint.set_normalizer(_identity_normalizer(shape))
    per.set_normalizer(_identity_normalizer(shape))
    j_cond = joint.encode_condition(joint._normalize_obs(batch["obs"]))
    p_cond = per.encode_condition(per._normalize_obs(batch["obs"]))
    assert j_cond.tokens is not None and p_cond.tokens is not None
    assert j_cond.tokens.shape == p_cond.tokens.shape == (2, 114, 64)
    out = per.compute_loss(batch)
    assert out["loss"].ndim == 0
    pred = per.predict_action(batch["obs"])
    assert pred["action"].shape == (2, 16, 10)

