"""
Unit tests for serve_obs — the wire<->UMI obs/action translation.

Pure numpy/scipy; no checkpoint or GPU. Run inside the container's umi env:
    python -m pytest test/test_serve_obs.py -q
(``python -m`` so the fork root is on sys.path and ``import serve_obs`` resolves.)
"""

# ruff: noqa: D103  - test functions are self-describing via names + inline comments

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from serve_obs import actions_rel_to_abs, agent_pos_to_pose_mat, wire_to_obs_dict
from umi.common.pose_util import mat_to_pose10d, pose_to_mat

# rot6d of the identity rotation: the first two rows of I3, flattened (mat_to_rot6d(I)).
IDENTITY_ROT6D = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])

TO = 2  # obs horizon (low_dim_obs_horizon == img_obs_horizon == 2)
H = W = 224


def _agent_pos(pos, rotvec, gripper=0.0):
    """Build a wire agent_pos row [x,y,z, qx,qy,qz,qw, gripper] from pos + axis-angle."""
    quat = Rotation.from_rotvec(rotvec).as_quat()  # xyzw
    return np.concatenate([pos, quat, [gripper]])


def _sample_agent_pos(rng, n=TO):
    rows = [
        _agent_pos(rng.uniform(-0.5, 0.5, 3), rng.uniform(-1.0, 1.0, 3), rng.uniform(0, 1))
        for _ in range(n)
    ]
    return np.stack(rows)


def _rot_close(quat, rotmat, tol=1e-6):
    """Return whether the xyzw quat and the rotation matrix are the same rotation (sign-agnostic)."""
    delta = Rotation.from_quat(quat).inv() * Rotation.from_matrix(rotmat)
    return delta.magnitude() < tol


def test_obs_keys_shapes_dtypes():
    rng = np.random.default_rng(0)
    image = rng.uniform(0, 1, (TO, H, W, 3)).astype(np.float32)
    agent_pos = _sample_agent_pos(rng)

    obs = wire_to_obs_dict(image, agent_pos)

    assert set(obs) == {
        'camera0_rgb',
        'robot0_eef_pos',
        'robot0_eef_rot_axis_angle',
        'robot0_eef_rot_axis_angle_wrt_start',
        'robot0_gripper_width',
    }
    expected = {
        'camera0_rgb': (1, TO, 3, H, W),
        'robot0_eef_pos': (1, TO, 3),
        'robot0_eef_rot_axis_angle': (1, TO, 6),
        'robot0_eef_rot_axis_angle_wrt_start': (1, TO, 6),
        'robot0_gripper_width': (1, TO, 1),
    }
    for key, shape in expected.items():
        assert obs[key].shape == shape, key
        assert obs[key].dtype == np.float32, key


def test_camera_channel_first_no_rescale():
    # A float image is already in [0,1]; the server must only move the channel axis for it.
    rng = np.random.default_rng(1)
    image = rng.uniform(0, 1, (TO, H, W, 3)).astype(np.float32)
    agent_pos = _sample_agent_pos(rng)

    cam = wire_to_obs_dict(image, agent_pos)['camera0_rgb'][0]  # [To,3,H,W]
    assert np.allclose(cam, np.moveaxis(image, -1, 1))
    assert cam.max() <= 1.0


def test_camera_uint8_normalized_here():
    # The client sends uint8 (a quarter of float32's bytes, and the dtype the dataset stores),
    # so the /255 happens here instead. Same values, not merely close ones: the two paths must be
    # indistinguishable to the policy or the wire change is a silent train/inference skew.
    rng = np.random.default_rng(11)
    image_u8 = rng.integers(0, 256, (TO, H, W, 3), dtype=np.uint8)
    agent_pos = _sample_agent_pos(rng)

    from_u8 = wire_to_obs_dict(image_u8, agent_pos)['camera0_rgb']
    from_f32 = wire_to_obs_dict(image_u8.astype(np.float32) / 255.0, agent_pos)['camera0_rgb']

    assert np.array_equal(from_u8, from_f32)
    assert from_u8.dtype == np.float32
    assert from_u8.max() <= 1.0


def test_main_obs_is_relative_to_current_pose():
    # The last obs step is the now-anchor, so its relative pose is the origin:
    # zero position and identity rotation, regardless of the absolute pose.
    rng = np.random.default_rng(2)
    image = np.zeros((TO, H, W, 3), np.float32)
    agent_pos = _sample_agent_pos(rng)

    obs = wire_to_obs_dict(image, agent_pos)
    assert np.allclose(obs['robot0_eef_pos'][0, -1], np.zeros(3), atol=1e-6)
    assert np.allclose(obs['robot0_eef_rot_axis_angle'][0, -1], IDENTITY_ROT6D, atol=1e-6)


def test_gripper_passthrough_into_obs():
    rng = np.random.default_rng(3)
    image = np.zeros((TO, H, W, 3), np.float32)
    agent_pos = _sample_agent_pos(rng)

    obs = wire_to_obs_dict(image, agent_pos)
    assert np.allclose(obs['robot0_gripper_width'][0, :, 0], agent_pos[:, 7], atol=1e-6)


def test_wrt_start_identity_on_fallback():
    # demo_start_pose6=None -> start anchor is the current pose, so the last step's wrt_start
    # rotation is identity (the documented fallback approximation).
    rng = np.random.default_rng(4)
    image = np.zeros((TO, H, W, 3), np.float32)
    agent_pos = _sample_agent_pos(rng)

    obs = wire_to_obs_dict(image, agent_pos, demo_start_pose6=None)
    assert np.allclose(obs['robot0_eef_rot_axis_angle_wrt_start'][0, -1], IDENTITY_ROT6D, atol=1e-6)


def test_wrt_start_uses_cached_start():
    # A real cached start pose yields the exact rotation of current-vs-start, not identity.
    rng = np.random.default_rng(5)
    image = np.zeros((TO, H, W, 3), np.float32)
    agent_pos = _sample_agent_pos(rng)

    # start pose rotated well away from the current poses so wrt_start is clearly non-identity.
    start_pose6 = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 1.2])  # pos + axis-angle (~69 deg about z)
    obs = wire_to_obs_dict(image, agent_pos, demo_start_pose6=start_pose6)

    got = obs['robot0_eef_rot_axis_angle_wrt_start'][0]  # [To,6]
    assert not np.allclose(got[-1], IDENTITY_ROT6D, atol=1e-2)

    # Recompute the expected wrt_start rot6d directly and compare.
    from diffusion_policy.common.pose_repr_util import convert_pose_mat_rep

    pose_mat = agent_pos_to_pose_mat(agent_pos)
    rel = convert_pose_mat_rep(pose_mat, pose_to_mat(start_pose6), pose_rep='relative', backward=False)
    expected = mat_to_pose10d(rel)[:, 3:]
    assert np.allclose(got, expected, atol=1e-6)


def test_rel_abs_roundtrip():
    # Absolute action poses -> relative (forward) -> actions_rel_to_abs -> recover absolute.
    rng = np.random.default_rng(6)
    from diffusion_policy.common.pose_repr_util import convert_pose_mat_rep

    base_mat = pose_to_mat(np.array([0.3, -0.1, 0.5, 0.2, 0.1, -0.3]))
    ta = 5
    abs_pose6 = np.concatenate(
        [rng.uniform(-0.4, 0.4, (ta, 3)), rng.uniform(-1.0, 1.0, (ta, 3))], axis=1
    )
    abs_mat = pose_to_mat(abs_pose6)  # [ta,4,4]
    gripper = np.linspace(0, 1, ta)[:, None]

    rel = convert_pose_mat_rep(abs_mat, base_mat, pose_rep='relative', backward=False)
    action_pred = np.concatenate([mat_to_pose10d(rel), gripper], axis=1)  # [ta,10]

    out = actions_rel_to_abs(action_pred, base_mat)  # [ta,8]
    assert out.shape == (ta, 8)
    assert np.allclose(out[:, :3], abs_mat[:, :3, 3], atol=1e-6)
    assert np.allclose(out[:, 7:8], gripper, atol=1e-6)
    for i in range(ta):
        assert _rot_close(out[i, 3:7], abs_mat[i, :3, :3]), i


def test_rel_abs_translation_equivariance():
    # Same relative chunk + same base rotation but base translation shifted by delta ->
    # absolute positions shift by exactly delta; absolute rotations unchanged.
    rng = np.random.default_rng(7)
    ta = 4
    # Build a valid relative action chunk from arbitrary relative pose mats.
    rel_pose6 = np.concatenate(
        [rng.uniform(-0.2, 0.2, (ta, 3)), rng.uniform(-0.6, 0.6, (ta, 3))], axis=1
    )
    action_pred = np.concatenate([mat_to_pose10d(pose_to_mat(rel_pose6)), np.zeros((ta, 1))], axis=1)

    base_rotvec = np.array([0.3, -0.2, 0.7])
    delta = np.array([0.11, -0.05, 0.2])
    base_a = pose_to_mat(np.concatenate([[0.1, 0.2, 0.3], base_rotvec]))
    base_b = pose_to_mat(np.concatenate([np.array([0.1, 0.2, 0.3]) + delta, base_rotvec]))

    out_a = actions_rel_to_abs(action_pred, base_a)
    out_b = actions_rel_to_abs(action_pred, base_b)

    assert np.allclose(out_b[:, :3] - out_a[:, :3], delta, atol=1e-6)
    assert np.allclose(out_a[:, 3:7], out_b[:, 3:7], atol=1e-6)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-q']))
