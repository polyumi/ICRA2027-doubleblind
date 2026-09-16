"""
Observation/action translation between the ROS wire contract and the UMI policy.

The ROS-side ``policy_client_node`` speaks the same HTTP contract as ``dummy_server``: absolute
EEF poses as ``[x, y, z, qx, qy, qz, qw, gripper]`` (quaternion, robot base frame) plus a stacked
image. The UMI policy consumes a *different* obs dict — name-matched keys, rotations as ``rot6d``,
poses expressed relative to the current EEF pose. These pure-numpy functions do that translation
both ways so they can be unit-tested without a checkpoint or a GPU; ``serve_policy.py`` wraps them
with torch/device handling.

The transform mirrors ``UmiDataset.__getitem__`` (diffusion_policy/dataset/umi_dataset.py) for the
single-robot case. ``pose_repr`` is ``relative`` for both obs and action (task/polyumi.yaml).

Note on ``mat_to_pose10d``: despite the name it returns **9** values (pos[3] + rot6d[6]); the
policy's 10-D action is that 9-vec plus a trailing gripper scalar.
"""

from __future__ import annotations  # PEP 604 unions (X | None) on the image's Python 3.9

import numpy as np
from scipy.spatial.transform import Rotation

from diffusion_policy.common.pose_repr_util import convert_pose_mat_rep
from umi.common.pose_util import mat_to_pose10d, pose10d_to_mat, pose_to_mat

# pose_repr from task/polyumi.yaml (obs_pose_repr == action_pose_repr == 'relative').
POSE_REPR = 'relative'


def agent_pos_to_pose6(agent_pos: np.ndarray) -> np.ndarray:
    """
    Convert a wire ``agent_pos`` ``[..., 8]`` (pos + quat_xyzw + gripper) to a 6-vec pose.

    Returns ``[..., 6]`` = ``[x, y, z, rvx, rvy, rvz]`` (position + axis-angle), the form
    ``pose_to_mat`` expects.
    """
    agent_pos = np.asarray(agent_pos, dtype=np.float64)
    pos = agent_pos[..., :3]
    rotvec = Rotation.from_quat(agent_pos[..., 3:7]).as_rotvec()
    return np.concatenate([pos, rotvec], axis=-1)


def agent_pos_to_pose_mat(agent_pos: np.ndarray) -> np.ndarray:
    """Convert wire ``agent_pos`` ``[To, 8]`` to homogeneous pose matrices ``[To, 4, 4]``."""
    return pose_to_mat(agent_pos_to_pose6(agent_pos))


def wire_to_obs_dict(
    image: np.ndarray | None,
    agent_pos: np.ndarray,
    demo_start_pose6: np.ndarray | None = None,
    finger_rgb: np.ndarray | None = None,
    mic_0: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """
    Translate one wire observation into the UMI policy's obs dict (batched ``[1, To, ...]``).

    Args:
        image: ``[To, H, W, 3]`` or ``None`` when the policy has no camera0_rgb (gripper-only).
            ``uint8`` in ``[0, 255]`` — what the dataset stores and what
            the client puts on the wire — or a float already normalized to ``[0, 1]``. Integer
            input is divided by 255 here; the two paths are bit-identical.
        agent_pos: ``[To, 8]`` absolute EEF poses ``[x, y, z, qx, qy, qz, qw, gripper]``.
        demo_start_pose6: cached episode-start pose ``[6]`` (pos + axis-angle) set via
            ``POST /reset``. ``None`` falls back to the current pose, which makes
            ``robot0_eef_rot_axis_angle_wrt_start`` collapse to identity — an approximation
            (see serve_policy.py) rather than the trained signal.
        finger_rgb: ``[To, H, W, 3]`` finger-camera frames, same dtype convention as ``image``.
            ``None`` omits the key, which is what a visuomotor checkpoint wants.
        mic_0: ``[rows, samples]`` raw contact-mic waveform. Its leading dim is audio ROWS
            (``audio_obs_horizon``), not ``To`` — the model's audio encoder consumes the window
            whole — so it is batched but never reshaped to the observation horizon.

    Returns:
        dict of ``np.float32`` arrays keyed by the exact ``shape_meta`` obs names.

    """
    if image is not None:
        image = np.asarray(image)
    if image is not None and np.issubdtype(image.dtype, np.integer):
        # The client sends uint8 because float32 is four times the bytes for no extra information
        # (the dataset itself stores camera0_rgb as uint8). Normalizing here rather than there is
        # the same arithmetic on the same values, so the policy cannot tell which side did it.
        image = image.astype(np.float32) / 255.0
    elif image is not None:
        image = image.astype(np.float32)
    agent_pos = np.asarray(agent_pos, dtype=np.float64)

    pose_mat = agent_pos_to_pose_mat(agent_pos)  # [To, 4, 4]
    # Now-anchor: the current EEF pose (agent_pos[-1]). This is the exact base for the main obs
    # AND for the rel->abs action conversion in serve_policy — it is what makes the policy
    # position-invariant.
    base = pose_mat[-1]

    rel = convert_pose_mat_rep(pose_mat, base, pose_rep=POSE_REPR, backward=False)
    o9 = mat_to_pose10d(rel)  # [To, 9] = pos[3] + rot6d[6]

    # wrt_start uses a *separate* base: the episode-start pose. It feeds only
    # robot0_eef_rot_axis_angle_wrt_start (the policy's coarse sense of orientation drift since the
    # episode began). Substituting the current pose here corrupts that signal as the arm moves, so
    # the real start pose is cached via /reset; None here is the explicit fallback.
    start_mat = base if demo_start_pose6 is None else pose_to_mat(
        np.asarray(demo_start_pose6, dtype=np.float64)
    )
    rel_start = convert_pose_mat_rep(pose_mat, start_mat, pose_rep=POSE_REPR, backward=False)
    o9_start = mat_to_pose10d(rel_start)  # [To, 9]

    # NB: the two rot keys say "axis_angle" but carry **rot6d** (6 values), not a 3-vec axis-angle.
    # The names come verbatim from the model's shape_meta (obs are named after the raw replay-buffer
    # fields, which UmiDataset then runs through a rotation_transformer to rot6d before the policy
    # sees them); we must key by those exact strings. Don't "fix" the names or truncate to 3.
    obs = {
        'robot0_eef_pos': o9[:, :3],  # [To, 3]
        'robot0_eef_rot_axis_angle': o9[:, 3:],  # [To, 6] rot6d (see note above re: name)
        'robot0_eef_rot_axis_angle_wrt_start': o9_start[:, 3:],  # [To, 6] rot6d (see note above)
        'robot0_gripper_width': agent_pos[:, 7:8],  # [To, 1]
    }
    if image is not None:
        obs['camera0_rgb'] = np.moveaxis(image, -1, 1)  # [To, 3, H, W], already [0,1]
    # Tactile channels, added only when the caller supplies them, so one function serves both a
    # visuomotor checkpoint and a tactile+audio one. Each reproduces exactly what UmiDataset does
    # at training time -- finger_rgb through the same channel-move and /255 as camera0_rgb
    # (umi_dataset.py:312), mic_0 passed through untouched under an identity normalizer. Anything
    # else here is train/serve skew: the shapes still match, so nothing raises, and the policy
    # merely receives a distribution it never saw.
    if finger_rgb is not None:
        finger = np.asarray(finger_rgb)
        if np.issubdtype(finger.dtype, np.integer):
            finger = finger.astype(np.float32) / 255.0
        else:
            finger = finger.astype(np.float32)
        obs['finger_rgb'] = np.moveaxis(finger, -1, 1)  # [To, 3, H, W]
    if mic_0 is not None:
        obs['mic_0'] = np.asarray(mic_0, dtype=np.float32)  # [rows, samples]
    # Add the batch dim and cast to the float32 the policy expects.
    return {k: v[None].astype(np.float32) for k, v in obs.items()}


def actions_rel_to_abs(action_pred: np.ndarray, base_pose_mat: np.ndarray) -> np.ndarray:
    """
    Convert a relative action chunk ``[Ta, 10]`` to absolute wire actions ``[Ta, 8]``.

    Args:
        action_pred: ``[Ta, 10]`` = pos[3] + rot6d[6] + gripper[1], relative to the current pose.
        base_pose_mat: ``[4, 4]`` current EEF pose the chunk is relative to (``agent_pos[-1]``).

    Returns:
        ``[Ta, 8]`` = ``[x, y, z, qx, qy, qz, qw, gripper]`` absolute in robot base frame.

    """
    action_pred = np.asarray(action_pred, dtype=np.float64)
    pose9 = action_pred[:, :9]  # pos[3] + rot6d[6]
    gripper = action_pred[:, 9:10]

    action_mat = pose10d_to_mat(pose9)  # [Ta, 4, 4]
    # backward transform: base @ mat (see convert_pose_mat_rep 'relative' branch).
    abs_mat = convert_pose_mat_rep(action_mat, base_pose_mat, pose_rep=POSE_REPR, backward=True)

    pos = abs_mat[:, :3, 3]
    quat = Rotation.from_matrix(abs_mat[:, :3, :3]).as_quat()  # xyzw
    return np.concatenate([pos, quat, gripper], axis=-1)  # [Ta, 8]
