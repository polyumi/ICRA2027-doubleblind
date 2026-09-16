"""
Inference server for the PolyUMI visuomotor diffusion policy.

The HTTP surface -- routes, the binary observation frame, the contract every request is checked
against, timing -- belongs to ``polyumi_inference``, the library the ROS-side ``policy_client_node``
also imports. This file is only the *backend*: load a checkpoint, run a forward pass, convert frames.
The dummy server in that same library is the same app with a sine oscillator in this file's place,
which is what makes "the bringup server refuses exactly what a checkpoint refuses" true by
construction rather than by two people keeping two listings in step.

    POST /predict_cartesian/   Content-Type: application/octet-stream
      one binary frame: [4B header length][JSON header][channel blobs]
      channels: camera0_rgb [To,H,W,3] uint8, agent_pos [To,8] float64
    -> {actions: [[8]], n_action_steps, server_total_ms, model_ms}

    POST /reset  {agent_pos: [8]}          # cache the episode-start EEF pose (see below)
    GET  /health

``camera0_rgb`` is uint8 -- what the dataset stores and what the client sends; a float array already
normalized to [0, 1] is accepted too (see ``serve_obs``).

Run it inside the training container (``docker/serve.sh``) -- that is the whole point of using one
image for both roles: the checkpoint is dill-pickled and must unpickle against the exact dependency
tree it was trained with, and the ``umi`` conda env has both ``diffusion_policy``/torch and
fastapi/uvicorn, so this process **direct-imports** the policy (no subprocess).

Two frame conversions happen here (see ``serve_obs.py``):
  - obs: absolute wire poses -> UMI's relative, rot6d, name-matched obs dict.
  - action: the policy's relative chunk -> absolute EEF targets (``convert_pose_mat_rep`` backward).

Episode-start pose: the policy consumes ``robot0_eef_rot_axis_angle_wrt_start`` -- orientation
relative to where the episode began. The wire ``agent_pos`` only carries the *current* pose, so the
client must ``POST /reset`` with the start pose once per rollout; it is cached here. Absent a reset,
``/predict_cartesian/`` falls back to the current pose (``wrt_start`` -> identity) and warns.
"""

from __future__ import annotations  # PEP 604 unions (X | None) on the image's Python 3.9

import logging
import os
import time

import numpy as np

from polyumi_inference import ActionChunk, Observation
from polyumi_inference.contract import REQUIRED_CHANNELS
from polyumi_inference.server import create_app

from serve_obs import (
    actions_rel_to_abs,
    agent_pos_to_pose6,
    agent_pos_to_pose_mat,
    wire_to_obs_dict,
)

# A child of uvicorn's own logger, NOT a bare getLogger('serve_policy'). Uvicorn configures
# handlers for the 'uvicorn*' loggers only and leaves the root logger bare, so a top-level logger
# here propagates to a root with no handler and every INFO line is silently discarded -- which is
# why 'loaded policy from ...' has never appeared in the container output. Hanging off
# 'uvicorn.error' inherits uvicorn's handler and format, so these lines interleave with the
# access log instead of needing a --log-config of their own.
logger = logging.getLogger('uvicorn.error').getChild('serve_policy')


def _sync_batchnorm_stats(source, target) -> int:
    """
    Copy BatchNorm running statistics from ``source`` into ``target``; return how many layers.

    ``EMAModel.step`` averages parameters only, so the EMA copy's ``running_mean``/``running_var``
    never leave their initial 0/1 while the trained model's track the data. In eval mode those
    buffers *are* the normalization, so serving the EMA weights unsynced feeds every later layer
    activations at the wrong scale. Validation runs ``workspace.model``, so val_loss never sees it.
    """
    import torch

    source_modules = dict(source.named_modules())
    n_synced = 0
    with torch.no_grad():
        for name, module in target.named_modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and module.track_running_stats:
                src = source_modules[name]
                module.running_mean.copy_(src.running_mean)
                module.running_var.copy_(src.running_var)
                module.num_batches_tracked.copy_(src.num_batches_tracked)
                n_synced += 1
    return n_synced


def _load_policy(ckpt_path: str):
    """
    Load the dill-pickled, self-describing checkpoint.

    Returns ``(policy, device)``. Mirrors ``base_workspace.load_payload`` + ``train.py`` -- the
    config travels inside the checkpoint, so only a path is needed.
    """
    import dill
    import hydra
    import torch

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with open(ckpt_path, 'rb') as f:
        payload = torch.load(f, pickle_module=dill, map_location='cpu')
    cfg = payload['cfg']
    workspace = hydra.utils.get_class(cfg._target_)(cfg)
    workspace.load_payload(payload)
    policy = workspace.ema_model  # EMA weights -- NOT workspace.model (eval uses EMA)
    n_bn = _sync_batchnorm_stats(workspace.model, policy)
    if n_bn:
        logger.info('synced BatchNorm running stats into the EMA weights (%d layers)', n_bn)
    policy.to(device)
    policy.eval()
    return policy, device


class UmiPolicyBackend:
    """Runs a trained UMI diffusion policy behind ``polyumi_inference``'s app."""

    def __init__(self, policy, device: str, ckpt_path: str) -> None:
        self._policy = policy
        self._device = device
        self._ckpt_path = ckpt_path
        # Episode-start pose (6-vec pos+rotvec), set via POST /reset. None -> current-pose fallback.
        self._demo_start_pose6 = None

    @classmethod
    def from_env(cls) -> UmiPolicyBackend:
        """
        Load the checkpoint named by ``$CKPT_PATH``.

        Fails loudly if the mount is misconfigured: a server that "looks healthy" but cannot serve
        is worse than one that never starts.
        """
        ckpt_path = os.environ.get('CKPT_PATH')
        if not ckpt_path or not os.path.isfile(ckpt_path):
            raise RuntimeError(
                f'CKPT_PATH must point to a checkpoint file; got {ckpt_path!r}. '
                'Set -e CKPT_PATH=/data/checkpoints/<name>.ckpt and mount the checkpoint dir.'
            )
        policy, device = _load_policy(ckpt_path)
        logger.info('loaded policy from %s on %s', ckpt_path, device)
        return cls(policy, device, ckpt_path)

    def reset(self, agent_pos: np.ndarray) -> None:
        """Cache the episode-start EEF pose. Called once at the start of each rollout."""
        self._demo_start_pose6 = agent_pos_to_pose6(np.asarray(agent_pos, dtype=np.float64))

    def describe(self) -> dict:
        """Report the checkpoint, device, and whether /reset has run."""
        return {
            'status': 'ready' if self._policy is not None else 'loading',
            'checkpoint': self._ckpt_path,
            'device': self._device,
            'episode_start_set': self._demo_start_pose6 is not None,
        }

    def predict(self, obs: Observation) -> ActionChunk:
        """Run the policy on one observation window and return an absolute EEF action chunk."""
        import torch

        image_arr = obs['camera0_rgb']
        # float64 because agent_pos_to_pose_mat builds rotations from it; the wire dtype is the
        # client's business, the precision the pose maths needs is ours.
        agent_pos = np.asarray(obs['agent_pos'], dtype=np.float64)

        start6 = self._demo_start_pose6
        if start6 is None:
            logger.warning(
                'no episode start set (POST /reset) -- approximating '
                'robot0_eef_rot_axis_angle_wrt_start with the current pose'
            )

        # Tactile channels are forwarded only when the request carried them. Which channels a
        # request MUST carry is declared once, to create_app, and enforced before predict() is
        # ever reached (Observation.require) — so a missing channel is a 4xx naming it, not a
        # forward pass quietly running on a modality that was never there.
        obs_np = wire_to_obs_dict(
            image_arr,
            agent_pos,
            demo_start_pose6=start6,
            finger_rgb=obs['finger_rgb'] if 'finger_rgb' in obs else None,
            mic_0=obs['mic_0'] if 'mic_0' in obs else None,
        )
        obs_dict = {k: torch.from_numpy(v).to(self._device) for k, v in obs_np.items()}

        # Timed through the .cpu() call, not just predict_action: CUDA kernels launch
        # asynchronously, so stopping the clock at the end of the `with` block would measure
        # queueing, not diffusion. The copy back to host is the synchronization point, and thus the
        # honest end of the work. The backend is the only place this can be measured at all, which
        # is why PolicyBackend asks for it rather than timing the call from outside.
        t_model = time.perf_counter()
        with torch.no_grad():
            action_pred = self._policy.predict_action(obs_dict)['action_pred']
        action_pred = action_pred[0].detach().cpu().numpy()  # [Ta, 10] relative to current pose
        model_ms = (time.perf_counter() - t_model) * 1e3

        # The current EEF pose (agent_pos[-1]) is the base the policy's chunk is relative to.
        base_pose_mat = agent_pos_to_pose_mat(agent_pos)[-1]
        # Truncation to what the client asked for is the app's; UMI's policy emits the full horizon
        # with no offset, so everything here is a legitimate action.
        return ActionChunk(actions_rel_to_abs(action_pred, base_pose_mat), model_ms=model_ms)


# Which channels a request must carry. The visuomotor default is the shared library's; the Vista
# family (vista, sparsh_x, polytouch, see_hear_feel) additionally indexes obs['finger_rgb'] and
# obs['mic_0'] unconditionally, so for those a request without them is not servable.
#
# An env var rather than the checkpoint's own shape_meta only because create_app fixes this before
# the backend loads. It MUST match the checkpoint, and it must match the ROS node's `send_tactile`;
# getting it wrong one way is a 4xx naming the missing channel, and the other way is a forward pass
# on a modality the model never trained with. Logged at startup for that reason.
_SERVE_TACTILE = os.environ.get('SERVE_TACTILE', '').lower() in ('1', 'true', 'yes')
_REQUIRED_CHANNELS = (
    (*REQUIRED_CHANNELS, 'finger_rgb', 'mic_0') if _SERVE_TACTILE else REQUIRED_CHANNELS
)
logger.info('required channels: %s', ', '.join(_REQUIRED_CHANNELS))

# from_env, not an instance: create_app calls it at startup, so a missing checkpoint is a startup
# failure rather than a health check that passes and a rollout that 500s.
app = create_app(
    UmiPolicyBackend.from_env,
    title='PolyUMI Inference Server',
    required_channels=_REQUIRED_CHANNELS,
)
