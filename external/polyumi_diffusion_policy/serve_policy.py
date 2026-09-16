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
from polyumi_inference.contract import AGENT_POS_CHANNEL
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
    never track the data the way the trained model's do. In eval mode those buffers *are* the
    normalization, so serving the EMA weights unsynced feeds every later layer activations at the
    wrong scale. Validation runs ``workspace.model``, so val_loss never sees it.
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
    # How this checkpoint's gripper action is expressed. Read from the checkpoint rather than a
    # serve-time flag: the answer is a property of the training run, and a flag that disagrees
    # with it is a silent offset on every command, not an error anyone would see.
    gripper_repr = 'abs'
    try:
        gripper_repr = cfg.task.pose_repr.get('action_gripper_repr', 'abs')
    except Exception:
        pass
    return policy, device, gripper_repr


class UmiPolicyBackend:
    """Runs a trained UMI diffusion policy behind ``polyumi_inference``'s app."""

    def __init__(self, policy, device: str, ckpt_path: str, wanted_keys: set | None = None,
                 gripper_repr: str = 'abs') -> None:
        self._policy = policy
        self._gripper_repr = gripper_repr
        self._device = device
        self._ckpt_path = ckpt_path
        # Obs keys the checkpoint's own shape_meta declares; drives which optional channels
        # predict() pulls off the wire. Empty means "visuomotor only", the pre-tactile behaviour.
        self._wanted_keys = set(wanted_keys or ())
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
        policy, device, gripper_repr = _load_policy(ckpt_path)
        # Which extra channels this checkpoint wants is a property of the CHECKPOINT, read from the
        # shape_meta it was trained with -- not a flag, and not something the client can assert.
        # A visuomotor checkpoint must keep working unchanged when the client happens to send
        # tactile channels, and a tactile one must refuse to run without them rather than quietly
        # encoding whatever it finds.
        # From the ENCODER, not the policy: DiffusionUnetTimmPolicy takes shape_meta as a
        # constructor argument but never stores it, so asking the policy silently yields an empty
        # set and every optional channel is dropped. The encoder's own key lists are the most
        # direct source available -- they are literally what its forward() iterates over.
        encoder = getattr(policy, 'obs_encoder', None)
        wanted = (
            set(getattr(encoder, 'rgb_keys', ()))
            | set(getattr(encoder, 'audio_keys', ()))
            | set(getattr(encoder, 'low_dim_keys', ()))
        )
        if not wanted:  # older encoder without the key lists
            wanted = set(getattr(encoder, 'shape_meta', {}).get('obs', {}))
        logger.info('loaded policy from %s on %s', ckpt_path, device)
        logger.info('checkpoint obs keys: %s', sorted(wanted))
        return cls(policy, device, ckpt_path, wanted, gripper_repr)

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
            'obs_keys': sorted(self._wanted_keys),
        }

    def predict(self, obs: Observation) -> ActionChunk:
        """Run the policy on one observation window and return an absolute EEF action chunk."""
        import torch

        # Only pulled when the checkpoint declares it: a gripper-only policy has no camera0_rgb in
        # its shape_meta, and the client omits it from the request to save the largest array on the
        # wire. Reading it unconditionally would KeyError before the gate below could explain why.
        image_arr = obs['camera0_rgb'] if 'camera0_rgb' in self._wanted_keys else None
        # float64 because agent_pos_to_pose_mat builds rotations from it; the wire dtype is the
        # client's business, the precision the pose maths needs is ours.
        agent_pos = np.asarray(obs['agent_pos'], dtype=np.float64)

        # Pulled only when the checkpoint declares them. Missing here is a hard error rather than a
        # silent omission: the alternative is a KeyError from deep inside the obs encoder's forward
        # pass, which says nothing about which side failed to hold up the contract.
        finger_rgb = mic_0 = None
        for key, target in (('finger_rgb', 'finger_rgb'), ('mic_0', 'mic_0')):
            if key not in self._wanted_keys:
                continue
            if key not in obs:
                raise KeyError(
                    f'checkpoint {self._ckpt_path} was trained with {key!r} but the observation '
                    f'does not carry it; the client has send_tactile off, or is older than this '
                    # names(), not sorted(obs): Observation defines __getitem__ and __contains__
                    # but no __iter__, so sorting it falls back to the legacy __getitem__(0)
                    # protocol and dies with KeyError: 0 -- the error handler crashing instead of
                    # reporting the error.
                    f'checkpoint. Wire keys present: {obs.names()}'
                )
            if target == 'finger_rgb':
                finger_rgb = obs[key]
            else:
                mic_0 = obs[key]

        start6 = self._demo_start_pose6
        if start6 is None:
            logger.warning(
                'no episode start set (POST /reset) -- approximating '
                'robot0_eef_rot_axis_angle_wrt_start with the current pose'
            )

        obs_np = wire_to_obs_dict(
            image_arr, agent_pos, demo_start_pose6=start6, finger_rgb=finger_rgb, mic_0=mic_0
        )
        # Keep only what this checkpoint was trained on. wire_to_obs_dict always builds the full
        # visuomotor set (poses, rot6d, wrt_start, gripper) because the wire carries agent_pos
        # regardless; a gripper-only policy's normalizer has no entry for the pose keys, and
        # LinearNormalizer raises AttributeError on the first one it does not recognise rather
        # than ignoring it. Filtering here keeps that contract in one place instead of teaching
        # wire_to_obs_dict about every policy shape.
        obs_dict = {
            k: torch.from_numpy(v).to(self._device)
            for k, v in obs_np.items()
            if not self._wanted_keys or k in self._wanted_keys
        }

        # Timed through the .cpu() call, not just predict_action: CUDA kernels launch
        # asynchronously, so stopping the clock at the end of the `with` block would measure
        # queueing, not diffusion. The copy back to host is the synchronization point, and thus the
        # honest end of the work. The backend is the only place this can be measured at all, which
        # is why PolicyBackend asks for it rather than timing the call from outside.
        t_model = time.perf_counter()
        with torch.no_grad():
            action_pred = self._policy.predict_action(obs_dict)['action_pred']
        action_pred = action_pred[0].detach().cpu().numpy()  # [Ta, 10] rel to current pose, or [Ta, 1] gripper-only
        model_ms = (time.perf_counter() - t_model) * 1e3

        # The current EEF pose (agent_pos[-1]) is the base the policy's chunk is relative to.
        base_pose_mat = agent_pos_to_pose_mat(agent_pos)[-1]

        if action_pred.shape[-1] == 1:
            # Gripper-only policy: it predicts width and nothing else. The wire action stays 8-wide
            # so the client, the contract and the dummy server are all untouched -- the pose columns
            # are filled with the CURRENT pose, i.e. "stay here". That is already what the client's
            # gripper_only mode commands, so the two agree by construction rather than by accident;
            # and a client without that mode still gets a hold rather than a jump to the origin.
            abs_actions = np.repeat(agent_pos[-1:, :8], action_pred.shape[0], axis=0)
            if self._gripper_repr == 'relative':
                # The policy predicted a CHANGE from the width it was conditioned on, which is
                # agent_pos[-1, 7] -- the same base UmiDataset subtracted at training time. The
                # wire stays absolute, so the conversion belongs here rather than in the client.
                abs_actions[:, 7] = agent_pos[-1, 7] + action_pred[:, 0]
            else:
                abs_actions[:, 7] = action_pred[:, 0]
            return ActionChunk(abs_actions, model_ms=model_ms)

        # Truncation to what the client asked for is the app's; UMI's policy emits the full horizon
        # with no offset, so everything here is a legitimate action.
        return ActionChunk(actions_rel_to_abs(action_pred, base_pose_mat), model_ms=model_ms)


# from_env, not an instance: create_app calls it at startup, so a missing checkpoint is a startup
# failure rather than a health check that passes and a rollout that 500s.
# required_channels is relaxed to agent_pos alone. The shared default also demands camera0_rgb,
# which a gripper-only checkpoint neither declares nor wants -- and the client omits it there, so
# the app-level check would reject every request before the backend saw one. The backend enforces
# the real requirement instead, from the checkpoint's own shape_meta, and its error names the
# missing channel and lists what the wire carried. agent_pos stays mandatory because the pose maths
# and /reset need it regardless of which modalities a policy consumes.
app = create_app(
    UmiPolicyBackend.from_env,
    title='PolyUMI Inference Server',
    required_channels=(AGENT_POS_CHANNEL,),
)
