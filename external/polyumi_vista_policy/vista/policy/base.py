"""Shared base for hardcoded Vista multimodal policies."""

from __future__ import annotations

from abc import abstractmethod
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy

PROPRIO_KEYS = (
    "robot0_eef_pos",
    "robot0_eef_rot_axis_angle",
    "robot0_gripper_width",
    "robot0_eef_rot_axis_angle_wrt_start",
)


class BaseVistaPolicy(BaseImagePolicy):
    """
    Normalize observations, expose encoder/head param groups, and define the
    train/predict interface. Concrete policies hardcode encoders + fusion + head.
    """

    def __init__(self, shape_meta: dict, n_obs_steps: int = 2):
        super().__init__()
        self.shape_meta = shape_meta
        action_shape = shape_meta["action"]["shape"]
        self.action_dim = int(action_shape[0])
        self.action_horizon = int(shape_meta["action"]["horizon"])
        self.n_obs_steps = int(n_obs_steps)
        self.normalizer = LinearNormalizer()
        self._encoder_modules: List[nn.Module] = []
        self._head_modules: List[nn.Module] = []

    def _register_encoder_modules(self, *modules: nn.Module) -> None:
        self._encoder_modules.extend(modules)

    def _register_head_modules(self, *modules: nn.Module) -> None:
        self._head_modules.extend(modules)

    def encoder_parameters(self) -> Iterable[nn.Parameter]:
        for module in self._encoder_modules:
            yield from module.parameters()

    def head_parameters(self) -> Iterable[nn.Parameter]:
        for module in self._head_modules:
            yield from module.parameters()

    def set_normalizer(self, normalizer: LinearNormalizer) -> None:
        self.normalizer.load_state_dict(normalizer.state_dict())

    def _normalize_obs(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.normalizer.normalize(obs_dict)

    def _concat_proprio(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Stack proprio keys into (B, N, D_total)."""
        parts = []
        for key in PROPRIO_KEYS:
            if key in obs:
                parts.append(obs[key])
        if not parts:
            raise KeyError("No proprio keys found in observation dict")
        return torch.cat(parts, dim=-1)

    @abstractmethod
    def encode_condition(self, obs: Dict[str, torch.Tensor]):
        """Return whatever the head needs (vector, tokens, or a Condition)."""

    @abstractmethod
    def compute_loss(self, batch: dict) -> Dict[str, torch.Tensor]:
        """Return dict with at least ``loss``."""

    @abstractmethod
    def predict_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
        fixed_action_prefix: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Return ``{"action": ..., "action_pred": ...}``."""

    def forward(self, batch):
        return self.compute_loss(batch)
