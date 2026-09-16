"""DDPM diffusion objective."""

from typing import Optional

import torch
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from vista.heads.base import PolicyHead
from vista.models.condition import Condition
from vista.objectives.base import Objective


class DiffusionObjective(Objective):
    """Standard epsilon-prediction diffusion loss."""

    def __init__(
        self,
        noise_scheduler: DDPMScheduler,
        num_inference_steps: Optional[int] = None,
        input_perturb: float = 0.1,
    ):
        super().__init__()
        self.noise_scheduler = noise_scheduler
        self.num_inference_steps = num_inference_steps or noise_scheduler.config.num_train_timesteps
        self.input_perturb = input_perturb

    def compute_loss(
        self,
        head: PolicyHead,
        condition: Condition,
        action: torch.Tensor,
    ) -> torch.Tensor:
        noise = torch.randn_like(action)
        if self.input_perturb > 0:
            noise = noise + self.input_perturb * torch.randn_like(action)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (action.shape[0],),
            device=action.device,
        ).long()
        noisy = self.noise_scheduler.add_noise(action, noise, timesteps)
        pred = head(condition, noisy_action=noisy, timestep=timesteps)
        target = noise if self.noise_scheduler.config.prediction_type == "epsilon" else action
        return F.mse_loss(pred, target)

    @torch.no_grad()
    def predict(
        self,
        head: PolicyHead,
        condition: Condition,
        action_shape: tuple,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        scheduler = self.noise_scheduler
        trajectory = torch.randn(action_shape, device=device, dtype=dtype)
        # device= matters: without it diffusers builds `timesteps` on the CPU, and the CPU `t`
        # then meets CUDA weights inside the head's step encoder. Training never exercises this
        # path -- it draws its own timesteps with torch.randint(..., device=action.device) -- so
        # the mismatch only ever surfaces at inference.
        scheduler.set_timesteps(self.num_inference_steps, device=device)
        for t in scheduler.timesteps:
            pred = head(condition, noisy_action=trajectory, timestep=t.expand(trajectory.shape[0]))
            trajectory = scheduler.step(pred, t, trajectory).prev_sample
        return trajectory
