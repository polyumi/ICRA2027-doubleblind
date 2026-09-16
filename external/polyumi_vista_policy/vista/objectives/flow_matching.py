"""Flow matching objective."""

import torch
import torch.nn.functional as F

from vista.heads.base import PolicyHead
from vista.models.condition import Condition
from vista.objectives.base import Objective


class FlowMatchingObjective(Objective):
    """Conditional flow matching with linear interpolation path."""

    def __init__(self, n_inference_steps: int = 10):
        super().__init__()
        self.n_inference_steps = n_inference_steps

    def compute_loss(
        self,
        head: PolicyHead,
        condition: Condition,
        action: torch.Tensor,
    ) -> torch.Tensor:
        action = action.float()
        b = action.shape[0]
        t = torch.rand(b, device=action.device, dtype=action.dtype)
        x0 = torch.randn_like(action)
        t_view = t.reshape(b, *([1] * (action.ndim - 1)))
        xt = (1 - t_view) * x0 + t_view * action
        target = action - x0
        t_emb = (t * 1000).long()
        pred = head(condition, noisy_action=xt, timestep=t_emb)
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
        x = torch.randn(action_shape, device=device, dtype=dtype)
        dt = 1.0 / self.n_inference_steps
        for i in range(self.n_inference_steps):
            t_val = i / self.n_inference_steps
            t_emb = torch.full((action_shape[0],), int(t_val * 1000), device=device, dtype=torch.long)
            v = head(condition, noisy_action=x, timestep=t_emb)
            x = x + dt * v
        return x
