"""Direct regression objective."""

import torch.nn.functional as F

from vista.heads.base import PolicyHead
from vista.models.condition import Condition
from vista.objectives.base import Objective


class RegressionObjective(Objective):
    """MSE between head output and target actions."""

    def compute_loss(
        self,
        head: PolicyHead,
        condition: Condition,
        action,
    ):
        pred = head(condition)
        return F.mse_loss(pred, action)

    def predict(
        self,
        head: PolicyHead,
        condition: Condition,
        action_shape: tuple,
        device,
        dtype,
    ):
        return head(condition)
