"""Training objective base."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from vista.heads.base import PolicyHead
from vista.models.condition import Condition


class Objective(nn.Module, ABC):
    """Compute training loss and run inference sampling."""

    @abstractmethod
    def compute_loss(
        self,
        head: PolicyHead,
        condition: Condition,
        action: torch.Tensor,
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def predict(
        self,
        head: PolicyHead,
        condition: Condition,
        action_shape: tuple,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        pass
