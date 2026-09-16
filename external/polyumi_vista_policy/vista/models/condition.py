"""Conditioning output from ObservationConditioner."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Callable, Optional

import torch


@dataclass
class Condition:
    """
    Fusion output consumed by policy heads.

    ``vector`` is computed lazily so DiT heads never pay for pooling and UNet
    heads can drop token tensors after pooling.
    """

    tokens: Optional[torch.Tensor]
    pad_mask: Optional[torch.Tensor]
    _pool_fn: Optional[Callable[[Optional[torch.Tensor], Optional[torch.Tensor]], torch.Tensor]] = None
    _vector: Optional[torch.Tensor] = None

    @cached_property
    def vector(self) -> torch.Tensor:
        if self._vector is not None:
            return self._vector
        if self._pool_fn is None:
            raise RuntimeError("Condition has no pool_fn and no precomputed vector.")
        return self._pool_fn(self.tokens, self.pad_mask)

    @classmethod
    def from_vector(cls, vector: torch.Tensor) -> Condition:
        return cls(tokens=None, pad_mask=None, _vector=vector)
