"""Masking utilities for self-supervised pretraining."""

from typing import Tuple

import torch


def random_tube_masking(
    tokens: torch.Tensor,
    mask_ratio: float = 0.5,
    tube_len: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Random tube mask over token sequence.

    Returns masked tokens (zeroed at masked positions) and boolean mask.
    """
    b, n, d = tokens.shape
    mask = torch.zeros(b, n, dtype=torch.bool, device=tokens.device)
    n_mask = max(1, int(n * mask_ratio))
    for i in range(b):
        starts = torch.randperm(max(1, n - tube_len + 1), device=tokens.device)[
            : max(1, n_mask // tube_len)
        ]
        for s in starts:
            mask[i, s : s + tube_len] = True
    masked = tokens.clone()
    masked[mask] = 0.0
    return masked, mask
