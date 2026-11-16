"""Small tensor operations Fish Audio's models and codec share."""

from __future__ import annotations

import torch
from torch import Tensor


def rms_norm(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    """Normalized in float32, scaled in the input's precision (as the checkpoint was trained)."""
    x32 = x.float()
    return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)).type_as(x) * weight


def rope_table(positions: int, head_dim: int, base: float, device=None) -> Tensor:
    """[positions, head_dim / 2, 2] cos / sin of each pair's angle, rounded to bfloat16
    (computed on the CPU, as the checkpoint's own code does)."""
    inv = 1.0 / base ** (torch.arange(0, head_dim, 2).float() / head_dim)
    angles = torch.outer(torch.arange(positions).float(), inv)
    return torch.stack([angles.cos(), angles.sin()], -1).to(torch.bfloat16).to(device)


def rotate(x: Tensor, table: Tensor) -> Tensor:
    """Rotary embedding on interleaved pairs (x[2i], x[2i+1]); x: [B, T, H, D], table: [T, D/2, 2]."""
    pairs = x.float().unflatten(-1, (-1, 2))
    cos, sin = table.float()[None, :, None, :, 0], table.float()[None, :, None, :, 1]
    a, b = pairs[..., 0], pairs[..., 1]
    return torch.stack([a * cos - b * sin, b * cos + a * sin], -1).flatten(-2).type_as(x)
