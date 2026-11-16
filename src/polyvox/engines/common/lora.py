"""LoRA added to a frozen layer by a forward hook, so the model's modules (and their names) stay
as they are. Each engine picks its target layers and saves / merges the weights its own way."""

from __future__ import annotations

import math

import torch
from torch import nn


class LoRA(nn.Module):
    """``scale * B(A(dropout(x)))`` added to a frozen layer's output by a forward hook."""

    def __init__(self, layer: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.A = nn.Parameter(torch.empty(rank, layer.in_features, device=layer.weight.device))
        self.B = nn.Parameter(torch.zeros(layer.out_features, rank, device=layer.weight.device))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.enabled = True  # off while the LoRA is merged into the weights (samples)

    def hook(self, module: nn.Module, args: tuple, output: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return output
        x = self.dropout(args[0]).to(self.A.dtype)
        return output + (x @ self.A.t() @ self.B.t() * self.scale).to(output.dtype)
