"""LoRA on v4's mel DiT, as upstream's s2_train_v3_lora.py trains it with peft
(not installed here): rank r on every attention projection (to_q, to_k, to_v,
to_out.0), alpha = r (scale 1), A Kaiming-uniform and B zero at the start (an
untrained LoRA changes nothing). The base layers stay frozen; the LoRA joins
them through forward hooks, so the model's own weights and keys are untouched.

Saved under peft's names (``cfm.base_model.model.<layer>.lora_A.default.weight``)
so the GPT-SoVITS webui loads voices trained here; synth.merge_lora folds them
back into plain weights.
"""

from __future__ import annotations

import math

import torch
from torch import nn

TARGETS = ("to_q", "to_k", "to_v", "to_out.0")
PEFT_PREFIX = "cfm.base_model.model."


class DitLora(nn.Module):
    def __init__(self, cfm: nn.Module, rank: int):
        super().__init__()
        self.rank = rank
        self.names: list[str] = []
        self.a = nn.ParameterList()
        self.b = nn.ParameterList()
        self._hooks = []
        for name, layer in cfm.named_modules():
            if isinstance(layer, nn.Linear) and name.endswith(TARGETS):
                a = nn.Parameter(torch.empty(rank, layer.in_features, device=layer.weight.device))
                nn.init.kaiming_uniform_(a, a=math.sqrt(5))  # peft's default
                b = nn.Parameter(torch.zeros(layer.out_features, rank, device=layer.weight.device))
                self.names.append(name)
                self.a.append(a)
                self.b.append(b)
                self._hooks.append(layer.register_forward_hook(self._hook(a, b)))
        if not self.names:
            raise ValueError("No attention projections to put a LoRA on.")

    @staticmethod
    def _hook(a: nn.Parameter, b: nn.Parameter):
        def add(layer, inputs, out):
            return out + nn.functional.linear(nn.functional.linear(inputs[0], a), b)

        return add

    def remove(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def peft_state(self) -> dict[str, torch.Tensor]:
        """The LoRA under peft's key names (as the webui saves it)."""
        out = {}
        for name, a, b in zip(self.names, self.a, self.b, strict=True):
            out[f"{PEFT_PREFIX}{name}.lora_A.default.weight"] = a.detach()
            out[f"{PEFT_PREFIX}{name}.lora_B.default.weight"] = b.detach()
        return out

    def merged_into(self, weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """``weights`` (a SoVITS state dict with the base DiT) with the LoRA folded in."""
        out = dict(weights)
        for name, a, b in zip(self.names, self.a, self.b, strict=True):
            key = f"cfm.{name}.weight"
            w = out[key]
            out[key] = (w.float() + b.detach().float() @ a.detach().float()).to(w.dtype)
        return out
