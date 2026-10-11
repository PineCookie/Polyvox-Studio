"""LoRA for dots.tts, shared by its training worker and its synthesis.

dots.tts has no LoRA of its own. Training adds ``B @ A`` through forward hooks
on the attention projections, so the model's modules (and their names) stay as
they are; synthesis merges ``scale * B @ A`` into the weights instead: its
first generation rewires the diffusion head in place for inference
(dit_inference.fuse_dit_for_inference: q/k/v and the adaLN layers copied into
fused layers), where hooks would not reach. ``Merger`` undoes that rewiring
before it changes weights, so the next generation fuses the merged ones.

Checkpoint: lora_weights.safetensors with ``<module>.lora_A`` (r x in) and
``<module>.lora_B`` (out x r) per layer (module names as in DotsTtsModel), and
lora_config.json {"engine": "dots_tts", "base_model": ..., "lora_config":
{"r", "alpha", "dropout", "enable_lm", "enable_dit"}}.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch import nn

ENGINE = "dots_tts"
# network -> (module name prefix, projection suffixes); the networks are also the
# optimizer's parameter groups.
TARGETS = {
    "llm": ("core.llm.model.layers.", ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj")),
    "dit": ("core.velocity_field_predictor.blocks.", ("attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.o_proj")),
}


def quiet_logging() -> None:
    """dots.tts logs every request and decode stage through loguru (DEBUG by
    default): keep warnings only."""
    import sys

    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, level="WARNING")


def target_layers(model: nn.Module, networks: tuple[str, ...] = tuple(TARGETS)) -> dict[str, tuple[str, nn.Linear]]:
    """{module name: (network, layer)} of the layers LoRA trains."""
    out = {}
    for net in networks:
        prefix, suffixes = TARGETS[net]
        for name, module in model.named_modules():
            if name.startswith(prefix) and name.endswith(suffixes) and isinstance(module, nn.Linear):
                out[name] = (net, module)
    return out


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


def attach(model: nn.Module, rank: int, alpha: float, dropout: float, networks: tuple[str, ...]) -> dict:
    """Hook a new LoRA onto each target layer: {module name: (network, LoRA)}."""
    loras = {}
    for name, (net, layer) in target_layers(model, networks).items():
        lora = LoRA(layer, rank, alpha, dropout)
        layer.register_forward_hook(lora.hook)
        loras[name] = (net, lora)
    return loras


def state(loras: dict) -> dict[str, torch.Tensor]:
    return {
        f"{name}.{key}": getattr(lora, part).detach().float().cpu().contiguous()
        for name, (_, lora) in loras.items()
        for key, part in (("lora_A", "A"), ("lora_B", "B"))
    }


def write_config(ckpt_dir: Path, config: dict, base_model: str) -> None:
    info = {"engine": ENGINE, "base_model": base_model, "lora_config": config}
    (ckpt_dir / "lora_config.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


class Unfuse:
    """Snapshot of the diffusion head's modules before inference fuses them;
    calling it puts them back (and drops the solvers built on the fused copy)."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.dit = model.core.velocity_field_predictor
        self.blocks = [(block, block.attn, block.adaLN_modulation) for block in self.dit.blocks]
        self.output = (self.dit.output_layer, self.dit.output_layer.adaLN_modulation)

    def __call__(self) -> None:
        if self.model.core.velocity_field_predictor is self.dit and not self.model._fm_dit_solvers:
            return
        self.model.core.velocity_field_predictor = self.dit
        for block, attn, modulation in self.blocks:
            block.attn, block.adaLN_modulation = attn, modulation
        self.output[0].adaLN_modulation = self.output[1]
        self.model._fm_dit_solvers.clear()


class Merger:
    """Puts one LoRA at a time into a model's weights, keeping the originals of
    the layers it changed so the next LoRA (or none) starts from them. Create it
    before the model's first generation (see Unfuse).

    ``keep_fusion=False`` (training samples) undoes the inference fusion first.
    ``keep_fusion=True`` (synthesis) leaves it and also writes the fused q/k/v
    copies, in place: the optimized runtime's CUDA graphs keep the addresses of
    the tensors they were recorded with, so every tensor must stay the same one.
    """

    def __init__(self, model: nn.Module, keep_fusion: bool = False):
        self.model = model
        self.keep_fusion = keep_fusion
        self.unfuse = Unfuse(model)
        self._modules = dict(model.named_modules())
        self._originals: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def _sync_fused(self) -> None:
        """Copy the DiT blocks' q/k/v weights into their fused qkv_proj (if fused);
        o_proj stays the same module in the fused attention."""
        for block, attn, _ in self.unfuse.blocks:
            fused = getattr(block.attn, "qkv_proj", None)
            if block.attn is not attn and fused is not None:
                fused.weight.copy_(torch.cat([attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight]))

    @torch.no_grad()
    def apply(self, tensors: dict[str, torch.Tensor] | None, scale: float = 1.0) -> None:
        try:
            self._apply(tensors, scale)
        finally:
            if self.keep_fusion:
                self._sync_fused()

    def _apply(self, tensors: dict[str, torch.Tensor] | None, scale: float) -> None:
        if not self.keep_fusion:
            self.unfuse()
        for name, weight in self._originals.items():
            self._modules[name].weight.copy_(weight)
        if not tensors:
            return
        names = {k.removesuffix(".lora_A") for k in tensors if k.endswith(".lora_A")}
        if len(names) * 2 != len(tensors):
            raise ValueError("LoRA file has unpaired lora_A / lora_B tensors.")
        for name in names:
            layer = self._modules.get(name)
            a, b = tensors[f"{name}.lora_A"], tensors[f"{name}.lora_B"]
            if not isinstance(layer, nn.Linear) or a.shape[1] != layer.in_features or b.shape[0] != layer.out_features:
                self._apply(None, 1.0)
                raise ValueError(f"LoRA layer {name} does not fit this dots.tts model.")
        for name in names:
            weight = self._modules[name].weight
            if name not in self._originals:
                self._originals[name] = weight.detach().clone()
            a = tensors[f"{name}.lora_A"].to(weight.device, torch.float32)
            b = tensors[f"{name}.lora_B"].to(weight.device, torch.float32)
            weight.copy_((self._originals[name].float() + scale * (b @ a)).to(weight.dtype))
