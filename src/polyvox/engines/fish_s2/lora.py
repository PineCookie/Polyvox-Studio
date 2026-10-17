"""LoRA for S2-Pro's fast transformer, shared by its training worker and its synthesis.

By default only the fast transformer (the 4 layers that turn each frame's slow
hidden state into its 9 residual codes) is trained: its attention and
feed-forward layers. LoRA on the slow transformer's attention is an option
("slow_attn"), trained with AdamW at its own, much smaller rate (default
SLOW_LR): at Prodigy's own rate it learned the training clips
(teacher-forced loss down) but spoke worse on its own (sentences cut short,
long silences); with AdamW at 5e-4 / 2e-5 it spoke fine but sounded no more
like the voice. Prodigy can't take a smaller rate for one group: it estimates
its step size from how far the weights have moved, so a smaller multiplier
(or d_coef) kept the slow group's step at its starting 1e-6.

Training adds ``B @ A`` through forward hooks (the model's modules and names
stay as they are). Synthesis merges ``scale * B @ A`` into the weights instead
(Merger), in place, so the decoder's CUDA graphs stay valid and decoding costs
nothing extra.

Checkpoint: lora_weights.safetensors with ``<module>.lora_A`` (r x in) and
``<module>.lora_B`` (out x r) per layer (module names as in model.DualAR, e.g.
``fast_layers.0.attention.wqkv``), lora_config.json {"engine": "fish_s2",
"base_model": ..., "lora_config": {"r", "alpha", "targets"}}, and the voice's
default reference (reference.wav + reference.json).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from ..dots.lora import LoRA

ENGINE = "fish_s2"
# target -> (module name prefix, layer suffixes)
TARGETS = {
    "fast_attn": ("fast_layers.", ("attention.wqkv", "attention.wo")),
    "fast_mlp": ("fast_layers.", ("feed_forward.w1", "feed_forward.w2", "feed_forward.w3")),
    "slow_attn": ("layers.", ("attention.wqkv", "attention.wo")),  # an option: see above
}
DEFAULT_TARGETS = ("fast_attn", "fast_mlp")
SLOW_TARGETS = ("slow_attn",)
SLOW_LR = 2e-5  # the slow layers' AdamW learning rate unless the run sets one (fast: 5e-4 tested)


class LowLoRA(LoRA):
    """LoRA computed in the layer's precision (bfloat16): no float32 copies of the
    inputs kept for the backward pass. The parameters stay float32."""

    def hook(self, module: nn.Module, args: tuple, output: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return output
        x = self.dropout(args[0])
        return output + (x @ self.A.to(x.dtype).t() @ self.B.to(x.dtype).t()) * self.scale


def target_layers(model: nn.Module, targets: tuple[str, ...] = DEFAULT_TARGETS) -> dict[str, nn.Linear]:
    out = {}
    for target in targets:
        if target not in TARGETS:
            raise ValueError(f"Unknown S2-Pro LoRA target: {target}")
        prefix, suffixes = TARGETS[target]
        for name, module in model.named_modules():
            if name.startswith(prefix) and name.endswith(suffixes) and isinstance(module, nn.Linear):
                out[name] = module
    return out


def attach(model: nn.Module, rank: int, alpha: float, dropout: float = 0.0, targets=DEFAULT_TARGETS) -> dict:
    """Hook a new LoRA onto each target layer: {module name: LoRA}."""
    loras = {}
    for name, layer in target_layers(model, tuple(targets)).items():
        lora = LowLoRA(layer, rank, alpha, dropout)
        layer.register_forward_hook(lora.hook)
        loras[name] = lora
    return loras


def state(loras: dict) -> dict[str, torch.Tensor]:
    return {
        f"{name}.{key}": getattr(lora, part).detach().float().cpu().contiguous()
        for name, lora in loras.items()
        for key, part in (("lora_A", "A"), ("lora_B", "B"))
    }


def write_config(ckpt_dir: Path, config: dict, base_model: str) -> None:
    info = {"engine": ENGINE, "base_model": base_model, "lora_config": config}
    (ckpt_dir / "lora_config.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


# The voice's default reference, as GPT-SoVITS voices keep theirs (gpt_sovits.voices.read_reference reads both).
REFERENCE_WAV, REFERENCE_JSON = "reference.wav", "reference.json"
REFERENCE_CLIPS = 3  # dataset clips joined into one: ~20 s cloned clearly better than one clip
MAX_REFERENCE_S = 30.0


def _write_info(folder: Path, text: str, source: str) -> None:
    info = {"text": text.strip(), "source": source}
    (folder / REFERENCE_JSON).write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")


def write_reference_clips(folder: Path, clips: list, source: str) -> str:
    """Dataset clips (data.references.Clip) joined into the voice's reference; returns its transcript."""
    from ...data import references

    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder / f".{REFERENCE_WAV}.tmp"
    text = references.join(clips, tmp)
    tmp.replace(folder / REFERENCE_WAV)
    _write_info(folder, text, source)
    return text


def write_reference_file(folder: Path, source: Path, text: str, origin: str = "") -> None:
    """Any readable clip of at most MAX_REFERENCE_S as the voice's reference."""
    import numpy as np
    import soundfile as sf

    from ...data import audio_io

    if not text.strip():
        raise ValueError("S2-Pro needs the reference clip's transcript.")
    audio, rate = audio_io.read(source)
    mono = audio.mean(axis=0) if audio.ndim == 2 else audio
    if len(mono) / rate > MAX_REFERENCE_S:
        raise ValueError(f"The reference clip is {len(mono) / rate:.0f} s; S2-Pro takes at most {MAX_REFERENCE_S:g} s.")
    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder / f".{REFERENCE_WAV}.tmp"
    sf.write(tmp, np.asarray(mono, dtype=np.float32), rate, format="WAV", subtype="PCM_16")
    tmp.replace(folder / REFERENCE_WAV)
    _write_info(folder, text, origin)


class Merger:
    """Puts one LoRA at a time into the model's weights, in place. The original
    weights of the layers it changed are kept (on the CPU: the fast transformer's
    layers are ~0.8 GB) so the next LoRA, or none, starts from them."""

    def __init__(self, model: nn.Module):
        self._modules = dict(model.named_modules())
        self._originals: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def apply(self, tensors: dict[str, torch.Tensor] | None, scale: float = 1.0) -> None:
        for name, weight in self._originals.items():
            self._modules[name].weight.copy_(weight)
        if not tensors:
            return
        names = {k.removesuffix(".lora_A") for k in tensors if k.endswith(".lora_A")}
        if not names or len(names) * 2 != len(tensors):
            raise ValueError("LoRA file has unpaired lora_A / lora_B tensors.")
        for name in names:
            layer = self._modules.get(name)
            a, b = tensors[f"{name}.lora_A"], tensors[f"{name}.lora_B"]
            if not isinstance(layer, nn.Linear) or a.shape[1] != layer.in_features or b.shape[0] != layer.out_features:
                raise ValueError(f"LoRA layer {name} does not fit this S2-Pro model.")
        for name in names:
            weight = self._modules[name].weight
            if name not in self._originals:
                self._originals[name] = weight.detach().cpu().clone()
            original = self._originals[name].to(weight.device, torch.float32)
            a = tensors[f"{name}.lora_A"].to(weight.device, torch.float32)
            b = tensors[f"{name}.lora_B"].to(weight.device, torch.float32)
            weight.copy_((original + scale * (b @ a)).to(weight.dtype))
