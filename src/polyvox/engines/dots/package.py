"""A dots.tts LoRA as a shareable package (common/lora_package.py), with a merge snippet in its README."""

from __future__ import annotations

from pathlib import Path

from ...lora_library import LoraCheckpoint
from ..common import lora_package
from ..common.lora_package import number

ENGINE = "dots_tts"


def config(raw: dict | None) -> dict:
    """The lora_config fields its loader reads, checked."""
    r, alpha = lora_package.rank_alpha(raw)
    raw = raw or {}
    return {
        "r": r,
        "alpha": alpha,
        "dropout": float(raw.get("dropout") or 0.0),
        "enable_lm": bool(raw.get("enable_lm", True)),
        "enable_dit": bool(raw.get("enable_dit", True)),
    }


def files(ckpt: LoraCheckpoint, name: str, base_model: str, half: bool) -> dict[str, bytes]:
    lora = config(ckpt.raw_config)
    return lora_package.files(ckpt, base_model, half, lora, readme(name, base_model, lora, ckpt.step, half), ENGINE)


def unpack(folder: Path, name: str, files: dict[str, bytes]) -> str:
    return lora_package.unpack(folder, name, files, config)


def readme(name: str, base_model: str, config: dict, step: int | None, half: bool) -> str:
    layers = ", ".join(
        label for key, label in (("enable_lm", "language model"), ("enable_dit", "diffusion head")) if config.get(key)
    )
    alpha = number(config["alpha"])
    return f"""# {name}

A LoRA for [dots.tts](https://github.com/studio-dots-ai/dots.tts), trained on top of
[{base_model}](https://huggingface.co/{base_model}). It only works with that base model.

| | |
|---|---|
| Base model | `{base_model}` |
| Rank / alpha | {config["r"]} / {alpha} |
| Layers | attention q/k/v/o of the {layers} |
| Training steps | {step if step is not None else "unknown"} |
| Weights | {lora_package.dtype(half)} |

## Use

**Polyvox Studio:** Voice panel -> Import, then choose this zip (or unzip it into
the `lora` folder). It appears as the voice "{name}" (dots.tts).

**Python** (`pip install dots.tts`): dots.tts has no LoRA loading of its own, so
merge the weights into the model before the first generation. Each layer
`<name>` has `<name>.lora_A` (rank x in) and `<name>.lora_B` (out x rank); the
change to its weight is `alpha / rank * B @ A`.

```python
import torch
from safetensors.torch import load_file
from dots_tts.runtime import DotsTtsRuntime

runtime = DotsTtsRuntime.from_pretrained("{base_model}", precision="bfloat16")
lora = load_file("{name}/lora_weights.safetensors")
modules = dict(runtime.model.named_modules())
with torch.no_grad():
    for key in [k for k in lora if k.endswith(".lora_A")]:
        layer = key.removesuffix(".lora_A")
        weight = modules[layer].weight
        delta = lora[layer + ".lora_B"].float() @ lora[key].float() * ({alpha} / {config["r"]})
        weight.copy_((weight.float() + delta.to(weight.device)).to(weight.dtype))

audio = runtime.generate(text="Hello!", language="en")["audio"]
```

Generating without a reference clip uses the voice the LoRA learned.
"""
