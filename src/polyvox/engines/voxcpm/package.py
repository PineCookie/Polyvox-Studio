"""A VoxCPM LoRA as a shareable package (common/lora_package.py): voxcpm's own loader reads it."""

from __future__ import annotations

from pathlib import Path

from ...lora_library import LoraCheckpoint, default_lora_config
from ..common import lora_package
from ..common.lora_package import number


def files(ckpt: LoraCheckpoint, name: str, base_model: str, half: bool) -> dict[str, bytes]:
    config = (ckpt.config or default_lora_config()).model_dump()  # no "engine": voxcpm reads this file too
    return lora_package.files(ckpt, base_model, half, config, readme(name, base_model, config, ckpt.step, half))


def unpack(folder: Path, name: str, files: dict[str, bytes]) -> str:
    from voxcpm.model.voxcpm import LoRAConfig

    return lora_package.unpack(folder, name, files, lambda raw: LoRAConfig(**raw))


def readme(name: str, base_model: str, config: dict, step: int | None, half: bool) -> str:
    layers = ", ".join(
        label
        for key, label in (("enable_lm", "LM"), ("enable_dit", "DiT"), ("enable_proj", "projections"))
        if config.get(key)
    )
    cli_flags = f"--lora-r {config['r']} --lora-alpha {config['alpha']}"
    cli_flags += "" if config.get("enable_lm", True) else " --lora-disable-lm"
    cli_flags += "" if config.get("enable_dit", True) else " --lora-disable-dit"
    cli_flags += " --lora-enable-proj" if config.get("enable_proj") else ""
    return f"""# {name}

A LoRA for [VoxCPM](https://github.com/OpenBMB/VoxCPM), trained on top of
[{base_model}](https://huggingface.co/{base_model}). It only works with that base model.

| | |
|---|---|
| Base model | `{base_model}` |
| Rank / alpha | {config["r"]} / {number(config["alpha"])} |
| Layers | {layers} |
| Training steps | {step if step is not None else "unknown"} |
| Weights | {lora_package.dtype(half)} |

## Use

**Polyvox Studio:** Voice panel -> Import, then choose this zip (or unzip it into
the `lora` folder). It appears as the voice "{name}".

**Python** (`pip install voxcpm`): pass this folder; `lora_config.json` is read from it.

```python
from voxcpm import VoxCPM

model = VoxCPM.from_pretrained("{base_model}", lora_weights_path="{name}")
wav = model.generate(text="Hello!")
```

**Command line:** the CLI does not read `lora_config.json`, so pass the layout:

```
voxcpm design --hf-model-id {base_model} --lora-path {name} {cli_flags} --text "Hello!" --output out.wav
```
"""
