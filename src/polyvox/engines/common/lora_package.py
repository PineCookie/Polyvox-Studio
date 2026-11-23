"""Shareable packages of LoRA checkpoints (VoxCPM, dots.tts, S2-Pro; Engine.package_files /
import_package): the weights without metadata, lora_config.json ({"base_model": <HF id>,
"lora_config": {...}}, + "engine" but for VoxCPM), a README, and files of the engine's own."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from ...lora_library import CONFIG_FILE, WEIGHTS_FILE, LoraCheckpoint
from ...models import HUB_ID

README_FILE = "README.md"


def number(value) -> int | float:
    """A config number as a README shows it (32.0 -> 32)."""
    return int(value) if float(value).is_integer() else value


def files(
    ckpt: LoraCheckpoint,
    base_model: str,
    half: bool,
    lora_config: dict,
    readme: str,
    engine: str | None = None,
    extra: dict[str, bytes] | None = None,
) -> dict[str, bytes]:
    """A LoRA package's files: ``engine`` named in its lora_config.json (none: VoxCPM, whose own
    loader reads the file too)."""
    import torch
    from safetensors.torch import load_file, save

    if not HUB_ID.fullmatch(base_model):
        raise ValueError(f"Base model must be a Hugging Face id such as 'org/model', not '{base_model}'.")
    info = {**({"engine": engine} if engine else {}), "base_model": base_model, "lora_config": lora_config}
    tensors = load_file(str(ckpt.path / WEIGHTS_FILE))
    if half:  # inference runs in bfloat16 and loading casts to it: half the download, the same output
        tensors = {k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in tensors.items()}
    return {
        WEIGHTS_FILE: save(tensors),  # no metadata: tensor names and shapes only
        CONFIG_FILE: (json.dumps(info, indent=2) + "\n").encode(),
        **(extra or {}),
        README_FILE: readme.encode(),
    }


def dtype(half: bool) -> str:
    return "bfloat16" if half else "float32"


def unpack(folder: Path, name: str, files: dict[str, bytes], check: Callable[[dict], object], keep=()) -> str:
    """A LoRA package's weights, config, README and ``keep`` files into ``folder/<name>/``;
    ``check(lora_config)`` raises on a config its engine can't load."""
    from safetensors.torch import load

    if WEIGHTS_FILE not in files or CONFIG_FILE not in files:
        raise ValueError(f"A LoRA package needs {WEIGHTS_FILE} and {CONFIG_FILE}.")
    try:
        tensors = load(files[WEIGHTS_FILE])
    except Exception as e:
        raise ValueError(f"{WEIGHTS_FILE} is not a valid safetensors file: {e}") from e
    if not tensors or not all("lora_" in k for k in tensors):
        raise ValueError(f"{WEIGHTS_FILE} does not contain LoRA weights.")
    try:
        check(json.loads(files[CONFIG_FILE])["lora_config"])
    except Exception as e:
        raise ValueError(f"{CONFIG_FILE} is invalid: {e}") from e
    target = folder / name
    if target.exists():
        raise FileExistsError(f"A LoRA named '{name}' already exists.")
    target.mkdir(parents=True)
    for file in (WEIGHTS_FILE, CONFIG_FILE, README_FILE, *keep):
        if file in files:
            (target / file).write_bytes(files[file])
    return name


def rank_alpha(raw: dict | None) -> tuple[int, float]:
    """A LoRA config's rank and alpha, checked."""
    raw = raw or {}
    r, alpha = raw.get("r"), raw.get("alpha")
    if not isinstance(r, int) or r < 1 or not isinstance(alpha, int | float) or alpha <= 0:
        raise ValueError(f"{CONFIG_FILE} needs a rank 'r' and an 'alpha'.")
    return r, float(alpha)
