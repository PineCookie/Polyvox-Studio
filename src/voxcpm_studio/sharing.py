"""Shareable LoRA packages: export a checkpoint, import someone else's.

A package is a zip holding one folder::

    <name>/lora_weights.safetensors   the weights, without metadata
    <name>/lora_config.json           {"base_model": "<HF id>", "lora_config": {...}}
    <name>/README.md                  what it is and how to use it

Nothing else from a run goes in: no optimizer state, local paths, logs,
manifests (transcripts) or dataset audio.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

from .lora_library import CONFIG_FILE, WEIGHTS_FILE, LoraCheckpoint, default_lora_config

README_FILE = "README.md"
PACKAGE_NAME = r"^[^\W_][\w.-]{0,79}$"  # Unicode letters allowed, like run names
HUB_ID = re.compile(r"[A-Za-z0-9][\w.-]*/[\w.-]+")
MAX_PACKAGE_BYTES = 2 * 1024**3


def hub_id(base_model: str | None) -> str | None:
    """Best guess at the Hugging Face id of a recorded base model.

    ``...\\models\\openbmb__VoxCPM2`` (a downloaded snapshot) -> ``openbmb/VoxCPM2``.
    """
    if not base_model:
        return None
    last = re.split(r"[\\/]", base_model.rstrip("\\/"))[-1]
    if "__" in last:
        org, _, repo = last.partition("__")
        return f"{org}/{repo}"
    return base_model if HUB_ID.fullmatch(base_model) else None


def checkpoint_step(ckpt: LoraCheckpoint) -> int | None:
    if m := re.search(r"step_(\d+)$", ckpt.id):
        return int(m.group(1))
    try:
        return int(json.loads((ckpt.path / "training_state.json").read_text(encoding="utf-8"))["step"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _readme(name: str, base_model: str, config: dict, step: int | None, dtype: str) -> str:
    layers = ", ".join(
        label
        for key, label in (("enable_lm", "LM"), ("enable_dit", "DiT"), ("enable_proj", "projections"))
        if config.get(key)
    )
    cli_flags = f"--lora-r {config['r']} --lora-alpha {config['alpha']}"
    cli_flags += "" if config.get("enable_lm", True) else " --lora-disable-lm"
    cli_flags += "" if config.get("enable_dit", True) else " --lora-disable-dit"
    cli_flags += " --lora-enable-proj" if config.get("enable_proj") else ""
    alpha = config["alpha"]
    alpha = int(alpha) if float(alpha).is_integer() else alpha
    return f"""# {name}

A LoRA for [VoxCPM](https://github.com/OpenBMB/VoxCPM), trained on top of
[{base_model}](https://huggingface.co/{base_model}). It only works with that base model.

| | |
|---|---|
| Base model | `{base_model}` |
| Rank / alpha | {config["r"]} / {alpha} |
| Layers | {layers} |
| Training steps | {step if step is not None else "unknown"} |
| Weights | {dtype} |

## Use

**VoxCPM Studio:** Voice panel -> Import, then choose this zip (or unzip it into
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


def export_package(ckpt: LoraCheckpoint, name: str, base_model: str, half: bool = True) -> Path:
    """Write a package for ``ckpt`` to a temporary zip file and return its path."""
    import torch
    from safetensors.torch import load_file, save

    if not re.fullmatch(PACKAGE_NAME, name):
        raise ValueError("Package name may contain letters (any language), digits, '.', '_' and '-' only.")
    if not HUB_ID.fullmatch(base_model):
        raise ValueError("Base model must be a Hugging Face id such as 'openbmb/VoxCPM2'.")

    tensors = load_file(str(ckpt.path / WEIGHTS_FILE))
    if half:
        # Inference runs in bfloat16 and loading casts to it anyway, so this
        # halves the download without changing the output.
        tensors = {k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in tensors.items()}
    weights = save(tensors)  # no metadata: tensor names and shapes only

    config = (ckpt.config or default_lora_config()).model_dump()
    info = {"base_model": base_model, "lora_config": config}
    readme = _readme(name, base_model, config, checkpoint_step(ckpt), "bfloat16" if half else "float32")

    fd, tmp = tempfile.mkstemp(prefix="voxcpm_lora_", suffix=".zip")
    os.close(fd)
    with zipfile.ZipFile(tmp, "w") as zf:
        zf.writestr(f"{name}/{WEIGHTS_FILE}", weights, compress_type=zipfile.ZIP_STORED)  # incompressible
        zf.writestr(f"{name}/{CONFIG_FILE}", json.dumps(info, indent=2) + "\n", compress_type=zipfile.ZIP_DEFLATED)
        zf.writestr(f"{name}/{README_FILE}", readme, compress_type=zipfile.ZIP_DEFLATED)
    return Path(tmp)


def _check_weights(data: bytes) -> None:
    from safetensors.torch import load

    try:
        tensors = load(data)
    except Exception as e:
        raise ValueError(f"{WEIGHTS_FILE} is not a valid safetensors file: {e}") from e
    if not tensors or not all("lora_" in k for k in tensors):
        raise ValueError(f"{WEIGHTS_FILE} does not contain LoRA weights.")


def _check_config(data: bytes) -> None:
    from voxcpm.model.voxcpm import LoRAConfig

    try:
        info = json.loads(data)
        LoRAConfig(**info["lora_config"])
    except Exception as e:
        raise ValueError(f"{CONFIG_FILE} is invalid: {e}") from e


def import_package(lora_dir: Path, package: Path, filename: str) -> str:
    """Unpack a package into ``lora_dir/<name>/``; returns the new LoRA id.

    ``filename`` is the uploaded file's name; it names the LoRA when the zip
    holds the files without a folder.
    """
    try:
        zf = zipfile.ZipFile(package)
    except zipfile.BadZipFile as e:
        raise ValueError("Not a zip file.") from e
    with zf:
        files: dict[str, zipfile.ZipInfo] = {}
        folders = set()
        for info in zf.infolist():
            if info.is_dir():
                continue
            parts = PurePosixPath(info.filename).parts
            # Accept "<name>/<file>" or, from a re-zipped folder, a bare "<file>".
            if len(parts) > 2 or ".." in parts or info.filename.startswith(("/", "\\")):
                raise ValueError(f"Unexpected entry in package: {info.filename}")
            if parts[-1] in (WEIGHTS_FILE, CONFIG_FILE, README_FILE):
                files[parts[-1]] = info
                folders.add(parts[0] if len(parts) == 2 else PurePosixPath(filename.replace("\\", "/")).stem)
        if WEIGHTS_FILE not in files or CONFIG_FILE not in files:
            raise ValueError(f"A package needs {WEIGHTS_FILE} and {CONFIG_FILE}.")
        if len(folders) != 1:
            raise ValueError("A package holds exactly one LoRA.")
        if sum(i.file_size for i in files.values()) > MAX_PACKAGE_BYTES:
            raise ValueError("Package is too large.")

        name = folders.pop()
        if not re.fullmatch(PACKAGE_NAME, name):
            raise ValueError(f"Invalid package name: {name}")
        target = lora_dir.resolve() / name
        if target.exists():
            raise FileExistsError(f"A LoRA named '{name}' already exists.")

        contents = {file: zf.read(info) for file, info in files.items()}
    _check_weights(contents[WEIGHTS_FILE])
    _check_config(contents[CONFIG_FILE])

    target.mkdir(parents=True)
    for file, data in contents.items():
        (target / file).write_bytes(data)
    return name
