"""Shareable packages: export a checkpoint, import someone else's.

A package is a zip holding one folder. For a LoRA (VoxCPM, dots.tts, S2-Pro)::

    <name>/lora_weights.safetensors   the weights, without metadata
    <name>/lora_config.json           {"base_model": "<HF id>", "lora_config": {...}}
                                      (+ "engine" for engines other than VoxCPM)
    <name>/reference.wav, reference.json   S2-Pro: the voice's default reference and its transcript
    <name>/README.md                  what it is and how to use it

For a GPT-SoVITS voice (whole models; engines/gpt_sovits/voices.package_files)::

    <name>/sovits_weights.pth         SoVITS, as the webui saves it
    <name>/gpt_weights.ckpt           GPT (absent: the voice uses the base GPT)
    <name>/reference.wav, reference.json   the default reference clip and its transcript
    <name>/lora_config.json           {"engine": "gpt_sovits", "version": ..., "base_model": ...}
    <name>/README.md

Nothing else from a run goes in: no optimizer state, local paths (the
weights' saved training configs are cleaned of them), logs, manifests
(transcripts) or dataset audio besides the one reference clip.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

from .engines import DEFAULT as DEFAULT_ENGINE
from .engines.dots import lora as dots_lora
from .engines.fish_s2 import lora as s2_lora
from .engines.gpt_sovits import voices as gsv_voices
from .lora_library import CONFIG_FILE, WEIGHTS_FILE, LoraCheckpoint, default_lora_config
from .lora_library import engine_dir as engine_folder

README_FILE = "README.md"
GSV_GPT_FILE = "gpt_weights.ckpt"
# A GPT-SoVITS voice package's files besides the README (lora_config.json is rewritten on import).
GSV_PACKAGE_FILES = ("sovits_weights.pth", GSV_GPT_FILE, "reference.wav", "reference.json")
PACKAGE_FILES = (WEIGHTS_FILE, CONFIG_FILE, README_FILE, *GSV_PACKAGE_FILES)
# The engines a package can be for, and the base model the export suggests.
ENGINE_BASE_MODELS = {
    DEFAULT_ENGINE: "openbmb/VoxCPM2",
    dots_lora.ENGINE: "rednote-hilab/dots.tts-soar",
    s2_lora.ENGINE: "fishaudio/s2-pro",
}
ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)  # zip's earliest date: entries carry no export time
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


def _dots_readme(name: str, base_model: str, config: dict, step: int | None, dtype: str) -> str:
    layers = ", ".join(
        label for key, label in (("enable_lm", "language model"), ("enable_dit", "diffusion head")) if config.get(key)
    )
    alpha = config["alpha"]
    alpha = int(alpha) if float(alpha).is_integer() else alpha
    return f"""# {name}

A LoRA for [dots.tts](https://github.com/studio-dots-ai/dots.tts), trained on top of
[{base_model}](https://huggingface.co/{base_model}). It only works with that base model.

| | |
|---|---|
| Base model | `{base_model}` |
| Rank / alpha | {config["r"]} / {alpha} |
| Layers | attention q/k/v/o of the {layers} |
| Training steps | {step if step is not None else "unknown"} |
| Weights | {dtype} |

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


def _s2_readme(name: str, base_model: str, config: dict, step: int | None, dtype: str, has_reference: bool) -> str:
    alpha = config["alpha"]
    alpha = int(alpha) if float(alpha).is_integer() else alpha
    reference = (
        "`reference.wav` is the voice's default reference (a few of its clips joined); `reference.json` holds its\n"
        "transcript. S2-Pro speaks with it unless you give another clip: the LoRA refines the sound, the\n"
        "reference still sets much of the voice."
        if has_reference
        else "It comes without a reference clip: give one of the voice, with its transcript, when synthesizing."
    )
    return f"""# {name}

A LoRA for [Fish Audio S2-Pro](https://huggingface.co/{base_model}): its fast transformer
(the 4 layers that predict each audio frame's residual codes), trained on top of
`{base_model}`. It only works with that base model, whose weights are under the Fish Audio
Research License (research and non-commercial use).

| | |
|---|---|
| Base model | `{base_model}` |
| Rank / alpha | {config["r"]} / {alpha} |
| Layers | attention (wqkv, wo) and feed-forward (w1, w2, w3) of the fast transformer |
| Training steps | {step if step is not None else "unknown"} |
| Weights | {dtype} |

{reference}

## Use

**Polyvox Studio:** Voice panel -> Import, then choose this zip. It appears as the
S2-Pro voice "{name}", with its reference.

**Other code:** each layer `<name>` (e.g. `audio_decoder.layers.0.attention.wqkv` in the
checkpoint's naming is `fast_layers.0.attention.wqkv` here) has `<name>.lora_A`
(rank x in) and `<name>.lora_B` (out x rank); add `alpha / rank * B @ A` to its weight.
Training heard every clip with a faint noise floor (white noise at -72 dBFS); encode
references the same way.
"""


def _gsv_readme(name: str, version: str, has_gpt: bool, has_reference: bool, step: int | None) -> str:
    gpt_row = f"`{GSV_GPT_FILE}`" if has_gpt else "none: the base GPT (`s1v3.ckpt`) speaks for it"
    reference = (
        "`reference.wav` is the voice's default reference clip; `reference.json` holds its transcript."
        if has_reference
        else "It comes without a reference clip: give one (3-10 s of the voice, with its transcript) when synthesizing."
    )
    trained = (
        "the GPT fine-tuned in full, SoVITS as the webui trains v4 (a LoRA of its mel DiT plus\n"
        "the parts around it; it loads over the v4 base, `gsv-v4-pretrained/s2Gv4.pth`)"
        if version == "v4"
        else "both models fine-tuned\nin full"
    )
    return f"""# {name}

A [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) {version} voice: {trained} from the official
{version} base models ([lj1995/GPT-SoVITS](https://huggingface.co/lj1995/GPT-SoVITS)).

| | |
|---|---|
| Version | {version} |
| SoVITS weights | `sovits_weights.pth` |
| GPT weights | {gpt_row} |
| Training steps | {step if step is not None else "unknown"} |

{reference} GPT-SoVITS always clones a reference clip: the closer it is to how the
voice should sound, the better.

## Use

**Polyvox Studio:** Voice panel -> Import, then choose this zip. It appears as the
GPT-SoVITS voice "{name}", with its reference.

**GPT-SoVITS webui:** copy `sovits_weights.pth` into `SoVITS_weights_{version}/` and
`{GSV_GPT_FILE if has_gpt else "s1v3.ckpt (from the pretrained models)"}` into `GPT_weights_{version}/` (rename them
if you like), refresh the model lists and choose them; use the reference clip and
transcript as the reference audio and text.
"""


def export_package(ckpt: LoraCheckpoint, name: str, base_model: str, half: bool = True) -> Path:
    """Write a package for ``ckpt`` to a temporary zip file and return its path."""
    import torch
    from safetensors.torch import load_file, save

    if not re.fullmatch(PACKAGE_NAME, name):
        raise ValueError("Package name may contain letters (any language), digits, '.', '_' and '-' only.")
    if ckpt.engine == gsv_voices.ENGINE:  # whole models; base_model and half don't apply
        files = gsv_voices.package_files(ckpt.path, name)
        files[README_FILE] = _gsv_readme(name, json.loads(files[CONFIG_FILE])["version"], GSV_GPT_FILE in files,
                                         gsv_voices.REFERENCE_WAV in files, checkpoint_step(ckpt)).encode()  # fmt: skip
        return _write_zip(name, files)
    if not HUB_ID.fullmatch(base_model):
        example = ENGINE_BASE_MODELS.get(ckpt.engine, "openbmb/VoxCPM2")
        raise ValueError(f"Base model must be a Hugging Face id such as '{example}'.")
    if ckpt.engine not in ENGINE_BASE_MODELS:
        raise ValueError(f"LoRAs for '{ckpt.engine}' can't be exported.")

    tensors = load_file(str(ckpt.path / WEIGHTS_FILE))
    if half:
        # Inference runs in bfloat16 and loading casts to it anyway, so this
        # halves the download without changing the output.
        tensors = {k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in tensors.items()}
    weights = save(tensors)  # no metadata: tensor names and shapes only

    step, dtype = checkpoint_step(ckpt), "bfloat16" if half else "float32"
    if ckpt.engine == DEFAULT_ENGINE:
        config = (ckpt.config or default_lora_config()).model_dump()
        info = {"base_model": base_model, "lora_config": config}
        readme = _readme(name, base_model, config, step, dtype)
    else:
        config = _engine_config(ckpt.engine, ckpt.raw_config)
        info = {"engine": ckpt.engine, "base_model": base_model, "lora_config": config}
        readme = _dots_readme(name, base_model, config, step, dtype)

    files = {
        WEIGHTS_FILE: weights,
        CONFIG_FILE: (json.dumps(info, indent=2) + "\n").encode(),
    }
    if ckpt.engine == s2_lora.ENGINE:
        reference = gsv_voices.read_reference(ckpt.path)
        if reference:  # the transcript only: where it came from stays here
            files[s2_lora.REFERENCE_WAV] = reference[0].read_bytes()
            files[s2_lora.REFERENCE_JSON] = (
                json.dumps({"text": reference[1]}, ensure_ascii=False, indent=1) + "\n"
            ).encode()
        readme = _s2_readme(name, base_model, config, step, dtype, reference is not None)
    files[README_FILE] = readme.encode()
    return _write_zip(name, files)


STORED = {WEIGHTS_FILE, gsv_voices.SOVITS_FILE, gsv_voices.GPT_FILE}  # weights don't compress


def _write_zip(name: str, files: dict[str, bytes]) -> Path:
    fd, tmp = tempfile.mkstemp(prefix="polyvox_package_", suffix=".zip")
    os.close(fd)
    with zipfile.ZipFile(tmp, "w") as zf:
        for file, data in files.items():
            method = zipfile.ZIP_STORED if file in STORED else zipfile.ZIP_DEFLATED
            zf.writestr(_entry(f"{name}/{file}", method), data)
    return Path(tmp)


def _entry(path: str, compress_type: int) -> zipfile.ZipInfo:
    """A zip entry without the time it was made (zip's earliest date) and with
    plain file permissions: a package tells nothing about when or where it was
    exported, and the same checkpoint always packs to the same bytes."""
    info = zipfile.ZipInfo(path, date_time=ZIP_EPOCH)
    info.compress_type = compress_type
    info.external_attr = 0o644 << 16
    return info


def _check_weights(data: bytes) -> None:
    from safetensors.torch import load

    try:
        tensors = load(data)
    except Exception as e:
        raise ValueError(f"{WEIGHTS_FILE} is not a valid safetensors file: {e}") from e
    if not tensors or not all("lora_" in k for k in tensors):
        raise ValueError(f"{WEIGHTS_FILE} does not contain LoRA weights.")


def _engine_config(engine: str, raw: dict | None) -> dict:
    """A dots.tts / S2-Pro lora_config with the fields its loader reads, checked."""
    raw = raw or {}
    r, alpha = raw.get("r"), raw.get("alpha")
    if not isinstance(r, int) or r < 1 or not isinstance(alpha, int | float) or alpha <= 0:
        raise ValueError(f"{CONFIG_FILE} needs a rank 'r' and an 'alpha'.")
    if engine == s2_lora.ENGINE:
        targets = raw.get("targets") or list(s2_lora.DEFAULT_TARGETS)
        if not isinstance(targets, list) or not set(targets) <= set(s2_lora.TARGETS):
            raise ValueError(f"{CONFIG_FILE} has unknown S2-Pro LoRA targets.")
        return {"r": r, "alpha": float(alpha), "dropout": float(raw.get("dropout") or 0.0), "targets": targets}
    return {
        "r": r,
        "alpha": float(alpha),
        "dropout": float(raw.get("dropout") or 0.0),
        "enable_lm": bool(raw.get("enable_lm", True)),
        "enable_dit": bool(raw.get("enable_dit", True)),
    }


def _check_config(data: bytes) -> str:
    """Validate a package's lora_config.json; returns its engine."""
    from voxcpm.model.voxcpm import LoRAConfig

    try:
        info = json.loads(data)
        engine = str(info.get("engine") or DEFAULT_ENGINE)
        if engine not in ENGINE_BASE_MODELS:
            raise ValueError(f"unknown engine '{engine}'")
        if engine == DEFAULT_ENGINE:
            LoRAConfig(**info["lora_config"])
        else:
            _engine_config(engine, info["lora_config"])
        return engine
    except Exception as e:
        raise ValueError(f"{CONFIG_FILE} is invalid: {e}") from e


def import_package(lora_dir: Path, package: Path, filename: str) -> tuple[str, str]:
    """Unpack a package into ``lora_dir/<engine>/<name>/``; returns the new LoRA id and its engine.

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
            if parts[-1] in PACKAGE_FILES:
                files[parts[-1]] = info
                folders.add(parts[0] if len(parts) == 2 else PurePosixPath(filename.replace("\\", "/")).stem)
        voice = gsv_voices.SOVITS_FILE in files  # a GPT-SoVITS voice
        if not voice and (WEIGHTS_FILE not in files or CONFIG_FILE not in files):
            raise ValueError(f"A package needs {WEIGHTS_FILE} and {CONFIG_FILE}.")
        if len(folders) != 1:
            raise ValueError("A package holds exactly one LoRA.")
        if sum(i.file_size for i in files.values()) > MAX_PACKAGE_BYTES:
            raise ValueError("Package is too large.")

        name = folders.pop()
        if not re.fullmatch(PACKAGE_NAME, name):
            raise ValueError(f"Invalid package name: {name}")
        contents = {file: zf.read(info) for file, info in files.items()}
    lora_dir = lora_dir.resolve()
    if voice:
        voice_files = {f: d for f, d in contents.items() if f in GSV_PACKAGE_FILES}
        engine_dir = engine_folder(lora_dir, gsv_voices.ENGINE)
        return f"{gsv_voices.ENGINE}/{gsv_voices.import_package(engine_dir, name, voice_files)}", gsv_voices.ENGINE
    _check_weights(contents[WEIGHTS_FILE])
    engine = _check_config(contents[CONFIG_FILE])
    target = engine_folder(lora_dir, engine) / name
    if target.exists():
        raise FileExistsError(f"A LoRA named '{name}' already exists.")

    target.mkdir(parents=True)
    for file, data in contents.items():
        (target / file).write_bytes(data)
    return f"{engine}/{name}", engine
