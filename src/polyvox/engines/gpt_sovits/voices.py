"""GPT-SoVITS voices: folders in the LoRA folder, like the other engines' checkpoints.

A voice folder holds the two trained models, their config and a default
reference clip (GPT-SoVITS always clones from a clip; Synthesize uses this one
unless another is given):

    sovits_weights.pth   SoVITS weights, saved as the webui saves them (v4: the
                         trained parts and the DiT's LoRA, over the v4 base)
    gpt_weights.ckpt     GPT weights (none: the base GPT, e.g. a training
                         checkpoint from before the GPT stage)
    lora_config.json     {"engine": "gpt_sovits", "version": "v2ProPlus", "base_model": ...}
    reference.wav        the default reference clip (3-10 s)
    reference.json       {"text": its transcript, "source": where it came from}
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from pydantic import BaseModel

from ...data import references
from ...data.references import text_quality  # noqa: F401  (part of this module's API)
from ..common import reference
from ..common.reference import REFERENCE_JSON, REFERENCE_WAV, read_reference  # noqa: F401  (the voice's files)
from .synth import REF_MAX_S, REF_MIN_S, V4, VERSIONS, sovits_version

ENGINE = "gpt_sovits"
BASE_REPO = "lj1995/GPT-SoVITS"
SOVITS_FILE = "sovits_weights.pth"
GPT_FILE = "gpt_weights.ckpt"
CONFIG_FILE = "lora_config.json"  # as lora_library reads it for every engine
VOICE_NAME = r"^[^\W_][\w.-]{0,79}$"  # as run and package names


class WebuiImport(BaseModel):
    """Weights the GPT-SoVITS webui trained, as server-local paths (Engine.import_voice)."""

    name: str
    sovits: str  # SoVITS_weights_*/<voice>.pth
    gpt: str = ""  # GPT_weights_*/<voice>.ckpt; none: the base GPT
    # The default reference: picked from this dataset (manifest id), or this clip + transcript;
    # neither: none yet.
    dataset: str | None = None
    reference: str | None = None
    reference_text: str = ""


def write_reference(voice_dir: Path, source: Path, text: str, origin: str = "") -> None:
    """Make ``source`` (any readable audio, 3-10 s) the voice's default reference."""
    reference.write_file(
        voice_dir, source, text, origin, model="GPT-SoVITS", min_seconds=REF_MIN_S, max_seconds=REF_MAX_S,
        transcript=False,
    )  # fmt: skip


def pick_reference(manifest: Path, root: Path | None = None, seed: int = 0) -> tuple[Path, str]:
    """A good default reference from a dataset (references.pick); ValueError when no clip fits."""
    (clip,) = references.pick(manifest, 1, root, seed)
    return clip.audio, clip.text


def write_config(voice_dir: Path, version: str, base_model: str = BASE_REPO, **extra) -> None:
    """``extra``: e.g. a training checkpoint's stage and epoch."""
    info = {"engine": ENGINE, "version": version, "base_model": base_model, **extra}
    (voice_dir / CONFIG_FILE).write_text(json.dumps(info, indent=1), encoding="utf-8")


def import_webui(
    lora_dir: Path,
    name: str,
    sovits: Path,
    gpt: Path | None,
    reference: tuple[Path, str] | None,
    reference_origin: str = "",
) -> str:
    """A voice from weights the GPT-SoVITS webui trained (its SoVITS_weights_* /
    GPT_weights_* files), into ``lora_dir/<name>/``; returns the voice id.
    Without a reference it synthesizes only with a clip given each time, until
    one is set (write_reference). Without ``gpt`` it speaks with the base GPT
    (a package of a checkpoint from before training's GPT stage)."""
    from ...vendor.gpt_sovits.process_ckpt import load_sovits_new, load_weights

    if not re.fullmatch(VOICE_NAME, name):
        raise ValueError("Voice name may contain letters (any language), digits, '.', '_' and '-' only.")
    sovits, gpt = Path(sovits), Path(gpt) if gpt is not None else None
    for path, kind in ((sovits, "SoVITS"), (gpt, "GPT")):
        if path is not None and not path.is_file():
            raise ValueError(f"No {kind} weights at {path}.")
    try:
        version, lora = sovits_version(sovits)
    except Exception as e:
        raise ValueError(f"Unreadable weights: {e}") from e
    if version not in VERSIONS or (lora and version != V4):
        raise ValueError(f"{sovits.name} is GPT-SoVITS {version}; only {' / '.join(VERSIONS)} voices are supported.")
    try:  # both readable (and holding no pickled code) before anything is copied
        load_sovits_new(sovits)
        gpt_saved = load_weights(gpt) if gpt is not None else {"weight": {}, "config": {}}
    except Exception as e:
        raise ValueError(f"Unreadable weights: {e}") from e
    if not isinstance(gpt_saved, dict) or "weight" not in gpt_saved or "config" not in gpt_saved:
        raise ValueError(f"{gpt.name} is not a GPT-SoVITS GPT checkpoint.")
    target = lora_dir / name
    if target.exists():
        raise FileExistsError(f"A voice or run named '{name}' already exists.")
    lora_dir.mkdir(parents=True, exist_ok=True)
    staging = lora_dir / f".import-{name}"
    shutil.rmtree(staging, ignore_errors=True)
    try:
        staging.mkdir(parents=True)
        shutil.copyfile(sovits, staging / SOVITS_FILE)
        if gpt is not None:
            shutil.copyfile(gpt, staging / GPT_FILE)
        write_config(staging, version)
        if reference is not None:
            write_reference(staging, reference[0], reference[1], reference_origin)
        staging.replace(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return name


# ------------------------------------------------------------------ packages

# Training configs saved inside webui weights also record where they were made:
# experiment names, log / output folders, the pretrained and dataset files'
# paths. A package leaves these out; what loading reads (model, data and train
# settings) stays.
PRIVATE_CONFIG_KEYS = {
    "name", "exp_name", "exp_dir", "s2_ckpt_dir", "save_weight_dir", "half_weights_save_dir", "output_dir",
    "pretrained_s1", "pretrained_s2G", "pretrained_s2D", "pretrain", "resume_step", "train_semantic_path",
    "train_phoneme_path", "training_files", "validation_files", "gpu_numbers",
}  # fmt: skip


def plain_config(config) -> dict:
    """A saved config (dict or the webui's HParams) as plain dicts, without PRIVATE_CONFIG_KEYS."""
    if not hasattr(config, "items"):
        return config
    return {k: plain_config(v) for k, v in config.items() if k not in PRIVATE_CONFIG_KEYS}


def package_files(voice_dir: Path, name: str) -> dict[str, bytes]:
    """A voice's files for a shareable package (sharing.py), by file name:
    both weights re-saved with cleaned configs (the SoVITS with its version
    tag, as the webui saves it), the reference and only its transcript, the
    voice config."""
    import io

    import torch

    from ...vendor.gpt_sovits.process_ckpt import load_sovits_new, load_weights, model_version2byte

    sovits_path = voice_dir / SOVITS_FILE
    version, _ = sovits_version(sovits_path)
    if version not in VERSIONS:
        raise ValueError(f"Only {' / '.join(VERSIONS)} voices can be exported, not {version}.")

    def saved(obj, tag: bytes = b"") -> bytes:
        buf = io.BytesIO()
        torch.save(obj, buf)
        data = buf.getvalue()
        return tag + data[2:] if tag else data

    sovits = load_sovits_new(sovits_path)
    config = {**plain_config(sovits["config"]), "name": name}
    # v4 voices keep lora_rank: the webui reads it to tell the trained parts from full weights.
    lora = {"lora_rank": sovits["lora_rank"]} if sovits.get("lora_rank") else {}
    files = {
        SOVITS_FILE: saved(
            {"weight": sovits["weight"], "config": config, "info": sovits.get("info", ""), **lora},
            model_version2byte.get(version, b""),  # v2: no tag (recognized by size), as the webui saves it
        ),
        CONFIG_FILE: (
            json.dumps({"engine": ENGINE, "version": version, "base_model": BASE_REPO}, indent=2) + "\n"
        ).encode(),
    }
    if (voice_dir / GPT_FILE).is_file():
        gpt = load_weights(voice_dir / GPT_FILE)
        files[GPT_FILE] = saved(
            {"weight": gpt["weight"], "config": plain_config(gpt["config"]), "info": gpt.get("info", "")}
        )
    if reference := read_reference(voice_dir):
        files[REFERENCE_WAV] = reference[0].read_bytes()
        files[REFERENCE_JSON] = (json.dumps({"text": reference[1]}, ensure_ascii=False, indent=1) + "\n").encode()
    return files


def import_package(lora_dir: Path, name: str, files: dict[str, bytes]) -> str:
    """A voice from a package's files (package_files), checked as webui weights are."""
    import tempfile

    with tempfile.TemporaryDirectory(prefix="gpt_sovits_") as tmp:
        tmp = Path(tmp)
        for file, data in files.items():
            (tmp / file).write_bytes(data)
        reference = None
        if REFERENCE_WAV in files:
            text = json.loads(files.get(REFERENCE_JSON, b"{}") or b"{}").get("text", "")
            reference = (tmp / REFERENCE_WAV, str(text))
        gpt = tmp / GPT_FILE if GPT_FILE in files else None
        return import_webui(lora_dir, name, tmp / SOVITS_FILE, gpt, reference, "package")


def readme(name: str, version: str, has_gpt: bool, has_reference: bool, step: int | None) -> str:
    gpt_row = f"`{GPT_FILE}`" if has_gpt else "none: the base GPT (`s1v3.ckpt`) speaks for it"
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
`{GPT_FILE if has_gpt else "s1v3.ckpt (from the pretrained models)"}` into `GPT_weights_{version}/` (rename them
if you like), refresh the model lists and choose them; use the reference clip and
transcript as the reference audio and text.
"""
