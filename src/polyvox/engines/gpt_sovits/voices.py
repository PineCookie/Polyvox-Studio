"""GPT-SoVITS voices: folders in the LoRA folder, like the other engines' checkpoints.

A voice folder holds the two trained models, their config and a default
reference clip (GPT-SoVITS always clones from a clip; Synthesize uses this one
unless another is given):

    sovits_weights.pth   SoVITS weights, saved as the webui saves them
    gpt_weights.ckpt     GPT weights (none: the base GPT, e.g. a training
                         checkpoint from before the GPT stage)
    lora_config.json     {"engine": "gpt_sovits", "version": "v2ProPlus", "base_model": ...}
    reference.wav        the default reference clip (3-10 s)
    reference.json       {"text": its transcript, "source": where it came from}
"""

from __future__ import annotations

import json
import random
import re
import shutil
from pathlib import Path

import numpy as np

from ...data import audio_io, manifests
from .synth import REF_MAX_S, REF_MIN_S

ENGINE = "gpt_sovits"
BASE_REPO = "lj1995/GPT-SoVITS"
SOVITS_FILE = "sovits_weights.pth"
GPT_FILE = "gpt_weights.ckpt"
CONFIG_FILE = "lora_config.json"  # as lora_library reads it for every engine
REFERENCE_WAV = "reference.wav"
REFERENCE_JSON = "reference.json"
VOICE_NAME = r"^[^\W_][\w.-]{0,79}$"  # as run and package names
PICK_RANGE_S = (4.0, 10.0)  # a default reference picked from a dataset


def read_reference(voice_dir: Path) -> tuple[Path, str] | None:
    """The voice's default reference clip and its transcript, if it has one."""
    wav, info = voice_dir / REFERENCE_WAV, voice_dir / REFERENCE_JSON
    if not wav.is_file():
        return None
    try:
        text = str(json.loads(info.read_text(encoding="utf-8")).get("text") or "")
    except (OSError, ValueError):
        text = ""
    return wav, text


def write_reference(voice_dir: Path, source: Path, text: str, origin: str = "") -> None:
    """Make ``source`` (any readable audio, 3-10 s) the voice's default reference."""
    audio, rate = audio_io.read(source)
    mono = audio.mean(axis=0) if audio.ndim == 2 else audio
    seconds = len(mono) / rate
    if not REF_MIN_S <= seconds <= REF_MAX_S:
        raise ValueError(f"The reference clip is {seconds:.1f} s; GPT-SoVITS needs {REF_MIN_S:g}-{REF_MAX_S:g} s.")
    import soundfile as sf

    voice_dir.mkdir(parents=True, exist_ok=True)
    tmp = voice_dir / f".{REFERENCE_WAV}.tmp"
    sf.write(tmp, np.asarray(mono, dtype=np.float32), rate, format="WAV", subtype="PCM_16")
    tmp.replace(voice_dir / REFERENCE_WAV)
    info = {"text": text.strip(), "source": origin}
    (voice_dir / REFERENCE_JSON).write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")


HESITATIONS = "嗯呃唔额"  # never words or sentence particles (unlike 啊 / 哦 in 这样啊)
MIN_CONTENT_CHARS = 10  # enough words for the GPT to continue from
_CLAUSES = re.compile(r"[，。！？、；：,.!?;:…~—\s]+")
# A stutter: 2-4 characters said twice in a row (也是也是); single ones (谢谢, 看看) are words.
_REPEAT = re.compile(r"(\w{2,4})\1")


def text_quality(text: str) -> tuple[bool, bool]:
    """(free of hesitations, enough content) of a transcript: a reference with
    嗯 / 呃, filler-only clauses (哈哈，) or stutters tends to make the voice
    start its sentences with them, and a few words give the GPT little to go on."""
    from ...prep.fillers import FILLER_CHARS

    clauses = [c for c in _CLAUSES.split(text) if c]
    filler_only = [c for c in clauses if all(ch in FILLER_CHARS for ch in c)]
    hesitant = bool(filler_only) or any(ch in HESITATIONS for ch in text) or bool(_REPEAT.search(text))
    content = sum(ch not in HESITATIONS for c in clauses if c not in filler_only for ch in c)
    return not hesitant, content >= MIN_CONTENT_CHARS


def pick_reference(manifest: Path, root: Path | None = None, seed: int = 0) -> tuple[Path, str]:
    """A good default reference from a dataset, among clips of 4-10 s not
    dropped: preferably without hesitations and with enough words
    (text_quality), then checked without flags (align.json) with the best
    score, then nearest 6 s. Raises ValueError when no clip fits."""
    entries, _ = manifests.training_entries(manifest, root)
    manifest_dir = manifest.resolve().parent
    checks = manifests.read_sidecar(manifest_dir / manifests.ALIGN_FILE).get("clips", {})
    candidates = []
    for entry in entries:
        try:
            seconds, _ = audio_io.info(entry["audio"])
        except (OSError, RuntimeError, ValueError):
            continue
        if not PICK_RANGE_S[0] <= seconds <= PICK_RANGE_S[1]:
            continue
        check = checks.get(manifests.row_key(entry["audio"], manifest_dir))
        if check and check.get("text") != manifests.text_key(entry["text"]):
            check = None  # made for an earlier transcript
        if check and check.get("flags"):
            continue
        score = check.get("score") if check else None
        candidates.append((*text_quality(entry["text"]), score is not None, score or 0.0, -abs(seconds - 6.0), entry))
    if not candidates:
        raise ValueError(f"No clip of {PICK_RANGE_S[0]:g}-{PICK_RANGE_S[1]:g} s without flags in this dataset.")
    best = max(c[:-1] for c in candidates)
    tied = [c[-1] for c in candidates if c[:-1] == best]
    entry = random.Random(seed).choice(tied)
    return Path(entry["audio"]), entry["text"]


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
    from ...vendor.gpt_sovits.process_ckpt import get_sovits_version_from_path_fast, load_sovits_new, load_weights

    if not re.fullmatch(VOICE_NAME, name):
        raise ValueError("Voice name may contain letters (any language), digits, '.', '_' and '-' only.")
    sovits, gpt = Path(sovits), Path(gpt) if gpt is not None else None
    for path, kind in ((sovits, "SoVITS"), (gpt, "GPT")):
        if path is not None and not path.is_file():
            raise ValueError(f"No {kind} weights at {path}.")
    _, version, lora = get_sovits_version_from_path_fast(str(sovits))
    if lora or version not in ("v2Pro", "v2ProPlus"):
        raise ValueError(f"{sovits.name} is GPT-SoVITS {version}; only v2Pro / v2ProPlus voices are supported.")
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

    from ...vendor.gpt_sovits.process_ckpt import (
        get_sovits_version_from_path_fast,
        load_sovits_new,
        load_weights,
        model_version2byte,
    )

    sovits_path = voice_dir / SOVITS_FILE
    _, version, _ = get_sovits_version_from_path_fast(str(sovits_path))
    if version not in ("v2Pro", "v2ProPlus"):
        raise ValueError(f"Only v2Pro / v2ProPlus voices can be exported, not {version}.")

    def saved(obj, tag: bytes = b"") -> bytes:
        buf = io.BytesIO()
        torch.save(obj, buf)
        data = buf.getvalue()
        return tag + data[2:] if tag else data

    sovits = load_sovits_new(sovits_path)
    config = {**plain_config(sovits["config"]), "name": name}
    files = {
        SOVITS_FILE: saved(
            {"weight": sovits["weight"], "config": config, "info": sovits.get("info", "")},
            model_version2byte[version],
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
