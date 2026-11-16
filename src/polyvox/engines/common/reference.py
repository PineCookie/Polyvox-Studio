"""A voice's default reference: the clip it speaks with when a request gives none.

In the voice's (or run's / checkpoint's) folder:
    reference.wav    PCM16 mono at the clip's own rate
    reference.json   {"text": its transcript, "source": where it came from}

Engines that clone better from a longer reference (S2-Pro, Drama 3) join a dataset's best few
clips into one (Capabilities.voice_reference_clips: ~20 s cloned clearly better than one clip);
GPT-SoVITS keeps one clip of 3-10 s.

An engine whose voices are only such a reference (Drama 3: capabilities.reference_voices) saves
them with create_voice: no weights, a voice.json beside the reference marks the folder.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REFERENCE_WAV, REFERENCE_JSON = "reference.wav", "reference.json"
MAX_SECONDS = 30.0
VOICE_NAME = r"^[^\W_][\w.-]{0,79}$"  # as run and package names


def read_reference(folder: Path) -> tuple[Path, str] | None:
    """The default reference clip and its transcript, if the folder has one."""
    wav, info = folder / REFERENCE_WAV, folder / REFERENCE_JSON
    if not wav.is_file():
        return None
    try:
        text = str(json.loads(info.read_text(encoding="utf-8")).get("text") or "")
    except (OSError, ValueError):
        text = ""
    return wav, text


def create_voice(
    engine_dir: Path,
    name: str,
    engine: str,
    base_model: str | None,
    model: str,
    clips: list | None = None,
    audio: Path | None = None,
    text: str = "",
    source: str = "",
) -> Path:
    """A saved voice of an engine whose voices are references only (capabilities.reference_voices):
    ``<engine_dir>/<name>/`` with the reference (dataset clips joined, or one ``audio`` clip with
    its transcript ``text``) and lora_library.VOICE_FILE {"engine", "base_model"}, which lists it."""
    import re
    import shutil

    from ...lora_library import VOICE_FILE

    if not re.fullmatch(VOICE_NAME, name):
        raise ValueError("A voice name may contain letters (any language), digits, '.', '_' and '-' only.")
    folder = engine_dir / name
    if folder.exists():
        raise FileExistsError(f"A voice named '{name}' already exists.")
    staging = engine_dir / f".{name}.new"
    shutil.rmtree(staging, ignore_errors=True)
    try:
        if clips:
            write_joined(staging, clips, source)
        elif audio is not None:
            write_file(staging, audio, text, source, model=model)
        else:
            raise ValueError("Give a dataset to pick clips from, or a clip with its transcript.")
        info = {"engine": engine, "base_model": base_model}
        (staging / VOICE_FILE).write_text(json.dumps(info, indent=1), encoding="utf-8")
        staging.rename(folder)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return folder


def write_info(folder: Path, text: str, source: str) -> None:
    info = {"text": text.strip(), "source": source}
    (folder / REFERENCE_JSON).write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")


def write_joined(folder: Path, clips: list, source: str) -> str:
    """Dataset clips (data.references.Clip) joined into the reference; returns its transcript."""
    from ...data import references

    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder / f".{REFERENCE_WAV}.tmp"
    text = references.join(clips, tmp)
    tmp.replace(folder / REFERENCE_WAV)
    write_info(folder, text, source)
    return text


def write_file(
    folder: Path,
    source: Path,
    text: str,
    origin: str = "",
    *,
    model: str,
    min_seconds: float = 0.0,
    max_seconds: float = MAX_SECONDS,
    transcript: bool = True,
) -> None:
    """Any clip audio_io reads, of ``min_seconds``-``max_seconds``, as the reference (``model``:
    for messages; ``transcript``: whether it needs one)."""
    import soundfile as sf

    from ...data import audio_io

    if transcript and not text.strip():
        raise ValueError(f"{model} needs the reference clip's transcript.")
    audio, rate = audio_io.read(source)
    mono = audio.mean(axis=0) if audio.ndim == 2 else audio
    seconds = len(mono) / rate
    if not min_seconds <= seconds <= max_seconds:
        needs = f"{min_seconds:g}-{max_seconds:g} s" if min_seconds else f"at most {max_seconds:g} s"
        raise ValueError(f"The reference clip is {seconds:.1f} s; {model} needs {needs}.")
    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder / f".{REFERENCE_WAV}.tmp"
    sf.write(tmp, np.asarray(mono, dtype=np.float32), rate, format="WAV", subtype="PCM_16")
    tmp.replace(folder / REFERENCE_WAV)
    write_info(folder, text, origin)
