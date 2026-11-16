"""Saved Drama 3 voices: a name and a reference, no training (the tower already renders a voice
from its reference as well as the recordings do; what Drama 3 misses comes from the backbone).

A voice folder in the LoRA folder (``fish_drama3/<name>/``):
    voice.json       {"engine": "fish_drama3", "base_model": ...}: marks the folder as a voice
    reference.wav    the reference: a dataset's best few clips joined, or one clip
    reference.json   {"text": its transcript, "source": where it came from}
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from ...lora_library import VOICE_FILE
from ..fish_s2 import lora as s2lora

ENGINE = "fish_drama3"
BASE_MODEL = "fishaudio/drama-3"
NAME = "Fish Audio Drama 3"
VOICE_NAME = r"^[^\W_][\w.-]{0,79}$"  # as run and package names


def create(
    engine_dir: Path, name: str, clips: list | None = None, audio: Path | None = None, text: str = "", source: str = ""
) -> Path:
    """A new voice folder from dataset clips (data.references.Clip, joined) or one ``audio`` clip
    with its transcript ``text``."""
    if not re.fullmatch(VOICE_NAME, name):
        raise ValueError("A voice name may contain letters (any language), digits, '.', '_' and '-' only.")
    folder = engine_dir / name
    if folder.exists():
        raise FileExistsError(f"A voice named '{name}' already exists.")
    staging = engine_dir / f".{name}.new"
    shutil.rmtree(staging, ignore_errors=True)
    try:
        if clips:
            s2lora.write_reference_clips(staging, clips, source)
        elif audio is not None:
            s2lora.write_reference_file(staging, audio, text, source, NAME)
        else:
            raise ValueError("Give a dataset to pick clips from, or a clip with its transcript.")
        info = {"engine": ENGINE, "base_model": BASE_MODEL}
        (staging / VOICE_FILE).write_text(json.dumps(info, indent=1), encoding="utf-8")
        staging.rename(folder)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return folder
