"""Reference clips for cloning, picked from a dataset, and several joined into one.

Several clips of a voice clone it better than one: engines that take one clip
get them joined (with short pauses) into a single reference and transcript.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import audio_io, manifests

PICK_RANGE_S = (4.0, 10.0)  # a reference clip picked from a dataset (GPT-SoVITS takes 3-10 s)
JOIN_PAUSE_S = 0.3  # between joined clips
HESITATIONS = "嗯呃唔额"  # never words or sentence particles (unlike 啊 / 哦 in 这样啊)
MIN_CONTENT_CHARS = 10  # enough words for the model to continue from
_CLAUSES = re.compile(r"[，。！？、；：,.!?;:…~—\s]+")
# A stutter: 2-4 characters said twice in a row (也是也是); single ones (谢谢, 看看) are words.
_REPEAT = re.compile(r"(\w{2,4})\1")
_ENDS = "。！？.!?…"


@dataclass(frozen=True)
class Clip:
    index: int  # in the dataset (manifests.clip_entries order)
    audio: Path
    text: str
    seconds: float


def text_quality(text: str) -> tuple[bool, bool]:
    """(free of hesitations, enough content) of a transcript: a reference with
    嗯 / 呃, filler-only clauses (哈哈，) or stutters tends to make the voice
    start its sentences with them, and a few words give the model little to go on."""
    from ..prep.fillers import FILLER_CHARS

    clauses = [c for c in _CLAUSES.split(text) if c]
    filler_only = [c for c in clauses if all(ch in FILLER_CHARS for ch in c)]
    hesitant = bool(filler_only) or any(ch in HESITATIONS for ch in text) or bool(_REPEAT.search(text))
    content = sum(ch not in HESITATIONS for c in clauses if c not in filler_only for ch in c)
    return not hesitant, content >= MIN_CONTENT_CHARS


def pick(
    manifest: Path, count: int = 1, root: Path | None = None, seed: int = 0, exclude: frozenset[int] = frozenset()
) -> list[Clip]:
    """The ``count`` best reference clips of a dataset, best first, among clips of
    4-10 s with a transcript, not dropped and not in ``exclude``: preferably
    without hesitations and with enough words (text_quality), then checked
    without flags (align.json) with the best score, then nearest 6 s; ties in a
    random order (``seed``). Raises ValueError when no clip fits."""
    manifest_dir = manifest.resolve().parent
    checks = manifests.read_sidecar(manifest_dir / manifests.ALIGN_FILE).get("clips", {})
    dropped = manifests.excluded_keys(manifest_dir)
    candidates = []
    for index, entry in enumerate(manifests.clip_entries(manifest, root)):
        key = manifests.row_key(entry["audio"], manifest_dir)
        if index in exclude or key in dropped or not entry["text"].strip():
            continue
        try:
            seconds, _ = audio_io.info(entry["audio"])
        except (OSError, RuntimeError, ValueError):
            continue
        if not PICK_RANGE_S[0] <= seconds <= PICK_RANGE_S[1]:
            continue
        check = checks.get(key)
        if check and check.get("text") != manifests.text_key(entry["text"]):
            check = None  # made for an earlier transcript
        if check and check.get("flags"):
            continue
        score = check.get("score") if check else None
        rank = (*text_quality(entry["text"]), score is not None, score or 0.0, -abs(seconds - 6.0))
        candidates.append((rank, Clip(index, Path(entry["audio"]), entry["text"], seconds)))
    if not candidates:
        raise ValueError(f"No clip of {PICK_RANGE_S[0]:g}-{PICK_RANGE_S[1]:g} s without flags in this dataset.")
    random.Random(seed).shuffle(candidates)
    candidates.sort(key=lambda c: c[0], reverse=True)  # stable: ties keep the shuffled order
    return [clip for _, clip in candidates[:count]]


def clips(manifest: Path, indices: list[int], root: Path | None = None) -> list[Clip]:
    """Chosen clips of a dataset by index; each needs its audio and a transcript."""
    entries = manifests.clip_entries(manifest, root)
    chosen = []
    for index in indices:
        if not 0 <= index < len(entries) or not Path(entries[index]["audio"]).is_file():
            raise KeyError(f"No clip {index} in this dataset.")
        audio, text = Path(entries[index]["audio"]), entries[index]["text"].strip()
        if not text:
            raise ValueError(f"Clip {index} has no transcript yet.")
        chosen.append(Clip(index, audio, text, audio_io.info(audio)[0]))
    return chosen


def join(chosen: list[Clip], dest: Path) -> str:
    """Write the clips one after another into ``dest`` (mono WAV at the highest
    of their rates, JOIN_PAUSE_S apart); returns their transcripts joined."""
    import soundfile as sf
    import torch
    import torchaudio

    parts, rate = [], 0
    for clip in chosen:
        audio, sr = audio_io.read(clip.audio)
        parts.append((np.asarray(audio.mean(axis=0) if audio.ndim == 2 else audio, dtype=np.float32), sr))
        rate = max(rate, sr)
    pause = np.zeros(int(JOIN_PAUSE_S * rate), dtype=np.float32)
    out = []
    for audio, sr in parts:
        if sr != rate:
            audio = torchaudio.functional.resample(torch.from_numpy(audio), sr, rate).numpy()
        out += [audio, pause]
    sf.write(dest, np.concatenate(out[:-1]), rate, format="WAV", subtype="PCM_16")
    return "".join(c.text if c.text[-1] in _ENDS else c.text + "。" for c in chosen)
