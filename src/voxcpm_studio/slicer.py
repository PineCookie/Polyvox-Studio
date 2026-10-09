"""Slice long recordings into training clips at pauses.

1. Voice activity detection (Silero VAD) finds speech segments separated by
   pauses of at least ``VAD_MIN_SILENCE_S``.
2. Segments still longer than the maximum clip length (talking without any
   pause) are split at their quietest point.
3. Cut points are chosen by dynamic programming over the pauses: cutting at a
   long pause (likely a sentence end) is rewarded, clips far from the target
   length are penalized, and a clip never exceeds the maximum length or
   contains a pause longer than ``max_inner_gap``.
4. Each clip is padded into the surrounding silence and written at the source
   sample rate.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

VAD_RATE = 16_000
VAD_MIN_SILENCE_S = 0.1
ENERGY_FRAME_S = 0.02
CUT_REWARD_CAP_S = 1.0  # pauses beyond this are all "clearly a boundary"
CUT_WEIGHT = 2.0

# 16 kHz mono audio -> speech segments as (start, end) in seconds.
Vad = Callable[[np.ndarray], list[tuple[float, float]]]


@dataclass(frozen=True)
class SliceOptions:
    min_seconds: float = 2.0  # shorter clips are dropped
    max_seconds: float = 15.0
    target_seconds: float = 8.0
    max_inner_gap: float = 1.0  # a longer pause always ends a clip
    pad_seconds: float = 0.15


_vad_model = None
_vad_lock = threading.Lock()


def silero_vad(wav16k: np.ndarray) -> list[tuple[float, float]]:
    """Speech segments with Silero VAD (bundled model, runs on CPU)."""
    global _vad_model
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad

    with _vad_lock:  # the model is stateful
        if _vad_model is None:
            _vad_model = load_silero_vad()
        stamps = get_speech_timestamps(
            torch.from_numpy(np.ascontiguousarray(wav16k)),
            _vad_model,
            sampling_rate=VAD_RATE,
            min_silence_duration_ms=int(VAD_MIN_SILENCE_S * 1000),
            speech_pad_ms=0,
        )
    return [(s["start"] / VAD_RATE, s["end"] / VAD_RATE) for s in stamps]


def split_long(
    segments: list[tuple[float, float]], wav16k: np.ndarray, max_seconds: float
) -> list[tuple[float, float]]:
    """Split segments longer than ``max_seconds`` at their quietest frame
    (searched in the middle half, so both parts stay reasonably long)."""
    frame = int(ENERGY_FRAME_S * VAD_RATE)
    out: list[tuple[float, float]] = []
    todo = list(segments)
    while todo:
        start, end = todo.pop(0)
        if end - start <= max_seconds:
            out.append((start, end))
            continue
        lo = int((start + (end - start) * 0.25) * VAD_RATE)
        hi = int((start + (end - start) * 0.75) * VAD_RATE)
        window = wav16k[lo:hi]
        n = max(1, len(window) // frame)
        rms = np.sqrt(np.mean(window[: n * frame].reshape(n, frame) ** 2, axis=1)) if len(window) >= frame else [0]
        cut = (lo + int(np.argmin(rms)) * frame + frame // 2) / VAD_RATE
        todo[:0] = [(start, cut), (cut, end)]
    return sorted(out)


def plan_clips(segments: list[tuple[float, float]], opts: SliceOptions) -> list[tuple[float, float]]:
    """Group consecutive speech segments into clips (start, end) in seconds."""
    n = len(segments)
    if n == 0:
        return []
    gaps = [segments[k + 1][0] - segments[k][1] for k in range(n - 1)]
    inf = float("inf")
    best = [inf] * (n + 1)  # best[j + 1]: cost of segments[0..j]
    back = [0] * (n + 1)
    best[0] = 0.0
    for j in range(n):
        for i in range(j, -1, -1):
            if i < j and gaps[i] > opts.max_inner_gap:
                break  # this pause must end a clip
            dur = segments[j][1] - segments[i][0]
            if dur > opts.max_seconds and i < j:
                break
            cost = ((dur - opts.target_seconds) / opts.target_seconds) ** 2
            if dur < opts.min_seconds:
                cost += 10.0 * (opts.min_seconds - dur)
            if j < n - 1:
                cost -= CUT_WEIGHT * min(gaps[j], CUT_REWARD_CAP_S)
            if best[i] + cost < best[j + 1]:
                best[j + 1], back[j + 1] = best[i] + cost, i
    clips = []
    j = n
    while j > 0:
        i = back[j]
        clips.append((segments[i][0], segments[j - 1][1]))
        j = i
    return clips[::-1]


def pad_clips(clips: list[tuple[float, float]], pad: float, duration: float) -> list[tuple[float, float]]:
    """Extend clips into the silence around them, at most halfway to a neighbor."""
    out = []
    for k, (start, end) in enumerate(clips):
        lo = (clips[k - 1][1] + start) / 2 if k > 0 else 0.0
        hi = (end + clips[k + 1][0]) / 2 if k + 1 < len(clips) else duration
        out.append((max(start - pad, lo), min(end + pad, hi)))
    return out


def find_clips(wav16k: np.ndarray, opts: SliceOptions, vad: Vad = silero_vad) -> list[tuple[float, float]]:
    duration = len(wav16k) / VAD_RATE
    segments = split_long(vad(wav16k), wav16k, opts.max_seconds)
    clips = pad_clips(plan_clips(segments, opts), opts.pad_seconds, duration)
    return [(s, e) for s, e in clips if e - s >= opts.min_seconds]


def _read_native(path: Path) -> tuple[np.ndarray, int]:
    """Mono float32 at the file's own sample rate."""
    try:
        audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
        return audio.mean(axis=1), rate
    except Exception:
        import torchaudio

        tensor, rate = torchaudio.load(str(path))
        return tensor.mean(dim=0).numpy(), rate


def slice_recording(
    path: Path, out_dir: Path, opts: SliceOptions, vad: Vad = silero_vad
) -> list[Path]:
    """Write ``<stem>_<start ms>_<end ms>.wav`` clips (mono, source rate) into ``out_dir``."""
    from .asr import load_audio

    clips = find_clips(load_audio(path), opts, vad)
    audio, rate = _read_native(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for start, end in clips:
        file = out_dir / f"{path.stem}_{int(start * 1000):08d}_{int(end * 1000):08d}.wav"
        sf.write(file, audio[int(start * rate) : int(end * rate)], rate, subtype="PCM_16")
        files.append(file)
    return files
