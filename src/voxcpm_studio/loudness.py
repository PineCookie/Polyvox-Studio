"""Loudness (ITU-R BS.1770-4 / EBU R128, via pyloudnorm) and normalization of clips.

Normalizing applies one gain per clip, lowered where needed so the sample peak
stays at or below a ceiling (no limiter: it would distort; such clips end a
little quieter than the target).
"""

from __future__ import annotations

import math
import shutil
import statistics
from collections.abc import Callable
from pathlib import Path

import numpy as np
import soundfile as sf

from .manifests import read_manifest, row_key

TARGET_LUFS = -20.0
PEAK_CEILING_DB = -1.0  # dBFS, sample peak after normalizing
CLIPPED_LEVEL = 0.999  # |sample| at or above this counts as at full scale
UNCHANGED_DB = 0.5  # a smaller gain is not worth rewriting a file
PROGRESS_EVERY = 50  # clips


def integrated(audio: np.ndarray, rate: int) -> float:
    """Integrated loudness in LUFS of float audio, (frames,) or (channels,
    frames); -inf for silence or a clip shorter than one 400 ms block."""
    import pyloudnorm

    meter = pyloudnorm.Meter(rate)
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim == 2:
        x = x.T  # pyloudnorm takes (frames, channels)
    if x.shape[0] < int(meter.block_size * rate):
        return -math.inf
    return float(meter.integrated_loudness(x))


def peak_db(audio: np.ndarray) -> float:
    peak = float(np.abs(audio).max()) if audio.size else 0.0
    return 20 * math.log10(peak) if peak > 0 else -math.inf


def clipped_share(audio: np.ndarray) -> float:
    """Share of samples at full scale."""
    return float((np.abs(audio) >= CLIPPED_LEVEL).mean()) if audio.size else 0.0


def gain_db(audio: np.ndarray, rate: int, target: float = TARGET_LUFS, ceiling: float = PEAK_CEILING_DB) -> float:
    """Gain bringing ``audio`` to ``target`` LUFS, capped so its sample peak
    stays at or below ``ceiling`` dBFS; 0 when its loudness can't be measured."""
    lufs = integrated(audio, rate)
    if not math.isfinite(lufs):
        return 0.0
    return min(target - lufs, ceiling - peak_db(audio))


def normalize(
    audio: np.ndarray, rate: int, target: float = TARGET_LUFS, ceiling: float = PEAK_CEILING_DB
) -> np.ndarray:
    """``audio`` (float) at ``target`` LUFS, its peak at most ``ceiling`` dBFS."""
    return (audio * 10 ** (gain_db(audio, rate, target, ceiling) / 20)).astype(audio.dtype, copy=False)


def _spread(values: list[float]) -> list[float] | None:
    """[p10, median, p90], rounded."""
    if not values:
        return None
    values = sorted(values)
    return [
        round(values[len(values) // 10], 1),
        round(statistics.median(values), 1),
        round(values[9 * len(values) // 10], 1),
    ]


def normalize_dataset(
    manifest: Path,
    root: Path | None,
    target: float = TARGET_LUFS,
    progress: Callable[[str, int, int], None] | None = None,
) -> dict:
    """Bring every clip of a manifest to ``target`` LUFS in place, in its own
    format and channels (peaks capped at PEAK_CEILING_DB). Originals are kept
    under checks.ORIGINALS_DIR next to the manifest (the first version only, as
    trimming does). Clips outside the manifest's folder, in a format soundfile
    can't write back, or too short to measure are skipped. A gain changes no
    check (timings and relative levels stay), so align.json is left as it is.

    Returns {"normalized", "unchanged" (within UNCHANGED_DB), "peak_limited"
    (ended below the target), "skipped", "before" / "after": [p10, median, p90]
    LUFS}."""
    from .checks import ORIGINALS_DIR, TRIM_IN_PLACE_FORMATS

    progress = progress or (lambda phase, done, total: None)
    entries, _ = read_manifest(manifest, root)
    manifest_dir = manifest.resolve().parent
    counts = {"normalized": 0, "unchanged": 0, "peak_limited": 0, "skipped": 0}
    before: list[float] = []
    after: list[float] = []
    for n, entry in enumerate(entries):
        if n % PROGRESS_EVERY == 0:
            progress("normalizing", n, len(entries))
        audio = Path(entry["audio"])
        if not audio.is_relative_to(manifest_dir) or audio.suffix.lower() not in TRIM_IN_PLACE_FORMATS:
            counts["skipped"] += 1
            continue
        try:
            info = sf.info(str(audio))
            data, rate = sf.read(str(audio), dtype="float32", always_2d=True)
        except (OSError, RuntimeError):
            counts["skipped"] += 1
            continue
        lufs = integrated(data.T, rate)
        if not math.isfinite(lufs):
            counts["skipped"] += 1
            continue
        gain = min(target - lufs, PEAK_CEILING_DB - peak_db(data))
        before.append(lufs)
        after.append(lufs + gain if abs(gain) >= UNCHANGED_DB else lufs)
        if abs(gain) < UNCHANGED_DB:
            counts["unchanged"] += 1
            continue
        backup = manifest_dir / ORIGINALS_DIR / row_key(str(audio), manifest_dir)
        if not backup.exists():
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(audio, backup)
        tmp = audio.with_name(f".{audio.name}.tmp")
        sf.write(str(tmp), data * 10 ** (gain / 20), rate, subtype=info.subtype, format=info.format)
        tmp.replace(audio)
        counts["normalized"] += 1
        counts["peak_limited"] += target - (lufs + gain) >= UNCHANGED_DB
    progress("normalizing", len(entries), len(entries))
    return {**counts, "before": _spread(before), "after": _spread(after)}
