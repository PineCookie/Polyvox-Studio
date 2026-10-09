"""Training manifests: discovery, path normalization and inspection.

Manifests are JSONL files of ``{"audio": ..., "text": ...}`` (optionally
``ref_audio``). voxcpm's loader resolves relative audio paths against the
manifest's directory, but manifests written by ``lora_ft_webui`` /
``convert_dataset.py`` are relative to the VoxCPM checkout the webui ran from
(``data/<name>/wavs/...``). ``legacy_root`` is that checkout; runs get copies of
their manifests with absolute paths, so both layouts train correctly.
"""

from __future__ import annotations

import functools
import hashlib
import json
import random
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

MANIFEST_GLOB = "*.jsonl"
ASR_FILE = "asr.json"  # next to a manifest: {"<audio as written>": {"language": ...}} for ASR-made texts
ALIGN_FILE = "align.json"  # next to a manifest: per-clip alignment checks (checks.py)
MAX_REPORTED = 20  # errors / warnings listed individually
SHORT_CLIP_S = 0.3
LONG_CLIP_S = 30.0


def legacy_root(data_dir: Path) -> Path:
    """The webui's working directory: the folder that holds ``data/`` and ``lora/``."""
    return data_dir.resolve().parent


def scan(data_dir: Path) -> list[str]:
    """Return manifest ids (POSIX paths relative to ``data_dir``)."""
    data_dir = data_dir.resolve()
    if not data_dir.is_dir():
        return []
    ids = (p.relative_to(data_dir) for p in data_dir.rglob(MANIFEST_GLOB))
    # Hidden folders hold work in progress, e.g. an import's staging dir.
    return sorted(p.as_posix() for p in ids if not any(part.startswith(".") for part in p.parts))


def resolve(data_dir: Path, dataset_id: str) -> Path:
    """Look up one manifest by id, refusing paths outside ``data_dir``."""
    data_dir = data_dir.resolve()
    path = (data_dir / dataset_id).resolve()
    if not path.is_relative_to(data_dir) or path.suffix != ".jsonl" or not path.is_file():
        raise KeyError(f"Unknown dataset: {dataset_id}")
    return path


def _resolve_audio(value: str, manifest_dir: Path, root: Path | None) -> str:
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    candidates = [manifest_dir / path] + ([root / path] if root else [])
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())
    return str(candidates[0].resolve())


def read_manifest(path: Path, root: Path | None = None) -> tuple[list[dict], list[str]]:
    """Parse a manifest, making audio paths absolute.

    Returns the usable entries and one error per malformed line.
    """
    entries, errors = [], []
    manifest_dir = path.resolve().parent
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"Line {lineno}: invalid JSON ({e.msg})")
                continue
            if not isinstance(entry, dict) or not isinstance(entry.get("audio"), str) or not entry["audio"]:
                errors.append(f"Line {lineno}: missing 'audio' path")
                continue
            if not isinstance(entry.get("text"), str) or not entry["text"].strip():
                errors.append(f"Line {lineno}: missing 'text'")
                continue
            entry["audio"] = _resolve_audio(entry["audio"], manifest_dir, root)
            if isinstance(entry.get("ref_audio"), str) and entry["ref_audio"]:
                entry["ref_audio"] = _resolve_audio(entry["ref_audio"], manifest_dir, root)
            entries.append(entry)
    return entries, errors


def write_manifest(entries: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


@dataclass
class ClipRow:
    index: int  # position among the manifest's usable entries (stable ids for audio)
    file: str  # audio path as it appears relative to the manifest, or absolute
    text: str
    seconds: float | None  # None: unreadable or missing
    asr: bool  # text made by ASR (listed in asr.json)
    score: float | None = None  # alignment check (align.json); None: not checked for this text
    flags: list[str] = field(default_factory=list)
    speech: tuple[float, float] | None = None  # aligned words' span in seconds, when checked
    peaks: list[float] | None = None  # waveform outline (clip_peaks); None: unreadable


@dataclass
class ClipPage:
    total: int  # clips matching the search / filter
    rows: list[ClipRow]
    # Over the whole dataset: {"aligner", "checked", "flagged", "flags": {flag: count}}, or None.
    checks: dict | None = None


def row_key(audio: str, manifest_dir: Path) -> str:
    """A clip's key in the sidecar files: its path relative to the manifest, else as is."""
    path = Path(audio)
    return path.relative_to(manifest_dir).as_posix() if path.is_relative_to(manifest_dir) else audio


def text_key(text: str) -> str:
    """Identifies the transcript a check was made for."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


@functools.lru_cache(maxsize=32)
def _sidecar(path: str, mtime_ns: int) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_sidecar(path: Path) -> dict:
    """A JSON file next to a manifest ({} if absent), cached until it changes."""
    try:
        return _sidecar(str(path), path.stat().st_mtime_ns)
    except FileNotFoundError:
        return {}


@functools.lru_cache(maxsize=16)
def _review_entries(path: str, mtime_ns: int, size: int, root: str) -> tuple[tuple[str, str], ...]:
    """(absolute audio, text) per usable entry; cached until the manifest changes.

    Parsing resolves every clip's path on disk: ~0.4 s for 2k clips, too slow per page.
    """
    entries, _ = read_manifest(Path(path), Path(root) if root else None)
    return tuple((e["audio"], e["text"]) for e in entries)


def _entries_for_review(path: Path, root: Path | None) -> tuple[tuple[str, str], ...]:
    path = path.resolve()
    st = path.stat()
    return _review_entries(str(path), st.st_mtime_ns, st.st_size, str(root) if root else "")


@functools.lru_cache(maxsize=200_000)
def _clip_seconds(audio: str) -> float | None:
    try:
        return sf.info(audio).duration
    except Exception:
        return None


PEAK_BINS = 120


@functools.lru_cache(maxsize=8192)
def _peaks(audio: str, mtime_ns: int, bins: int) -> tuple[float, ...] | None:
    try:
        data, _ = sf.read(audio, dtype="float32", always_2d=True)
    except Exception:
        return None
    level = np.abs(data).max(axis=1)
    edges = np.linspace(0, len(level), bins + 1).astype(int)
    return tuple(
        round(float(level[a:b].max()), 3) if b > a else 0.0 for a, b in zip(edges[:-1], edges[1:], strict=True)
    )


def clip_peaks(audio: str, bins: int = PEAK_BINS) -> list[float] | None:
    """Peak level (0..1) per equal slice of a clip, to draw its waveform; cached
    until the file changes. None when the file is missing or unreadable."""
    try:
        mtime = Path(audio).stat().st_mtime_ns
    except OSError:
        return None
    peaks = _peaks(audio, mtime, bins)
    return list(peaks) if peaks is not None else None


def read_rows(
    path: Path,
    root: Path | None = None,
    offset: int = 0,
    limit: int = 50,
    query: str = "",
    flagged: bool = False,
    worst_first: bool = False,
) -> ClipPage:
    """One page of a manifest's clips for review.

    ``query`` matches text or file name, case-insensitively; ``flagged`` keeps
    clips with a check flag; ``worst_first`` orders by alignment score
    (unchecked clips last).
    """
    manifest_dir = path.resolve().parent
    entries = _entries_for_review(path, root)
    asr = read_sidecar(manifest_dir / ASR_FILE)
    aligned = read_sidecar(manifest_dir / ALIGN_FILE)
    checks = aligned.get("clips", {})

    def check_of(key: str, text: str) -> dict | None:
        check = checks.get(key)
        return check if check and check.get("text") == text_key(text) else None

    q = query.strip().lower()
    matches = []
    summary = {"checked": 0, "flagged": 0, "flags": {}}
    for i, (audio, text) in enumerate(entries):
        key = row_key(audio, manifest_dir)
        check = check_of(key, text)
        if check:
            summary["checked"] += 1
            summary["flagged"] += bool(check["flags"])
            for f in check["flags"]:
                summary["flags"][f] = summary["flags"].get(f, 0) + 1
        if q and q not in text.lower() and q not in Path(audio).name.lower():
            continue
        if flagged and not (check and check["flags"]):
            continue
        matches.append((i, audio, text, key, check))
    if worst_first:
        matches.sort(key=lambda m: (m[4] is None, m[4]["score"] if m[4] else 0.0, m[0]))

    rows = [
        ClipRow(
            index=i,
            file=key,
            text=text,
            seconds=_clip_seconds(audio),
            asr=key in asr,
            score=check["score"] if check else None,
            flags=check["flags"] if check else [],
            speech=(check["start"], check["end"]) if check and "start" in check else None,
            peaks=clip_peaks(audio),
        )
        for i, audio, text, key, check in matches[offset : offset + limit]
    ]
    has_checks = summary["checked"] > 0
    return ClipPage(len(matches), rows, {"aligner": aligned.get("aligner"), **summary} if has_checks else None)


def clip_audio(path: Path, root: Path | None, index: int) -> Path:
    """The audio file of a manifest's ``index``-th usable entry."""
    entries = _entries_for_review(path, root)
    if not 0 <= index < len(entries) or not Path(entries[index][0]).is_file():
        raise KeyError(f"No clip {index} in this dataset.")
    return Path(entries[index][0])


def split_holdout(entries: list[dict], count: int, seed: int = 0) -> tuple[list[dict], list[dict]]:
    """Move ``count`` reproducibly chosen entries into a validation set."""
    count = min(count, len(entries) - 1)
    if count <= 0:
        return entries, []
    held = set(random.Random(seed).sample(range(len(entries)), count))
    train = [e for i, e in enumerate(entries) if i not in held]
    val = [e for i, e in enumerate(entries) if i in held]
    return train, val


@dataclass
class DatasetReport:
    samples: int = 0  # well-formed lines
    valid: int = 0  # well-formed lines whose audio is readable
    total_seconds: float = 0.0
    min_seconds: float | None = None
    median_seconds: float | None = None
    max_seconds: float | None = None
    # Training resamples to the model rate, so mixed rates are fine.
    sample_rates: dict[int, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _capped(messages: list[str], total: int, what: str) -> list[str]:
    if total > len(messages):
        messages.append(f"... and {total - len(messages)} more {what}")
    return messages


def inspect(path: Path, root: Path | None = None) -> DatasetReport:
    entries, line_errors = read_manifest(path, root)
    report = DatasetReport(samples=len(entries))
    durations: list[float] = []
    rates: Counter[int] = Counter()
    # Malformed lines are left out of a run's manifest copy, so they only warn;
    # unreadable audio would crash training mid-epoch.
    errors: list[str] = []
    error_count = 0
    short = long = 0

    for entry in entries:
        try:
            info = sf.info(entry["audio"])
        except Exception as e:
            error_count += 1
            if len(errors) < MAX_REPORTED:
                reason = "file not found" if not Path(entry["audio"]).is_file() else str(e)
                errors.append(f"{entry['audio']}: {reason}")
            continue
        durations.append(info.duration)
        rates[info.samplerate] += 1
        short += info.duration < SHORT_CLIP_S
        long += info.duration > LONG_CLIP_S

    report.valid = len(durations)
    report.errors = _capped(errors, error_count, "errors")
    report.warnings = _capped([f"Skipped: {e}" for e in line_errors[:MAX_REPORTED]], len(line_errors), "skipped lines")
    if short:
        report.warnings.append(f"{short} clips shorter than {SHORT_CLIP_S} s")
    if long:
        report.warnings.append(f"{long} clips longer than {LONG_CLIP_S:.0f} s (may run out of memory)")
    if durations:
        report.total_seconds = sum(durations)
        report.min_seconds = min(durations)
        report.median_seconds = statistics.median(durations)
        report.max_seconds = max(durations)
    report.sample_rates = dict(sorted(rates.items()))
    return report
