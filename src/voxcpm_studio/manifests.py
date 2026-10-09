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
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

MANIFEST_GLOB = "*.jsonl"
ASR_FILE = "asr.json"  # next to a manifest: {"<audio as written>": {"language": ...}} for ASR-made texts
ALIGN_FILE = "align.json"  # next to a manifest: per-clip alignment checks (checks.py)
# Next to a manifest: {"clips": ["<audio as written>", ...]} dropped in review. They
# stay in the manifest (stable indices, easy to restore) but not in training runs.
EXCLUDED_FILE = "excluded.json"
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


def _parse_line(line: str) -> tuple[dict | None, str | None]:
    """(entry, None) for a usable manifest line, (None, why not) otherwise,
    (None, None) for a blank one. The one rule for what counts as a clip."""
    if not line.strip():
        return None, None
    try:
        entry = json.loads(line)
    except json.JSONDecodeError as e:
        return None, f"invalid JSON ({e.msg})"
    if not isinstance(entry, dict) or not isinstance(entry.get("audio"), str) or not entry["audio"]:
        return None, "missing 'audio' path"
    if not isinstance(entry.get("text"), str) or not entry["text"].strip():
        return None, "missing 'text'"
    return entry, None


def read_manifest(path: Path, root: Path | None = None) -> tuple[list[dict], list[str]]:
    """Parse a manifest, making audio paths absolute.

    Returns the usable entries and one error per malformed line.
    """
    entries, errors = [], []
    manifest_dir = path.resolve().parent
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            entry, error = _parse_line(line)
            if error:
                errors.append(f"Line {lineno}: {error}")
            if entry is None:
                continue
            entry["audio"] = _resolve_audio(entry["audio"], manifest_dir, root)
            if isinstance(entry.get("ref_audio"), str) and entry["ref_audio"]:
                entry["ref_audio"] = _resolve_audio(entry["ref_audio"], manifest_dir, root)
            entries.append(entry)
    return entries, errors


def rewrite_manifest(path: Path, change) -> int:
    """Apply ``change(entry) -> entry | None`` (None: unchanged) to each usable
    entry, in place: other lines and keys stay as written. The first time, the
    original is kept as ``<name>.orig``. Returns the number of entries changed."""
    lines = path.read_text(encoding="utf-8").splitlines()
    changed = 0
    for n, line in enumerate(lines):
        entry, _ = _parse_line(line)
        if entry is not None and (new := change(entry)) is not None:
            lines[n] = json.dumps(new, ensure_ascii=False)
            changed += 1
    if changed:
        backup = path.with_name(path.name + ".orig")
        if not backup.exists():
            backup.write_bytes(path.read_bytes())
        tmp = path.with_name(f".{path.name}.tmp")
        tmp.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        tmp.replace(path)
    return changed


def excluded_keys(manifest_dir: Path) -> set[str]:
    return set(read_sidecar(manifest_dir / EXCLUDED_FILE).get("clips", []))


def set_dropped(path: Path, root: Path | None, index: int, dropped: bool) -> None:
    """Drop the ``index``-th clip from training (or restore it)."""
    entries = _entries_for_review(path, root)
    if not 0 <= index < len(entries):
        raise KeyError(f"No clip {index} in this dataset.")
    manifest_dir = path.resolve().parent
    keys = excluded_keys(manifest_dir)
    key = row_key(entries[index][0], manifest_dir)
    _write_excluded(manifest_dir, keys | {key} if dropped else keys - {key})


def drop_flagged(path: Path, root: Path | None, flag: str) -> int:
    """Drop every clip whose current check has ``flag``; returns how many were
    newly dropped."""
    manifest_dir = path.resolve().parent
    checks = read_sidecar(manifest_dir / ALIGN_FILE).get("clips", {})
    keys = excluded_keys(manifest_dir)
    new = set()
    for audio, text in _entries_for_review(path, root):
        key = row_key(audio, manifest_dir)
        check = checks.get(key)
        if check and check.get("text") == text_key(text) and flag in check["flags"] and key not in keys:
            new.add(key)
    if new:
        _write_excluded(manifest_dir, keys | new)
    return len(new)


def _write_excluded(manifest_dir: Path, keys: set[str]) -> None:
    tmp = manifest_dir / f".{EXCLUDED_FILE}.tmp"
    tmp.write_text(json.dumps({"clips": sorted(keys)}, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(manifest_dir / EXCLUDED_FILE)


def training_entries(path: Path, root: Path | None = None) -> tuple[list[dict], list[str]]:
    """``read_manifest`` without the clips dropped in review."""
    entries, errors = read_manifest(path, root)
    manifest_dir = path.resolve().parent
    excluded = excluded_keys(manifest_dir)
    return [e for e in entries if row_key(e["audio"], manifest_dir) not in excluded], errors


def set_text(path: Path, index: int, text: str) -> None:
    """Replace the transcript of a manifest's ``index``-th usable entry."""
    text = text.strip()
    if not text:
        raise ValueError("A transcript can't be empty; drop the clip instead.")
    seen = -1

    def change(entry: dict) -> dict | None:
        nonlocal seen
        seen += 1
        return {**entry, "text": text} if seen == index and entry["text"] != text else None

    rewrite_manifest(path, change)
    if not 0 <= index <= seen:
        raise KeyError(f"No clip {index} in this dataset.")


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
    dropped: bool = False  # dropped in review: not used for training (excluded.json)
    trim: tuple[float, float] | None = None  # what trimming would keep (trim_of), seconds


@dataclass
class ClipPage:
    total: int  # clips matching the search / filter
    rows: list[ClipRow]
    # Over the whole dataset: {"aligner", "checked", "flagged", "flags": {flag: count}}, or None.
    checks: dict | None = None
    dropped: int = 0  # clips dropped in review, in the whole dataset


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
def _seconds(audio: str, mtime_ns: int) -> float | None:
    try:
        return sf.info(audio).duration
    except Exception:
        return None


def _clip_seconds(audio: str) -> float | None:
    """A clip's duration, cached until the file changes (trimming rewrites clips)."""
    try:
        return _seconds(audio, Path(audio).stat().st_mtime_ns)
    except OSError:
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


# A saved check -> what trimming would keep (checks.ClipCheck.trim_range); set by the caller,
# as checks.py builds on this module.
TrimOf = Callable[[dict], tuple[float, float] | None]


def read_rows(
    path: Path,
    root: Path | None = None,
    offset: int = 0,
    limit: int = 50,
    query: str = "",
    flag: str | None = None,
    worst_first: bool = False,
    trim_of: TrimOf | None = None,
) -> ClipPage:
    """One page of a manifest's clips for review.

    ``query`` matches text or file name, case-insensitively. ``flag`` keeps
    clips with that check flag, "any" with any flag (both: not dropped), or
    "dropped" the clips dropped in review. ``worst_first`` orders by alignment
    score (unchecked clips last). Dropped clips don't count in the summary.
    """
    manifest_dir = path.resolve().parent
    entries = _entries_for_review(path, root)
    asr = read_sidecar(manifest_dir / ASR_FILE)
    aligned = read_sidecar(manifest_dir / ALIGN_FILE)
    checks = aligned.get("clips", {})
    excluded = excluded_keys(manifest_dir)

    def check_of(key: str, text: str) -> dict | None:
        check = checks.get(key)
        return check if check and check.get("text") == text_key(text) else None

    q = query.strip().lower()
    matches = []
    summary = {"checked": 0, "flagged": 0, "flags": {}}
    dropped_count = 0
    for i, (audio, text) in enumerate(entries):
        key = row_key(audio, manifest_dir)
        check = check_of(key, text)
        dropped = key in excluded
        dropped_count += dropped
        if check and not dropped:
            summary["checked"] += 1
            summary["flagged"] += bool(check["flags"])
            for f in check["flags"]:
                summary["flags"][f] = summary["flags"].get(f, 0) + 1
        if q and q not in text.lower() and q not in Path(audio).name.lower():
            continue
        if flag == "dropped" and not dropped:
            continue
        if flag and flag != "dropped" and (dropped or not check or not check["flags"]):
            continue
        if flag not in (None, "any", "dropped") and flag not in check["flags"]:
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
            dropped=key in excluded,
            trim=trim_of(check) if trim_of and check else None,
        )
        for i, audio, text, key, check in matches[offset : offset + limit]
    ]
    has_checks = summary["checked"] > 0
    checks_summary = {"aligner": aligned.get("aligner"), **summary} if has_checks else None
    return ClipPage(len(matches), rows, checks_summary, dropped_count)


DETAIL_BINS = 800


@dataclass
class ClipDetail:
    row: ClipRow  # with a finer waveform (DETAIL_BINS slices)
    aligner: str | None  # that made `words`
    words: list[list]  # [text, start, end, confidence] from the check; [] when not checked
    asr: dict | None = None  # what ASR heard (checks.compare_asr), when the check compared
    check: dict | None = None  # the clip's current check as align.json holds it


def clip_detail(path: Path, root: Path | None, index: int, trim_of: TrimOf | None = None) -> ClipDetail:
    """One clip for a closer look: a finer waveform and the checked word timings."""
    entries = _entries_for_review(path, root)
    if not 0 <= index < len(entries):
        raise KeyError(f"No clip {index} in this dataset.")
    audio, text = entries[index]
    manifest_dir = path.resolve().parent
    key = row_key(audio, manifest_dir)
    aligned = read_sidecar(manifest_dir / ALIGN_FILE)
    check = aligned.get("clips", {}).get(key)
    if check and check.get("text") != text_key(text):
        check = None  # made for an earlier transcript
    row = ClipRow(
        index=index,
        file=key,
        text=text,
        seconds=_clip_seconds(audio),
        asr=key in read_sidecar(manifest_dir / ASR_FILE),
        score=check["score"] if check else None,
        flags=check["flags"] if check else [],
        speech=(check["start"], check["end"]) if check and "start" in check else None,
        peaks=clip_peaks(audio, DETAIL_BINS),
        dropped=key in excluded_keys(manifest_dir),
        trim=trim_of(check) if trim_of and check else None,
    )
    return ClipDetail(
        row,
        aligned.get("aligner") if check else None,
        check.get("words", []) if check else [],
        check.get("asr") if check else None,
        check,
    )


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
    samples: int = 0  # well-formed lines (dropped clips not counted)
    dropped: int = 0  # clips dropped in review: not in training, nor in the numbers below
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
    all_entries, line_errors = read_manifest(path, root)
    entries, _ = training_entries(path, root)
    report = DatasetReport(samples=len(entries), dropped=len(all_entries) - len(entries))
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
