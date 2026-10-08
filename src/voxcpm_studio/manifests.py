"""Training manifests: discovery, path normalization and inspection.

Manifests are JSONL files of ``{"audio": ..., "text": ...}`` (optionally
``ref_audio``). voxcpm's loader resolves relative audio paths against the
manifest's directory, but manifests written by ``lora_ft_webui`` /
``convert_dataset.py`` are relative to the VoxCPM checkout the webui ran from
(``data/<name>/wavs/...``). ``legacy_root`` is that checkout; runs get copies of
their manifests with absolute paths, so both layouts train correctly.
"""

from __future__ import annotations

import json
import random
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import soundfile as sf

MANIFEST_GLOB = "*.jsonl"
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
    return sorted(p.relative_to(data_dir).as_posix() for p in data_dir.rglob(MANIFEST_GLOB))


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
