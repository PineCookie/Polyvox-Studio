"""Import datasets from other tools into ``data/<name>/train.jsonl``.

Sources (detected from the path):

- GPT-SoVITS ``.list``: ``path|speaker|lang|text``
- LJSpeech-style ``.csv`` / ``.txt``: ``file_or_id|text[|...]``
- ``.jsonl`` manifests: ``{"audio": ..., "text": ...}``
- a folder of audio files with same-name ``.txt`` / ``.lab`` transcripts

Audio paths in lists are often stale (another machine, a deleted folder), so each
one is looked up as written, then relative to the list (and its ``wavs/``
folder), then by file name inside an optional audio folder. Rows whose audio
cannot be found are skipped: lists shared by other people often come without it.

Clips that have audio but no transcript can be transcribed with ASR during the
import; those are listed in ``<name>/asr.json`` for later review.
"""

from __future__ import annotations

import json
import re
import shutil
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
from typing import Callable

import soundfile as sf

from .manifests import legacy_root, write_manifest

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a"}
TRANSCRIPT_EXTENSIONS = (".lab", ".txt")
# Unicode-aware: voice names are often Chinese or Japanese.
DATASET_NAME = r"^[^\W_][\w.-]{0,79}$"
ASR_FILE = "asr.json"
EXAMPLES = 5
MISSING_EXAMPLES = 2  # sliced file names are long
TRANSCRIBE_CHUNK = 32  # clips per ASR call; progress and cancel granularity

# Audio files -> (text, detected language) per file.
Transcriber = Callable[[list[Path]], list[tuple[str, str | None]]]
# (phase, done, total); may raise to cancel.
Progress = Callable[[str, int, int], None]


@dataclass
class Row:
    ref: str  # audio path or file name as written in the source
    text: str  # empty when the source has no transcript for it
    audio: Path | None = None  # resolved file


@dataclass
class ImportPreview:
    kind: str  # gsv | csv | jsonl | folder
    source: str
    rows: int = 0
    found: int = 0  # audio and transcript
    untranscribed: int = 0  # audio but no transcript (ASR can fill it in)
    missing: int = 0  # no audio
    duplicates: int = 0
    unreadable: int = 0
    total_seconds: float = 0.0  # of found + untranscribed clips
    median_seconds: float | None = None
    sample_rates: dict[int, int] = field(default_factory=dict)
    examples: list[dict] = field(default_factory=list)  # {"text", "audio"}
    missing_examples: list[str] = field(default_factory=list)
    suggested_name: str = ""


@dataclass
class ImportResult:
    id: str
    imported: int
    transcribed: int  # of `imported`, texts made by ASR
    preview: ImportPreview


def clean_path(value: str) -> Path:
    """A path as pasted by a user: Explorer's "Copy as path" adds quotes."""
    return Path(value.strip().strip('"').strip("'")).expanduser()


def detect(source: Path) -> str:
    if source.is_dir():
        return "folder"
    if not source.is_file():
        raise FileNotFoundError(f"Not found: {source}")
    kind = {".list": "gsv", ".csv": "csv", ".txt": "csv", ".jsonl": "jsonl"}.get(source.suffix.lower())
    if kind is None:
        raise ValueError(f"Unsupported source: {source.name} (use .list, .csv, .txt, .jsonl or a folder)")
    return kind


def _read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def parse(source: Path, kind: str) -> list[Row]:
    rows = []
    if kind == "gsv":
        for line in _read_lines(source):
            parts = line.split("|", 3)
            if len(parts) == 4:
                rows.append(Row(parts[0].strip(), parts[3].strip()))
    elif kind == "csv":
        for line in _read_lines(source):
            parts = line.split("|")
            if len(parts) >= 2:
                rows.append(Row(parts[0].strip(), parts[1].strip()))
    elif kind == "jsonl":
        for line in _read_lines(source):
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(entry, dict) and isinstance(entry.get("audio"), str):
                rows.append(Row(entry["audio"], str(entry.get("text") or "").strip()))
    else:
        for audio in sorted(p for p in source.rglob("*") if p.suffix.lower() in AUDIO_EXTENSIONS):
            transcript = next((audio.with_suffix(e) for e in TRANSCRIPT_EXTENSIONS if audio.with_suffix(e).is_file()), None)
            text = transcript.read_text(encoding="utf-8-sig").strip() if transcript else ""
            rows.append(Row(str(audio), text, audio))
    return rows


def _file_name(ref: str) -> str:
    # Handles "C:\a\b.wav" and "/root/a/b.wav" on any OS.
    return PureWindowsPath(ref).name


def _index(folder: Path | None) -> dict[str, Path]:
    """File name (and stem, for id-only lists) -> path, for audio under ``folder``."""
    index: dict[str, Path] = {}
    if folder is None:
        return index
    for path in folder.rglob("*"):
        if path.suffix.lower() in AUDIO_EXTENSIONS:
            index.setdefault(path.name.lower(), path)
            index.setdefault(path.stem.lower(), path)
    return index


def resolve_audio(rows: list[Row], source: Path, audio_dir: Path | None, root: Path | None) -> None:
    base = source if source.is_dir() else source.parent
    index = _index(audio_dir)
    for row in rows:
        if row.audio is not None:
            continue
        ref, name = Path(row.ref), _file_name(row.ref)
        candidates = [ref] if ref.is_absolute() else [base / ref] + ([root / ref] if root else [])
        # By file name next to the list, as GPT-SoVITS datasets are usually kept.
        candidates += [base / name, base / "wavs" / name]
        found = next((c for c in candidates if c.is_file()), None)
        if found is None and index:
            name = _file_name(row.ref).lower()
            found = index.get(name) or index.get(Path(name).stem)
        row.audio = found.resolve() if found else None


def _suggest_name(source: Path) -> str:
    stem = source.name if source.is_dir() else source.stem
    if source.name.lower() in ("train.jsonl", "metadata.csv"):
        stem = source.parent.name
    stem = re.sub(r"\(\d+\)$", "", stem).strip()  # "speaker(1)" -> "speaker"
    return re.sub(r"[^\w.\-]+", "_", stem).strip("._-") or "dataset"


def _scan(source: Path, audio_dir: Path | None, data_dir: Path) -> tuple[ImportPreview, list[Row]]:
    """Summary, and the importable rows in source order (with or without text)."""
    kind = detect(source)
    rows = parse(source, kind)
    resolve_audio(rows, source, audio_dir, legacy_root(data_dir))

    preview = ImportPreview(kind=kind, source=str(source), rows=len(rows), suggested_name=_suggest_name(source))
    keep, seen, durations, rates = [], set(), [], Counter()
    for row in rows:
        if row.audio is None:
            preview.missing += 1
            if len(preview.missing_examples) < MISSING_EXAMPLES:
                preview.missing_examples.append(row.ref)
            continue
        if row.audio in seen:
            preview.duplicates += 1
            continue
        try:
            info = sf.info(str(row.audio))
        except Exception:
            preview.unreadable += 1
            continue
        seen.add(row.audio)
        keep.append(row)
        durations.append(info.duration)
        rates[info.samplerate] += 1
        if row.text:
            preview.found += 1
            if len(preview.examples) < EXAMPLES:
                preview.examples.append({"text": row.text, "audio": str(row.audio)})
        else:
            preview.untranscribed += 1

    preview.total_seconds = sum(durations)
    preview.median_seconds = statistics.median(durations) if durations else None
    preview.sample_rates = dict(sorted(rates.items()))
    return preview, keep


def preview_import(source: str, audio_dir: str | None, data_dir: Path) -> ImportPreview:
    return _scan(clean_path(source), clean_path(audio_dir) if audio_dir else None, data_dir)[0]


def _unique_names(paths: list[Path]) -> list[str]:
    """Target file names in ``wavs/``; clashing names get a numeric suffix."""
    used: set[str] = set()
    names = []
    for path in paths:
        name, i = path.name, 2
        while name.lower() in used:
            name = f"{path.stem}_{i}{path.suffix}"
            i += 1
        used.add(name.lower())
        names.append(name)
    return names


def check_name(name: str, data_dir: Path) -> None:
    if not re.fullmatch(DATASET_NAME, name):
        raise ValueError("Dataset name may contain letters (any language), digits, '.', '_' and '-' only.")
    if (data_dir.resolve() / name).exists():
        raise FileExistsError(f"A dataset named '{name}' already exists.")


def _transcribe(rows: list[Row], transcriber: Transcriber, progress: Progress) -> dict[int, str | None]:
    """Fill in missing texts; returns {row index: detected language} of transcribed rows."""
    todo = [i for i, row in enumerate(rows) if not row.text]
    languages: dict[int, str | None] = {}
    for start in range(0, len(todo), TRANSCRIBE_CHUNK):
        progress("transcribing", start, len(todo))
        chunk = todo[start : start + TRANSCRIBE_CHUNK]
        for i, (text, language) in zip(chunk, transcriber([rows[i].audio for i in chunk])):
            rows[i].text = text.strip()
            if rows[i].text:
                languages[i] = language
    progress("transcribing", len(todo), len(todo))
    return languages


def import_dataset(
    source: str,
    audio_dir: str | None,
    name: str,
    data_dir: Path,
    copy: bool = True,
    transcriber: Transcriber | None = None,
    progress: Progress | None = None,
) -> ImportResult:
    """Write ``data_dir/<name>/train.jsonl``.

    With ``copy``, audio goes to ``<name>/wavs/`` and the manifest uses relative
    paths, so the folder is self-contained; otherwise it references the files.
    With a ``transcriber``, clips without a transcript get one from ASR;
    otherwise they are left out. ``progress`` may raise to cancel; nothing is
    left behind then.
    """
    progress = progress or (lambda phase, done, total: None)
    check_name(name, data_dir)
    data_dir = data_dir.resolve()
    target = data_dir / name

    progress("scanning", 0, 0)
    preview, rows = _scan(clean_path(source), clean_path(audio_dir) if audio_dir else None, data_dir)
    languages = _transcribe(rows, transcriber, progress) if transcriber else {}
    rows_kept = [(i, row) for i, row in enumerate(rows) if row.text]
    if not rows_kept:
        raise ValueError("No clips with both audio and a transcript; nothing to import.")

    # Build next to the target and rename, so a failure leaves nothing half-written.
    staging = data_dir / f".import-{name}"
    shutil.rmtree(staging, ignore_errors=True)
    try:
        if copy:
            (staging / "wavs").mkdir(parents=True)
            files = _unique_names([row.audio for _, row in rows_kept])
            for n, ((_, row), file) in enumerate(zip(rows_kept, files)):
                if n % 50 == 0:
                    progress("copying", n, len(rows_kept))
                shutil.copy2(row.audio, staging / "wavs" / file)
            paths = [f"wavs/{file}" for file in files]
        else:
            paths = [str(row.audio) for _, row in rows_kept]
        write_manifest([{"audio": p, "text": row.text} for p, (_, row) in zip(paths, rows_kept)], staging / "train.jsonl")
        asr_rows = {p: {"language": languages[i]} for p, (i, _) in zip(paths, rows_kept) if i in languages}
        if asr_rows:
            (staging / ASR_FILE).write_text(json.dumps(asr_rows, ensure_ascii=False, indent=1), encoding="utf-8")
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return ImportResult(id=f"{name}/train.jsonl", imported=len(rows_kept), transcribed=len(asr_rows), preview=preview)
