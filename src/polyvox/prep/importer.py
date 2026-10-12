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

``import_dataset`` adds a source to a new dataset and runs the processing
steps asked for (steps.py: clean-up, slicing, transcription, checks, tidying).
"""

from __future__ import annotations

import json
import re
import shutil
import statistics
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath

import soundfile as sf

from ..data import record
from ..data.manifests import ASR_FILE, legacy_root, read_sidecar, training_entries
from . import checks, fillers, slicer
from .checks import Checker
from .slicer import SliceOptions, Vad

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a"}
TRANSCRIPT_EXTENSIONS = (".lab", ".txt")
# Unicode-aware: voice names are often Chinese or Japanese.
DATASET_NAME = r"^[^\W_][\w.-]{0,79}$"
EXAMPLES = 5
MISSING_EXAMPLES = 2  # sliced file names are long
LONG_RECORDING_S = 20.0  # longer audio counts as a recording to slice, not a clip

# Audio files -> (text, detected language) per file.
Transcriber = Callable[[list[Path]], list[tuple[str, str | None]]]
# (phase, done, total); may raise to cancel.
Progress = Callable[[str, int, int], None]


@dataclass
class Row:
    ref: str  # audio path or file name as written in the source
    text: str  # empty when the source has no transcript for it
    audio: Path | None = None  # resolved file
    seconds: float = 0.0
    rate: int | None = None  # sample rate of the file
    asr: bool = False  # text made by speech recognition
    language: str | None = None  # as ASR detected (or was told)


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
    long_recordings: int = 0  # longer than a training clip should be: slice them
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
    sliced: int  # long recordings cut into clips
    preview: ImportPreview
    trimmed: int = 0  # clips whose silent edges were cut
    checks: dict | None = None  # checks.summarize() of the alignment checks


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
            transcript = next(
                (audio.with_suffix(e) for e in TRANSCRIPT_EXTENSIONS if audio.with_suffix(e).is_file()), None
            )
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


def _probe(path: Path) -> tuple[float, int]:
    """(seconds, sample rate); soundfile's header read is fast, other formats
    (m4a, mp3, ...) go through the decoder."""
    try:
        info = sf.info(str(path))
        return info.duration, info.samplerate
    except Exception:
        return slicer.probe(path)


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
            seconds, rate = _probe(row.audio)
        except Exception:
            preview.unreadable += 1
            continue
        seen.add(row.audio)
        row.seconds, row.rate = seconds, rate
        keep.append(row)
        durations.append(seconds)
        rates[rate] += 1
        preview.long_recordings += seconds > LONG_RECORDING_S
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


def scan_source(source: str, audio_dir: str | None, data_dir: Path) -> tuple[ImportPreview, list[Row]]:
    return _scan(clean_path(source), clean_path(audio_dir) if audio_dir else None, data_dir)


def preview_import(source: str, audio_dir: str | None, data_dir: Path) -> ImportPreview:
    return scan_source(source, audio_dir, data_dir)[0]


def check_name(name: str, data_dir: Path) -> None:
    if not re.fullmatch(DATASET_NAME, name):
        raise ValueError("Dataset name may contain letters (any language), digits, '.', '_' and '-' only.")
    if (data_dir.resolve() / name).exists():
        raise FileExistsError(f"A dataset named '{name}' already exists.")


def import_dataset(
    source: str,
    audio_dir: str | None,
    name: str,
    data_dir: Path,
    copy: bool = True,
    transcriber: Transcriber | None = None,
    progress: Progress | None = None,
    slicing: SliceOptions | None = None,
    vad: Vad = slicer.silero_vad,
    checker: Checker | None = None,
    aligner: str = "qwen3",
    trim: bool = False,
    guide: slicer.Guide | None = None,
    separation=None,
    tidy_fillers: bool = False,
    normalize: float | None = None,
    embed: checks.Embed | None = None,
    steps: dict[str, dict] | None = None,
    cleanup_models: list[str] | None = None,
) -> ImportResult:
    """Make dataset ``name`` from a source and run the steps asked for, in order
    (steps.py): add the source (``copy``: into ``<name>/wavs/``), clean up with
    the ``separation`` chain, slice recordings longer than LONG_RECORDING_S
    (``slicing``; without it they are imported as clips), transcribe clips
    without text, tidy fillers, check (``checker`` for ``aligner``; ``embed``
    compares voices), trim (``trim``) and even out loudness (``normalize``
    LUFS). ``steps``: the settings to note in the dataset's record. Each step
    saves as it goes: a failure or cancel keeps what was done (only a dataset
    with nothing in it is removed).
    """
    from . import steps as st
    from .asr import load_audio

    progress = progress or (lambda phase, done, total: None)
    if separation is not None and not copy:
        raise ValueError("Separation writes new audio: it needs copying.")
    if trim and not (copy and checker):
        raise ValueError("Trimming rewrites the copied clips at their checked edges: it needs copying and checking.")
    if normalize is not None and not copy:
        raise ValueError("Normalizing loudness rewrites the clips: it needs copying.")
    folder = st.create(data_dir, name)
    try:
        added = st.add_source(folder, source, audio_dir, data_dir, copy, slicing is None, progress)
        if not added.clips and not added.recordings:
            raise ValueError("No clips with audio found; nothing to import.")
    except BaseException:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    for step, settings in (steps or {}).items():
        record.remember(folder, step, settings)
    if separation is not None:
        st.clean_up(folder, separation, cleanup_models or [], progress)
    sliced = st.slice_recordings(folder, slicing, vad, guide, separation, progress)["sliced"] if slicing else 0
    if transcriber:
        st.transcribe(folder, transcriber, tidy_fillers, progress, (steps or {}).get("transcribe"))
    elif tidy_fillers:
        fillers.tidy_manifest(st.manifest_of(folder))
    if checker:
        st.check(folder, checker, aligner, load_audio, embed, progress, (steps or {}).get("check"))
    tidied = st.tidy(folder, trim, normalize, progress) if trim or normalize is not None else {}
    manifest = st.manifest_of(folder)
    summary = st.check_summary(folder) if checker else None
    return ImportResult(
        id=f"{name}/{st.MANIFEST}",
        imported=len(training_entries(manifest)[0]),
        transcribed=len(read_sidecar(folder / ASR_FILE)),
        sliced=sliced,
        preview=added.preview,
        trimmed=tidied.get("trimmed", 0),
        checks=summary,
    )
