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

Long recordings can be sliced into clips at pauses (slicer.py), and clips that
have audio but no transcript can be transcribed with ASR during the import;
ASR-made texts are listed in ``<name>/asr.json`` for later review. Finally each
clip can be checked against its text by forced alignment (checks.py), and its
silent edges trimmed while it is copied.
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

from .. import timing
from ..data.manifests import ASR_FILE, legacy_root, write_manifest
from . import checks, fillers, loudness, slicer
from .checks import Checker, ClipCheck
from .slicer import SliceOptions, Vad

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a"}
TRANSCRIPT_EXTENSIONS = (".lab", ".txt")
# Unicode-aware: voice names are often Chinese or Japanese.
DATASET_NAME = r"^[^\W_][\w.-]{0,79}$"
EXAMPLES = 5
MISSING_EXAMPLES = 2  # sliced file names are long
TRANSCRIBE_CHUNK = 32  # clips per ASR call; progress and cancel granularity
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
        row.seconds = seconds
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


def _transcribe(rows: list[Row], transcriber: Transcriber, progress: Progress) -> None:
    """Fill in missing texts, marking them as made by ASR."""
    todo = [i for i, row in enumerate(rows) if not row.text]
    for start in range(0, len(todo), TRANSCRIBE_CHUNK):
        progress("transcribing", start, len(todo))
        chunk = todo[start : start + TRANSCRIBE_CHUNK]
        for i, (text, language) in zip(chunk, transcriber([rows[i].audio for i in chunk]), strict=True):
            rows[i].text = text.strip()
            rows[i].asr, rows[i].language = bool(rows[i].text), language
    progress("transcribing", len(todo), len(todo))


def _slice(
    rows: list[Row],
    out_dir: Path,
    opts: SliceOptions,
    vad: Vad,
    guide: slicer.Guide | None,
    progress: Progress,
    chain=None,
) -> tuple[list[Row], int]:
    """Replace recordings longer than LONG_RECORDING_S by their clips. With a
    ``guide`` the clips are cut between words and keep their ASR text; without
    one they have no text. (A recording's own transcript is not kept: it is not
    known which part of it each chunk holds.) Progress counts seconds of audio.
    Returns the new rows and the number of recordings sliced. With a ``chain``
    the voice is separated window by window too (phase "separating_slicing")."""
    phase = "separating_slicing" if chain is not None else "slicing"
    total = int(sum(row.seconds for row in rows if row.seconds > LONG_RECORDING_S))
    out: list[Row] = []
    done, seconds = 0, 0.0
    for row in rows:
        if row.seconds <= LONG_RECORDING_S:
            out.append(row)
            continue
        progress(phase, int(seconds), total)
        base = seconds
        clips = slicer.slice_recording(
            row.audio,
            out_dir / f"{done:05d}",
            opts,
            vad,
            guide,
            progress=lambda at, base=base: progress(phase, int(base + at), total),
            chain=chain,
        )
        for clip in clips:
            out.append(
                Row(
                    ref=str(clip.path),
                    text=clip.text,
                    audio=clip.path,
                    seconds=sf.info(str(clip.path)).duration,
                    asr=bool(clip.text),
                    language=clip.language,
                )
            )
        done += 1
        seconds += row.seconds
    progress(phase, total, total)
    return out, done


def _separate(rows: list[Row], out_dir: Path, chain, progress: Progress, sliced: bool) -> None:
    """Replace each clip's audio by its separated voice, written to ``out_dir``
    as PCM16 WAV at the models' rate (same file name). Recordings that will be
    ``sliced`` are separated window by window while slicing instead."""
    todo = [row for row in rows if not (sliced and row.seconds > LONG_RECORDING_S)]
    for n, row in enumerate(todo):
        progress("separating", n, len(todo))
        src = slicer.Source(row.audio)
        voice = chain.process(src.read_channels(0.0, src.duration, chain.rate, chain.channels)).mean(axis=0)
        target = out_dir / f"{n:06d}" / f"{row.audio.stem}.wav"
        target.parent.mkdir(parents=True)
        sf.write(target, voice, chain.rate, subtype="PCM_16")
        row.audio, row.ref = target, str(target)
    progress("separating", len(todo), len(todo))


def _check(rows: list[Row], checker: Checker, progress: Progress) -> list[ClipCheck]:
    """Alignment checks of rows (all with text), in order."""
    from .asr import load_audio

    out: list[ClipCheck] = []
    for start in range(0, len(rows), checks.CHECK_CHUNK):
        progress("aligning", start, len(rows))
        chunk = rows[start : start + checks.CHECK_CHUNK]
        texts, made = [row.text for row in chunk], [row.asr for row in chunk]
        out += checker([load_audio(row.audio) for row in chunk], texts, [not m for m in made], made)
    progress("aligning", len(rows), len(rows))
    return out


def _write_clip(source: Path, target: Path, cut: tuple[float, float] | None, normalize: float | None) -> None:
    """``source`` (mono) as PCM16 WAV: cut to [start, end) seconds, at ``normalize`` LUFS."""
    src = slicer.Source(source)
    audio = src.read(*(cut or (0.0, src.duration)))
    if normalize is not None:
        audio = loudness.normalize(audio, src.rate, normalize)
    sf.write(target, audio, src.rate, subtype="PCM_16")


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
) -> ImportResult:
    """Write ``data_dir/<name>/train.jsonl``.

    With ``copy``, audio goes to ``<name>/wavs/`` and the manifest uses relative
    paths, so the folder is self-contained; otherwise it references the files.
    With ``slicing``, recordings longer than LONG_RECORDING_S are cut into clips
    at pauses (needs ``copy`` and a ``transcriber``: the clips are new files
    without text). With a ``transcriber``, clips without a transcript get one
    from ASR; otherwise they are left out. With a ``checker`` (made for the
    ``aligner`` named), every clip is checked against its text and the results
    saved as align.json; ``trim`` then cuts long silent edges from the copies
    (needs ``copy``; trimmed clips are written as PCM16 WAV). ``normalize``
    brings each copied clip to that many LUFS (loudness.normalize; mono PCM16
    WAV). ``progress`` may raise to cancel; nothing is left behind then.
    """
    progress = progress or (lambda phase, done, total: None)
    if slicing and not (copy and transcriber):
        raise ValueError("Slicing writes new clips without text: it needs copying and transcription.")
    if separation is not None and not copy:
        raise ValueError("Separation writes new audio: it needs copying.")
    if trim and not (copy and checker):
        raise ValueError("Trimming rewrites the copied clips at their checked edges: it needs copying and checking.")
    if normalize is not None and not copy:
        raise ValueError("Normalizing loudness rewrites the clips: it needs copying.")
    check_name(name, data_dir)
    data_dir = data_dir.resolve()
    target = data_dir / name

    progress("scanning", 0, 0)
    with timing.stage("scanning"):
        preview, rows = _scan(clean_path(source), clean_path(audio_dir) if audio_dir else None, data_dir)

    # Build next to the target and rename, so a failure leaves nothing half-written.
    staging = data_dir / f".import-{name}"
    shutil.rmtree(staging, ignore_errors=True)
    try:
        sliced = 0
        if separation is not None:
            _separate(rows, staging / "_separated", separation, progress, sliced=slicing is not None)
        if slicing:
            rows, sliced = _slice(rows, staging / "_slices", slicing, vad, guide, progress, separation)
        if transcriber:
            _transcribe(rows, transcriber, progress)
        if tidy_fillers:
            for row in rows:
                row.text = fillers.tidy(row.text)
        rows_kept = [(i, row) for i, row in enumerate(rows) if row.text]
        if not rows_kept:
            raise ValueError("No clips with both audio and a transcript; nothing to import.")
        with timing.stage("checking"):  # alignment, VAD, ... count as themselves
            clip_checks = _check([row for _, row in rows_kept], checker, progress) if checker else []
        # (start, end) to keep per clip, for those worth trimming.
        cuts = [c.trim_range() for c in clip_checks] if trim else [None] * len(rows_kept)

        if copy:
            (staging / "wavs").mkdir(parents=True, exist_ok=True)
            # Rewritten clips (trimmed, normalized) are WAV, whatever the source format.
            names = [
                r.audio.with_suffix(".wav") if cut or normalize is not None else r.audio
                for (_, r), cut in zip(rows_kept, cuts, strict=True)
            ]
            files = _unique_names(names)
            for n, ((_, row), file, cut) in enumerate(zip(rows_kept, files, cuts, strict=True)):
                if n % 50 == 0:
                    progress("copying", n, len(rows_kept))
                with timing.stage("writing clips"):
                    if cut or normalize is not None:
                        _write_clip(row.audio, staging / "wavs" / file, cut, normalize)
                        if cut:
                            clip_checks[n] = clip_checks[n].trimmed(*cut)
                    elif row.audio.is_relative_to(staging):
                        row.audio.replace(staging / "wavs" / file)  # a fresh slice: move, don't copy
                    else:
                        shutil.copy2(row.audio, staging / "wavs" / file)
            shutil.rmtree(staging / "_slices", ignore_errors=True)
            shutil.rmtree(staging / "_separated", ignore_errors=True)
            paths = [f"wavs/{file}" for file in files]
        else:
            paths = [str(row.audio) for _, row in rows_kept]
        pairs = list(zip(paths, rows_kept, strict=True))
        write_manifest([{"audio": p, "text": row.text} for p, (_, row) in pairs], staging / "train.jsonl")
        asr_rows = {p: {"language": row.language} for p, (_, row) in pairs if row.asr}
        if asr_rows:
            (staging / ASR_FILE).write_text(json.dumps(asr_rows, ensure_ascii=False, indent=1), encoding="utf-8")
        if clip_checks:
            checks.write_checks(
                staging,
                aligner,
                {p: (row.text, c) for (p, (_, row)), c in zip(pairs, clip_checks, strict=True)},
            )
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return ImportResult(
        id=f"{name}/train.jsonl",
        imported=len(rows_kept),
        transcribed=len(asr_rows),
        sliced=sliced,
        preview=preview,
        trimmed=sum(cut is not None for cut in cuts),
        checks=checks.summarize(clip_checks) if clip_checks else None,
    )
