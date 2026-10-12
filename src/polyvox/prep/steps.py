"""Processing steps of a dataset, each run in place on its folder.

A dataset is built in steps, in this order; each works on what still needs it
(``status`` counts that), saves as it goes and can run again later (after
another source was added, or a run stopped halfway):

1. ``add_source``: clips from a list or folder are copied in (``wavs/``,
   texts as the source has them, possibly none); long recordings are listed in
   the record (record.py), to be sliced.
2. ``clean_up`` (optional): separation / de-reverb / denoise of the clips in
   place; recordings are cleaned while they are sliced.
3. ``slice``: unsliced recordings are cut into clips between words (with the
   ASR text of each clip).
4. ``transcribe``: clips without text get ASR text; tidying fillers goes with it.
5. ``check``: clips without a current check are aligned with their text.
6. ``tidy``: checked clips are trimmed at their edges; loudness is evened out.

Rewritten clips keep their first version in ``originals/`` (checks.ORIGINALS_DIR).
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from .. import timing
from ..data import manifests, record
from ..data.manifests import ASR_FILE, clip_entries, read_sidecar, row_key
from . import checks, fillers, importer, loudness, slicer
from .checks import ORIGINALS_DIR, TRIM_IN_PLACE_FORMATS, TRIM_REVIEW_FIRST, Checker, ClipCheck
from .importer import LONG_RECORDING_S, ImportPreview, Progress, Transcriber
from .slicer import SliceOptions, Vad

MANIFEST = "train.jsonl"
WAVS = "wavs"
TRANSCRIBE_CHUNK = 32  # clips per ASR call: progress, cancel and save granularity
STEPS = ("sources", "clean_up", "slice", "transcribe", "check", "tidy")
# Each step's settings until it has run on a dataset (then its record has them).
DEFAULTS: dict[str, dict] = {
    "clean_up": {"models": []},
    "slice": {"min_seconds": 2.0, "max_seconds": 15.0},
    "transcribe": {"language": "auto", "prompt": None, "tidy_fillers": True},
    "check": {"aligner": "qwen3", "compare_asr": False, "detect_singing": False},
    "tidy": {"trim": True, "normalize_lufs": None},
}


def _quiet(phase: str, done: int, total: int) -> None:
    pass


def manifest_of(folder: Path) -> Path:
    return folder / MANIFEST


def create(data_dir: Path, name: str) -> Path:
    """A new, empty dataset folder."""
    importer.check_name(name, data_dir)
    folder = data_dir.resolve() / name
    folder.mkdir(parents=True)
    manifest_of(folder).write_text("", encoding="utf-8")
    record.write_record(folder, record.Record())
    return folder


@dataclass
class Added:
    clips: int  # added to the manifest (with or without text)
    recordings: int  # listed to be sliced
    preview: ImportPreview


def _wav_name(path: Path, used: set[str]) -> str:
    """A free name in wavs/; formats the in-place steps can't write back become WAV."""
    stem, suffix = path.stem, path.suffix if path.suffix.lower() in TRIM_IN_PLACE_FORMATS else ".wav"
    name, i = f"{stem}{suffix}", 2
    while name.lower() in used:
        name, i = f"{stem}_{i}{suffix}", i + 1
    used.add(name.lower())
    return name


def add_source(
    folder: Path,
    source: str,
    audio_dir: str | None,
    data_dir: Path,
    copy: bool = True,
    long_as_clips: bool = False,
    progress: Progress = _quiet,
) -> Added:
    """Add a source's clips and recordings to a dataset (audio already in it
    is skipped). With ``copy``, clips go to ``wavs/`` (as WAV unless WAV /
    FLAC / OGG), else the manifest points at them. Audio longer than
    LONG_RECORDING_S is a recording to slice (never copied: slicing reads it
    where it is), unless ``long_as_clips``."""
    folder = folder.resolve()
    progress("scanning", 0, 0)
    with timing.stage("scanning"):
        preview, rows = importer.scan_source(source, audio_dir, data_dir)
    manifest = manifest_of(folder)
    rec = record.read_record(folder)
    have = {e["audio"] for e in clip_entries(manifest)} | {r.audio for r in rec.recordings}
    have |= {o.recording for o in rec.origins.values()}  # source files copied in before
    rows = [r for r in rows if str(r.audio) not in have]
    long = [r for r in rows if r.seconds > LONG_RECORDING_S and not long_as_clips]
    clips = [r for r in rows if r not in long]

    entries = []
    if copy:
        wavs = folder / WAVS
        wavs.mkdir(exist_ok=True)
        used = {p.name.lower() for p in wavs.iterdir()}
        for n, row in enumerate(clips):
            if n % 50 == 0:
                progress("copying", n, len(clips))
            name = _wav_name(row.audio, used)
            with timing.stage("writing clips"):
                if Path(name).suffix == row.audio.suffix:
                    shutil.copy2(row.audio, wavs / name)
                else:
                    src = slicer.Source(row.audio)
                    sf.write(wavs / name, src.read(0.0, src.duration), src.rate, subtype="PCM_16")
            entries.append({"audio": f"{WAVS}/{name}", "text": row.text})
            rec.origins[f"{WAVS}/{name}"] = record.Origin(str(row.audio), 0.0, round(row.seconds, 3))
        progress("copying", len(clips), len(clips))
    else:
        entries = [{"audio": str(row.audio), "text": row.text} for row in clips]
    manifests.append_entries(manifest, entries)
    for row in long:
        rec.recordings.append(record.Recording(str(row.audio), round(row.seconds, 3), row.rate, row.text))
    record.write_record(folder, rec)
    return Added(len(entries), len(long), preview)


def _writable(audio: Path, folder: Path) -> bool:
    return audio.is_relative_to(folder) and audio.suffix.lower() in TRIM_IN_PLACE_FORMATS


def clean_up(folder: Path, chain, models: list[str], progress: Progress = _quiet) -> dict:
    """Run a separation chain (separation.Chain) over the clips not cleaned yet,
    in place: the voice, mono, at the models' rate (originals kept). Clips
    outside the folder or in other formats are skipped. Recordings get the
    same chain when they are sliced (``slice``: steps["clean_up"])."""
    folder = folder.resolve()
    manifest = manifest_of(folder)
    rec = record.read_record(folder)
    todo = [Path(e["audio"]) for e in clip_entries(manifest)]
    todo = [a for a in todo if row_key(str(a), folder) not in rec.cleaned]
    cleaned = skipped = 0
    for n, audio in enumerate(todo):
        progress("separating", n, len(todo))
        if not _writable(audio, folder):
            skipped += 1
            continue
        src = slicer.Source(audio)
        voice = chain.process(src.read_channels(0.0, src.duration, chain.rate, chain.channels)).mean(axis=0)
        key = row_key(str(audio), folder)
        backup = folder / ORIGINALS_DIR / key
        if not backup.exists():
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(audio, backup)
        tmp = audio.with_name(f".{audio.name}.tmp")
        sf.write(tmp, voice, chain.rate, format=sf.info(str(audio)).format)
        tmp.replace(audio)
        rec.cleaned[key] = list(models)
        cleaned += 1
        if cleaned % 20 == 0:  # saved as it goes
            record.write_record(folder, rec)
    progress("separating", len(todo), len(todo))
    rec.steps["clean_up"] = {"models": list(models)}
    record.write_record(folder, rec)
    return {"cleaned": cleaned, "skipped": skipped}


def _add_asr(folder: Path, languages: dict[str, str | None]) -> None:
    """Mark texts as made by ASR (asr.json), with the language heard."""
    if not languages:
        return
    made = dict(read_sidecar(folder / ASR_FILE))
    made.update({key: {"language": language} for key, language in languages.items()})
    tmp = folder / f".{ASR_FILE}.tmp"
    tmp.write_text(json.dumps(made, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(folder / ASR_FILE)


def slice_recordings(
    folder: Path,
    opts: SliceOptions,
    vad: Vad = slicer.silero_vad,
    guide: slicer.Guide | None = None,
    chain=None,
    progress: Progress = _quiet,
) -> dict:
    """Cut the recordings not sliced yet into clips in ``wavs/`` (between words
    with a ``guide``, which also gives each clip its text; with a separation
    ``chain``, from the cleaned voice). Each recording is saved when done.
    Progress counts seconds of audio. Returns {"sliced", "clips", "missing"}."""
    folder = folder.resolve()
    rec = record.read_record(folder)
    todo = [r for r in rec.recordings if r.sliced is None]
    total = int(sum(r.seconds for r in todo))
    settings = {"min_seconds": opts.min_seconds, "max_seconds": opts.max_seconds}
    phase = "separating_slicing" if chain is not None else "slicing"
    work = folder / ".slicing"
    wavs = folder / WAVS
    wavs.mkdir(exist_ok=True)
    sliced = added = missing = 0
    seconds = 0.0
    for recording in todo:
        path = Path(recording.audio)
        if not path.is_file():
            missing += 1
            continue
        base = seconds
        progress(phase, int(seconds), total)
        shutil.rmtree(work, ignore_errors=True)
        clips = slicer.slice_recording(
            path, work, opts, vad, guide, lambda at, base=base: progress(phase, int(base + at), total), chain
        )
        used = {p.name.lower() for p in wavs.iterdir()}
        entries, languages = [], {}
        rec = record.read_record(folder)  # the steps may run while nothing else writes; re-read to be safe
        for clip in clips:
            name = _wav_name(clip.path, used)
            clip.path.replace(wavs / name)
            key = f"{WAVS}/{name}"
            entries.append({"audio": key, "text": clip.text})
            rec.origins[key] = record.Origin(recording.audio, clip.start, clip.end)
            if clip.text:
                languages[key] = clip.language
        manifests.append_entries(manifest_of(folder), entries)
        _add_asr(folder, languages)
        for r in rec.recordings:
            if r.audio == recording.audio:
                r.sliced = settings
        rec.steps["slice"] = settings
        record.write_record(folder, rec)
        sliced += 1
        added += len(entries)
        seconds += recording.seconds
    shutil.rmtree(work, ignore_errors=True)
    progress(phase, total, total)
    return {"sliced": sliced, "clips": added, "missing": missing}


def transcribe(
    folder: Path,
    transcriber: Transcriber,
    tidy_fillers: bool = False,
    progress: Progress = _quiet,
    settings: dict | None = None,
) -> dict:
    """Give the clips without text (not dropped) ASR text, saved chunk by
    chunk; clips ASR hears nothing in are dropped (review can bring them
    back). With ``tidy_fillers``, every text's runs of fillers are tidied
    after. Returns {"transcribed", "dropped", "tidied"}."""
    folder = folder.resolve()
    manifest = manifest_of(folder)
    excluded = manifests.excluded_keys(folder)
    todo = [Path(e["audio"]) for e in clip_entries(manifest) if not e["text"]]
    todo = [a for a in todo if row_key(str(a), folder) not in excluded]
    done = empty = 0
    for start in range(0, len(todo), TRANSCRIBE_CHUNK):
        progress("transcribing", start, len(todo))
        chunk = todo[start : start + TRANSCRIBE_CHUNK]
        texts, languages, nothing = {}, {}, set()
        for audio, (text, language) in zip(chunk, transcriber(chunk), strict=True):
            key = row_key(str(audio), folder)
            if text.strip():
                texts[key], languages[key] = text.strip(), language
            else:
                nothing.add(key)
        manifests.set_texts(manifest, texts)
        _add_asr(folder, languages)
        manifests.drop_keys(folder, nothing)
        done += len(texts)
        empty += len(nothing)
    progress("transcribing", len(todo), len(todo))
    tidied = fillers.tidy_manifest(manifest) if tidy_fillers else 0
    record.remember(folder, "transcribe", {**(settings or {}), "tidy_fillers": tidy_fillers})
    return {"transcribed": done, "dropped": empty, "tidied": tidied}


def check(
    folder: Path,
    checker: Checker,
    kind: str,
    load: Callable[[Path], np.ndarray],
    embed: checks.Embed | None = None,
    progress: Progress = _quiet,
    settings: dict | None = None,
    everything: bool = False,
) -> dict:
    """Check the clips without a current check (``everything``: all clips again)."""
    summary = checks.check_dataset(
        manifest_of(folder), None, checker, kind, load, progress, embed, only_needed=not everything
    )
    record.remember(folder, "check", {"aligner": kind, **(settings or {})})
    return summary


def check_summary(folder: Path) -> dict:
    """checks.summarize over the dataset's current checks (dropped clips too)."""
    saved = read_sidecar(folder / manifests.ALIGN_FILE).get("clips", {})
    current = []
    for e in clip_entries(manifest_of(folder)):
        old = saved.get(row_key(e["audio"], folder.resolve()))
        if e["text"] and old and old.get("text") == manifests.text_key(e["text"]):
            current.append(ClipCheck.from_saved(old))
    return checks.summarize(current)


def _trimmable(check: ClipCheck) -> tuple[float, float] | None:
    """What trimming keeps of a checked clip, unless it should be reviewed first."""
    if any(f in check.flags for f in TRIM_REVIEW_FIRST) or checks.squeezed_edge_filler(check):
        return None
    return check.trim_range()


def tidy(folder: Path, trim: bool = True, normalize_lufs: float | None = None, progress: Progress = _quiet) -> dict:
    """Trim checked clips at their edges (those flagged for review first are
    left), then bring every clip to ``normalize_lufs``. Both in place
    (originals kept); running again changes nothing more."""
    folder = folder.resolve()
    manifest = manifest_of(folder)
    result: dict = {}
    if trim:
        progress("trimming", 0, 0)
        result["trimmed"] = checks.trim_clips(manifest, None, lambda _, c: _trimmable(c) if c else None)["trimmed"]
    if normalize_lufs is not None:
        result["normalized"] = loudness.normalize_dataset(manifest, None, normalize_lufs, progress)["normalized"]
    record.remember(folder, "tidy", {"trim": trim, "normalize_lufs": normalize_lufs})
    return result


def status(folder: Path) -> dict:
    """Per step, what still needs it (clips dropped in review don't count)."""
    folder = folder.resolve()
    manifest = manifest_of(folder)
    rec = record.read_record(folder)
    excluded = manifests.excluded_keys(folder)
    clips = [e for e in clip_entries(manifest) if row_key(e["audio"], folder) not in excluded]
    saved = read_sidecar(folder / manifests.ALIGN_FILE).get("clips", {})
    unchecked = trimmable = 0
    for e in clips:
        if not e["text"]:
            continue
        old = saved.get(row_key(e["audio"], folder))
        if not old or old.get("text") != manifests.text_key(e["text"]):
            unchecked += 1
        elif _trimmable(ClipCheck.from_saved(old)):
            trimmable += 1
    unsliced = [r for r in rec.recordings if r.sliced is None]
    return {
        "sources": {"clips": len(clips), "recordings": len(rec.recordings), "dropped": len(excluded)},
        "clean_up": {"clips": sum(row_key(e["audio"], folder) not in rec.cleaned for e in clips)},
        "slice": {"recordings": len(unsliced), "seconds": round(sum(r.seconds for r in unsliced), 1)},
        "transcribe": {"clips": sum(not e["text"] for e in clips)},
        "check": {"clips": unchecked},
        "tidy": {"trimmable": trimmable},
        "settings": rec.steps,
    }


def remaining(folder: Path) -> list[str]:
    """The steps with work left, in order (clean-up and tidy are optional: not listed)."""
    s = status(folder)
    return [
        step for step, key in (("slice", "recordings"), ("transcribe", "clips"), ("check", "clips")) if s[step][key]
    ]
