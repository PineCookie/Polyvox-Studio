"""Export a prepared dataset for other TTS training tools.

Formats (``FORMATS``): GPT-SoVITS ``.list``, LJSpeech ``metadata.csv``, a
``audio_file|text`` CSV with a header (F5-TTS's custom-dataset layout), Kaldi
style ``wav.scp`` / ``text`` / ``utt2spk`` / ``spk2utt``, and ``.lab`` files next
to the audio (folder layout; the importer reads it back too).

Clips dropped in review are left out, and so are clips outside an optional
length range. Audio is written as WAV under ``wavs/`` with plain numbered
names (``<speaker>_00001.wav``: no spaces or odd characters for Kaldi-style
tools), copied as is or resampled. The export is built in a hidden staging
folder and renamed when complete, so a failed or cancelled one leaves nothing.
"""

from __future__ import annotations

import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from ..data import audio_io, manifests
from .asr import language_of

PROGRESS_EVERY = 10


@dataclass(frozen=True)
class Format:
    id: str
    label: str
    description: str
    # Suggested clip length range (seconds; None = no limit), prefilled in the UI.
    min_seconds: float | None = None
    max_seconds: float | None = None


FORMATS: tuple[Format, ...] = (
    Format(
        "gpt_sovits",
        "GPT-SoVITS (.list)",
        "path|speaker|language|text per line, absolute paths; language zh / ja / en / ko per clip "
        "(GPT-SoVITS's zh covers Chinese mixed with English). It works best with short clips.",
        1.0,
        15.0,
    ),
    Format("ljspeech", "LJSpeech (metadata.csv)", "id|text|text per line, audio in wavs/<id>.wav."),
    Format(
        "csv",
        "CSV with header (F5-TTS)",
        "metadata.csv with an audio_file|text header, paths relative to it (wavs/...).",
    ),
    Format("kaldi", "Kaldi (wav.scp, text, utt2spk)", "Kaldi-style data folder, as CosyVoice-type recipes use."),
    Format("lab", "Folder with .lab files", "Each wavs/<id>.wav with its transcript in wavs/<id>.lab."),
)
BY_ID = {f.id: f for f in FORMATS}

# asr.language_of's names -> GPT-SoVITS language codes
LANGUAGE_CODES = {"Chinese": "zh", "Japanese": "ja", "Korean": "ko"}
GPT_SOVITS_LANGUAGES = ("zh", "ja", "en", "ko", "yue")


def clean_text(text: str) -> str:
    """One line, no field separators: | and tabs / line breaks become spaces."""
    return re.sub(r"\s+", " ", re.sub(r"[|\t\r\n]", " ", text)).strip()


def clip_language(text: str) -> str:
    """A GPT-SoVITS language code from the text's script: kana ja, hangul ko,
    Han zh; Latin letters only en; otherwise zh."""
    named = language_of(text)
    if named:
        return LANGUAGE_CODES[named]
    return "en" if re.search(r"[A-Za-z]", text) else "zh"


def default_dir(data_dir: Path, dataset_id: str, fmt: str) -> Path:
    """<the folder holding the data dir>/exports/<dataset>_<format>."""
    name = Path(dataset_id).parent.name or Path(dataset_id).stem
    return data_dir.resolve().parent / "exports" / f"{name}_{fmt}"


def export_dataset(
    manifest: Path,
    root: Path | None,
    out_dir: Path,
    fmt: str,
    speaker: str,
    *,
    min_seconds: float | None = None,
    max_seconds: float | None = None,
    sample_rate: int | None = None,
    language: str | None = None,
    progress: Callable[[str, int, int], None] | None = None,
) -> dict:
    """Write ``manifest``'s training clips to ``out_dir`` in format ``fmt``.

    ``sample_rate`` resamples (None: each clip keeps its own rate); ``language``
    forces GPT-SoVITS's language code (None: per clip from its text).
    Returns {"dir", "format", "clips", "seconds", "skipped": {"short", "long",
    "unreadable"}}."""
    if fmt not in BY_ID:
        raise ValueError(f"Unknown export format: {fmt}")
    if language is not None and language not in GPT_SOVITS_LANGUAGES:
        raise ValueError(f"Unknown language code: {language}")
    out_dir = out_dir.resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} already exists and is not empty.")
    progress = progress or (lambda phase, done, total: None)
    speaker = re.sub(r"[^\w.-]+", "_", speaker).strip("_") or "speaker"

    entries, _ = manifests.training_entries(manifest, root)
    stage = out_dir.with_name(f".{out_dir.name}.export-tmp")
    if stage.exists():
        shutil.rmtree(stage)
    (stage / "wavs").mkdir(parents=True)
    rows: list[tuple[str, Path, str]] = []  # (utterance id, written wav, text)
    skipped = {"short": 0, "long": 0, "unreadable": 0}
    seconds = 0.0
    try:
        for n, entry in enumerate(entries):
            if n % PROGRESS_EVERY == 0:
                progress("exporting", n, len(entries))
            source, text = Path(entry["audio"]), clean_text(str(entry.get("text", "")))
            try:
                duration, rate = audio_io.info(source)
            except (ValueError, OSError):
                skipped["unreadable"] += 1
                continue
            if min_seconds is not None and duration < min_seconds:
                skipped["short"] += 1
                continue
            if max_seconds is not None and duration > max_seconds:
                skipped["long"] += 1
                continue
            utt = f"{speaker}_{len(rows) + 1:05d}"
            target = stage / "wavs" / f"{utt}.wav"
            try:
                _write_wav(source, target, rate, sample_rate)
            except (ValueError, OSError, RuntimeError):
                skipped["unreadable"] += 1
                continue
            rows.append((utt, target, text))
            seconds += duration
        progress("exporting", len(entries), len(entries))
        _write_index(stage, out_dir, fmt, speaker, rows, language)
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        if out_dir.exists():
            out_dir.rmdir()  # empty (checked above)
        stage.rename(out_dir)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return {"dir": str(out_dir), "format": fmt, "clips": len(rows), "seconds": round(seconds, 1), "skipped": skipped}


def _write_wav(source: Path, target: Path, rate: int, sample_rate: int | None) -> None:
    if source.suffix.lower() == ".wav" and (sample_rate is None or sample_rate == rate):
        shutil.copyfile(source, target)
        return
    audio, rate = audio_io.read(source)
    if sample_rate is not None and sample_rate != rate:
        audio, rate = audio_io.resample(audio, rate, sample_rate), sample_rate
    sf.write(str(target), np.clip(audio, -1.0, 1.0).T, rate, subtype="PCM_16")


def _write_index(
    stage: Path, out_dir: Path, fmt: str, speaker: str, rows: list[tuple[str, Path, str]], language: str | None
) -> None:
    """The format's index files, written into ``stage``; paths that must be
    absolute point to where the files will be (``out_dir``)."""

    def lines(path: Path, items: list[str]) -> None:
        with path.open("w", encoding="utf-8", newline="\n") as f:
            f.writelines(f"{item}\n" for item in items)

    final = {utt: out_dir / "wavs" / wav.name for utt, wav, _ in rows}
    if fmt == "gpt_sovits":
        items = [f"{final[utt]}|{speaker}|{language or clip_language(text)}|{text}" for utt, _, text in rows]
        lines(stage / f"{speaker}.list", items)
    elif fmt == "ljspeech":
        lines(stage / "metadata.csv", [f"{utt}|{text}|{text}" for utt, _, text in rows])
    elif fmt == "csv":
        lines(stage / "metadata.csv", ["audio_file|text", *(f"wavs/{wav.name}|{text}" for _, wav, text in rows)])
    elif fmt == "kaldi":
        lines(stage / "wav.scp", [f"{utt} {final[utt]}" for utt, _, _ in rows])
        lines(stage / "text", [f"{utt} {text}" for utt, _, text in rows])
        lines(stage / "utt2spk", [f"{utt} {speaker}" for utt, _, _ in rows])
        lines(stage / "spk2utt", [f"{speaker} {' '.join(utt for utt, _, _ in rows)}"] if rows else [])
    elif fmt == "lab":
        for _, wav, text in rows:
            wav.with_suffix(".lab").write_text(text, encoding="utf-8")
