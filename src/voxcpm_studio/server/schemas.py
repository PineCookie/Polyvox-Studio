"""Request and response bodies of the HTTP API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ..prep import loudness


class LoraConfigOut(BaseModel):
    r: int
    alpha: float
    dropout: float
    enable_lm: bool
    enable_dit: bool
    enable_proj: bool
    target_modules_lm: list[str]
    target_modules_dit: list[str]


class LoraOut(BaseModel):
    id: str
    engine: str
    base_model: str | None
    # Hugging Face id guessed from base_model; the default when exporting.
    base_model_id: str | None
    step: int | None
    rank: int | None
    alpha: float | None
    config: LoraConfigOut | None  # VoxCPM's layout (VoxCPM checkpoints only)


class DatasetReportOut(BaseModel):
    id: str
    samples: int
    dropped: int
    valid: int
    total_seconds: float
    min_seconds: float | None
    median_seconds: float | None
    max_seconds: float | None
    sample_rates: dict[int, int]
    errors: list[str]
    warnings: list[str]


class ImportSourceIn(BaseModel):
    # Server-local paths: a list / CSV / JSONL file or a folder of audio + transcripts.
    source: str
    # Where to look up audio by file name when the source's paths are stale.
    audio_dir: str | None = None


class ImportIn(ImportSourceIn):
    name: str
    copy_audio: bool = True
    # Fill in missing transcripts with ASR (otherwise those clips are left out).
    transcribe: bool = False
    # e.g. "zh"; None = detect per clip; "auto" = detect until the clips agree, then force that (faster)
    language: str | None = "auto"
    prompt: str | None = None  # names / terms the recognizer should prefer
    # Cut long recordings into clips at pauses (needs copy_audio and transcribe).
    slice: bool = False
    slice_min_seconds: float = Field(2.0, ge=0.5, le=30)
    slice_max_seconds: float = Field(15.0, ge=3, le=60)
    # Check each clip against its text by forced alignment (align.json).
    check: bool = False
    aligner: Literal["qwen3", "ctc"] = "qwen3"
    # Cut long silent edges from the copied clips (needs copy_audio and check).
    trim: bool = False
    detect_singing: bool = False  # with check: flag sung clips (slow; few datasets have songs)
    # Clean-up before anything else (needs copy_audio and the model configured):
    separate: bool = False  # keep only the voice (remove music / background)
    dereverb: bool = False
    denoise: bool = False
    # Shorten runs of fillers / laughs (哈哈哈哈 -> 哈哈) in every transcript.
    tidy_fillers: bool = True
    # Bring each copied clip to this loudness in LUFS (needs copy_audio); None = as recorded.
    normalize_lufs: float | None = Field(None, ge=-40, le=-10)


class DatasetIn(BaseModel):
    id: str  # dataset manifest id


class NormalizeIn(BaseModel):
    id: str  # dataset manifest id
    target_lufs: float = Field(loudness.TARGET_LUFS, ge=-40, le=-10)


class ExportIn(BaseModel):
    id: str  # dataset manifest id
    format: str  # exporter.FORMATS id
    # A folder on the server; empty: exporter.default_dir.
    out_dir: str = ""
    min_seconds: float | None = Field(None, ge=0)
    max_seconds: float | None = Field(None, gt=0)
    sample_rate: int | None = Field(None, ge=8000, le=96000)  # None: keep each clip's rate
    language: str | None = None  # GPT-SoVITS code; None: per clip from the text


class CheckIn(BaseModel):
    id: str  # dataset manifest id
    aligner: Literal["qwen3", "ctc"] = "qwen3"
    # Also transcribe each clip and flag those where ASR hears words the text lacks.
    compare_asr: bool = False
    # Flag sung clips (karaoke, covers): not wanted in TTS training data. Slow; few datasets have songs.
    detect_singing: bool = False


class ClipTextIn(BaseModel):
    id: str
    index: int
    text: str


class DropFlaggedIn(BaseModel):
    id: str
    flag: str  # a check flag, e.g. "singing"


class ClipDropIn(BaseModel):
    id: str
    index: int
    dropped: bool = True


class ClipTrimIn(BaseModel):
    id: str
    index: int
    # The part to keep, in seconds; both omitted: the checked edges (ClipCheck.trim_range).
    start: float | None = Field(None, ge=0)
    end: float | None = Field(None, gt=0)


class ClipCheckIn(BaseModel):
    id: str
    index: int
    compare_asr: bool = False  # also compare with what ASR hears (otherwise as the dataset's checks did)


class ClipRangeIn(BaseModel):
    id: str
    index: int
    start: float = Field(ge=0)
    end: float = Field(gt=0)


class ImportPreviewOut(BaseModel):
    kind: str
    source: str
    rows: int
    found: int
    untranscribed: int
    missing: int
    duplicates: int
    unreadable: int
    long_recordings: int
    total_seconds: float
    median_seconds: float | None
    sample_rates: dict[int, int]
    examples: list[dict]
    missing_examples: list[str]
    suggested_name: str


class ClipRowOut(BaseModel):
    index: int
    file: str
    text: str
    seconds: float | None
    asr: bool
    score: float | None
    flags: list[str]
    speech: tuple[float, float] | None
    peaks: list[float] | None
    dropped: bool  # dropped in review: left out of training runs
    # What trimming keeps, (start, end) seconds; None when not checked or there is little to trim.
    trim: tuple[float, float] | None


class WordOut(BaseModel):
    text: str
    start: float
    end: float
    confidence: float
    weak: bool  # below checks.WEAK_WORD_BELOW for the aligner: likely not what was said


class HeardOut(BaseModel):
    text: str  # what ASR heard
    extra: int  # words ASR heard that the transcript lacks
    missing: int  # transcript words ASR did not hear
    changed: int
    # (equal | missing | extra | changed, transcript part, ASR part) stretches
    diff: list[tuple[str, str, str]]


class ClipDetailOut(ClipRowOut):
    aligner: str | None
    words: list[WordOut]  # empty when not checked, or checked before word timings were saved
    heard: HeardOut | None  # when the check compared with ASR
    # Untranscribed sound a pause separates from the words (edge_sound): it lies
    # before / after this second. None when there is none.
    sound_before: float | None
    sound_after: float | None


class ClipPageOut(BaseModel):
    total: int
    rows: list[ClipRowOut]
    # Alignment checks over the whole dataset; None when it was never checked.
    checks: dict | None
    dropped: int  # clips dropped in review, whole dataset


class JobOut(BaseModel):
    id: str
    kind: str
    title: str
    state: str
    phase: str
    done: int
    total: int
    error: str | None
    result: dict
    started: float
    finished: float | None


class ModelOut(BaseModel):
    setting: str
    name: str
    purpose: str
    repo: str
    hub: str  # huggingface | modelscope
    size_mb: int
    license: str
    required: bool
    # downloaded: complete in the models folder; custom: configured explicitly;
    # missing: not downloaded (a Hugging Face model would then download to its cache on first use).
    state: Literal["downloaded", "custom", "missing"]
    path: str  # what the setting resolves to
    # When missing, loading still works: from the hub's cache (else the feature is unavailable).
    fallback: bool


class ModelsOut(BaseModel):
    models_dir: str
    endpoint: str  # the server's default download source ("" = huggingface.co)
    models: list[ModelOut]


class DownloadIn(BaseModel):
    settings: list[str] = Field(min_length=1)
    # Hugging Face or a mirror, e.g. https://hf-mirror.com; None = the server's default.
    endpoint: str | None = None


class ResumeIn(BaseModel):
    num_iters: int | None = Field(None, ge=1, le=1_000_000)
