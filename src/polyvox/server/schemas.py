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
    # The run / voice it belongs to: its folder's id (e.g. "gpt_sovits/speaker_a") and name.
    voice_path: str
    voice: str
    base_model: str | None
    # Hugging Face id guessed from base_model; the default when exporting.
    base_model_id: str | None
    step: int | None
    rank: int | None
    alpha: float | None
    config: LoraConfigOut | None  # VoxCPM's layout (VoxCPM checkpoints only)
    # Its default reference clip's transcript (GPT-SoVITS voices); None: it has none.
    reference_text: str | None = None
    version: str | None = None  # the model version, when the checkpoint records one (e.g. v2ProPlus)


class GptSovitsImportIn(BaseModel):
    """Weights the GPT-SoVITS webui trained, as server-local paths."""

    name: str
    sovits: str  # SoVITS_weights_*/<voice>.pth
    gpt: str  # GPT_weights_*/<voice>.ckpt
    # The default reference: picked from this dataset (manifest id), or this clip + transcript;
    # neither: none yet.
    dataset: str | None = None
    reference: str | None = None
    reference_text: str = ""


class DatasetReportOut(BaseModel):
    id: str
    samples: int
    dropped: int
    untranscribed: int = 0  # clips without text yet: not in training
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


class AddSourceIn(ImportSourceIn):
    id: str  # dataset manifest id
    copy_audio: bool = True


class StepIn(BaseModel):
    """One processing step on a dataset (prep/steps.py). Settings left out
    come from the step's last run on this dataset (its record), else defaults."""

    id: str  # dataset manifest id
    step: Literal["clean_up", "slice", "transcribe", "check", "tidy"]
    # clean_up
    models: list[Literal["separate", "dereverb", "denoise"]] | None = None
    # slice
    min_seconds: float | None = Field(None, ge=0.5, le=30)
    max_seconds: float | None = Field(None, ge=3, le=60)
    # slice (its clips' texts) and transcribe
    language: str | None = None  # "auto" (default), "" = per clip, or a code
    prompt: str | None = None
    tidy_fillers: bool | None = None
    # check
    aligner: Literal["qwen3", "ctc"] | None = None
    compare_asr: bool | None = None
    detect_singing: bool | None = None
    everything: bool = False  # check every clip again, not only those without a current check
    # tidy
    trim: bool | None = None
    normalize_lufs: float | None = Field(None, ge=-40, le=-10)


class StepStatusOut(BaseModel):
    """Per step what still needs it (prep/steps.status), and its last settings."""

    sources: dict
    clean_up: dict
    slice: dict
    transcribe: dict
    check: dict
    tidy: dict
    settings: dict


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
    speaker: float | None = None  # similarity of the voice to the dataset's, when scored
    clipped: int | None = None  # flat tops at the peak, when measured


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
