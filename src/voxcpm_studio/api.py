"""HTTP API and static frontend."""

from __future__ import annotations

import dataclasses
import importlib.metadata
import io
import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from . import audio_io, checks, fillers, importer, lora_library, loudness, manifests, models, sharing, slicer
from .aligner import AlignService
from .asr import SAMPLE_RATE as ASR_RATE
from .asr import AsrService, LanguageLock, language_of, load_audio
from .config import Settings
from .inference import InferenceService, SynthesisRequest
from .jobs import Job, JobRunner
from .separation import SeparationService
from .singing import SingingService
from .slicer import SliceOptions
from .training import RunDetail, RunSummary, TrainingConflict, TrainingManager, TrainParams

WEB_DIST = Path(__file__).resolve().parents[2] / "web" / "dist"


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
    base_model: str | None
    # Hugging Face id guessed from base_model; the default when exporting.
    base_model_id: str | None
    step: int | None
    config: LoraConfigOut | None


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


MIN_KEPT_S = 0.3  # a trimmed clip keeps at least this much


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


def _cuda_index(device: str) -> int | None:
    if not device.startswith("cuda"):
        return None
    index = device.partition(":")[2]
    return int(index) if index else 0


def free_gpu_for_training(
    service: InferenceService,
    train_gpu: int | None,
    *helpers: AsrService | AlignService | SeparationService | SingingService,
) -> None:
    """Unload the models that sit on the GPU a training job will use."""
    if _cuda_index(service.device) == (train_gpu or 0):
        service.unload()
    for helper in helpers:
        if helper.loaded and _cuda_index(helper.device) == (train_gpu or 0):
            helper.unload()


def create_app(
    settings: Settings | None = None,
    service: InferenceService | None = None,
    training: TrainingManager | None = None,
    asr: AsrService | None = None,
    jobs: JobRunner | None = None,
    align: AlignService | None = None,
    separation: SeparationService | None = None,
    singing: SingingService | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    audio_io.patch_torchaudio()  # voxcpm's torchaudio.load calls, without a system FFmpeg
    service = service or InferenceService(settings)
    asr = asr or AsrService(settings)
    align = align or AlignService(settings)
    separation = separation or SeparationService(lambda: asr.device)
    singing = singing or SingingService(settings, lambda: asr.device)
    jobs = jobs or JobRunner()
    training = training or TrainingManager(
        settings,
        before_start=lambda: free_gpu_for_training(service, settings.train_gpu, asr, align, separation, singing),
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        training.shutdown()  # saves a checkpoint

    app = FastAPI(title="VoxCPM Studio", lifespan=lifespan)

    @app.exception_handler(TrainingConflict)
    def conflict(request: Request, exc: TrainingConflict) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=409)

    def cleanup_models() -> dict[str, str]:
        """Import clean-up option -> its RoFormer checkpoint ("" = not available). Looked up
        each time: a model downloaded meanwhile is used."""
        return {
            "separate": settings.model_path("vocals_model"),
            "dereverb": settings.model_path("dereverb_model"),
            "denoise": settings.model_path("denoise_model"),
        }

    def status_with_model_id() -> dict:
        # The hub id lets the UI flag LoRAs trained on a different base model.
        return {
            **service.status(),
            "model_id": sharing.hub_id(settings.model_path("model")),
            # The PyTorch build (e.g. 2.14.1+cu130): which of run.bat / run.sh's variants runs.
            "torch": importlib.metadata.version("torch"),
            "asr_loaded": asr.loaded,
            "aligner_loaded": align.loaded,
            "separation_loaded": separation.loaded,
            "singing_loaded": singing.loaded,
            # Import clean-up options whose model is configured.
            "cleanup": [k for k, v in cleanup_models().items() if v],
            # Not in the models folder (loading would download it to the Hugging Face cache).
            "missing_required": [
                s.name
                for s in models.MODELS
                if s.required and not settings.configured(s.setting) and not s.downloaded(settings.models_dir)
            ],
        }

    @app.get("/api/status")
    def status() -> dict:
        return status_with_model_id()

    @app.get("/api/loras")
    def list_loras() -> list[LoraOut]:
        return [
            LoraOut(
                id=c.id,
                base_model=c.base_model,
                base_model_id=sharing.hub_id(c.base_model),
                step=sharing.checkpoint_step(c),
                config=LoraConfigOut(**c.config.model_dump()) if c.config else None,
            )
            for c in lora_library.scan(settings.lora_dir)
        ]

    @app.get("/api/loras/export", responses={200: {"content": {"application/zip": {}}}})
    def export_lora(id: str, name: str, base_model: str, half: bool = True) -> FileResponse:
        ckpt = _found(lora_library.resolve, settings.lora_dir, id)
        path = _checked(sharing.export_package, ckpt, name, base_model, half)
        return FileResponse(
            path, media_type="application/zip", filename=f"{name}.zip", background=BackgroundTask(path.unlink)
        )

    @app.post("/api/loras/import")
    def import_lora(package: Annotated[UploadFile, File()]) -> dict:
        path = _save_upload(package)
        try:
            return {"id": sharing.import_package(settings.lora_dir, path, package.filename or "lora.zip")}
        except FileExistsError as e:
            raise HTTPException(409, str(e)) from e
        except (ValueError, OSError) as e:
            raise HTTPException(422, str(e)) from e
        finally:
            path.unlink(missing_ok=True)

    @app.post("/api/tts", responses={200: {"content": {"audio/wav": {}}}})
    def tts(
        text: Annotated[str, Form()],
        control: Annotated[str, Form()] = "",
        lora_id: Annotated[str | None, Form()] = None,
        prompt_text: Annotated[str | None, Form()] = None,
        cfg_value: Annotated[float, Form(ge=0.5, le=5.0)] = 2.0,
        inference_timesteps: Annotated[int, Form(ge=1, le=50)] = 10,
        normalize: Annotated[bool, Form()] = True,
        denoise: Annotated[bool, Form()] = True,
        seed: Annotated[int | None, Form()] = None,
        reference_audio: Annotated[UploadFile | None, File()] = None,
    ) -> Response:
        # As WAV when soundfile can't read it (e.g. m4a): voxcpm can't without a system FFmpeg.
        ref_path = audio_io.to_wav(_save_upload(reference_audio)) if reference_audio else None
        try:
            result = service.synthesize(
                SynthesisRequest(
                    text=text,
                    control=control,
                    lora_id=lora_id or None,
                    reference_wav=ref_path,
                    prompt_text=prompt_text,
                    cfg_value=cfg_value,
                    inference_timesteps=inference_timesteps,
                    normalize=normalize,
                    denoise=denoise,
                    seed=seed,
                )
            )
        except KeyError as e:
            raise HTTPException(404, str(e.args[0])) from e
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        finally:
            if ref_path:
                ref_path.unlink(missing_ok=True)

        buf = io.BytesIO()
        sf.write(buf, result.audio, result.sample_rate, format="WAV")
        headers = {"X-Seed": str(result.seed)} if result.seed is not None else {}
        return Response(buf.getvalue(), media_type="audio/wav", headers=headers)

    @app.post("/api/model/unload")
    def unload() -> dict:
        # Frees all GPU memory the server holds: TTS, ASR and the aligner.
        service.unload()
        asr.unload()
        align.unload()
        separation.unload()
        singing.unload()
        return status_with_model_id()

    @app.post("/api/asr")
    def transcribe(
        audio: Annotated[UploadFile, File()],
        language: Annotated[str | None, Form()] = None,
        prompt: Annotated[str | None, Form()] = None,
    ) -> dict:
        path = _save_upload(audio)
        try:
            (result,) = asr.transcribe([load_audio(path)], language=language or None, prompt=prompt or None)
        except (ValueError, RuntimeError) as e:
            raise HTTPException(422, str(e)) from e
        finally:
            path.unlink(missing_ok=True)
        return {"text": result.text, "language": result.language}

    @app.get("/api/datasets")
    def list_datasets() -> list[str]:
        return manifests.scan(settings.data_dir)

    @app.post("/api/datasets/import/preview")
    def preview_import(body: ImportSourceIn) -> ImportPreviewOut:
        preview = _checked(importer.preview_import, body.source, body.audio_dir, settings.data_dir)
        return ImportPreviewOut(**dataclasses.asdict(preview))

    @app.post("/api/datasets/import", status_code=202)
    def import_dataset(body: ImportIn) -> JobOut:
        """Start an import job; poll /api/jobs/{id}. The job's result is
        {"id", "imported", "transcribed", "sliced"}."""
        available = cleanup_models()
        cleanup = [k for k in available if getattr(body, k)]
        # Fail fast on what can be checked now.
        try:
            if body.slice and not (body.copy_audio and body.transcribe):
                raise ValueError("Slicing makes new clips without text: it needs copying and transcription.")
            if body.slice and body.slice_min_seconds >= body.slice_max_seconds:
                raise ValueError("The minimum clip length must be below the maximum.")
            if body.trim and not (body.copy_audio and body.check):
                raise ValueError("Trimming rewrites the copied clips: it needs copying and checking.")
            for option in cleanup:
                if not available[option]:
                    raise ValueError(f"The model for '{option}' is not downloaded (see Models).")
            if cleanup and not body.copy_audio:
                raise ValueError("Clean-up writes new audio: it needs copying.")
            if body.normalize_lufs is not None and not body.copy_audio:
                raise ValueError("Normalizing loudness rewrites the clips: it needs copying.")
            importer.check_name(body.name, settings.data_dir)
            importer.detect(importer.clean_path(body.source))
        except FileExistsError as e:
            raise HTTPException(409, str(e)) from e
        except (ValueError, OSError) as e:
            raise HTTPException(422, str(e)) from e
        refuse_if_busy(gpu=body.transcribe or body.check or bool(cleanup))

        lock = LanguageLock() if body.language == "auto" else None

        def heard(wavs: list) -> list[tuple[str, str | None]]:
            forced = lock.language if lock else body.language
            results = asr.transcribe(wavs, language=forced, prompt=body.prompt)
            if lock:
                lock.add(wavs, results)
            # A forced language is not detected, so record what was forced.
            return [(r.text, r.language or forced) for r in results]

        def transcriber(paths: list[Path]) -> list[tuple[str, str | None]]:
            return heard([load_audio(p) for p in paths])

        def run(job: Job) -> dict:
            result = importer.import_dataset(
                body.source,
                body.audio_dir,
                body.name,
                settings.data_dir,
                copy=body.copy_audio,
                transcriber=transcriber if body.transcribe else None,
                progress=job.progress,
                slicing=(
                    SliceOptions(
                        min_seconds=body.slice_min_seconds,
                        max_seconds=body.slice_max_seconds,
                        target_seconds=(body.slice_min_seconds + body.slice_max_seconds) / 2,
                    )
                    if body.slice
                    else None
                ),
                checker=checker(body.aligner, detect_singing=body.detect_singing) if body.check else None,
                aligner=body.aligner,
                trim=body.trim,
                guide=slicer.Guide(transcribe=heard, align=lambda wavs, texts: align.align(wavs, texts, body.aligner))
                if body.slice
                else None,
                separation=separation.chain([available[k] for k in cleanup]) if cleanup else None,
                tidy_fillers=body.tidy_fillers,
                normalize=body.normalize_lufs,
            )
            return {
                "id": result.id,
                "imported": result.imported,
                "transcribed": result.transcribed,
                "sliced": result.sliced,
                "trimmed": result.trimmed,
                "checks": result.checks,
            }

        return JobOut(**jobs.start("import", f"Import {body.name}", run).public())

    def checker(kind: str, compare_asr: bool = False, detect_singing: bool = False) -> checks.Checker:
        def heard_texts(wavs: list) -> list[str]:
            return [r.text for r in asr.transcribe(wavs)]

        return checks.make_checker(
            lambda wavs, texts: align.align(wavs, texts, kind),
            kind,
            transcribe=heard_texts if compare_asr else None,
            singing=singing.scores if detect_singing else None,
        )

    def refuse_if_busy(gpu: bool) -> None:
        """One preparation job at a time; GPU stages wait for training on their GPU."""
        if running := jobs.running("import") or jobs.running("check") or jobs.running("normalize"):
            raise HTTPException(409, f"Wait for “{running.title}” to finish.")
        if gpu and training.active_run and _cuda_index(asr.device) == (settings.train_gpu or 0):
            raise HTTPException(
                409,
                f"Run '{training.active_run}' is training on the GPU that ASR and alignment use. Wait for it, or "
                "set VOXCPM_STUDIO_ASR_DEVICE to another GPU.",
            )

    @app.post("/api/datasets/tidy-fillers")
    def tidy_fillers(body: DatasetIn) -> dict:
        """Shorten runs of fillers in a dataset's transcripts (fillers.tidy), in
        place; the first time, the original manifest is kept as <name>.orig.
        Checks made for the old texts become stale."""
        manifest = _found(manifests.resolve, settings.data_dir, body.id)
        return {"changed": _checked(fillers.tidy_manifest, manifest)}

    @app.post("/api/datasets/check", status_code=202)
    def check_dataset(body: CheckIn) -> JobOut:
        """Check every clip of a dataset against its text (forced alignment) and
        save the results as align.json. The job's result is the summary
        {"aligner", "checked", "flagged", "flags": {flag: count}}."""
        manifest = _found(manifests.resolve, settings.data_dir, body.id)
        refuse_if_busy(gpu=True)

        def run(job: Job) -> dict:
            return checks.check_dataset(
                manifest,
                manifests.legacy_root(settings.data_dir),
                checker(body.aligner, body.compare_asr, body.detect_singing),
                body.aligner,
                load_audio,
                job.progress,
            )

        return JobOut(**jobs.start("check", f"Check {body.id}", run).public())

    @app.post("/api/datasets/normalize", status_code=202)
    def normalize_dataset(body: NormalizeIn) -> JobOut:
        """Bring every clip of a dataset to ``target_lufs`` in place (originals
        kept in originals/; checks stay valid). The job's result is
        loudness.normalize_dataset's summary."""
        manifest = _found(manifests.resolve, settings.data_dir, body.id)
        refuse_if_busy(gpu=False)

        def run(job: Job) -> dict:
            root = manifests.legacy_root(settings.data_dir)
            return loudness.normalize_dataset(manifest, root, body.target_lufs, job.progress)

        return JobOut(**jobs.start("normalize", f"Normalize {body.id}", run).public())

    @app.post("/api/dataset-clip/text")
    def set_clip_text(body: ClipTextIn) -> ClipDetailOut:
        """Change one clip's transcript. When the dataset has checks, the clip is
        checked again right away (with the dataset's aligner, compared with ASR
        if its last check was); otherwise its old check just goes stale."""
        manifest = _found(manifests.resolve, settings.data_dir, body.id)
        refuse_if_busy(gpu=False)
        _checked(manifests.set_text, manifest, body.index, body.text)
        if has_checks(manifest) and gpu_free():
            check_clip_again(manifest, body.index)
        return dataset_clip(body.index, id=body.id)

    def has_checks(manifest: Path) -> bool:
        return manifests.read_sidecar(manifest.parent / manifests.ALIGN_FILE).get("aligner") in ("qwen3", "ctc")

    def gpu_free() -> bool:
        """ASR and alignment can run: no training on their GPU."""
        return not (training.active_run and _cuda_index(asr.device) == (settings.train_gpu or 0))

    def check_clip_again(manifest: Path, index: int, compare_asr: bool | None = None) -> None:
        """Check one clip with the dataset's aligner (Qwen3 when never checked).
        ``compare_asr`` None: compare with ASR if the dataset's checks did; True
        also compares a text ASR made. Flags sung clips if the checks did."""
        saved = manifests.read_sidecar(manifest.parent / manifests.ALIGN_FILE)
        kind = saved.get("aligner") if saved.get("aligner") in ("qwen3", "ctc") else "qwen3"
        clips = saved.get("clips", {}).values()
        compared = any(c.get("asr") for c in clips) if compare_asr is None else compare_asr
        sang = any(c.get("singing") for c in clips)
        root = manifests.legacy_root(settings.data_dir)
        checker_ = checker(kind, compared, sang)
        checks.check_one(manifest, root, index, checker_, kind, load_audio, compare=True if compare_asr else None)

    @app.post("/api/dataset-clip/check")
    def check_clip(body: ClipCheckIn) -> ClipDetailOut:
        """Check one clip again now (alignment; with ``compare_asr``, also what ASR hears)."""
        manifest = _found(manifests.resolve, settings.data_dir, body.id)
        refuse_if_busy(gpu=True)
        _found(check_clip_again, manifest, body.index, body.compare_asr or None)
        return dataset_clip(body.index, id=body.id)

    @app.post("/api/datasets/drop-flagged")
    def drop_flagged(body: DropFlaggedIn) -> dict:
        """Drop every clip whose check has ``flag`` (e.g. all sung clips)."""
        manifest = _found(manifests.resolve, settings.data_dir, body.id)
        if body.flag not in checks.FLAGS:
            raise HTTPException(422, f"Unknown flag '{body.flag}'.")
        return {"dropped": manifests.drop_flagged(manifest, manifests.legacy_root(settings.data_dir), body.flag)}

    @app.post("/api/datasets/trim-flagged")
    def trim_flagged(body: DropFlaggedIn) -> dict:
        """Trim every clip with ``flag`` (edge_sound or long_silence) at its
        checked edges, in place (originals kept in originals/). Clips also
        flagged mismatch / asr_more are left for review. Returns {"trimmed",
        "skipped" (format or location can't be rewritten), "review"}."""
        manifest = _found(manifests.resolve, settings.data_dir, body.id)
        if body.flag not in ("edge_sound", "long_silence"):
            raise HTTPException(422, "Only clips flagged for edge sound or long silence can be trimmed.")
        refuse_if_busy(gpu=False)
        return checks.trim_flagged(manifest, manifests.legacy_root(settings.data_dir), body.flag)

    @app.post("/api/dataset-clip/trim")
    def trim_clip(body: ClipTrimIn) -> ClipDetailOut:
        """Trim one clip in place (see trim-flagged): to [start, end) seconds when
        given, else at its checked edges."""
        manifest = _found(manifests.resolve, settings.data_dir, body.id)
        root = manifests.legacy_root(settings.data_dir)
        refuse_if_busy(gpu=False)
        keep: tuple[float, float] | None = None
        if body.start is not None or body.end is not None:
            seconds = _found(manifests.clip_detail, manifest, root, body.index).row.seconds or 0.0
            keep = (max(0.0, body.start or 0.0), min(seconds, body.end if body.end is not None else seconds))
            if keep[1] - keep[0] < MIN_KEPT_S:
                raise HTTPException(422, f"Keep at least {MIN_KEPT_S} s of the clip.")
            if keep[0] < 0.01 and keep[1] > seconds - 0.01:
                raise HTTPException(409, "Nothing to trim: the range is the whole clip.")

        def cut_for(index: int, check: checks.ClipCheck | None) -> tuple[float, float] | None:
            if index != body.index:
                return None
            return keep or (check.trim_range() if check else None)

        result = checks.trim_clips(manifest, root, cut_for)
        if result["skipped"]:
            raise HTTPException(
                422, "This clip can't be trimmed in place (outside the dataset folder, or not WAV / FLAC / OGG)."
            )
        if not result["trimmed"]:
            raise HTTPException(409, "Nothing to trim: check the clip first.")
        # Align the trimmed clip afresh rather than keep the shifted old check.
        if has_checks(manifest) and gpu_free():
            check_clip_again(manifest, body.index)
        return dataset_clip(body.index, id=body.id)

    @app.post("/api/dataset-clip/hear")
    def hear_clip_part(body: ClipRangeIn) -> dict:
        """What ASR hears in [start, end) seconds of one clip, e.g. an edge sound
        the transcript lacks: {"text"}."""
        manifest = _found(manifests.resolve, settings.data_dir, body.id)
        audio = _found(manifests.clip_audio, manifest, manifests.legacy_root(settings.data_dir), body.index)
        refuse_if_busy(gpu=True)
        part = load_audio(audio)[int(body.start * ASR_RATE) : int(body.end * ASR_RATE)]
        if len(part) < ASR_RATE // 10:
            raise HTTPException(422, "That part is too short to hear.")
        text = manifests.clip_detail(manifest, manifests.legacy_root(settings.data_dir), body.index).row.text
        return {"text": asr.transcribe([part], language=language_of(text))[0].text}

    @app.post("/api/dataset-clip/drop")
    def drop_clip(body: ClipDropIn) -> dict:
        """Leave one clip out of training (``dropped``), or bring it back."""
        manifest = _found(manifests.resolve, settings.data_dir, body.id)
        _found(manifests.set_dropped, manifest, manifests.legacy_root(settings.data_dir), body.index, body.dropped)
        return {"index": body.index, "dropped": body.dropped}

    def review_manifest(id: str | None, run: str | None) -> Path:
        """A dataset by manifest id (e.g. "speaker/train.jsonl"), or the one a run trains on."""
        if run:
            return _found(training.train_manifest, run)
        if id:
            return _found(manifests.resolve, settings.data_dir, id)
        raise HTTPException(422, "Give a dataset id or a run name.")

    @app.get("/api/dataset-report")
    def dataset_report(id: str | None = None, run: str | None = None) -> DatasetReportOut:
        report = manifests.inspect(review_manifest(id, run), manifests.legacy_root(settings.data_dir))
        return DatasetReportOut(id=id or f"run:{run}", **dataclasses.asdict(report))

    def trim_of(saved: dict) -> tuple[float, float] | None:
        return checks.ClipCheck.from_saved(saved).trim_range()

    @app.get("/api/dataset-rows")
    def dataset_rows(
        id: str | None = None,
        run: str | None = None,
        offset: int = 0,
        limit: int = Query(50, ge=1, le=500),
        q: str = "",
        flag: str | None = None,
        order: Literal["file", "worst"] = "file",
        keep: Annotated[list[int] | None, Query()] = None,
    ) -> ClipPageOut:
        """A page of a dataset's clips for review. ``flag``: a check flag, "any"
        flag, or "dropped" clips; ``order=worst`` puts the lowest scores first."""
        path = review_manifest(id, run)
        page = manifests.read_rows(
            path,
            manifests.legacy_root(settings.data_dir),
            max(0, offset),
            limit,
            q,
            flag or None,
            order == "worst",
            trim_of,
            frozenset(keep or ()),
        )
        return ClipPageOut(
            total=page.total,
            rows=[ClipRowOut(**dataclasses.asdict(r)) for r in page.rows],
            checks=page.checks,
            dropped=page.dropped,
        )

    @app.get("/api/dataset-clip")
    def dataset_clip(index: int, id: str | None = None, run: str | None = None) -> ClipDetailOut:
        """One clip with a finer waveform and, when checked, where each word is."""
        path = review_manifest(id, run)
        detail = _found(manifests.clip_detail, path, manifests.legacy_root(settings.data_dir), index, trim_of)
        weak_below = checks.WEAK_WORD_BELOW.get(detail.aligner or "", 0.0)
        check = checks.ClipCheck.from_saved(detail.check) if detail.check else None
        return ClipDetailOut(
            sound_before=check.sound_before[0] if check and check.sound_before else None,
            sound_after=check.sound_after[0] if check and check.sound_after else None,
            **dataclasses.asdict(detail.row),
            aligner=detail.aligner,
            words=[WordOut(text=t, start=s, end=e, confidence=c, weak=c < weak_below) for t, s, e, c in detail.words],
            heard=HeardOut(
                text=detail.asr["text"],
                extra=detail.asr["extra"],
                missing=detail.asr["missing"],
                changed=detail.asr["changed"],
                diff=checks.asr_diff(detail.row.text, detail.asr["text"]),
            )
            if detail.asr
            else None,
        )

    @app.get("/api/dataset-audio")
    def dataset_audio(index: int, id: str | None = None, run: str | None = None) -> FileResponse:
        """Audio of one clip, by its index in a dataset (never by path)."""
        path = review_manifest(id, run)
        return FileResponse(_found(manifests.clip_audio, path, manifests.legacy_root(settings.data_dir), index))

    @app.get("/api/models")
    def list_models() -> ModelsOut:
        out = []
        for spec in models.MODELS:
            path = settings.model_path(spec.setting)
            if settings.configured(spec.setting):
                state = "custom"
            else:
                state = "downloaded" if spec.downloaded(settings.models_dir) else "missing"
            out.append(
                ModelOut(
                    setting=spec.setting,
                    name=spec.name,
                    purpose=spec.purpose,
                    repo=spec.repo,
                    hub=spec.hub,
                    size_mb=spec.size_mb,
                    license=spec.license,
                    required=spec.required,
                    state=state,
                    path=path,
                    fallback=bool(Settings.__dataclass_fields__[spec.setting].default),
                )
            )
        return ModelsOut(models_dir=str(settings.models_dir.resolve()), endpoint=settings.hf_endpoint, models=out)

    @app.post("/api/models/download", status_code=202)
    def download_models(body: DownloadIn) -> JobOut:
        """Download models into the models folder, one after another (job phase:
        the setting being downloaded; done / total in MB)."""
        unknown = [s for s in body.settings if s not in models.BY_SETTING]
        if unknown:
            raise HTTPException(422, f"Unknown model: {', '.join(unknown)}")
        endpoint = (settings.hf_endpoint if body.endpoint is None else body.endpoint).strip().rstrip("/")
        if endpoint and not endpoint.startswith(("http://", "https://")):
            raise HTTPException(422, "The download source must be an http(s) URL.")
        if running := jobs.running("download"):
            raise HTTPException(409, f"Wait for “{running.title}” to finish.")
        specs = [models.BY_SETTING[s] for s in body.settings]

        def run(job: Job) -> dict:
            for spec in specs:
                models.download(
                    spec,
                    settings.models_dir,
                    endpoint or None,
                    lambda done, total, spec=spec: job.progress(spec.setting, done, total),
                )
            return {"downloaded": [s.setting for s in specs]}

        names = ", ".join(s.name for s in specs)
        return JobOut(**jobs.start("download", f"Download {names}", run).public())

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str) -> JobOut:
        return JobOut(**_found(jobs.get, job_id).public())

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str) -> JobOut:
        return JobOut(**_found(jobs.cancel, job_id).public())

    @app.get("/api/datasets/{dataset_id:path}")
    def inspect_dataset(dataset_id: str) -> DatasetReportOut:
        path = _found(manifests.resolve, settings.data_dir, dataset_id)
        report = manifests.inspect(path, manifests.legacy_root(settings.data_dir))
        return DatasetReportOut(id=dataset_id, **dataclasses.asdict(report))

    @app.get("/api/training/runs")
    def list_runs() -> list[RunSummary]:
        return training.list_runs()

    @app.post("/api/training/runs")
    def start_run(params: TrainParams) -> RunDetail:
        return _checked(training.start, params)

    @app.get("/api/training/runs/{name}")
    def run_detail(name: str) -> RunDetail:
        return _found(training.detail, name)

    @app.post("/api/training/runs/{name}/resume")
    def resume_run(name: str, body: ResumeIn | None = None) -> RunDetail:
        return _checked(training.resume, name, body.num_iters if body else None)

    @app.post("/api/training/stop")
    def stop_training() -> dict:
        training.stop()
        return {"stopping": training.active_run}

    @app.get("/api/training/runs/{name}/samples/{file}")
    def run_sample(name: str, file: str) -> FileResponse:
        return FileResponse(_found(training.sample_file, name, file), media_type="audio/wav")

    if WEB_DIST.is_dir():
        app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")

    return app


def _found(fn, *args):
    try:
        return fn(*args)
    except KeyError as e:
        raise HTTPException(404, str(e.args[0])) from e


def _checked(fn, *args):
    try:
        return _found(fn, *args)
    except (ValueError, OSError) as e:
        raise HTTPException(422, str(e)) from e


def _save_upload(upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "").suffix or ".wav"
    fd, name = tempfile.mkstemp(prefix="voxcpm_ref_", suffix=suffix)
    with os.fdopen(fd, "wb") as out:
        shutil.copyfileobj(upload.file, out)
    return Path(name)
