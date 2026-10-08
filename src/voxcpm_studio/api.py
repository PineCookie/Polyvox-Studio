"""HTTP API and static frontend."""

from __future__ import annotations

import dataclasses
import io
import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from . import importer, lora_library, manifests, sharing
from .asr import AsrService, load_audio
from .config import Settings
from .inference import InferenceService, SynthesisRequest
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


class ImportPreviewOut(BaseModel):
    kind: str
    source: str
    rows: int
    found: int
    missing: int
    skipped_empty: int
    duplicates: int
    unreadable: int
    total_seconds: float
    median_seconds: float | None
    sample_rates: dict[int, int]
    examples: list[dict]
    missing_examples: list[str]
    suggested_name: str


class ImportOut(BaseModel):
    id: str
    imported: int
    preview: ImportPreviewOut


class ResumeIn(BaseModel):
    num_iters: int | None = Field(None, ge=1, le=1_000_000)


def _cuda_index(device: str) -> int | None:
    if not device.startswith("cuda"):
        return None
    index = device.partition(":")[2]
    return int(index) if index else 0


def free_gpu_for_training(service: InferenceService, train_gpu: int | None, asr: AsrService | None = None) -> None:
    """Unload the models that sit on the GPU a training job will use."""
    if _cuda_index(service.device) == (train_gpu or 0):
        service.unload()
    if asr is not None and asr.loaded and _cuda_index(asr.device) == (train_gpu or 0):
        asr.unload()


def create_app(
    settings: Settings | None = None,
    service: InferenceService | None = None,
    training: TrainingManager | None = None,
    asr: AsrService | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    service = service or InferenceService(settings)
    asr = asr or AsrService(settings)
    training = training or TrainingManager(
        settings, before_start=lambda: free_gpu_for_training(service, settings.train_gpu, asr)
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        training.shutdown()  # saves a checkpoint

    app = FastAPI(title="VoxCPM Studio", lifespan=lifespan)

    @app.exception_handler(TrainingConflict)
    def conflict(request: Request, exc: TrainingConflict) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=409)

    def status_with_model_id() -> dict:
        # The hub id lets the UI flag LoRAs trained on a different base model.
        return {**service.status(), "model_id": sharing.hub_id(settings.model), "asr_loaded": asr.loaded}

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
        ref_path = _save_upload(reference_audio) if reference_audio else None
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
        # Frees all GPU memory the server holds: TTS and ASR.
        service.unload()
        asr.unload()
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

    @app.post("/api/datasets/import")
    def import_dataset(body: ImportIn) -> ImportOut:
        try:
            dataset_id, preview = importer.import_dataset(
                body.source, body.audio_dir, body.name, settings.data_dir, copy=body.copy_audio
            )
        except FileExistsError as e:
            raise HTTPException(409, str(e)) from e
        except (ValueError, OSError) as e:
            raise HTTPException(422, str(e)) from e
        return ImportOut(id=dataset_id, imported=preview.found, preview=ImportPreviewOut(**dataclasses.asdict(preview)))

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
