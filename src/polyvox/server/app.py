"""The FastAPI app: services, routes (one module per area) and the built frontend."""

from __future__ import annotations

import functools
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .. import engines, vram
from ..config import Settings
from ..data import audio_io
from ..engines import Engine
from ..engines.voxcpm.engine import VoxCPMEngine
from ..engines.voxcpm.inference import InferenceService
from ..jobs import JobRunner
from ..prep import slicer
from ..prep.aligner import AlignService
from ..prep.asr import AsrService
from ..prep.separation import SeparationService
from ..prep.separation import load_models as separation_models
from ..prep.singing import SingingService
from ..speaker import SpeakerService
from ..training import TrainingConflict, TrainingManager
from . import clips, datasets, loras, status, tts
from . import training as training_routes
from .context import Studio, free_gpu_for_training

WEB_DIST = Path(__file__).resolve().parents[3] / "web" / "dist"

logger = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None,
    service: InferenceService | None = None,
    training: TrainingManager | None = None,
    asr: AsrService | None = None,
    jobs: JobRunner | None = None,
    align: AlignService | None = None,
    separation: SeparationService | None = None,
    singing: SingingService | None = None,
    speaker: SpeakerService | None = None,
    engines_: dict[str, Engine] | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    audio_io.patch_torchaudio()  # voxcpm's torchaudio.load calls, without a system FFmpeg
    service = service or InferenceService(settings)
    # Every engine; VoxCPM's wraps ``service`` (the status fields and GPU rules use it).
    registry = engines_ or engines.create(settings, voxcpm=VoxCPMEngine(settings, service))
    others = [e for e in registry.values() if e.id != "voxcpm"]
    asr = asr or AsrService(settings)
    align = align or AlignService(settings)
    slicer.vad_device = lambda: asr.device  # VAD of long recordings runs where ASR does
    # Batch sizes and model sharing for the GPU's memory (vram.py).
    gpu_gb, profile = vram.detect(asr.device)
    for device in {asr.device, service.device}:
        vram.apply_limit(device)
    asr.batch_seconds, align.batch_seconds = profile.asr_batch_seconds, profile.align_batch_seconds
    service.denoise_on_cpu = profile.denoise_on_cpu
    logger.info("GPU memory %s GB: profile '%s'", f"{gpu_gb:.1f}" if gpu_gb else "-", profile.name)
    # Compiled on CUDA unless POLYVOX_OPTIMIZE is off (as the TTS model).
    load_separation = functools.partial(
        separation_models, compile=settings.optimize is not False, batch=profile.separation_batch
    )
    separation = separation or SeparationService(lambda: asr.device, load_separation)
    singing = singing or SingingService(settings, lambda: asr.device)
    training = training or TrainingManager(
        settings,
        before_start=lambda: free_gpu_for_training(
            service, settings.train_gpu, *others, asr, align, separation, singing
        ),
        engines_=registry,
    )
    studio = Studio(
        settings=settings,
        service=service,
        registry=registry,
        asr=asr,
        align=align,
        separation=separation,
        singing=singing,
        speaker=speaker or SpeakerService(settings),
        jobs=jobs or JobRunner(),
        training=training,
        gpu_gb=gpu_gb,
        profile=profile,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        studio.training.shutdown()  # saves a checkpoint

    app = FastAPI(title="Polyvox Studio", lifespan=lifespan)

    @app.exception_handler(TrainingConflict)
    def conflict(request: Request, exc: TrainingConflict) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=409)

    for area in (status, tts, loras, datasets, clips, training_routes):
        app.include_router(area.router(studio))
    if WEB_DIST.is_dir():
        app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")
    return app
