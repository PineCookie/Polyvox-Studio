"""HTTP API and static frontend."""

from __future__ import annotations

import io
import os
import shutil
import tempfile
from pathlib import Path
from typing import Annotated

import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import lora_library
from .config import Settings
from .inference import InferenceService, SynthesisRequest

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
    config: LoraConfigOut | None


def create_app(settings: Settings | None = None, service: InferenceService | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    service = service or InferenceService(settings)
    app = FastAPI(title="VoxCPM Studio")

    @app.get("/api/status")
    def status() -> dict:
        return service.status()

    @app.get("/api/loras")
    def list_loras() -> list[LoraOut]:
        return [
            LoraOut(
                id=c.id,
                base_model=c.base_model,
                config=LoraConfigOut(**c.config.model_dump()) if c.config else None,
            )
            for c in lora_library.scan(settings.lora_dir)
        ]

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
        service.unload()
        return service.status()

    if WEB_DIST.is_dir():
        app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")

    return app


def _save_upload(upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "").suffix or ".wav"
    fd, name = tempfile.mkstemp(prefix="voxcpm_ref_", suffix=suffix)
    with os.fdopen(fd, "wb") as out:
        shutil.copyfileobj(upload.file, out)
    return Path(name)
