"""Speech synthesis, its progress, and ASR of a reference clip."""

from __future__ import annotations

import io
import json
import os
import tempfile
from pathlib import Path
from typing import Annotated

import soundfile as sf
from fastapi import APIRouter, File, Form, HTTPException, Response, UploadFile
from pydantic import ValidationError

from .. import engines, vram
from ..data import audio_io, manifests, references
from ..prep.asr import load_audio
from .context import Studio, checked, found, save_upload
from .schemas import ReferenceClipsIn


def temporary_wav() -> Path:
    fd, name = tempfile.mkstemp(prefix="polyvox_reference_", suffix=".wav")
    os.close(fd)
    return Path(name)


def router(s: Studio) -> APIRouter:
    api = APIRouter()

    def dataset_clips(value: str) -> list[references.Clip]:
        try:
            chosen = ReferenceClipsIn.model_validate_json(value)
        except ValidationError as e:
            raise HTTPException(422, f"Invalid reference_clips: {e.errors()[0]['msg']}") from e
        manifest = found(manifests.resolve, s.settings.data_dir, chosen.dataset)
        return checked(references.clips, manifest, chosen.indices, manifests.legacy_root(s.settings.data_dir))

    @api.post("/api/tts", responses={200: {"content": {"audio/wav": {}}}})
    def tts(
        text: Annotated[str, Form()],
        engine: Annotated[str, Form()] = engines.DEFAULT,
        # The engine's settings (its params in /api/engines) as a JSON object.
        options: Annotated[str | None, Form()] = None,
        control: Annotated[str, Form()] = "",  # style / voice description
        lora_id: Annotated[str | None, Form()] = None,
        prompt_text: Annotated[str | None, Form()] = None,
        # VoxCPM's settings as separate fields (as before engines); override ``options``.
        cfg_value: Annotated[float | None, Form(ge=0.5, le=5.0)] = None,
        inference_timesteps: Annotated[int | None, Form(ge=1, le=50)] = None,
        normalize: Annotated[bool | None, Form()] = None,
        denoise: Annotated[bool | None, Form()] = None,
        seed: Annotated[int | None, Form()] = None,
        reference_audio: Annotated[UploadFile | None, File()] = None,
        # Instead of reference_audio: dataset clips with their transcripts, as JSON (ReferenceClipsIn).
        reference_clips: Annotated[str | None, Form()] = None,
    ) -> Response:
        tts_engine = s.registry.get(engine)
        if tts_engine is None:
            raise HTTPException(404, f"Unknown engine '{engine}'.")
        if reason := tts_engine.unavailable():
            raise HTTPException(503, f"{tts_engine.name} can't run here: {reason}")
        try:
            values = json.loads(options) if options else {}
            if not isinstance(values, dict):
                raise ValueError("options must be a JSON object.")
        except ValueError as e:
            raise HTTPException(422, f"Invalid options: {e}") from e
        legacy = dict(
            cfg_value=cfg_value, inference_timesteps=inference_timesteps, normalize=normalize, denoise=denoise
        )
        values |= {k: v for k, v in legacy.items() if v is not None}
        clips = dataset_clips(reference_clips) if reference_clips else []
        if clips and reference_audio:
            raise HTTPException(422, "Give reference audio or dataset clips, not both.")
        if s.profile.exclusive and (running := s.jobs.running("import") or s.jobs.running("check")):
            raise HTTPException(409, f"This GPU fits one model group at a time: wait for “{running.title}”.")
        s.make_room_for_tts(tts_engine)
        temporary: list[Path] = []  # removed afterwards (never the dataset's own clips)
        ref_path, extra = None, []
        try:
            if clips and tts_engine.capabilities.several_references:
                (first, *rest) = clips
                ref_path, prompt_text, extra = first.audio, first.text, [c.audio for c in rest]
            elif clips:  # one reference made of them all: several clips clone a voice better than one
                ref_path = temporary_wav()
                temporary.append(ref_path)
                prompt_text = references.join(clips, ref_path)
            elif reference_audio:
                # As WAV when soundfile can't read it (e.g. m4a): voxcpm can't without a system FFmpeg.
                ref_path = audio_io.to_wav(save_upload(reference_audio))
                temporary.append(ref_path)
            result = tts_engine.synthesize(
                engines.SynthesisRequest(
                    text=text,
                    style=control,
                    lora_id=lora_id or None,
                    reference_wav=ref_path,
                    prompt_text=prompt_text,
                    extra_references=extra,
                    seed=seed,
                    options=values,
                )
            )
        except KeyError as e:
            raise HTTPException(404, str(e.args[0])) from e
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        except Exception as e:
            if not vram.is_oom(e):
                raise
            vram.free_cache()
            raise HTTPException(
                507,
                "Out of GPU memory. Speech synthesis needs about 8 GB; free the GPU (other programs, Unload "
                "models) or try a shorter text.",
            ) from e
        finally:
            for path in temporary:
                path.unlink(missing_ok=True)

        buf = io.BytesIO()
        sf.write(buf, result.audio, result.sample_rate, format="WAV")
        headers = {"X-Seed": str(result.seed)} if result.seed is not None else {}
        return Response(buf.getvalue(), media_type="audio/wav", headers=headers)

    @api.get("/api/tts/progress")
    def tts_progress() -> dict:
        """The running synthesis: phase (idle | loading | preparing | compiling |
        generating), step and expected steps (an estimate; the text decides when
        it stops), attempt (VoxCPM retries a run-away generation) and elapsed seconds."""
        for engine in s.registry.values():
            if (progress := engine.progress())["phase"] != "idle":
                return progress
        return s.service.progress_public()

    @api.post("/api/asr")
    def transcribe(
        audio: Annotated[UploadFile, File()],
        language: Annotated[str | None, Form()] = None,
        prompt: Annotated[str | None, Form()] = None,
    ) -> dict:
        s.make_room_for_prep()
        path = save_upload(audio)
        try:
            (result,) = s.asr.transcribe([load_audio(path)], language=language or None, prompt=prompt or None)
        except (ValueError, RuntimeError) as e:
            raise HTTPException(422, str(e)) from e
        finally:
            path.unlink(missing_ok=True)
        return {"text": result.text, "language": result.language}

    return api
