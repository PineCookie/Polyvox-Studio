"""Voice conversion (the Convert tab): songs or vocals sung again in a voice, as a job; rebuilding
a voice's index."""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Annotated

import numpy as np
import soundfile as sf
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .. import engines
from ..data import audio_io
from ..jobs import Job
from ..prep.karaoke import Karaoke
from .context import Studio, save_upload
from .schemas import JobOut

SONG_RATE = 44_100  # separation runs at this rate; covers are written at it
RESULTS = Path(tempfile.gettempdir()) / "polyvox-convert"  # one folder per job; older ones pruned
KEEP_RESULTS = 8
FILES = ("cover", "vocals_converted", "instrumental", "vocals", "lead", "backing")
BACKING = ("keep", "convert")  # the backing vocals of a split: as sung, or converted on their own


class IndexIn(BaseModel):
    id: str
    index_dataset: str | None = None
    index_logs: str | None = None


def prune_results() -> None:
    if not RESULTS.is_dir():
        return
    folders = sorted((p for p in RESULTS.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in folders[KEEP_RESULTS:]:
        shutil.rmtree(old, ignore_errors=True)


def level_matched(converted: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Converted audio at the level of ``reference`` (what it replaces: RVC sings at its training
    level, whatever the input's), cut or padded to its length."""
    n = reference.shape[-1]
    converted = np.pad(converted[..., :n], [(0, 0)] * (converted.ndim - 1) + [(0, n - min(n, converted.shape[-1]))])
    gain = np.sqrt(np.mean(reference**2) / max(float(np.mean(converted**2)), 1e-12))
    return (gain * converted).astype(np.float32)


def remix(*tracks: np.ndarray) -> np.ndarray:
    """Tracks ((2, n) or mono (n,)) added up as (2, n), peaks kept below 0.99."""
    n = min(t.shape[-1] for t in tracks)
    mix = sum(np.broadcast_to(t[..., :n], (2, n)) for t in tracks)
    return mix * min(1.0, 0.99 / max(float(np.abs(mix).max()), 1e-12))


def shift_key(audio: np.ndarray, semitones: int) -> np.ndarray:
    """(channels, n) moved by ``semitones`` (librosa's phase vocoder), same length."""
    import librosa

    return np.stack([librosa.effects.pitch_shift(ch, sr=SONG_RATE, n_steps=semitones) for ch in audio])


def router(s: Studio) -> APIRouter:
    api = APIRouter()
    settings = s.settings
    karaoke = Karaoke()  # on the CPU; kept loaded between songs (a small model)

    def rvc():
        """The engine that converts recordings (capabilities.conversion)."""
        engine = next((e for e in s.registry.values() if e.capabilities.conversion), None)
        if engine is None:
            raise HTTPException(404, "No voice conversion here.")
        if reason := engine.unavailable():
            raise HTTPException(503, f"{engine.name} can't run here: {reason}")
        return engine

    @api.post("/api/convert/index")
    def rebuild_index(body: IndexIn) -> dict:
        """A voice's index (its training features) built again from a dataset or RVC WebUI logs folder."""
        engine = rvc()
        if not body.index_dataset and not body.index_logs:
            raise HTTPException(422, "Give a dataset or an RVC WebUI logs folder.")
        try:
            rows = engine.index_from(body.id, body.index_dataset, body.index_logs, lambda: s.make_room_for_tts(engine))
        except KeyError as e:
            raise HTTPException(404, str(e)) from e
        except (ValueError, OSError) as e:
            raise HTTPException(422, str(e)) from e
        return {"id": body.id, "index_rows": rows}

    @api.post("/api/convert", status_code=202)
    def convert(
        audio: Annotated[UploadFile, File()],
        voice: Annotated[str, Form()],  # an RVC voice's id
        options: Annotated[str | None, Form()] = None,  # the engine's params as a JSON object
        separate: Annotated[bool, Form()] = True,  # a song: take the vocals out first, remix after
        dereverb: Annotated[bool, Form()] = True,
        # Split lead and backing vocals: only the lead is converted unless backing is "convert".
        split: Annotated[bool, Form()] = False,
        backing: Annotated[str, Form()] = "keep",
        shift_instrumental: Annotated[bool, Form()] = True,  # a key change moves the instrumental too
        seed: Annotated[int | None, Form()] = None,
    ) -> JobOut:
        """Converts a recording in a job (steps: separate, de-reverb, split, convert, mix); its
        result lists the files (GET /api/convert/<job>/<file>)."""
        engine = rvc()
        try:
            values = json.loads(options) if options else {}
            engine.checkpoint(voice)
            opts = engine.options(values)
        except KeyError as e:
            raise HTTPException(404, str(e)) from e
        except (ValueError, TypeError) as e:
            raise HTTPException(422, str(e)) from e
        if backing not in BACKING:
            raise HTTPException(422, f"backing is one of {', '.join(BACKING)}.")
        models = {**s.cleanup_models(), "split": settings.model_path("karaoke_model")}
        needed = [m for m, on in (("separate", separate), ("dereverb", dereverb), ("split", split)) if on]
        if missing := [m for m in needed if not models.get(m)]:
            name = {"separate": "vocal separation", "dereverb": "de-reverb", "split": "karaoke (lead / backing)"}
            raise HTTPException(422, f"The {name[missing[0]]} model isn't downloaded (Models).")
        s.refuse_if_busy(gpu=False)
        if running := s.jobs.running("convert"):
            raise HTTPException(409, f"Wait for “{running.title}” to finish.")
        path = save_upload(audio)
        title = Path(audio.filename or "audio").stem[:60]

        def run(job: Job) -> dict:
            try:
                return work(job)
            finally:
                path.unlink(missing_ok=True)

        def work(job: Job) -> dict:
            started = time.time()
            prune_results()
            out = RESULTS / job.id
            out.mkdir(parents=True, exist_ok=True)
            job.progress("reading", 0, 1)
            data, rate = audio_io.read(path)
            stereo = audio_io.resample(np.vstack([data, data]) if data.shape[0] == 1 else data[:2], rate, SONG_RATE)
            stereo = np.ascontiguousarray(stereo, dtype=np.float32)
            instrumental = None
            vocals = stereo
            if separate:
                job.progress("separating", 0, 1)
                s.make_room_for_prep()
                vocals = s.separated(s.separation.chain([models["separate"]])).process(stereo)
                instrumental = stereo - vocals
            if dereverb:
                job.progress("dereverb", 0, 1)
                s.make_room_for_prep()
                vocals = s.separated(s.separation.chain([models["dereverb"]])).process(vocals)
            s.separation.unload()
            files = {"vocals": vocals}
            lead, back = vocals, None
            if split:
                job.progress("splitting", 0, 1)
                lead, back = karaoke.split(models["split"], vocals, lambda d, t: job.progress("splitting", d, t))
                files |= {"lead": lead, "backing": back}
            s.make_room_for_tts(engine)

            def converted(part: np.ndarray, phase: str) -> tuple[np.ndarray, int]:
                """A (2, n) part sung in the voice (mono, at the part's level and length) and the seed."""
                job.progress(phase, 0, 1)
                mono = part.mean(0)
                request = engines.ConvertRequest(
                    mono, SONG_RATE, voice, opts, seed, on_progress=lambda d, t: job.progress(phase, d, t)
                )
                result = engine.convert(request)
                audio = audio_io.resample(result.audio[None].astype(np.float32), result.sample_rate, SONG_RATE)[0]
                return level_matched(audio, mono), result.seed

            sung, used_seed = converted(lead, "converting")
            parts = [sung]
            if back is not None:
                parts.append(converted(back, "converting_backing")[0] if backing == "convert" else back)
            job.progress("mixing", 0, 1)
            files["vocals_converted"] = remix(*parts)
            if instrumental is not None:
                if opts["shift"] and shift_instrumental:
                    instrumental = shift_key(instrumental, int(opts["shift"]))
                files["instrumental"] = instrumental
                files["cover"] = remix(instrumental, *parts)
            for name, audio_data in files.items():
                sf.write(out / f"{name}.wav", audio_data.T, SONG_RATE)
            return {
                "files": [n for n in FILES if n in files],
                "seconds": round(stereo.shape[1] / SONG_RATE, 1),
                "took": round(time.time() - started, 1),
                "seed": used_seed,
                "voice": voice,
            }

        return JobOut(**s.jobs.start("convert", f"Convert {title}", run, voice, "convert").public())

    @api.get("/api/convert/{job_id}/{name}")
    def result_file(job_id: str, name: str) -> FileResponse:
        if name not in FILES or not re.fullmatch(r"[\w-]+", job_id):
            raise HTTPException(404, "No such file.")
        path = RESULTS / job_id / f"{name}.wav"
        if not path.is_file():
            raise HTTPException(404, "No such file (results are kept for the last few conversions).")
        return FileResponse(path, media_type="audio/wav", filename=f"{name}.wav")

    return api
