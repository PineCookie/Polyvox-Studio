"""Voice conversion (the Convert tab): songs or vocals sung again in an RVC voice, as a job;
importing RVC WebUI voices and building their index."""

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

from .. import engines, lora_library
from ..data import audio_io, manifests
from ..jobs import Job
from ..prep import importer
from .context import Studio, found, save_upload
from .schemas import JobOut

VOICE_NAME = r"^[^\W_][\w.-]{0,79}$"  # as run and package names
SONG_RATE = 44_100  # separation runs at this rate; covers are written at it
RESULTS = Path(tempfile.gettempdir()) / "polyvox-convert"  # one folder per job; older ones pruned
KEEP_RESULTS = 8
FILES = ("cover", "vocals_converted", "instrumental", "vocals")


class RvcImportIn(BaseModel):
    name: str
    path: str  # the voice's .pth on this computer
    index_dataset: str | None = None  # build its index from a dataset's clips
    index_logs: str | None = None  # or from an RVC WebUI logs folder (its total_fea.npy)


class RvcIndexIn(BaseModel):
    id: str
    index_dataset: str | None = None
    index_logs: str | None = None


def prune_results() -> None:
    if not RESULTS.is_dir():
        return
    folders = sorted((p for p in RESULTS.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in folders[KEEP_RESULTS:]:
        shutil.rmtree(old, ignore_errors=True)


def remix(instrumental: np.ndarray, vocals: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """(2, n) instrumental + mono converted vocals at the level of ``reference`` (the vocals they
    replace: RVC sings at its training level, whatever the input's); peaks kept below 0.99."""
    n = min(instrumental.shape[1], len(vocals))
    gain = np.sqrt(np.mean(reference[:n] ** 2) / max(float(np.mean(vocals[:n] ** 2)), 1e-12))
    mix = instrumental[:, :n] + gain * vocals[None, :n]
    return mix * min(1.0, 0.99 / max(float(np.abs(mix).max()), 1e-12))


def shift_key(audio: np.ndarray, semitones: int) -> np.ndarray:
    """(channels, n) moved by ``semitones`` (librosa's phase vocoder), same length."""
    import librosa

    return np.stack([librosa.effects.pitch_shift(ch, sr=SONG_RATE, n_steps=semitones) for ch in audio])


def router(s: Studio) -> APIRouter:
    api = APIRouter()
    settings = s.settings

    def rvc():
        engine = s.registry.get("rvc")
        if engine is None:
            raise HTTPException(404, "RVC isn't available here.")
        if reason := engine.unavailable():
            raise HTTPException(503, f"RVC can't run here: {reason}")
        return engine

    def dataset_clips(dataset: str) -> list[Path]:
        manifest = found(manifests.resolve, settings.data_dir, dataset)
        entries = manifests.training_entries(manifest)[0]
        if not entries:
            raise HTTPException(422, f"'{dataset}' has no clips to build an index from.")
        return [Path(e["audio"]) for e in entries]

    def build_index(engine, lora_id: str, dataset: str | None, logs: str | None) -> int | None:
        from ..engines.rvc import voices

        if dataset:
            clips = dataset_clips(dataset)
            s.make_room_for_tts(engine)
            return engine.build_index(lora_id, clips)
        if logs:
            path = Path(importer.clean_path(logs))
            if not (path / "total_fea.npy").is_file() and not (path.is_file() and path.suffix == ".npy"):
                raise HTTPException(422, "That folder has no total_fea.npy (an RVC WebUI logs/<run> folder has one).")
            return engine.save_index_from(lora_id, voices.webui_features(path))
        return None

    @api.post("/api/rvc/import")
    def import_voice(body: RvcImportIn) -> dict:
        """An RVC WebUI voice (.pth, v2, 40 kHz), its index rebuilt from a dataset or a WebUI
        logs folder (the WebUI's faiss .index isn't read)."""
        from ..engines.rvc import voices

        engine = rvc()
        name = body.name.strip()
        if not re.match(VOICE_NAME, name):
            raise HTTPException(422, "Use letters, digits, '.', '-' or '_' for the name (up to 80).")
        folder = lora_library.engine_dir(settings.lora_dir, voices.ENGINE) / name
        if folder.exists():
            raise HTTPException(409, f"A voice named '{name}' already exists.")
        try:
            voices.import_webui(folder, Path(importer.clean_path(body.path)))
        except (ValueError, OSError) as e:
            raise HTTPException(422, str(e)) from e
        lora_id = f"{voices.ENGINE}/{name}"
        try:
            rows = build_index(engine, lora_id, body.index_dataset, body.index_logs)
        except (ValueError, OSError) as e:  # the voice stays; its index can be built later
            raise HTTPException(422, f"Imported '{name}', but its index failed: {e}") from e
        return {"id": lora_id, "engine": voices.ENGINE, "index_rows": rows}

    @api.post("/api/rvc/index")
    def rebuild_index(body: RvcIndexIn) -> dict:
        """A voice's index (its training features) built again from a dataset or WebUI logs folder."""
        engine = rvc()
        if not body.index_dataset and not body.index_logs:
            raise HTTPException(422, "Give a dataset or an RVC WebUI logs folder.")
        try:
            rows = build_index(engine, body.id, body.index_dataset, body.index_logs)
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
        shift_instrumental: Annotated[bool, Form()] = True,  # a key change moves the instrumental too
        seed: Annotated[int | None, Form()] = None,
    ) -> JobOut:
        """Converts a recording in a job; its result lists the files (GET /api/convert/<job>/<file>)."""
        engine = rvc()
        try:
            values = json.loads(options) if options else {}
            engine.checkpoint(voice)
            opts = engine.options(values)
        except KeyError as e:
            raise HTTPException(404, str(e)) from e
        except (ValueError, TypeError) as e:
            raise HTTPException(422, str(e)) from e
        models = s.cleanup_models()
        needed = [m for m, on in (("separate", separate), ("dereverb", dereverb)) if on]
        if missing := [m for m in needed if not models.get(m)]:
            name = {"separate": "vocal separation", "dereverb": "de-reverb"}[missing[0]]
            raise HTTPException(422, f"The {name} model isn't downloaded (Models).")
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
            lead = vocals.mean(0)
            job.progress("converting", 0, 1)
            s.make_room_for_tts(engine)
            result = engine.convert(
                engines.ConvertRequest(
                    lead, SONG_RATE, voice, opts, seed, on_progress=lambda d, t: job.progress("converting", d, t)
                )
            )
            converted = audio_io.resample(result.audio[None].astype(np.float32), result.sample_rate, SONG_RATE)[0]
            job.progress("mixing", 0, 1)
            files = {"vocals_converted": converted[None], "vocals": vocals}
            if instrumental is not None:
                if opts["shift"] and shift_instrumental:
                    instrumental = shift_key(instrumental, int(opts["shift"]))
                files["instrumental"] = instrumental
                files["cover"] = remix(instrumental, converted, lead)
            for name, audio_data in files.items():
                sf.write(out / f"{name}.wav", audio_data.T, SONG_RATE)
            return {
                "files": [n for n in FILES if n in files],
                "seconds": round(stereo.shape[1] / SONG_RATE, 1),
                "took": round(time.time() - started, 1),
                "seed": result.seed,
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
