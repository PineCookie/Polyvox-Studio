"""What the routes share: the services and the rules for who may use the GPU."""

from __future__ import annotations

import dataclasses
import importlib.metadata
import logging
import os
import shutil
import tempfile
from pathlib import Path

from fastapi import HTTPException, UploadFile

from .. import models, sharing, timing, vram
from ..config import Settings
from ..data import manifests
from ..engines import Engine
from ..engines.voxcpm.inference import InferenceService
from ..jobs import JobRunner
from ..prep import checks
from ..prep import separation as separation_mod
from ..prep.aligner import AlignService
from ..prep.asr import AsrService, load_audio
from ..prep.separation import SeparationService
from ..prep.singing import SingingService
from ..speaker import SpeakerService
from ..training import TrainingManager

logger = logging.getLogger(__name__)


def cuda_index(device: str) -> int | None:
    if not device.startswith("cuda"):
        return None
    index = device.partition(":")[2]
    return int(index) if index else 0


def same_gpu(a: str, b: str) -> bool:
    return a.startswith("cuda") and b.startswith("cuda") and cuda_index(a) == cuda_index(b)


def free_gpu_for_training(
    service: InferenceService,
    train_gpu: int | None,
    *helpers: Engine | AsrService | AlignService | SeparationService | SingingService,
) -> None:
    """Unload the models that sit on the GPU a training job will use (``service``:
    VoxCPM's; other engines among the helpers)."""
    if cuda_index(service.device) == (train_gpu or 0):
        service.unload()
    for helper in helpers:
        if helper.loaded and cuda_index(helper.device) == (train_gpu or 0):
            helper.unload()


@dataclasses.dataclass
class Studio:
    settings: Settings
    service: InferenceService  # VoxCPM's (status fields predate engines)
    registry: dict[str, Engine]
    asr: AsrService
    align: AlignService
    separation: SeparationService
    singing: SingingService
    speaker: SpeakerService
    jobs: JobRunner
    training: TrainingManager
    gpu_gb: float | None
    profile: vram.Profile

    def cleanup_models(self) -> dict[str, str]:
        """Import clean-up option -> its RoFormer checkpoint ("" = not available). Looked up
        each time: a model downloaded meanwhile is used."""
        return {
            "separate": self.settings.model_path("vocals_model"),
            "dereverb": self.settings.model_path("dereverb_model"),
            "denoise": self.settings.model_path("denoise_model"),
        }

    def status(self) -> dict:
        settings = self.settings
        # The hub id lets the UI flag LoRAs trained on a different base model.
        return {
            **self.service.status(),
            # Engines with a model in memory (one per GPU at a time).
            "loaded_engines": [e.id for e in self.registry.values() if e.loaded],
            "model_id": sharing.hub_id(settings.model_path("model")),
            # The PyTorch build (e.g. 2.14.1+cu130): which of run.bat / run.sh's variants runs.
            "torch": importlib.metadata.version("torch"),
            "asr_loaded": self.asr.loaded,
            "aligner_loaded": self.align.loaded,
            "separation_loaded": self.separation.loaded,
            "singing_loaded": self.singing.loaded,
            # Import clean-up options whose model is configured.
            "cleanup": [k for k, v in self.cleanup_models().items() if v],
            # The GPU's memory and the settings picked for it (vram.Profile).
            "gpu": {"total_gb": round(self.gpu_gb, 1) if self.gpu_gb else None, **self.profile.public()},
            # Not in the models folder (loading would download it to the Hugging Face cache).
            "missing_required": [
                s.name
                for s in models.MODELS
                if s.required and not settings.configured(s.setting) and not s.downloaded(settings.models_dir)
            ],
        }

    def unload_all(self) -> None:
        # Frees all GPU memory the server holds: TTS engines, ASR and the aligner.
        for engine in self.registry.values():
            engine.unload()
        self.asr.unload()
        self.align.unload()
        self.separation.unload()
        self.singing.unload()

    # ------------------------------------------------------------ GPU rules
    def make_room_for_prep(self) -> None:
        """On a card that fits one model group (vram.Profile.exclusive), unload TTS
        before ASR / alignment / separation work on its GPU."""
        if self.profile.exclusive:
            for engine in self.registry.values():
                if engine.loaded and same_gpu(engine.device, self.asr.device):
                    engine.unload()

    def make_room_for_tts(self, engine: Engine) -> None:
        # One TTS engine per GPU: each takes 5-8 GB.
        for other in self.registry.values():
            if other is not engine and other.loaded and same_gpu(other.device, engine.device):
                other.unload()
        if self.profile.exclusive:
            for helper in (self.asr, self.align, self.separation, self.singing):
                if helper.loaded and same_gpu(helper.device, engine.device):
                    helper.unload()

    def take_gpu(self, owner) -> None:
        """On cards that fit one preparation model at a time (Profile.swap_prep_models),
        park the others in CPU memory before ``owner`` runs."""
        if self.profile.swap_prep_models:
            for helper in (self.asr, self.align, self.separation, self.singing):
                if helper is not owner:
                    helper.park()

    def gpu_free(self) -> bool:
        """ASR and alignment can run: no training on their GPU."""
        return not (self.training.active_run and cuda_index(self.asr.device) == (self.settings.train_gpu or 0))

    def refuse_if_busy(self, gpu: bool) -> None:
        """One preparation job at a time; GPU stages wait for training on their GPU."""
        jobs = self.jobs
        if running := next((j for k in ("import", "step", "check", "normalize") if (j := jobs.running(k))), None):
            raise HTTPException(409, f"Wait for “{running.title}” to finish.")
        if gpu and not self.gpu_free():
            raise HTTPException(
                409,
                f"Run '{self.training.active_run}' is training on the GPU that ASR and alignment use. Wait for it, "
                "or set POLYVOX_ASR_DEVICE to another GPU.",
            )
        if gpu:
            self.make_room_for_prep()

    # ------------------------------------------------------------ prep models, taking turns
    def aligned(self, wavs: list, texts: list[str], kind: str) -> list:
        self.take_gpu(self.align)
        return self.align.align(wavs, texts, kind)

    def sung(self, wavs: list) -> list:
        self.take_gpu(self.singing)
        return self.singing.scores(wavs)

    def voices(self, wavs: list):
        """Speaker embeddings for the dataset check, on the prep GPU (CAM++ is small)."""
        return self.speaker.embed_wavs(wavs, self.asr.device)

    def separated(self, chain: separation_mod.Chain) -> separation_mod.Chain:
        def process(audio):
            self.take_gpu(self.separation)
            return chain.process(audio)

        return dataclasses.replace(chain, process=process)

    def checker(self, kind: str, compare_asr: bool = False, detect_singing: bool = False) -> checks.Checker:
        def heard_texts(wavs: list) -> list[str]:
            self.take_gpu(self.asr)
            return [r.text for r in self.asr.transcribe(wavs)]

        return checks.make_checker(
            lambda wavs, texts: self.aligned(wavs, texts, kind),
            kind,
            transcribe=heard_texts if compare_asr else None,
            singing=self.sung if detect_singing else None,
        )

    def check_clip_again(self, manifest: Path, index: int, compare_asr: bool | None = None) -> None:
        """Check one clip with the dataset's aligner (Qwen3 when never checked).
        ``compare_asr`` None: compare with ASR if the dataset's checks did; True
        also compares a text ASR made. Flags sung clips if the checks did."""
        saved = manifests.read_sidecar(manifest.parent / manifests.ALIGN_FILE)
        kind = saved.get("aligner") if saved.get("aligner") in ("qwen3", "ctc") else "qwen3"
        clips = saved.get("clips", {}).values()
        compared = any(c.get("asr") for c in clips) if compare_asr is None else compare_asr
        sang = any(c.get("singing") for c in clips)
        self.make_room_for_prep()
        root = manifests.legacy_root(self.settings.data_dir)
        checker_ = self.checker(kind, compared, sang)
        compare = True if compare_asr else None
        checks.check_one(manifest, root, index, checker_, kind, load_audio, compare, self.voices)


def has_checks(manifest: Path) -> bool:
    return manifests.read_sidecar(manifest.parent / manifests.ALIGN_FILE).get("aligner") in ("qwen3", "ctc")


def logged_timings(what: str, times: timing.Timings) -> dict[str, float]:
    """Seconds per stage (timing.Timings.summary), also logged."""
    summary = times.summary()
    logger.info("%s took %s", what, ", ".join(f"{k} {v:.1f} s" for k, v in summary.items()))
    return summary


def found(fn, *args):
    try:
        return fn(*args)
    except KeyError as e:
        raise HTTPException(404, str(e.args[0])) from e


def checked(fn, *args):
    try:
        return found(fn, *args)
    except (ValueError, OSError) as e:
        raise HTTPException(422, str(e)) from e


def save_upload(upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "").suffix or ".wav"
    fd, name = tempfile.mkstemp(prefix="polyvox_upload_", suffix=suffix)
    with os.fdopen(fd, "wb") as out:
        shutil.copyfileobj(upload.file, out)
    return Path(name)
