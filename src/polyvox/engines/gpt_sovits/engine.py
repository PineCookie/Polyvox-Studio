"""GPT-SoVITS v2Pro / v2ProPlus (RVC-Boss/GPT-SoVITS, vendor/gpt_sovits) as an engine.

A GPT turns text into semantic tokens, continuing a reference clip's; SoVITS
decodes them to 32 kHz audio in the reference's timbre. It always clones a
clip: the request's, else the voice's own default (voices.py). Voices are full
weights (~300 MB), loaded when chosen; the base is v2ProPlus.
"""

from __future__ import annotations

import gc
import logging
import random
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ... import lora_library
from ...config import Settings
from ..base import Capabilities, Engine, Param, SynthesisRequest, SynthesisResult
from . import voices
from .synth import Options, split_text

if TYPE_CHECKING:
    from ...training import TrainParams

logger = logging.getLogger(__name__)

TOKENS_PER_CHAR = 5.5  # first guess of semantic tokens per character; refined from finished generations
MIN_TOKENS = 12
LEARN_MIN_CHARS = 10
BASE_VERSION = "v2ProPlus"

# (base models folder, G2PW folder, device) -> synth.Pipeline
PipelineLoader = Callable[[Path, Path, str], object]


def load_pipeline(base_dir: Path, g2pw_dir: Path, device: str):
    from .synth import Pipeline

    return Pipeline(base_dir, g2pw_dir, device)


@dataclass
class Progress:
    phase: str = "idle"  # idle | loading | preparing | generating
    step: int = 0  # semantic tokens so far (~25 per second of audio)
    expected: int | None = None
    attempt: int = 0
    started: float = 0.0


class GptSovitsEngine(Engine):
    id = voices.ENGINE
    name = "GPT-SoVITS"
    capabilities = Capabilities(
        sample_rate=32_000,
        training_rate=32_000,
        languages=("zh", "en"),
        reference_audio=True,
        reference_transcript=True,
        style=False,
        tags=False,
        lora_hot_swap=False,
        reference_required=True,
        full_finetune=True,
    )
    params = (
        Param("temperature", "float", 1.0, 0.05, 2.0, 0.05),
        Param("top_k", "int", 15, 1, 100, 1),
        Param("top_p", "float", 1.0, 0.05, 1.0, 0.05),
        # Against loops and stutters: penalizes semantic tokens the GPT just made.
        Param("repetition_penalty", "float", 1.35, 1.0, 2.0, 0.05),
        Param("speed", "float", 1.0, 0.6, 1.65, 0.05),
    )
    model_setting = "gsv_model"
    worker_module = "polyvox.engines.gpt_sovits.worker"

    def __init__(self, settings: Settings, loader: PipelineLoader = load_pipeline):
        self.settings = settings
        self._loader = loader
        self._lock = threading.Lock()
        self._pipeline = None
        self._voice = None  # synth.Voice, its key = (checkpoint id or "base", weights mtime)
        self._references: dict[object, object] = {}  # default references of the loaded voice, by key
        self._progress = Progress()
        self._tokens_per_char = TOKENS_PER_CHAR

    @property
    def device(self) -> str:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def loaded(self) -> bool:
        return self._pipeline is not None

    def unavailable(self) -> str | None:
        if not self.settings.model_path("gsv_model") or not self.settings.model_path("gsv_g2pw_model"):
            return "download its models first (Models: GPT-SoVITS and G2PW)."
        return None

    def status(self) -> dict:
        return {
            "device": self.device,
            "model": self.settings.model_path(self.model_setting),
            "loaded": self.loaded,
            "active_lora": self._voice.key[0] if self._voice is not None else None,
        }

    def progress(self) -> dict:
        p = self._progress
        return {**p.__dict__, "elapsed": round(time.time() - p.started, 1) if p.started else 0.0}

    def synthesize(self, req: SynthesisRequest) -> SynthesisResult:
        text = req.text.strip()
        if not split_text(text):
            raise ValueError("Text to synthesize is empty.")
        if reason := self.unavailable():
            raise ValueError(f"GPT-SoVITS can't run yet: {reason}")
        options = Options(**self.options(req.options))
        ckpt = lora_library.resolve(self.settings.lora_dir, req.lora_id) if req.lora_id else None
        if ckpt and ckpt.engine != self.id:
            raise ValueError(f"'{ckpt.id}' was trained for another model ({ckpt.engine}), not GPT-SoVITS.")
        default_ref = voices.read_reference(ckpt.path) if ckpt else None
        if req.reference_wav is None and default_ref is None:
            raise ValueError("GPT-SoVITS clones a reference clip: add one (3-10 s), or choose a voice that has one.")

        with self._lock:
            self._progress = Progress(phase="loading" if self._pipeline is None else "preparing", started=time.time())
            try:
                pipeline = self._ensure_pipeline()
                voice = self._ensure_voice(pipeline, ckpt)
                if req.reference_wav is not None:
                    ref = pipeline.reference(voice, req.reference_wav, req.prompt_text)
                else:
                    ref = self._default_reference(pipeline, voice, *default_ref)
                seed = req.seed if req.seed is not None else random.randrange(2**31)
                chars = len(re.sub(r"\W", "", text))
                self._progress.expected = max(MIN_TOKENS, round(chars * self._tokens_per_char))
                self._progress.phase = "generating"
                audio = pipeline.synthesize(voice, ref, text, options, seed, on_token=self._count)
                if chars >= LEARN_MIN_CHARS and options.speed == 1.0:
                    observed = self._progress.step / chars
                    self._tokens_per_char = 0.7 * self._tokens_per_char + 0.3 * observed
                return SynthesisResult(voice.sample_rate, audio, seed)
            finally:
                self._progress.phase = "idle"

    def _count(self) -> None:
        self._progress.step += 1

    def _ensure_pipeline(self):
        if self._pipeline is None:
            base = Path(self.settings.model_path("gsv_model"))
            g2pw = Path(self.settings.model_path("gsv_g2pw_model"))
            t0 = time.time()
            self._pipeline = self._loader(base, g2pw, self.device)
            logger.info("GPT-SoVITS front ends loaded in %.1f s", time.time() - t0)
        return self._pipeline

    def _ensure_voice(self, pipeline, ckpt: lora_library.LoraCheckpoint | None):
        if ckpt is None:
            key = ("base", BASE_VERSION)
            sovits, gpt = pipeline.base_weights(BASE_VERSION)
        else:
            key = (ckpt.id, ckpt.weights_mtime)
            sovits, gpt = ckpt.path / voices.SOVITS_FILE, ckpt.path / voices.GPT_FILE
            if not gpt.is_file():  # trained SoVITS only so far
                gpt = pipeline.base_weights(ckpt.version or BASE_VERSION)[1]
        if self._voice is None or self._voice.key != key:
            self._voice = self._references = None
            gc.collect()
            self._voice = pipeline.load_voice(sovits, gpt, key=key)
            self._references = {}
        return self._voice

    def _default_reference(self, pipeline, voice, wav: Path, text: str):
        key = (str(wav), wav.stat().st_mtime, text)
        if key not in self._references:
            self._references[key] = pipeline.reference(voice, wav, text, key=key)
        return self._references[key]

    def unload(self) -> None:
        with self._lock:
            if self._pipeline is None:
                return
            self._pipeline = self._voice = None
            self._references = {}
            gc.collect()
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def training_config(
        self, params: TrainParams, run_dir: Path, train_manifest: Path, val_manifest: Path | None
    ) -> dict:
        """Full fine-tuning in two stages (worker.py): sovits_epochs at
        batch_size and learning_rate, then gpt_epochs at gpt_batch_size (the
        GPT's optimizer is upstream's); num_iters follows from them (estimated
        here, exact once the worker has the clips' features). The run's
        reference clip (samples, checkpoints) is the dataset's best one."""
        if reason := self.unavailable():
            raise ValueError(f"GPT-SoVITS can't train yet: {reason}")
        if not params.sovits_epochs and not params.gpt_epochs:
            raise ValueError("Train SoVITS, the GPT or both: set their epochs.")
        from ...data import manifests
        from .worker import plan_steps

        dataset = manifests.resolve(self.settings.data_dir, params.dataset)
        audio, text = voices.pick_reference(dataset, manifests.legacy_root(self.settings.data_dir))
        clips = len(manifests.read_manifest(train_manifest)[0])
        sovits_steps, gpt_steps = plan_steps(
            clips, params.batch_size, params.sovits_epochs, clips, params.gpt_batch_size, params.gpt_epochs
        )
        return {
            "version": BASE_VERSION,
            "pretrained_path": self.settings.model_path("gsv_model"),
            "g2pw_path": self.settings.model_path("gsv_g2pw_model"),
            "train_manifest": str(train_manifest),
            "val_manifest": str(val_manifest) if val_manifest else "",
            "save_path": str(run_dir / "checkpoints"),
            "num_iters": sovits_steps + gpt_steps,
            "sovits_epochs": params.sovits_epochs,
            "batch_size": params.batch_size,
            "learning_rate": params.learning_rate,
            "gpt_epochs": params.gpt_epochs,
            "gpt_batch_size": params.gpt_batch_size,
            "log_interval": params.log_interval,
            "reference": {"audio": str(audio), "text": text, "source": f"{params.dataset}: {audio.name}"},
        }
