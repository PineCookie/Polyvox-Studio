"""dots.tts (rednote-hilab / studio-dots-ai, vendor/dots_tts) as an engine.

2B continuous autoregressive TTS: a Qwen2.5 1.5B language model over 48 kHz
audio-VAE latents and a diffusion head; it hears and writes 48 kHz audio, so a
LoRA learns a voice's whole band. Clones from a reference clip, with or without
its transcript. LoRAs: lora.py; training: worker.py.
"""

from __future__ import annotations

import gc
import logging
import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ... import lora_library
from ...config import Settings
from ..base import Capabilities, Engine, Param, ReferenceInput, SynthesisRequest, SynthesisResult
from . import lora as dots_lora

if TYPE_CHECKING:
    from ...training import TrainParams

logger = logging.getLogger(__name__)

PATCHES_PER_TOKEN = 1.6  # first guess of audio patches per text token; refined from finished generations
MIN_PATCHES = 6
LEARN_MIN_TOKENS = 10

# (model path, optimize) -> DotsTtsRuntime, not warmed up yet
RuntimeLoader = Callable[[str, bool], object]


def load_runtime(model: str, optimize: bool):
    """dots.tts's runtime; ``optimize``: torch.compile + CUDA graphs per decode
    step (~4x faster here: 0.19 vs 0.8 s per second of audio). Its warm-up (the
    compilation, minutes with a cold Inductor cache) is left to the caller, which
    must snapshot the model before it (dots_lora.Merger)."""
    from ...vendor.dots_tts.runtime import DotsTtsRuntime

    dots_lora.quiet_logging()
    return DotsTtsRuntime.from_pretrained(model, precision="bfloat16", optimize=optimize, warmup_on_optimize=False)


@dataclass
class Progress:
    phase: str = "idle"  # idle | loading | compiling | preparing | generating
    step: int = 0  # audio patches so far (6.25 per second of audio)
    expected: int | None = None
    attempt: int = 0
    started: float = 0.0


class DotsEngine(Engine):
    id = dots_lora.ENGINE
    name = "dots.tts"
    capabilities = Capabilities(
        sample_rate=48_000,
        training_rate=48_000,
        languages=("zh", "en"),
        reference_audio=True,
        reference_transcript=True,
        style=False,
        tags=False,
        lora_hot_swap=True,
        # One clip: its speaker embedding, and with its transcript also speech to continue.
        reference_inputs=(ReferenceInput("clip", "prompt", "optional"),),
    )
    params = (
        Param("guidance_scale", "float", 1.2, 0.0, 5.0, 0.1),
        Param("num_steps", "int", 10, 1, 50, 1),
        # How strongly the reference clip's speaker embedding steers the voice.
        Param("speaker_scale", "float", 1.5, 0.0, 3.0, 0.1),
        Param("normalize", "bool", False),  # spell out numbers, dates, ... first
    )
    model_setting = "dots_model"
    worker_module = "polyvox.engines.dots.worker"

    def __init__(self, settings: Settings, loader: RuntimeLoader = load_runtime):
        self.settings = settings
        self._loader = loader
        self._lock = threading.Lock()
        self._runtime = None
        self._merger: dots_lora.Merger | None = None
        self._lora: tuple[str, float] | None = None  # (id, weights mtime) merged into the weights
        self._progress = Progress()
        self._patches_per_token = PATCHES_PER_TOKEN

    @property
    def device(self) -> str:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def loaded(self) -> bool:
        return self._runtime is not None

    def unavailable(self) -> str | None:
        import torch

        return None if torch.cuda.is_available() else "it needs a CUDA GPU (bfloat16)."

    def status(self) -> dict:
        return {
            "device": self.device,
            "model": self.settings.model_path(self.model_setting),
            "loaded": self.loaded,
            "optimize": self.optimize,
            "active_lora": self._lora[0] if self._lora else None,
        }

    def progress(self) -> dict:
        p = self._progress
        return {**p.__dict__, "elapsed": round(time.time() - p.started, 1) if p.started else 0.0}

    def synthesize(self, req: SynthesisRequest) -> SynthesisResult:
        from ...vendor.dots_tts.utils import text as dots_text

        text = req.text.strip()
        if not text:
            raise ValueError("Text to synthesize is empty.")
        options = self.options(req.options)
        ckpt = lora_library.resolve(self.settings.lora_dir, req.lora_id) if req.lora_id else None
        if ckpt and ckpt.engine != self.id:
            raise ValueError(f"LoRA '{ckpt.id}' was trained for another model ({ckpt.engine}), not dots.tts.")
        if req.prompt_wav and req.reference_wav and req.prompt_wav != req.reference_wav:
            raise ValueError("dots.tts takes one reference clip: the prompt to continue from is the voice's clip.")

        with self._lock:
            self._progress = Progress(phase="loading" if self._runtime is None else "preparing", started=time.time())
            try:
                runtime = self._ensure_runtime()
                self._apply_lora(ckpt)
                seed = req.seed if req.seed is not None else random.randrange(2**31)
                kwargs = {}
                # One clip gives the speaker embedding and, with its transcript, the prompt.
                if req.prompt_wav:
                    kwargs["prompt_audio_path"] = str(req.prompt_wav)
                    kwargs["prompt_text"] = (req.prompt_text or "").strip()
                elif req.reference_wav:
                    kwargs["prompt_audio_path"] = str(req.reference_wav)
                tokens = len(runtime.model.tokenizer(text)["input_ids"])
                self._progress.expected = max(MIN_PATCHES, round(tokens * self._patches_per_token))
                audio = self._generate(runtime, text, seed, dots_text.detect(text), options, kwargs)
                if tokens >= LEARN_MIN_TOKENS and not kwargs.get("prompt_text"):
                    observed = self._progress.step / tokens
                    self._patches_per_token = 0.7 * self._patches_per_token + 0.3 * observed
                return SynthesisResult(runtime.sample_rate, audio, seed)
            finally:
                self._progress.phase = "idle"

    def _generate(self, runtime, text: str, seed: int, language: str | None, options: dict, kwargs: dict):
        import torch

        model = runtime.model
        stream = type(model)._generate_latents_stream
        progress = self._progress

        def counted(*args, **kw):
            progress.phase = "generating"
            for patch in stream(model, *args, **kw):
                progress.step += 1
                yield patch

        torch.manual_seed(seed)
        model._generate_latents_stream = counted  # this call only: progress per audio patch
        try:
            result = runtime.generate(
                text=text,
                language=language,
                guidance_scale=float(options["guidance_scale"]),
                num_steps=int(options["num_steps"]),
                speaker_scale=float(options["speaker_scale"]),
                normalize_text=bool(options["normalize"]),
                **kwargs,
            )
        finally:
            del model._generate_latents_stream
        return result["audio"].detach().float().cpu().numpy().reshape(-1)

    def unload(self) -> None:
        with self._lock:
            if self._runtime is None:
                return
            self._runtime = self._merger = self._lora = None
            gc.collect()
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @property
    def optimize(self) -> bool:
        return self.device == "cuda" if self.settings.optimize is None else self.settings.optimize

    def _ensure_runtime(self):
        if self._runtime is None:
            model = self.settings.model_path(self.model_setting)
            logger.info("Loading dots.tts from %s (optimize=%s)", model, self.optimize)
            runtime = self._loader(model, self.optimize)
            # Before the warm-up fuses the diffusion head; LoRAs then go into the
            # fused copies in place, so the recorded CUDA graphs stay valid.
            self._merger = dots_lora.Merger(runtime.model, keep_fusion=True)
            self._lora = None
            if self.optimize:
                self._progress.phase = "compiling"
                t0 = time.time()
                runtime.run_warmup()
                logger.info("dots.tts compiled in %.0f s", time.time() - t0)
            self._runtime = runtime
        return self._runtime

    def _apply_lora(self, ckpt: lora_library.LoraCheckpoint | None) -> None:
        key = (ckpt.id, ckpt.weights_mtime) if ckpt else None
        if key == self._lora:
            return
        self._lora = None
        if ckpt is None:
            self._merger.apply(None)
            return
        from safetensors.torch import load_file

        try:
            tensors = load_file(str(ckpt.path / lora_library.WEIGHTS_FILE))
        except (OSError, RuntimeError) as e:
            raise RuntimeError(f"Unable to load LoRA '{ckpt.id}': {e}") from e
        rank, alpha = ckpt.rank, ckpt.alpha
        self._merger.apply(tensors, (alpha / rank) if rank and alpha else 1.0)
        self._lora = key

    def training_config(
        self, params: TrainParams, run_dir: Path, train_manifest: Path, val_manifest: Path | None
    ) -> dict:
        if not (params.enable_lm or params.enable_dit):
            raise ValueError("LoRA needs the language model, the diffusion head or both.")
        return {
            "pretrained_path": self.settings.model_path(self.model_setting),
            "train_manifest": str(train_manifest),
            "val_manifest": str(val_manifest) if val_manifest else "",
            "save_path": str(run_dir / "checkpoints"),
            "num_iters": params.num_iters,
            "batch_size": params.batch_size,
            "grad_accum_steps": params.grad_accum_steps,
            "num_workers": params.num_workers,
            "optimizer": params.optimizer,
            "learning_rate": params.learning_rate,
            "prodigy_steps": params.prodigy_steps,
            "weight_decay": params.weight_decay,
            "warmup_steps": params.warmup_steps,
            "max_grad_norm": params.max_grad_norm,
            "log_interval": params.log_interval,
            "valid_interval": params.valid_interval,
            "save_interval": params.save_interval,
            "lora": {
                "r": params.lora_rank,
                "alpha": params.lora_alpha,
                "dropout": params.lora_dropout,
                "enable_lm": params.enable_lm,
                "enable_dit": params.enable_dit,
            },
        }
