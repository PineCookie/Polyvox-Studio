"""Model lifecycle and synthesis with LoRA hot-swapping."""

from __future__ import annotations

import gc
import logging
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import lora_library, vram
from .config import Settings
from .lora_library import LoraCheckpoint

logger = logging.getLogger(__name__)

ENGINE = "voxcpm"  # checkpoints of other engines are not loaded here

# (settings, device, lora_config, optimize) -> voxcpm.VoxCPM
ModelLoader = Callable[..., object]

# VoxCPM's generation loop is `for i in tqdm(range(max_len))`, one step per audio
# patch, max_len = int(text tokens * RATIO_LIMIT + 10) (its retry_badcase_ratio_threshold
# default); it stops much earlier, after about STEPS_PER_TOKEN steps per token.
RATIO_LIMIT = 6.0
STEPS_PER_TOKEN = 1.1  # first guess; refined from finished generations
MIN_STEPS = 12  # even "好的。" takes ~13 steps
LEARN_MIN_TOKENS = 15  # shorter texts don't tell the pace (their fixed overhead dominates)

_step_hook: Callable[[int, int], None] | None = None  # (step index, max_len) of the running generation


def _tracked(iterable, *args, **kwargs):
    """Stands in for tqdm in voxcpm's generation loop: reports each step."""
    total = len(iterable) if hasattr(iterable, "__len__") else 0
    for i, item in enumerate(iterable):
        if _step_hook is not None:
            _step_hook(i, total)
        yield item


def _hook_generation_loop() -> None:
    try:
        from voxcpm.model import voxcpm as v1
        from voxcpm.model import voxcpm2 as v2
    except ImportError:  # pragma: no cover
        return
    v1.tqdm = v2.tqdm = _tracked


@dataclass
class Progress:
    phase: str = "idle"  # idle | loading | preparing | compiling | generating
    step: int = 0
    expected: int | None = None  # steps this text should take (estimate)
    attempt: int = 0  # voxcpm retries a generation that ran far too long
    started: float = 0.0  # time.time() of the request


def load_voxcpm(settings: Settings, device: str, lora_config, optimize: bool):
    import voxcpm

    model = voxcpm.VoxCPM.from_pretrained(
        settings.model_path("model"),
        load_denoiser=settings.load_denoiser,
        zipenhancer_model_id=settings.model_path("denoiser_model"),
        # Compile after loading so a LoRA layout change never reuses an
        # incompatible torch.compile graph.
        optimize=False,
        device=device,
        lora_config=lora_config,
    )
    if optimize:
        model.tts_model.optimize(disable=False)
    return model


# LoRA hot-swap without rebuilding the model.
#
# voxcpm sizes each LoRALinear's lora_A (r x in) and lora_B (out x r) at build
# time, so a different rank used to mean a new model and a new torch.compile.
# Instead the model is built once with a layout covering every checkpoint
# (highest rank, union of targets) and a smaller LoRA is loaded zero-padded:
# B[:, :r] @ A[:r] equals its own B @ A, and the padding contributes nothing.
# Each LoRA's alpha / r goes into the per-layer `scaling` buffer, which voxcpm
# keeps as a tensor so changing it in place does not recompile.


def _lora_layers(tts_model):
    """(state-dict style name, LoRALinear) for every LoRA layer in the model."""
    from voxcpm.modules.layers.lora import LoRALinear

    for name, module in tts_model.named_modules():
        if isinstance(module, LoRALinear) and module.lora_A is not None:
            # torch.compile wraps modules as `<name>._orig_mod`.
            yield name.replace("._orig_mod", "").replace("_orig_mod.", ""), module


def load_lora_padded(tts_model, ckpt: LoraCheckpoint) -> float:
    """Copy a checkpoint's LoRA weights into the model; returns its alpha / r.

    Raises ValueError, leaving LoRA weights zeroed, when the checkpoint does not
    fit the model (another base model, or a layout the model lacks).
    """
    import torch
    from safetensors.torch import load_file

    state = load_file(str(ckpt.path / lora_library.WEIGHTS_FILE))
    used = 0
    with torch.no_grad():
        for name, layer in _lora_layers(tts_model):
            layer.lora_A.zero_()
            layer.lora_B.zero_()
            a, b = state.get(f"{name}.lora_A"), state.get(f"{name}.lora_B")
            if a is None or b is None:
                continue
            r = a.shape[0]
            if r > layer.r or a.shape[1] != layer.in_features or tuple(b.shape) != (layer.out_features, r):
                raise ValueError(f"LoRA '{ckpt.id}' does not fit the loaded model ({name}: rank {r}).")
            layer.lora_A[:r].copy_(a)
            layer.lora_B[:, :r].copy_(b)
            used += 2
        if used != len(state):
            for _, layer in _lora_layers(tts_model):
                layer.lora_A.zero_()
                layer.lora_B.zero_()
            raise ValueError(
                f"LoRA '{ckpt.id}' does not match the loaded model: {len(state) - used} of {len(state)} tensors "
                "have no matching layer (trained on another base model?)."
            )
    config = ckpt.config or lora_library.default_lora_config()
    return config.alpha / config.r


def set_lora_scale(tts_model, scale: float) -> None:
    """Set every LoRA layer's alpha / r; 0 disables LoRA."""
    for _, layer in _lora_layers(tts_model):
        layer._base_scaling = scale  # what voxcpm's set_enabled(True) restores
        layer.scaling.fill_(scale)


@dataclass
class SynthesisRequest:
    text: str
    # Voice design / style instruction, rendered as "(control)text".
    control: str = ""
    lora_id: str | None = None
    reference_wav: Path | None = None
    # Transcript of reference_wav; enables continuation-style cloning.
    prompt_text: str | None = None
    cfg_value: float = 2.0
    inference_timesteps: int = 10
    normalize: bool = True
    denoise: bool = True
    seed: int | None = None


@dataclass
class SynthesisResult:
    sample_rate: int
    audio: np.ndarray
    seed: int | None


def compose_text(text: str, control: str) -> str:
    text = text.strip()
    if not text:
        raise ValueError("Text to synthesize is empty.")
    # Parentheses would break the "(control)text" prompt format.
    control = re.sub(r"[()（）]", "", control).strip()
    return f"({control}){text}" if control else text


def _denoiser_to_cpu(model) -> None:
    """Run voxcpm's ZipEnhancer (a ModelScope pipeline, GPU by default) on the CPU:
    a few seconds of reference audio, quick there, and ~2.7 GB less on the GPU."""
    denoiser = getattr(model, "denoiser", None)
    if denoiser is None or not hasattr(denoiser, "_pipeline"):
        return
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks

    denoiser._pipeline = pipeline(Tasks.acoustic_noise_suppression, model=denoiser.model_path, device="cpu")
    vram.free_cache()


class InferenceService:
    def __init__(self, settings: Settings, loader: ModelLoader = load_voxcpm):
        from voxcpm.model.utils import resolve_runtime_device

        self.settings = settings
        self.device = resolve_runtime_device(settings.device, "cuda")
        self.optimize = self.device.startswith("cuda") if settings.optimize is None else settings.optimize
        self._loader = loader
        self._lock = threading.Lock()
        self._model = None
        self._lora_config = None  # LoRA layout the loaded model was built with
        self._loaded_lora: tuple[str, float] | None = None  # (id, weights mtime) in the model
        self._lora_scale = 0.0  # alpha / r of the loaded LoRA
        self._lora_active = False  # the loaded LoRA's weights stay resident while it is off
        # The reference denoiser needs ~2.7 GB more on the GPU; small cards run it on the CPU (vram.py).
        self.denoise_on_cpu = False
        self.progress = Progress()
        self._steps_per_token = STEPS_PER_TOKEN
        self._fresh = False  # loaded but not generated yet: the first steps compile

    def status(self) -> dict:
        return {
            "device": self.device,
            "optimize": self.optimize,
            "model": self.settings.model_path("model"),
            "loaded": self._model is not None,
            "lora_rank": self._lora_config.r if self._lora_config else None,
            "active_lora": self._loaded_lora[0] if self._loaded_lora and self._lora_active else None,
        }

    def synthesize(self, req: SynthesisRequest) -> SynthesisResult:
        final_text = compose_text(req.text, req.control)
        ckpt = lora_library.resolve(self.settings.lora_dir, req.lora_id) if req.lora_id else None
        if ckpt and ckpt.engine != ENGINE:
            raise ValueError(f"LoRA '{ckpt.id}' was trained for another model ({ckpt.engine}), not VoxCPM.")

        with self._lock:
            self.progress = Progress(phase="loading" if self._model is None else "preparing", started=time.time())
            try:
                return self._synthesize(req, final_text, ckpt)
            finally:
                self.progress.phase = "idle"

    def _synthesize(self, req: SynthesisRequest, final_text: str, ckpt: LoraCheckpoint | None) -> SynthesisResult:
        global _step_hook
        model = self._ensure_model(ckpt)
        self._apply_lora(model, ckpt)
        self.progress.phase = "preparing"  # reference denoising and encoding come before the first step

        kwargs = dict(
            text=final_text,
            reference_wav_path=str(req.reference_wav) if req.reference_wav else None,
            cfg_value=float(req.cfg_value),
            inference_timesteps=int(req.inference_timesteps),
            normalize=req.normalize,
            denoise=req.denoise,
            seed=req.seed,
        )
        prompt_text = (req.prompt_text or "").strip()
        if prompt_text and req.reference_wav:
            kwargs["prompt_wav_path"] = str(req.reference_wav)
            kwargs["prompt_text"] = prompt_text

        logger.info("Synthesizing (lora=%s): %.80s", req.lora_id or "none", final_text)
        _hook_generation_loop()
        attempts: list[tuple[float, int]] = []  # (text tokens, step limit) per attempt

        def on_step(i: int, total: int) -> None:
            p = self.progress
            if i == 0:
                p.attempt += 1
                attempts.append((max(1.0, (total - 10) / RATIO_LIMIT), total))
                p.expected = max(MIN_STEPS, round(attempts[-1][0] * self._steps_per_token))
                # The first steps after loading wait for torch.compile.
                p.phase = "compiling" if self._fresh and self.optimize else "generating"
            elif i == 1:
                p.phase, self._fresh = "generating", False
            p.step = i + 1

        _step_hook = on_step
        try:
            audio = model.generate(**kwargs)
        finally:
            _step_hook = None
        tokens, limit = attempts[0] if attempts else (0.0, 0)
        if len(attempts) == 1 and tokens >= LEARN_MIN_TOKENS and self.progress.step < limit:  # learn the pace
            observed = self.progress.step / tokens
            self._steps_per_token = 0.7 * self._steps_per_token + 0.3 * observed
        seed = getattr(model.tts_model, "last_successful_seed", req.seed)
        return SynthesisResult(model.tts_model.sample_rate, audio, seed)

    def progress_public(self) -> dict:
        p = self.progress
        return {**p.__dict__, "elapsed": round(time.time() - p.started, 1) if p.started else 0.0}

    def unload(self) -> None:
        with self._lock:
            self._release()

    def _ensure_model(self, ckpt: LoraCheckpoint | None):
        needed = (ckpt.config or lora_library.default_lora_config()) if ckpt else None
        # Base output only needs LoRA disabled, so any loaded layout works.
        if self._model is not None and (needed is None or lora_library.covers(self._lora_config, needed)):
            return self._model

        # Cover every checkpoint present, so switching among them never reloads;
        # only a LoRA with a larger rank or new targets added later does.
        configs = [
            c.config or lora_library.default_lora_config()
            for c in lora_library.scan(self.settings.lora_dir)
            if c.engine == ENGINE
        ]
        target = lora_library.layout_covering(configs + ([needed] if needed else []))

        self._release()
        logger.info(
            "Loading %s (LoRA layout rank %s, optimize=%s)", self.settings.model_path("model"), target.r, self.optimize
        )
        self._model = self._loader(self.settings, self.device, target, self.optimize)
        self._lora_config = target
        self._fresh = True
        if self.denoise_on_cpu:
            _denoiser_to_cpu(self._model)
        return self._model

    def _apply_lora(self, model, ckpt: LoraCheckpoint | None) -> None:
        self._lora_active = False
        if ckpt is None:
            set_lora_scale(model.tts_model, 0.0)
            return
        key = (ckpt.id, ckpt.weights_mtime)
        if self._loaded_lora != key:
            self._loaded_lora = None
            try:
                self._lora_scale = load_lora_padded(model.tts_model, ckpt)
            except (OSError, RuntimeError) as e:  # unreadable file; ValueError (mismatch) passes through
                raise RuntimeError(f"Unable to load LoRA '{ckpt.id}': {e}") from e
            self._loaded_lora = key
        set_lora_scale(model.tts_model, self._lora_scale)
        self._lora_active = True

    def _release(self) -> None:
        if self._model is None:
            return
        self._model = None
        self._lora_config = None
        self._loaded_lora = None
        gc.collect()
        if self.device.startswith("cuda"):
            import torch

            torch.cuda.empty_cache()
