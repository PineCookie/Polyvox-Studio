"""Model lifecycle and synthesis with LoRA hot-swapping."""

from __future__ import annotations

import gc
import logging
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from . import lora_library
from .config import Settings
from .lora_library import LoraCheckpoint

logger = logging.getLogger(__name__)

# (settings, device, lora_config, optimize) -> voxcpm.VoxCPM
ModelLoader = Callable[..., object]


def load_voxcpm(settings: Settings, device: str, lora_config, optimize: bool):
    import voxcpm

    model = voxcpm.VoxCPM.from_pretrained(
        settings.model,
        load_denoiser=settings.load_denoiser,
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

    def status(self) -> dict:
        return {
            "device": self.device,
            "optimize": self.optimize,
            "model": self.settings.model,
            "loaded": self._model is not None,
            "lora_rank": self._lora_config.r if self._lora_config else None,
            "active_lora": self._loaded_lora[0] if self._loaded_lora and self._lora_active else None,
        }

    def synthesize(self, req: SynthesisRequest) -> SynthesisResult:
        final_text = compose_text(req.text, req.control)
        ckpt = lora_library.resolve(self.settings.lora_dir, req.lora_id) if req.lora_id else None

        with self._lock:
            model = self._ensure_model(ckpt)
            self._apply_lora(model, ckpt)

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
            audio = model.generate(**kwargs)
            seed = getattr(model.tts_model, "last_successful_seed", req.seed)
            return SynthesisResult(model.tts_model.sample_rate, audio, seed)

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
        configs = [c.config or lora_library.default_lora_config() for c in lora_library.scan(self.settings.lora_dir)]
        target = lora_library.layout_covering(configs + ([needed] if needed else []))

        self._release()
        logger.info("Loading %s (LoRA layout rank %s, optimize=%s)", self.settings.model, target.r, self.optimize)
        self._model = self._loader(self.settings, self.device, target, self.optimize)
        self._lora_config = target
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
