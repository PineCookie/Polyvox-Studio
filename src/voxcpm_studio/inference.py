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

    def status(self) -> dict:
        return {
            "device": self.device,
            "optimize": self.optimize,
            "model": self.settings.model,
            "loaded": self._model is not None,
            "lora_rank": self._lora_config.r if self._lora_config else None,
            "active_lora": self._loaded_lora[0] if self._loaded_lora else None,
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
        if ckpt is not None:
            target = ckpt.config or lora_library.default_lora_config()
        elif self._model is not None:
            # Base output only needs LoRA disabled, so any loaded layout works.
            target = self._lora_config
        else:
            target = lora_library.default_lora_config()

        if self._model is not None and self._lora_config == target:
            return self._model

        self._release()
        logger.info("Loading %s (LoRA rank %s, optimize=%s)", self.settings.model, target.r, self.optimize)
        self._model = self._loader(self.settings, self.device, target, self.optimize)
        self._lora_config = target
        return self._model

    def _apply_lora(self, model, ckpt: LoraCheckpoint | None) -> None:
        if ckpt is None:
            model.set_lora_enabled(False)
            return
        key = (ckpt.id, ckpt.weights_mtime)
        if self._loaded_lora != key:
            try:
                model.load_lora(str(ckpt.path))
            except Exception as e:
                self._loaded_lora = None
                raise RuntimeError(f"Unable to load LoRA '{ckpt.id}': {e}") from e
            self._loaded_lora = key
        model.set_lora_enabled(True)

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
