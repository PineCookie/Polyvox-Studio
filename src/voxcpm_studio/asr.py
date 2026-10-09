"""Speech recognition with Qwen3-ASR, native in transformers (no extra package).

The ``qwen-asr`` package pins an older transformers than voxcpm needs; the
transformers-native ``Qwen3ASRForConditionalGeneration`` with the ``*-hf``
checkpoints runs in the same environment.
"""

from __future__ import annotations

import gc
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import Settings

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
BATCH_SIZE = 32  # larger batches are much faster; transcripts barely change
BATCH_SECONDS = 400  # padded audio per batch, so long chunks don't run out of memory
MAX_NEW_TOKENS = 512


@dataclass
class Transcript:
    text: str
    language: str | None  # detected (e.g. "Chinese"), or None when it was forced


# (model, device) -> (processor, model)
AsrLoader = Callable[[str, str], tuple]


def load_qwen3_asr(model: str, device: str):
    import torch
    from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

    processor = AutoProcessor.from_pretrained(model)
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    asr = Qwen3ASRForConditionalGeneration.from_pretrained(model, dtype=dtype).to(device).eval()
    return processor, asr


def load_audio(path: Path) -> np.ndarray:
    """Mono float32 at 16 kHz, from any format soundfile or torchaudio can read."""
    try:
        import soundfile as sf

        wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
        wav = wav.mean(axis=1)
    except Exception:
        import torchaudio

        tensor, sr = torchaudio.load(str(path))
        wav = tensor.mean(dim=0).numpy()
    if sr != SAMPLE_RATE:
        import torch
        import torchaudio.functional as F

        wav = F.resample(torch.from_numpy(np.ascontiguousarray(wav)), sr, SAMPLE_RATE).numpy()
    return wav.astype(np.float32)


def _batches(order: list[int], lengths: list[int]) -> list[list[int]]:
    """Runs of ``order`` (sorted by length) of at most BATCH_SIZE clips and
    BATCH_SECONDS of padded audio."""
    out: list[list[int]] = []
    for i in order:
        batch = out[-1] if out else None
        if batch and len(batch) < BATCH_SIZE and (len(batch) + 1) * lengths[i] <= BATCH_SECONDS * SAMPLE_RATE:
            batch.append(i)
        else:
            out.append([i])
    return out


class AsrService:
    def __init__(self, settings: Settings, loader: AsrLoader = load_qwen3_asr):
        self.settings = settings
        self._loader = loader
        self._lock = threading.Lock()
        self._processor = None
        self._model = None

    @property
    def device(self) -> str:
        from voxcpm.model.utils import resolve_runtime_device

        device = self.settings.device if self.settings.asr_device == "auto" else self.settings.asr_device
        return resolve_runtime_device(device, "cuda")

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def transcribe(
        self, audios: list[np.ndarray], language: str | None = None, prompt: str | None = None
    ) -> list[Transcript]:
        """Transcribe 16 kHz mono clips.

        ``language`` forces a language ("zh", "Chinese", ...); ``prompt`` gives
        context such as names and terms the recognizer should prefer.
        """
        import torch

        results: list[Transcript | None] = [None] * len(audios)
        # Similar lengths per batch keep padding small.
        order = sorted(range(len(audios)), key=lambda i: len(audios[i]))
        with self._lock:
            processor, model = self._ensure_loaded()
            for idx in _batches(order, [len(a) for a in audios]):
                inputs = processor.apply_transcription_request(
                    [audios[i] for i in idx],
                    language=language or None,
                    prompt=[prompt] * len(idx) if prompt else None,
                ).to(model.device, model.dtype)
                with torch.inference_mode():
                    out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
                parsed = processor.decode(
                    out[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True, return_format="parsed"
                )
                for i, p in zip(idx, parsed if isinstance(parsed, list) else [parsed], strict=True):
                    results[i] = Transcript(text=p["transcription"].strip(), language=p["language"])
        return results  # type: ignore[return-value]

    def unload(self) -> None:
        with self._lock:
            if self._model is None:
                return
            self._processor = self._model = None
            gc.collect()
            if self.device.startswith("cuda"):
                import torch

                torch.cuda.empty_cache()

    def _ensure_loaded(self):
        if self._model is None:
            logger.info("Loading ASR model %s on %s", self.settings.asr_model, self.device)
            self._processor, self._model = self._loader(self.settings.asr_model, self.device)
        return self._processor, self._model
