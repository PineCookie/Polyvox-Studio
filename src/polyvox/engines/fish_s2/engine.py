"""Fish Audio S2-Pro as an engine (zero-shot cloning; no training here yet).

model.py, codec.py, prompt.py and generate.py are our own implementation of the
published architecture; the weights (models\\fishaudio__s2-pro) are under the
Fish Audio Research License: research and non-commercial use only.

The model is large (bfloat16 language model 9.1 GB + codec): a 16 GB GPU only.
It clones from a reference clip with its transcript; several clips are joined
into one (about 20 s cloned clearly better than one short clip).
"""

from __future__ import annotations

import gc
import logging
import random
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ... import vram
from ...config import Settings
from ..base import Capabilities, Engine, Param, ReferenceInput, SynthesisRequest, SynthesisResult, TagGroup

if TYPE_CHECKING:
    from ...training import TrainParams

logger = logging.getLogger(__name__)

ENGINE = "fish_s2"
MIN_GB = 15.0  # a 16 GB card reports a little under 16
FRAMES_PER_CHAR = 5.0  # first guess (Chinese ~4.5 characters per second, 21.5 frames per second); then learned
MIN_FRAMES = 10
LEARN_MIN_CHARS = 10
MAX_FRAMES = 1024  # per part of the text (~48 s)
PART_CHARS = 150  # longer texts are spoken in parts of whole sentences, each continuing the last
_SENTENCE_END = re.compile(r"(?<=[。！？!?；;…\n])|(?<=\.)\s")


@dataclass
class Progress:
    phase: str = "idle"  # idle | loading | preparing | generating
    step: int = 0  # audio frames so far (~21.5 per second of audio)
    expected: int | None = None
    attempt: int = 0
    started: float = 0.0


def text_parts(text: str, limit: int = PART_CHARS) -> list[str]:
    """Whole sentences, joined up to ``limit`` characters a part (a longer sentence stays whole)."""
    parts: list[str] = []
    for sentence in (s.strip() for s in _SENTENCE_END.split(text)):
        if not sentence:
            continue
        if parts and len(parts[-1]) + len(sentence) <= limit:
            parts[-1] += sentence
        else:
            parts.append(sentence)
    return parts


class Runtime:
    """The loaded model, codec and tokenizer."""

    def __init__(self, model_dir: Path, device: str, graphs: bool = True):
        import torch
        from transformers import AutoTokenizer

        from . import codec, decode, model

        t0 = time.time()
        self.model = model.load(model_dir, device, torch.bfloat16)
        self.codec = codec.load(model_dir / "codec.pth", device)
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.end_token = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        # One CUDA graph per frame (~1.6x faster than eager; graphs=False: the same steps eagerly).
        self.decoder = decode.Decoder(self.model, self.end_token, graphs=graphs)
        self.device = device
        logger.info("S2-Pro loaded in %.1f s", time.time() - t0)

    def encode(self, wav: Path):
        import torch
        import torchaudio

        from ...data import audio_io

        audio, rate = audio_io.read(wav)
        mono = torch.from_numpy(np.asarray(audio.mean(axis=0) if audio.ndim == 2 else audio, dtype=np.float32))
        mono = torchaudio.functional.resample(mono.to(self.device), rate, self.codec.sample_rate)
        return self.codec.encode(mono).cpu()


class FishS2Engine(Engine):
    id = ENGINE
    name = "Fish Audio S2-Pro"
    capabilities = Capabilities(
        sample_rate=44_100,
        training_rate=44_100,
        languages=("zh", "en", "ja", "ko"),
        reference_audio=True,
        reference_transcript=True,
        style=False,
        tags=True,
        # Fish Audio's well-tested tags; any [natural-language description] works too. A tag
        # acts on the words after it, until the next tag or the end of the sentence.
        tag_groups=(
            TagGroup("emotion", ("[excited]", "[delight]", "[surprised]", "[shocked]", "[sad]", "[angry]")),
            TagGroup(
                "sounds",
                (
                    "[laughing]",
                    "[chuckling]",
                    "[giggle]",
                    "[sigh]",
                    "[inhale]",
                    "[exhale]",
                    "[gasp]",
                    "[panting]",
                    "[clearing throat]",
                    "[sobbing]",
                    "[crying]",
                ),
            ),
            TagGroup(
                "delivery",
                (
                    "[whisper]",
                    "[soft voice]",
                    "[low voice]",
                    "[loud voice]",
                    "[shouting]",
                    "[screaming]",
                    "[emphasis]",
                    "[singing]",
                ),
            ),
            TagGroup("pacing", ("[short pause]", "[pause]", "[long pause]")),
        ),
        free_tags=True,
        lora_hot_swap=False,
        training=False,
        # A clip with its transcript (several are joined into one).
        reference_inputs=(ReferenceInput("reference", "prompt", "required"),),
    )
    params = (
        Param("temperature", "float", 0.8, 0.1, 1.5, 0.05),
        Param("top_p", "float", 0.8, 0.1, 1.0, 0.05),
        Param("top_k", "int", 30, 1, 100, 1),
    )
    model_setting = "s2_model"
    worker_module = ""

    def __init__(self, settings: Settings, loader=Runtime):
        self.settings = settings
        self._loader = loader
        self._lock = threading.Lock()
        self._runtime: Runtime | None = None
        self._progress = Progress()
        self._frames_per_char = FRAMES_PER_CHAR

    @property
    def device(self) -> str:
        return "cuda"

    @property
    def loaded(self) -> bool:
        return self._runtime is not None

    def unavailable(self) -> str | None:
        if not self.settings.model_path(self.model_setting):
            return "download it first (Models: Fish Audio S2-Pro)."
        gb = vram.total_gb(self.device)
        if gb is None:
            return "it needs a CUDA GPU."
        if gb < MIN_GB:
            return f"it needs a 16 GB GPU (this one has {gb:.0f} GB)."
        return None

    def status(self) -> dict:
        return {
            "device": self.device,
            "model": self.settings.model_path(self.model_setting),
            "loaded": self.loaded,
            "active_lora": None,
        }

    def progress(self) -> dict:
        p = self._progress
        return {**p.__dict__, "elapsed": round(time.time() - p.started, 1) if p.started else 0.0}

    def synthesize(self, req: SynthesisRequest) -> SynthesisResult:
        import torch

        from . import generate, prompt

        text = re.sub(r"\s+", " ", req.text).strip()
        parts = text_parts(text)
        if not parts:
            raise ValueError("Text to synthesize is empty.")
        if req.lora_id:
            raise ValueError("Fish Audio S2-Pro has no trained voices here yet: choose its base model.")
        if reason := self.unavailable():
            raise ValueError(f"Fish Audio S2-Pro can't run: {reason}")
        clip = req.prompt_wav or req.reference_wav
        transcript = (req.prompt_text or "").strip()
        if clip is not None and not transcript:
            raise ValueError("Fish Audio S2-Pro needs the reference clip's transcript.")
        options = self.options(req.options)
        sampling = generate.Sampling(float(options["temperature"]), float(options["top_p"]), int(options["top_k"]))

        with self._lock:
            self._progress = Progress(phase="loading" if self._runtime is None else "preparing", started=time.time())
            try:
                rt = self._ensure_runtime()
                c = rt.model.config
                reference = (rt.encode(clip), transcript) if clip is not None else None
                seed = req.seed if req.seed is not None else random.randrange(2**31)
                torch.manual_seed(seed)
                chars = sum(len(re.sub(r"\W", "", p)) for p in parts)
                self._progress.expected = max(MIN_FRAMES, round(chars * self._frames_per_char))
                self._progress.phase = "generating"
                history: list[tuple[str, torch.Tensor]] = []
                for part in parts:
                    tokens = prompt.build(rt.tokenizer, c.codebooks, c.semantic_begin, part, reference, history[-1:])
                    codes = rt.decoder.generate(tokens, sampling, MAX_FRAMES, self._count)
                    history.append((part, codes))
                codes = torch.cat([h[1] for h in history], 1)
                if codes.shape[1] == 0:
                    raise ValueError("Fish Audio S2-Pro made no audio for this text.")
                audio = rt.codec.decode(codes.to(rt.device)).float().cpu().numpy()
                if chars >= LEARN_MIN_CHARS:
                    self._frames_per_char = 0.7 * self._frames_per_char + 0.3 * codes.shape[1] / chars
                return SynthesisResult(rt.codec.sample_rate, audio, seed)
            finally:
                self._progress.phase = "idle"

    def _count(self) -> None:
        self._progress.step += 1

    def _ensure_runtime(self) -> Runtime:
        if self._runtime is None:
            graphs = self.settings.optimize is not False
            self._runtime = self._loader(Path(self.settings.model_path(self.model_setting)), self.device, graphs)
        return self._runtime

    def unload(self) -> None:
        with self._lock:
            if self._runtime is None:
                return
            self._runtime = None
            gc.collect()
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def training_config(
        self, params: TrainParams, run_dir: Path, train_manifest: Path, val_manifest: Path | None
    ) -> dict:
        raise ValueError("Fish Audio S2-Pro can't be trained here yet.")
