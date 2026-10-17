"""Fish Audio S2-Pro as an engine: zero-shot cloning, and voices trained here
(LoRA on its fast transformer: lora.py, train.py, worker.py).

model.py, codec.py, prompt.py and generate.py are our own implementation of the
published architecture; the weights (models\\fishaudio__s2-pro) are under the
Fish Audio Research License: research and non-commercial use only.

The model is large (bfloat16 language model 9.1 GB + codec): a 16 GB GPU only.
It clones from a reference clip with its transcript; several clips are joined
into one (about 20 s cloned clearly better than one short clip). A trained
voice speaks with its default reference (a few of its clips joined) unless the
request gives another: its LoRA refines the sound, the reference still sets
much of the voice.
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

from ... import lora_library, vram
from ...config import Settings
from ..base import Capabilities, Engine, Param, ReferenceInput, SynthesisRequest, SynthesisResult, TagGroup
from . import lora as s2lora

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
    phase: str = "idle"  # idle | loading | preparing | compiling | generating
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
        # One CUDA graph per frame (~1.6x faster than eager; graphs=False: the same steps eagerly),
        # the small steps between its linear layers fused by torch.compile (~1.2x more).
        self.decoder = decode.Decoder(self.model, self.end_token, graphs=graphs, fused=graphs)
        self.device = device
        self.merger = s2lora.Merger(self.model)
        self.lora: tuple[str, float] | None = None  # (id, weights mtime) merged into the weights
        logger.info("S2-Pro loaded in %.1f s", time.time() - t0)

    def encode(self, wav: Path, noise_floor: bool = False):
        """Codes of a clip; ``noise_floor``: heard as trained voices learned their clips."""
        import torch
        import torchaudio

        from ...data import audio_io
        from .train import with_noise_floor

        audio, rate = audio_io.read(wav)
        mono = torch.from_numpy(np.asarray(audio.mean(axis=0) if audio.ndim == 2 else audio, dtype=np.float32))
        mono = torchaudio.functional.resample(mono.to(self.device), rate, self.codec.sample_rate)
        return self.codec.encode(with_noise_floor(mono) if noise_floor else mono).cpu()

    def generate(self, tokens, sampling, max_frames: int, on_frame):
        """decoder.generate; if fusing its steps fails (torch.compile: no working
        Triton, say), the rest of the process decodes unfused."""
        import torch

        from . import decode

        try:
            return self.decoder.generate(tokens, sampling, max_frames, on_frame)
        except Exception as e:
            if not self.decoder.fused or isinstance(e, torch.OutOfMemoryError):
                raise
            logger.warning("S2-Pro: fusing the decoding steps failed (%s); decoding unfused", e)
            self.decoder = decode.Decoder(self.model, self.end_token, graphs=self.decoder.graphs)
            return self.decoder.generate(tokens, sampling, max_frames, on_frame)

    def use_lora(self, ckpt: lora_library.LoraCheckpoint | None) -> None:
        """Merge the checkpoint's LoRA into the weights (None: the base model's)."""
        key = (ckpt.id, ckpt.weights_mtime) if ckpt else None
        if key == self.lora:
            return
        self.lora = None
        if ckpt is None:
            self.merger.apply(None)
            return
        from safetensors.torch import load_file

        try:
            tensors = load_file(str(ckpt.path / lora_library.WEIGHTS_FILE))
        except (OSError, RuntimeError) as e:
            raise RuntimeError(f"Unable to load voice '{ckpt.id}': {e}") from e
        rank, alpha = ckpt.rank, ckpt.alpha
        try:
            self.merger.apply(tensors, (alpha / rank) if rank and alpha else 1.0)
        except ValueError:
            self.merger.apply(None)
            raise
        self.lora = key


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
        lora_hot_swap=True,
        voice_reference=True,
        # A clip with its transcript (several are joined into one).
        reference_inputs=(ReferenceInput("reference", "prompt", "required"),),
    )
    params = (
        Param("temperature", "float", 0.8, 0.1, 1.5, 0.05),
        Param("top_p", "float", 0.8, 0.1, 1.0, 0.05),
        Param("top_k", "int", 30, 1, 100, 1),
    )
    model_setting = "s2_model"
    worker_module = "polyvox.engines.fish_s2.worker"

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
            "active_lora": self._runtime.lora[0] if self._runtime and self._runtime.lora else None,
        }

    def progress(self) -> dict:
        p = self._progress
        return {**p.__dict__, "elapsed": round(time.time() - p.started, 1) if p.started else 0.0}

    def synthesize(self, req: SynthesisRequest) -> SynthesisResult:
        import torch

        from ..gpt_sovits.voices import read_reference
        from . import generate, prompt

        text = re.sub(r"\s+", " ", req.text).strip()
        parts = text_parts(text)
        if not parts:
            raise ValueError("Text to synthesize is empty.")
        ckpt = lora_library.resolve(self.settings.lora_dir, req.lora_id) if req.lora_id else None
        if ckpt and ckpt.engine != self.id:
            raise ValueError(f"Voice '{ckpt.id}' was trained for another model ({ckpt.engine}), not S2-Pro.")
        if reason := self.unavailable():
            raise ValueError(f"Fish Audio S2-Pro can't run: {reason}")
        clip = req.prompt_wav or req.reference_wav
        transcript = (req.prompt_text or "").strip()
        if clip is None and ckpt and (own := read_reference(ckpt.path)) and own[1].strip():
            clip, transcript = own  # the voice's default reference
        if clip is not None and not transcript:
            raise ValueError("Fish Audio S2-Pro needs the reference clip's transcript.")
        options = self.options(req.options)
        sampling = generate.Sampling(float(options["temperature"]), float(options["top_p"]), int(options["top_k"]))

        with self._lock:
            self._progress = Progress(phase="loading" if self._runtime is None else "preparing", started=time.time())
            try:
                rt = self._ensure_runtime()
                rt.use_lora(ckpt)
                c = rt.model.config
                reference = (rt.encode(clip, noise_floor=ckpt is not None), transcript) if clip is not None else None
                seed = req.seed if req.seed is not None else random.randrange(2**31)
                torch.manual_seed(seed)
                chars = sum(len(re.sub(r"\W", "", p)) for p in parts)
                self._progress.expected = max(MIN_FRAMES, round(chars * self._frames_per_char))
                # The first frames of a process wait for torch.compile (until the first frame).
                self._progress.phase = "generating" if rt.decoder.warm else "compiling"
                history: list[tuple[str, torch.Tensor]] = []
                for part in parts:
                    tokens = prompt.build(rt.tokenizer, c.codebooks, c.semantic_begin, part, reference, history[-1:])
                    codes = rt.generate(tokens, sampling, MAX_FRAMES, self._count)
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
        self._progress.phase = "generating"
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
        """A LoRA on the fast transformer (lora.py; rank / alpha / optimizer from
        ``params``). The run's reference (samples, the voice's default) is the
        dataset's best few clips joined."""
        from ...data import manifests, references

        if reason := self.unavailable():
            raise ValueError(f"Fish Audio S2-Pro can't train here: {reason}")
        if params.s2_slow and params.optimizer != "adamw":
            raise ValueError("Training the slow transformer needs AdamW (Prodigy can't give it a smaller rate).")
        dataset = manifests.resolve(self.settings.data_dir, params.dataset)
        picked = references.pick(dataset, s2lora.REFERENCE_CLIPS, manifests.legacy_root(self.settings.data_dir))
        return {
            "pretrained_path": self.settings.model_path(self.model_setting),
            "train_manifest": str(train_manifest),
            "val_manifest": str(val_manifest) if val_manifest else "",
            "save_path": str(run_dir / "checkpoints"),
            "num_iters": params.num_iters,
            "batch_size": params.batch_size,
            "grad_accum_steps": params.grad_accum_steps,
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
                "targets": [*s2lora.DEFAULT_TARGETS, *(s2lora.SLOW_TARGETS if params.s2_slow else ())],
            },
            "compile": self.settings.optimize is not False,  # fuse the fast layers' training forward
            "cached_references": params.s2_cached_references,
            "offload": params.s2_offload,
            # Codes and slow transformer states, shared by the dataset's runs (hidden folders aren't datasets).
            "cache_file": str(dataset.parent / ".cache" / "fish_s2.pt"),
            # The slow transformer's own AdamW rate (learning_rate is the fast layers').
            **({"slow_learning_rate": params.s2_slow_lr} if params.s2_slow else {}),
            "reference": {
                "clips": [{"audio": str(c.audio), "text": c.text} for c in picked],
                "source": f"{params.dataset}: " + ", ".join(c.audio.name for c in picked),
            },
        }
