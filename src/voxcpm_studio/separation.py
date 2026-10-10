"""Vocal separation, and optional de-reverb / denoise, with RoFormer models.

The models are the BS-RoFormer / Mel-Band RoFormer checkpoints that
Music-Source-Separation-Training (MSST), MSST-WebUI and UVR5 run: a checkpoint
plus its yaml config. The architectures are vendored (vendor/roformer); the
inference below follows MSST's: overlapping chunks blended with linear fades.

For TTS training data, what counts is that no music is left in the vocal
("bleedless"), more than keeping every breath ("fullness"). Separation runs
before slicing, so VAD, ASR, the aligner and the edge checks all work on the
voice alone.
"""

from __future__ import annotations

import gc
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from . import timing, vram

logger = logging.getLogger(__name__)

# Chunks per position. MSST configs ask 4-8; 1 (chunks only crossfade at their
# seams) is about twice as fast as 2 with a difference far below audibility.
DEFAULT_OVERLAP = 1
BATCH_SIZE = 2


class _ConfigLoader(yaml.SafeLoader):
    """Safe YAML, plus the ``!!python/tuple`` tag MSST configs use."""


_ConfigLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple", lambda loader, node: tuple(loader.construct_sequence(node))
)


def load_config(path: Path) -> dict:
    return yaml.load(path.read_text(encoding="utf-8"), Loader=_ConfigLoader)


def resolve_model(spec: str) -> tuple[Path, Path]:
    """(checkpoint, config) from ``model.ckpt`` (config: same stem .yaml, or the
    only .yaml beside it) or ``model.ckpt|config.yaml``."""
    ckpt_part, _, config_part = spec.partition("|")
    ckpt = Path(ckpt_part.strip().strip('"')).expanduser()
    if not ckpt.is_file():
        raise FileNotFoundError(f"Separation model not found: {ckpt}")
    if config_part:
        config = Path(config_part.strip().strip('"')).expanduser()
    elif ckpt.with_suffix(".yaml").is_file():
        config = ckpt.with_suffix(".yaml")
    else:
        yamls = sorted(ckpt.parent.glob("*.yaml"))
        if len(yamls) != 1:
            raise ValueError(f"Can't tell which config belongs to {ckpt.name}; give it as 'model.ckpt|config.yaml'.")
        config = yamls[0]
    return ckpt, config


def _window(size: int, fade: int):
    import torch

    window = torch.ones(size)
    window[:fade] = torch.linspace(0, 1, fade)
    window[-fade:] = torch.linspace(1, 0, fade)
    return window


class RoformerModel:
    """One checkpoint: separates its target stem (vocals, noreverb, dry, ...)."""

    def __init__(self, ckpt: Path, config_path: Path, device: str, compile: bool = True, batch: int = BATCH_SIZE):
        import torch

        from .vendor.roformer import BSRoformer, MelBandRoformer

        config = load_config(config_path)
        params = dict(config["model"])
        model_cls = MelBandRoformer if "num_bands" in params else BSRoformer
        self.model = model_cls(**params)
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        for key in ("state_dict", "model", "model_state_dict"):
            if isinstance(state, dict) and key in state and isinstance(state[key], dict):
                state = state[key]
        self.model.load_state_dict(state)
        self.model.to(device).eval()
        self.device = device
        self.name = ckpt.stem
        audio, inference, training = config.get("audio", {}), config.get("inference", {}), config.get("training", {})
        self.rate = int(audio.get("sample_rate", 44100))
        self.channels = 2 if params.get("stereo", False) else 1
        self.chunk = int(inference.get("chunk_size") or audio["chunk_size"])
        self.overlap = min(int(inference.get("num_overlap", DEFAULT_OVERLAP)), DEFAULT_OVERLAP)
        instruments = list(training.get("instruments") or [])
        target = training.get("target_instrument")
        self.stem = target or (instruments[0] if instruments else "target")
        # Single-target models output one stem; multi-stem ones output all, in order.
        self.stem_index = 0 if target or params.get("num_stems", 1) == 1 else instruments.index(self.stem)
        self._compiled = compile_transformers(self.model) if compile and device.startswith("cuda") else []
        self.batch = batch  # chunks per model call; halved when the GPU runs out of memory

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """(channels, samples) at self.rate -> the target stem, same shape."""
        import torch

        def run(batch):
            out = self.model(batch).float()
            return out[:, self.stem_index] if out.dim() == 4 else out  # (batch, stems, channels, samples)

        x = np.ascontiguousarray(audio, dtype=np.float32)
        if x.shape[0] != self.channels:  # mono model on stereo input or the reverse
            x = np.repeat(x.mean(axis=0, keepdims=True), self.channels, axis=0)
        with torch.inference_mode(), torch.autocast("cuda", enabled=self.device.startswith("cuda")):
            try:
                return demix(run, x, self.chunk, self.overlap, self.device, self.batch)
            except Exception as e:
                if vram.is_oom(e) and self.batch > 1:
                    vram.free_cache()
                    self.batch //= 2
                    logger.warning("Separation ran out of GPU memory; %d chunks per batch from now on", self.batch)
                    return self(audio)
                if not self._compiled:
                    raise
                # e.g. no working Triton: the same model, not compiled, still works.
                logger.warning("Compiled separation failed; running it uncompiled", exc_info=True)
                for module in self._compiled:
                    module._compiled_call_impl = None  # what Module.compile() set
                self._compiled = []
                return demix(run, x, self.chunk, self.overlap, self.device, self.batch)


def compile_transformers(model) -> list:
    """torch.compile each transformer block of a RoFormer ("regional"
    compilation): the time and frequency transformers repeat with the same
    shapes, so all share two compiled graphs and compiling takes seconds, not
    the minutes a whole-model compile takes. About twice as fast on CUDA; the
    first window pays the compile. Returns the compiled modules."""
    modules = [m for m in model.modules() if type(m).__name__ == "Transformer"]
    for module in modules:
        module.compile()
    return modules


def demix(
    model: Callable, audio: np.ndarray, chunk: int, overlap: int, device: str = "cpu", batch_size: int = BATCH_SIZE
) -> np.ndarray:
    """Run ``model`` ((batch, channels, chunk) -> same shape) over long audio in
    overlapping chunks blended with linear fades, as MSST does."""
    import torch

    x = torch.from_numpy(audio)
    fade = chunk // 10
    # overlap 1: chunks follow each other and only crossfade over the fade length.
    step = chunk - fade if overlap <= 1 else chunk // overlap
    border = chunk - step
    length = x.shape[-1]
    padded = length > 2 * border
    if padded:  # so the first and last samples are covered by full chunks too
        x = torch.nn.functional.pad(x[None], (border, border), mode="reflect")[0]
    window = _window(chunk, fade).to(device)
    result = torch.zeros(x.shape, device=device)
    counter = torch.zeros(x.shape[-1], device=device)
    x = x.to(device)
    starts = list(range(0, x.shape[-1], step))
    for b in range(0, len(starts), batch_size):
        batch, spans = [], []
        for i in starts[b : b + batch_size]:
            part = x[:, i : i + chunk]
            n = part.shape[-1]
            mode = "reflect" if n > chunk // 2 else "constant"
            batch.append(torch.nn.functional.pad(part[None], (0, chunk - n), mode=mode)[0])
            spans.append((i, n))
        out = model(torch.stack(batch)).float()
        for k, (i, n) in enumerate(spans):
            w = window.clone()
            if i == 0:
                w[:fade] = 1
            if i + step >= x.shape[-1]:
                w[-fade:] = 1
            result[:, i : i + n] += out[k, :, :n] * w[:n]
            counter[i : i + n] += w[:n]
    out = (result / counter.clamp_min(1e-8)).cpu().numpy()
    if padded:
        out = out[:, border:-border]
    return np.nan_to_num(out[:, :length])


# (channels, samples) at `rate` -> processed audio, same shape and rate
Process = Callable[[np.ndarray], np.ndarray]


@dataclass
class Chain:
    """Models applied in order; all run at one sample rate."""

    rate: int
    channels: int
    process: Process


# (model specs, device) -> loaded models
ModelLoader = Callable[[list[str], str], list]


def load_models(specs: list[str], device: str, compile: bool = True, batch: int = BATCH_SIZE) -> list:
    return [RoformerModel(*resolve_model(spec), device, compile, batch) for spec in specs]


class SeparationService:
    """Loads the requested chain on first use (one chain resident at a time)."""

    def __init__(self, device: Callable[[], str], loader: ModelLoader = load_models):
        self._device = device
        self._loader = loader
        self._lock = threading.Lock()
        self._specs: tuple[str, ...] = ()
        self._models: list = []
        self._parked = False  # models in CPU memory (park)

    @property
    def loaded(self) -> bool:
        return bool(self._models)

    @property
    def device(self) -> str:
        return self._device()

    def chain(self, specs: list[str]) -> Chain:
        with self._lock:
            if tuple(specs) != self._specs:
                self._unload()
                logger.info("Loading separation models %s on %s", specs, self.device)
                with timing.stage("loading models"):
                    self._models, self._specs = self._loader(list(specs), self.device), tuple(specs)
            models = self._models
        rates = {m.rate for m in models}
        if len(rates) != 1:
            raise ValueError(f"Separation models use different sample rates: {sorted(rates)}")

        def process(audio: np.ndarray) -> np.ndarray:
            with self._lock, timing.stage("separation"):  # one GPU job at a time
                if self._parked:
                    for m in models:
                        m.model.to(m.device)
                    self._parked = False
                for m in models:
                    audio = m(audio)
            return audio

        return Chain(rate=rates.pop(), channels=max(m.channels for m in models), process=process)

    def park(self) -> None:
        """Move the models to CPU memory, back on the next call (AsrService.park)."""
        with self._lock:
            if self._models and not self._parked:
                for m in self._models:
                    m.model.to("cpu")
                self._parked = True
                vram.free_cache()

    def unload(self) -> None:
        with self._lock:
            self._unload()

    def _unload(self) -> None:
        if not self._models:
            return
        self._models, self._specs, self._parked = [], (), False
        gc.collect()
        if self.device.startswith("cuda"):
            import torch

            torch.cuda.empty_cache()
