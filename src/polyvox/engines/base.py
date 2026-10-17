"""What every TTS engine provides, so datasets, training runs, samples and the
UI work the same for each model family."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from ..training import TrainParams


@dataclass(frozen=True)
class Param:
    """One synthesis setting an engine accepts; the UI draws its control from this."""

    name: str
    kind: Literal["float", "int", "bool", "choice"]
    default: float | int | bool | str
    min: float | None = None
    max: float | None = None
    step: float | None = None
    choices: tuple[str, ...] | None = None  # kind "choice": the allowed values
    base_only: bool = False  # only used with the base model (no voice / LoRA chosen)
    versions: tuple[str, ...] = ()  # only used by these model versions (a voice's, or base_version); () = all

    def coerce(self, value) -> float | int | bool | str:
        if self.kind == "choice":
            if str(value) not in (self.choices or ()):
                raise ValueError(f"'{self.name}' must be one of {', '.join(self.choices or ())}.")
            return str(value)
        if self.kind == "bool":
            if isinstance(value, str):
                return value.strip().lower() in ("1", "true", "yes", "on")
            return bool(value)
        try:
            number = int(value) if self.kind == "int" else float(value)
        except (TypeError, ValueError):
            raise ValueError(f"'{self.name}' must be a number.") from None
        if (self.min is not None and number < self.min) or (self.max is not None and number > self.max):
            raise ValueError(f"'{self.name}' must be between {self.min} and {self.max}.")
        return number


@dataclass(frozen=True)
class Capabilities:
    sample_rate: int  # of the audio it outputs
    training_rate: int  # training audio is heard at this rate
    languages: tuple[str, ...]
    reference_audio: bool  # clones the voice of a reference clip
    reference_transcript: bool  # also takes the clip's transcript (continues from it)
    style: bool  # a free-text style / voice description
    tags: bool  # inline tags such as [laughing]
    lora_hot_swap: bool  # switching LoRAs needs no model reload
    # Speech always clones a reference clip: one given with the request, else the
    # voice's own (a checkpoint's reference.wav); none = refused.
    reference_required: bool = False
    training: bool = True  # New run offers it
    # Training updates every weight (no LoRA rank, no optimizer choice).
    full_finetune: bool = False
    # Takes further clips of the voice (SynthesisRequest.extra_references) besides
    # the reference; other engines get several clips joined into one reference.
    several_references: bool = False


@dataclass
class SynthesisRequest:
    text: str
    style: str = ""
    lora_id: str | None = None
    reference_wav: Path | None = None
    prompt_text: str | None = None  # transcript of reference_wav
    # More clips of the same voice (engines with capabilities.several_references).
    extra_references: list[Path] = field(default_factory=list)
    seed: int | None = None
    # Values for the engine's Params, by name; missing ones take their defaults.
    options: dict = field(default_factory=dict)


@dataclass
class SynthesisResult:
    sample_rate: int
    audio: np.ndarray
    seed: int | None


class Engine(ABC):
    """A model family: synthesis, and the training worker for its LoRAs.

    Checkpoints of every engine live in the LoRA folder; each records its
    engine in lora_config.json, each run in train_config.yaml (no entry:
    VoxCPM, which predates engines).
    """

    id: str
    name: str
    capabilities: Capabilities
    params: tuple[Param, ...]
    model_setting: str  # Settings field (and models.MODELS entry) of the base model
    worker_module: str  # run as `python -m <module> <config> --samples-dir ... --stop-file ...`

    @property
    @abstractmethod
    def device(self) -> str: ...

    @property
    @abstractmethod
    def loaded(self) -> bool: ...

    @abstractmethod
    def status(self) -> dict: ...

    @abstractmethod
    def synthesize(self, req: SynthesisRequest) -> SynthesisResult: ...

    @abstractmethod
    def progress(self) -> dict:
        """The running synthesis: {"phase", "step", "expected", "attempt", "elapsed"}."""

    @abstractmethod
    def unload(self) -> None: ...

    def unavailable(self) -> str | None:
        """Why the engine can't run here (a missing package), else None."""
        return None

    def options(self, values: dict) -> dict:
        """``values`` checked against the engine's params, defaults filled in."""
        by_name = {p.name: p for p in self.params}
        unknown = set(values) - set(by_name)
        if unknown:
            raise ValueError(f"{self.name} has no setting {', '.join(sorted(unknown))}.")
        return {p.name: p.coerce(values[p.name]) if p.name in values else p.default for p in self.params}

    @abstractmethod
    def training_config(
        self, params: TrainParams, run_dir: Path, train_manifest: Path, val_manifest: Path | None
    ) -> dict:
        """The worker's train_config.yaml. It must hold num_iters, save_path
        (``<run>/checkpoints``), train_manifest, val_manifest and, for the run
        list, lora.r / lora.alpha."""

    def adopt(self, run_dir: Path, config: dict) -> dict:
        """A run's config made ready to resume (e.g. paths of runs made elsewhere)."""
        return config

    def public(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "unavailable": self.unavailable(),
            "capabilities": asdict(self.capabilities),
            "params": [asdict(p) for p in self.params],
            "model_setting": self.model_setting,
            "loaded": self.loaded,
        }


def fuse_settings() -> None:
    """torch.compile settings for the training workers' fused layers: float
    attributes (norm eps, attention scaling, dropout p, a LoRA's scale) as
    constants (left variable, as dynamic shapes leave them, they enter the graph
    as CPU scalars, whose kernels need a C++ compiler), and the backward pass
    compiled with the forward (so a failure shows in the forward call, where
    training can fall back to unfused layers)."""
    import torch

    torch._dynamo.config.specialize_float = True
    torch._functorch.config.force_non_lazy_backward_lowering = True
