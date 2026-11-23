"""What every TTS engine provides, so datasets, training runs, samples and the
UI work the same for each model family."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from ..config import Settings
    from ..lora_library import LoraCheckpoint
    from ..models import ModelSpec
    from ..training import TrainParams
    from ..vram import Profile


# {another setting's name: the values it may have}: holds when each has one of them.
Condition = dict[str, tuple]


@dataclass(frozen=True)
class Param:
    """One setting an engine accepts (synthesis: Engine.params; a training run: Engine.train_form);
    the UI draws its control from this. Labels and hints are message keys the UI translates (a
    text it doesn't know is shown as it is)."""

    name: str
    kind: Literal["float", "int", "bool", "choice"]
    default: float | int | bool | str
    min: float | None = None
    max: float | None = None
    step: float | None = None
    choices: tuple[str | int | float, ...] | None = None  # kind "choice": the allowed values
    base_only: bool = False  # only used with the base model (no voice / LoRA chosen)
    versions: tuple[str, ...] = ()  # only used by these model versions (a voice's, or base_version); () = all
    # Training forms only:
    # "run": a TrainParams field; "option": the engine's own (TrainParams.options); "plan": only
    # sizes the run (Engine.plan, e.g. epochs), never sent to the worker.
    scope: Literal["run", "option", "plan"] = "run"
    group: Literal["main", "advanced"] = "main"  # advanced: folded away in a new run
    shown_if: Condition | None = None  # hidden unless this holds
    disabled_if: Condition | None = None  # fixed at its value while this holds
    disabled: str | None = None  # can't be changed here (on this GPU): why, as a message key
    implies: dict[str, dict] | None = None  # {its value: {setting: value}}: set when it takes that value
    labels: tuple[tuple[Condition, str], ...] = ()  # (condition, label key): the first that holds names it
    hint: str | None = None  # message key of a note shown with it
    config: str | None = None  # its key in a run's train_config.yaml ("lora.r"); None = its name
    legacy: float | int | bool | str | None = None  # its value for runs whose config lacks it (None: default)

    def coerce(self, value) -> float | int | bool | str:
        if self.kind == "choice":
            # The choice itself (its type kept: a number stays a number).
            match = next((c for c in self.choices or () if str(c) == str(value)), None)
            if match is None:
                raise ValueError(f"'{self.name}' must be one of {', '.join(map(str, self.choices or ()))}.")
            return match
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
class ReferenceInput:
    """One audio input of an engine's cloning, as its model takes it; Synthesize
    shows each with the engine's own name for it."""

    name: str  # e.g. "prompt", "timbre" (the UI's label for engine.name)
    # The request fields it fills: "reference" = reference_wav (+ extra_references),
    # "prompt" = prompt_wav + prompt_text. /api/tts takes <slot>_audio / <slot>_clips.
    slot: Literal["reference", "prompt"]
    transcript: Literal["none", "optional", "required"]
    several: bool = False  # several clips as they are; other inputs get several joined into one
    max_seconds: float | None = None  # one clip of at most this length each (never joined)
    versions: tuple[str, ...] = ()  # model versions that have it; () = all
    same_as: str | None = None  # the name of another input whose audio it may reuse


@dataclass(frozen=True)
class TagGroup:
    """Inline tags of one kind, as the engine's model knows them (inserted as written)."""

    name: str  # sounds | reactions | emotion | delivery | pacing (the UI's label)
    tags: tuple[str, ...]


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
    # Trained voices keep a default reference clip (reference.wav + reference.json), used
    # when a request gives none; changeable in the Voices panel.
    voice_reference: bool = False
    # How many of a dataset's best clips such a reference joins (more clone better on some models).
    voice_reference_clips: int = 1
    # What an exported voice package holds (sharing.py): "lora" = the LoRA weights + config (a
    # base model and precision to choose), "voice" = whole models; None = no export.
    package: Literal["lora", "voice"] | None = None
    # Its voices are saved references only (no training): made in the Voices panel from a
    # dataset's clips or an uploaded clip (lora_library.VOICE_FILE marks the folder).
    reference_voices: bool = False
    # A voice's reference from a dataset can be chosen by what it makes of a few candidate sets
    # (reference_search.py) rather than by the cleanest transcripts alone.
    reference_search: bool = False
    training: bool = True  # New run offers it
    # Training updates every weight (no LoRA rank, no optimizer choice).
    full_finetune: bool = False
    # The audio inputs its cloning takes (each engine its own).
    reference_inputs: tuple[ReferenceInput, ...] = ()
    # Inline tags (``tags``): the ones its model knows, and whether any [description] works.
    tag_groups: tuple[TagGroup, ...] = ()
    free_tags: bool = False
    # Edits recorded speech (Engine.edit): changed words spoken again, the rest kept.
    speech_edit: bool = False
    # Converts recorded speech or singing into a voice (Engine.convert), no text: the Convert tab,
    # not Synthesize.
    conversion: bool = False
    # A long text is spoken as separate takes, each from the reference again (Engine.takes says
    # where); a blank line in the text starts one.
    takes: bool = False
    # Voices from files another tool made (Engine.import_voice; the UI has a dialog for each).
    voice_import: bool = False

    def reference_input(self, slot: str) -> ReferenceInput | None:
        return next((i for i in self.reference_inputs if i.slot == slot), None)


@dataclass
class SynthesisRequest:
    """``reference_wav`` (+ ``extra_references``): who it should sound like, audio
    only. ``prompt_wav`` + ``prompt_text``: speech the model continues from, so it
    also takes on its pace, tone and recording. A transcript without a prompt clip
    makes the reference the prompt too (requests from before the two were apart)."""

    text: str
    style: str = ""
    lora_id: str | None = None
    reference_wav: Path | None = None
    prompt_text: str | None = None  # transcript of prompt_wav
    # More clips of the same voice (engines whose reference input takes several: ReferenceInput.several).
    extra_references: list[Path] = field(default_factory=list)
    seed: int | None = None
    # Values for the engine's Params, by name; missing ones take their defaults.
    options: dict = field(default_factory=dict)
    prompt_wav: Path | None = None

    def __post_init__(self):
        if self.prompt_wav is None and (self.prompt_text or "").strip():
            self.prompt_wav = self.reference_wav
        if self.prompt_wav is None:
            self.prompt_text = None


@dataclass
class EditRequest:
    """Speech in ``audio`` (transcript ``text``) made to say ``new_text``. ``align(audio, text)``
    gives (start, end) in seconds of each of the text's aligner words (aligner.split_words)."""

    audio: Path
    text: str
    new_text: str
    align: Callable[[Path, str], list[tuple[float, float]]]
    seed: int | None = None


@dataclass
class ConvertRequest:
    """Mono ``audio`` at ``rate`` (speech or singing) sung / spoken again in the voice ``lora_id``
    (capabilities.conversion); ``options`` = values for the engine's Params."""

    audio: np.ndarray
    rate: int
    lora_id: str
    options: dict = field(default_factory=dict)
    seed: int | None = None
    on_progress: Callable[[int, int], None] | None = None  # (done, total) pieces


@dataclass
class SynthesisResult:
    sample_rate: int
    audio: np.ndarray
    seed: int | None
    # An edit's new audio: (start, end) in seconds of the result.
    span: tuple[float, float] | None = None


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
    settings: Settings
    model_setting: str  # its base model's setting (one of model_specs)
    model_specs: tuple[ModelSpec, ...] = ()  # the models it uses (models.all_specs collects them)
    worker_module: str  # run as `python -m <module> <config> --samples-dir ... --stop-file ...`
    # The losses a run's charts show first (the rest follow as its worker logs them).
    chart_losses: tuple[str, ...] = ("loss/diff", "loss/stop")
    # The files of one of its checkpoints / voices in the LoRA folder (lora_library): the first
    # marks such a folder; all of them count for when its weights last changed.
    checkpoint_files: tuple[str, ...] = ()

    @classmethod
    def lora_layers(cls, lora_config: dict) -> list[str]:
        """The parts of the model a checkpoint's saved lora_config adapts (shown with its rank:
        "lm", "dit", ...); none for engines without LoRAs."""
        return []

    @classmethod
    def package_files(cls, ckpt: LoraCheckpoint, name: str, base_model: str, half: bool) -> dict[str, bytes]:
        """A checkpoint's shareable package (capabilities.package; sharing.py zips it): its files by
        name, a README and a config naming the engine (lora_config.json or voice.json) among them.
        ``base_model``: the Hugging Face id a LoRA was trained on; ``half``: LoRA weights in bfloat16."""
        raise ValueError(f"{cls.name}'s voices can't be exported.")

    @classmethod
    def import_package(cls, folder: Path, name: str, files: dict[str, bytes]) -> str:
        """A package's files (package_files) checked and written into ``folder/<name>/`` (``folder``:
        the engine's in the LoRA folder); returns the name used. FileExistsError when it is taken."""
        raise ValueError(f"{cls.name}'s voices can't be imported.")

    def import_voice(self, body: dict, make_room: Callable[[], None]) -> dict:
        """A voice from files another tool made (``body``: the engine's own fields, server paths);
        ``make_room`` frees the GPU before work on it. Returns {"id", "engine", ...}."""
        raise ValueError(f"{self.name} imports no voices from other tools.")

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

    def edit(self, req: EditRequest) -> SynthesisResult:
        """``req.audio`` saying ``req.new_text`` (capabilities.speech_edit)."""
        raise ValueError(f"{self.name} can't edit speech.")

    def convert(self, req: ConvertRequest) -> SynthesisResult:
        """``req.audio`` in the voice ``req.lora_id`` (capabilities.conversion)."""
        raise ValueError(f"{self.name} can't convert recordings.")

    def takes(self, text: str, cloning: bool) -> list[int]:
        """Where each take of ``text`` ends (offsets into it, the last one its end), spoken with a
        reference or a voice (``cloning``) or without (capabilities.takes)."""
        return []

    @abstractmethod
    def progress(self) -> dict:
        """The running synthesis: {"phase", "step", "expected", "attempt", "elapsed"}."""

    @abstractmethod
    def unload(self) -> None: ...

    def unavailable(self) -> str | None:
        """Why the engine can't run here (a missing package), else None."""
        return None

    def training_unavailable(self) -> str | None:
        """Why the engine can't train here, else None (by default: when it can't run)."""
        return self.unavailable()

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

    @classmethod
    def train_form(cls, profile: Profile | None = None) -> tuple[Param, ...]:
        """New run's fields for this engine, in order: TrainParams fields (scope "run"), its own
        settings ("option", in TrainParams.options) and what only sizes the run ("plan"), with
        the defaults that fit ``profile``'s GPU (None: no GPU known)."""
        return ()

    @classmethod
    def train_options(cls, options: dict) -> dict:
        """``options`` (TrainParams.options) checked against the engine's own settings, the
        missing ones at their defaults."""
        own = {p.name: p for p in cls.train_form() if p.scope == "option"}
        if unknown := set(options) - set(own):
            raise ValueError(f"{cls.name} has no training setting {', '.join(sorted(unknown))}.")
        return {name: p.coerce(options[name]) if name in options else p.default for name, p in own.items()}

    @classmethod
    def plan(cls, values: dict, clips: int, seconds: float, overrides: Iterable[str] = ()) -> dict:
        """A new run's size from the form's ``values`` (its fields by name) and its training clips
        (``clips``, ``seconds`` of audio in all): {"values": the fields that follow from them,
        num_iters etc., except ``overrides`` (set by hand); "summary": {"kind", ...} for the
        UI's description; "name_tag": the model part of a run name; "samples": {"every",
        "unit"}: how often validation samples come}."""
        raise ValueError(f"{cls.name} doesn't train here.")

    @classmethod
    def values_from_config(cls, config: dict) -> dict:
        """The form's values a run was started with (its train_config.yaml); a setting its
        config lacks takes its ``legacy`` value (what runs from before it used), else its default."""
        return {p.name: config_value(config, p) for p in cls.train_form() if p.scope != "plan"}

    @classmethod
    def base_repo(cls) -> str | None:
        """The Hugging Face repo of its base model (model_specs): the model's own, also when it
        is downloaded from a converted copy (Drama 3)."""
        return next((m.model_repo for m in cls.model_specs if m.setting == cls.model_setting), None)

    def use_profile(self, profile: Profile) -> None:
        """Settings that follow the GPU's size (vram.Profile), told once at startup."""
        return None  # most engines have none

    def public(self, profile: Profile | None = None) -> dict:
        """For /api/engines; ``profile``: the GPU's, for the training form's defaults. ``model_id``:
        the Hugging Face id of the model it runs (LoRAs record theirs: the UI flags a mismatch)."""
        from ..models import hub_id

        return {
            "id": self.id,
            "name": self.name,
            "unavailable": self.unavailable(),
            "training_unavailable": self.training_unavailable() if self.capabilities.training else None,
            "capabilities": asdict(self.capabilities),
            "params": [asdict(p) for p in self.params],
            "train_form": [asdict(p) for p in self.train_form(profile)] if self.capabilities.training else [],
            "model_setting": self.model_setting,
            "model_id": hub_id(settings.model_path(self.model_setting))
            if (settings := getattr(self, "settings", None))
            else None,
            "chart_losses": list(self.chart_losses),
            "base_repo": self.base_repo(),
            "loaded": self.loaded,
        }


def config_value(config: dict, param: Param):
    """``param``'s value in a run's config (Param.config: a dotted key), if it is one it can
    take; else its legacy value or default."""
    node = config
    for part in (param.config or param.name).split("."):
        node = node.get(part) if isinstance(node, dict) else None
    if param.kind == "bool" and isinstance(node, bool):
        return node
    if param.kind in ("int", "float") and isinstance(node, int | float) and not isinstance(node, bool):
        return node
    if param.kind == "choice" and node is not None:
        try:
            return param.coerce(node)
        except ValueError:
            pass
    return param.legacy if param.legacy is not None else param.default


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
