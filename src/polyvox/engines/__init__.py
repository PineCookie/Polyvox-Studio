"""TTS engines: each model family behind one interface (base.Engine).

Datasets, training runs (worker output, samples, similarity), checkpoints in
the LoRA folder and the UI are shared; an engine brings its model loading,
synthesis settings and training worker.
"""

from __future__ import annotations

from ..config import Settings
from .base import (
    Capabilities,
    ConvertRequest,
    EditRequest,
    Engine,
    Param,
    ReferenceInput,
    SynthesisRequest,
    SynthesisResult,
    TagGroup,
)

__all__ = [
    "Capabilities",
    "ConvertRequest",
    "EditRequest",
    "Engine",
    "Param",
    "ReferenceInput",
    "SynthesisRequest",
    "SynthesisResult",
    "TagGroup",
    "classes",
    "create",
    "default",
    "with_capability",
]


def classes() -> dict[str, type[Engine]]:
    from .dots.engine import DotsEngine
    from .fish_drama3.engine import FishDrama3Engine
    from .fish_s2.engine import FishS2Engine
    from .gpt_sovits.engine import GptSovitsEngine
    from .rvc.engine import RvcEngine
    from .voxcpm.engine import VoxCPMEngine

    return {
        VoxCPMEngine.id: VoxCPMEngine,
        DotsEngine.id: DotsEngine,
        GptSovitsEngine.id: GptSovitsEngine,
        FishS2Engine.id: FishS2Engine,
        FishDrama3Engine.id: FishDrama3Engine,
        RvcEngine.id: RvcEngine,
    }


def default() -> str:
    """The engine a request that names none uses: the first one."""
    return next(iter(classes()))


def with_capability(capability: str, engine: str | None = None) -> type[Engine] | None:
    """``engine``'s class if it has ``capability`` (a true Capabilities field), else None;
    without ``engine``, the first engine that has it."""
    found = classes()
    candidates = [found[engine]] if engine in found else [] if engine else list(found.values())
    return next((cls for cls in candidates if getattr(cls.capabilities, capability)), None)


def create(settings: Settings, **instances: Engine) -> dict[str, Engine]:
    """One engine of each kind; ``instances`` replaces some by id (tests, or a
    service made earlier)."""
    return {id: instances.get(id) or cls(settings) for id, cls in classes().items()}
