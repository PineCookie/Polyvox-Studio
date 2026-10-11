"""TTS engines: each model family behind one interface (base.Engine).

Datasets, training runs (worker output, samples, similarity), checkpoints in
the LoRA folder and the UI are shared; an engine brings its model loading,
synthesis settings and training worker.
"""

from __future__ import annotations

from ..config import Settings
from .base import Capabilities, Engine, Param, SynthesisRequest, SynthesisResult

DEFAULT = "voxcpm"  # runs and checkpoints that record no engine

__all__ = ["DEFAULT", "Capabilities", "Engine", "Param", "SynthesisRequest", "SynthesisResult", "classes", "create"]


def classes() -> dict[str, type[Engine]]:
    from .dots.engine import DotsEngine
    from .gpt_sovits.engine import GptSovitsEngine
    from .voxcpm.engine import VoxCPMEngine

    return {VoxCPMEngine.id: VoxCPMEngine, DotsEngine.id: DotsEngine, GptSovitsEngine.id: GptSovitsEngine}


def create(settings: Settings, **instances: Engine) -> dict[str, Engine]:
    """One engine of each kind; ``instances`` replaces some by id (tests, or a
    service made earlier)."""
    return {id: instances.get(id) or cls(settings) for id, cls in classes().items()}
