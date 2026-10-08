"""Discovery and metadata for trained LoRA checkpoints."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from voxcpm.model.voxcpm import LoRAConfig

logger = logging.getLogger(__name__)

WEIGHTS_FILE = "lora_weights.safetensors"
CONFIG_FILE = "lora_config.json"


@dataclass(frozen=True)
class LoraCheckpoint:
    # POSIX path relative to the LoRA directory, e.g. "speaker_r64/checkpoints/step_0005000".
    id: str
    path: Path
    config: LoRAConfig | None
    base_model: str | None

    @property
    def weights_mtime(self) -> float:
        return (self.path / WEIGHTS_FILE).stat().st_mtime


def default_lora_config() -> LoRAConfig:
    """Layout used when a checkpoint has no lora_config.json."""
    return LoRAConfig(
        enable_lm=True,
        enable_dit=True,
        r=32,
        alpha=16,
        target_modules_lm=["q_proj", "v_proj", "k_proj", "o_proj"],
        target_modules_dit=["q_proj", "v_proj", "k_proj", "o_proj"],
    )


def _read_metadata(ckpt_dir: Path) -> tuple[LoRAConfig | None, str | None]:
    config_file = ckpt_dir / CONFIG_FILE
    if not config_file.is_file():
        return None, None
    try:
        info = json.loads(config_file.read_text(encoding="utf-8"))
        cfg = info.get("lora_config")
        return (LoRAConfig(**cfg) if cfg else None), info.get("base_model")
    except (OSError, ValueError) as e:
        logger.warning("Ignoring unreadable %s: %s", config_file, e)
        return None, None


def _load(lora_dir: Path, ckpt_dir: Path) -> LoraCheckpoint:
    config, base_model = _read_metadata(ckpt_dir)
    return LoraCheckpoint(
        id=ckpt_dir.relative_to(lora_dir).as_posix(),
        path=ckpt_dir,
        config=config,
        base_model=base_model,
    )


def scan(lora_dir: Path) -> list[LoraCheckpoint]:
    """Return every checkpoint under ``lora_dir``, newest step first."""
    lora_dir = lora_dir.resolve()
    if not lora_dir.is_dir():
        return []
    ckpts = [_load(lora_dir, weights.parent) for weights in lora_dir.rglob(WEIGHTS_FILE)]
    return sorted(ckpts, key=lambda c: c.id, reverse=True)


def resolve(lora_dir: Path, lora_id: str) -> LoraCheckpoint:
    """Look up one checkpoint by id, refusing paths outside ``lora_dir``."""
    lora_dir = lora_dir.resolve()
    ckpt_dir = (lora_dir / lora_id).resolve()
    if not ckpt_dir.is_relative_to(lora_dir) or not (ckpt_dir / WEIGHTS_FILE).is_file():
        raise KeyError(f"Unknown LoRA checkpoint: {lora_id}")
    return _load(lora_dir, ckpt_dir)
