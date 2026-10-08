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


def _union(lists) -> list[str]:
    out: list[str] = []
    for names in lists:
        out += [n for n in names if n not in out]
    return out


def layout_covering(configs: list[LoRAConfig]) -> LoRAConfig:
    """The smallest LoRA layout any of ``configs`` fits into: highest rank,
    union of target layers. A smaller LoRA loads zero-padded (see inference)."""
    configs = configs or [default_lora_config()]
    r = max(c.r for c in configs)
    return LoRAConfig(
        enable_lm=any(c.enable_lm for c in configs),
        enable_dit=any(c.enable_dit for c in configs),
        enable_proj=any(c.enable_proj for c in configs),
        r=r,
        alpha=r,  # unused: each loaded LoRA sets its own alpha / r
        dropout=0.0,
        target_modules_lm=_union(c.target_modules_lm for c in configs if c.enable_lm),
        target_modules_dit=_union(c.target_modules_dit for c in configs if c.enable_dit),
        target_proj_modules=_union(c.target_proj_modules for c in configs if c.enable_proj),
    )


def covers(layout: LoRAConfig, config: LoRAConfig) -> bool:
    """Whether a model built with ``layout`` can hold a LoRA with ``config``."""
    return (
        layout.r >= config.r
        and (layout.enable_lm or not config.enable_lm)
        and (layout.enable_dit or not config.enable_dit)
        and (layout.enable_proj or not config.enable_proj)
        and (not config.enable_lm or set(config.target_modules_lm) <= set(layout.target_modules_lm))
        and (not config.enable_dit or set(config.target_modules_dit) <= set(layout.target_modules_dit))
        and (not config.enable_proj or set(config.target_proj_modules) <= set(layout.target_proj_modules))
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
