"""Discovery and metadata for trained checkpoints: LoRAs (VoxCPM, dots.tts) and
GPT-SoVITS voices (full weights; engines/gpt_sovits/voices.py), all in the LoRA folder.

The LoRA folder has one folder per engine (``voxcpm/``, ``dots_tts/``,
``gpt_sovits/``) holding its runs and voices; a checkpoint's id is its path
from the LoRA folder, e.g. ``gpt_sovits/speaker_a/checkpoints/step_0000100``.
Runs and voices from before that sat directly in the LoRA folder: ``migrate``
moves them into their engine's folder (still found where they are until then).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, replace
from pathlib import Path

from voxcpm.model.voxcpm import LoRAConfig

from .engines import DEFAULT as DEFAULT_ENGINE

logger = logging.getLogger(__name__)

WEIGHTS_FILE = "lora_weights.safetensors"
# A GPT-SoVITS voice's weights (its folder also holds a GPT checkpoint).
VOICE_WEIGHTS_FILES = ("sovits_weights.pth", "gpt_weights.ckpt")
MARKERS = (WEIGHTS_FILE, VOICE_WEIGHTS_FILES[0])  # a folder holding one of these is a checkpoint
# {"lora_config": {...}, "base_model": ..., "engine": ...}; no engine: VoxCPM.
CONFIG_FILE = "lora_config.json"


@dataclass(frozen=True)
class LoraCheckpoint:
    # POSIX path relative to the LoRA directory, e.g. "speaker_r64/checkpoints/step_0005000".
    id: str
    path: Path
    config: LoRAConfig | None  # VoxCPM checkpoints only
    base_model: str | None
    engine: str = DEFAULT_ENGINE
    # The saved lora_config as written, for any engine (rank "r", "alpha", ...).
    raw_config: dict | None = None
    version: str | None = None  # model version, when recorded (GPT-SoVITS: v2Pro / v2ProPlus)

    @property
    def voice_path(self) -> str:
        """The id of the run / voice folder holding it, e.g. ``gpt_sovits/speaker_a``."""
        parts = self.id.split("/")
        return "/".join(parts[: 2 if len(parts) > 1 and parts[0] in engine_folders() else 1])

    @property
    def voice(self) -> str:
        """The run / voice name (its folder's name)."""
        return self.voice_path.split("/")[-1]

    @property
    def weights_mtime(self) -> float:
        """When its weights last changed (any of its weight files)."""
        files = [self.path / f for f in (WEIGHTS_FILE, *VOICE_WEIGHTS_FILES)]
        return max(f.stat().st_mtime for f in files if f.is_file())

    @property
    def rank(self) -> int | None:
        r = (self.raw_config or {}).get("r")
        return int(r) if isinstance(r, int | float) else None

    @property
    def alpha(self) -> float | None:
        alpha = (self.raw_config or {}).get("alpha")
        return float(alpha) if isinstance(alpha, int | float) else None


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


def _load(lora_dir: Path, ckpt_dir: Path) -> LoraCheckpoint:
    ckpt = LoraCheckpoint(id=ckpt_dir.relative_to(lora_dir).as_posix(), path=ckpt_dir, config=None, base_model=None)
    config_file = ckpt_dir / CONFIG_FILE
    if not config_file.is_file():
        return ckpt
    try:
        info = json.loads(config_file.read_text(encoding="utf-8"))
        engine = str(info.get("engine") or DEFAULT_ENGINE)
        raw = info.get("lora_config") or None
        config = LoRAConfig(**raw) if raw and engine == DEFAULT_ENGINE else None
        version = info.get("version") if isinstance(info.get("version"), str) else None
        return replace(
            ckpt, config=config, base_model=info.get("base_model"), engine=engine, raw_config=raw, version=version
        )
    except (OSError, ValueError, TypeError) as e:
        logger.warning("Ignoring unreadable %s: %s", config_file, e)
        return ckpt


def scan(lora_dir: Path) -> list[LoraCheckpoint]:
    """Return every checkpoint under ``lora_dir``, newest step first."""
    lora_dir = lora_dir.resolve()
    if not lora_dir.is_dir():
        return []
    folders = {marker.parent for name in MARKERS for marker in lora_dir.rglob(name)}
    ckpts = [_load(lora_dir, folder) for folder in folders if not folder.name.startswith(".")]
    return sorted(ckpts, key=lambda c: c.id, reverse=True)


def resolve(lora_dir: Path, lora_id: str) -> LoraCheckpoint:
    """Look up one checkpoint by id, refusing paths outside ``lora_dir``."""
    lora_dir = lora_dir.resolve()
    ckpt_dir = (lora_dir / lora_id).resolve()
    if not ckpt_dir.is_relative_to(lora_dir) or not any((ckpt_dir / m).is_file() for m in MARKERS):
        raise KeyError(f"Unknown LoRA checkpoint: {lora_id}")
    return _load(lora_dir, ckpt_dir)


def engine_folders() -> set[str]:
    """The names of the LoRA folder's engine folders (the engine ids)."""
    from . import engines

    return set(engines.classes())


def engine_dir(lora_dir: Path, engine: str) -> Path:
    """The folder of an engine's runs and voices."""
    return lora_dir / engine


def folder_engine(folder: Path) -> str | None:
    """The engine of a run or voice folder (its train_config.yaml, else its
    checkpoints' lora_config.json), or None when it holds neither."""
    import yaml

    config = folder / "train_config.yaml"
    if config.is_file():
        try:
            data = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            return None
        return str(data.get("engine") or DEFAULT_ENGINE) if isinstance(data, dict) else None
    for ckpt in scan(folder):
        return ckpt.engine
    return None


def migrate(lora_dir: Path) -> list[tuple[str, str]]:
    """Move runs and voices lying directly in ``lora_dir`` into their engine's
    folder; returns the (from, to) names moved. Paths into the folder that a
    run's train_config records (absolute, or the webui's ``lora/<name>/...``)
    follow it. A folder whose engine can't be told, or whose name is taken in
    its engine folder, stays (and is logged)."""
    lora_dir = lora_dir.resolve()
    if not lora_dir.is_dir():
        return []
    folders, moved = engine_folders(), []
    for item in sorted(lora_dir.iterdir()):
        if not item.is_dir() or item.name.startswith(".") or item.name in folders:
            continue
        engine = folder_engine(item)
        if engine is None:
            logger.warning("Left %s in the LoRA folder: no run or checkpoint found in it", item.name)
            continue
        target = engine_dir(lora_dir, engine) / item.name
        if target.exists():
            logger.warning("Left %s in the LoRA folder: %s already exists", item.name, target)
            continue
        target.parent.mkdir(exist_ok=True)
        item.rename(target)
        _repoint_configs(target, item, lora_dir.name, engine)
        moved.append((item.name, f"{engine}/{item.name}"))
        logger.info("Moved %s into the LoRA folder's %s/", item.name, engine)
    return moved


def _repoint_configs(folder: Path, old: Path, lora_name: str, engine: str) -> None:
    """In a moved run's configs, paths into its old folder point into the new one."""
    import yaml

    # The old folder as an absolute path, or as the webui wrote it (relative to its working directory).
    old_forms = re.compile(
        "^(" + "|".join(re.escape(form) for form in (str(old), f"{lora_name}\\{old.name}", f"{lora_name}/{old.name}"))
        + r")(?=$|[\\/])"
    )  # fmt: skip

    def repoint(value):
        if isinstance(value, dict):
            return {k: repoint(v) for k, v in value.items()}
        if isinstance(value, list):
            return [repoint(v) for v in value]
        if isinstance(value, str) and (m := old_forms.match(value)):
            new = (
                str(folder)
                if m.group(1) == str(old)
                else f"{lora_name}{value[len(lora_name)]}{engine}{value[len(lora_name)]}{old.name}"
            )
            return new + value[m.end() :]
        return value  # fmt: skip

    for name in ("train_config.yaml", "train_config.orig.yaml"):
        path = folder / name
        if not path.is_file():
            continue
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        new = repoint(data)
        if new != data:
            path.write_text(yaml.safe_dump(new, sort_keys=False, allow_unicode=True), encoding="utf-8")
