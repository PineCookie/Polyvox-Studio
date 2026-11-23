"""Discovery and metadata for trained checkpoints and voices of every engine, all in the LoRA
folder: a folder holding a file one of the engines names first in its checkpoint_files (a LoRA's
weights, a GPT-SoVITS SoVITS file, a voice.json) is one; its config file names its engine.

The LoRA folder has one folder per engine (``voxcpm/``, ``dots_tts/``,
``gpt_sovits/``) holding its runs and voices; a checkpoint's id is its path
from the LoRA folder, e.g. ``gpt_sovits/speaker_a/checkpoints/step_0000100``.
Runs and voices from before that sat directly in the LoRA folder: ``migrate``
moves them into their engine's folder (still found where they are until then).
"""

from __future__ import annotations

import functools
import json
import logging
import re
from dataclasses import dataclass, replace
from pathlib import Path

logger = logging.getLogger(__name__)

WEIGHTS_FILE = "lora_weights.safetensors"  # a LoRA's weights (VoxCPM, dots.tts, S2-Pro)
# A voice's config when it isn't a LoRA (RVC, saved-reference voices): {"engine": ..., "base_model": ...},
# read like lora_config.json.
VOICE_FILE = "voice.json"
CONFIG_FILE = "lora_config.json"  # {"engine": ..., "base_model": ..., "lora_config": {...}}
RUN_CONFIGS = ("train_config.yaml", "train_config.orig.yaml")
# Runs and checkpoints from before there were engines (they record none) are VoxCPM's.
LEGACY_ENGINE = "voxcpm"


@functools.cache
def _files() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(files that mark a checkpoint folder, every engine's weight files): Engine.checkpoint_files."""
    from . import engines

    kinds = [cls.checkpoint_files for cls in engines.classes().values() if cls.checkpoint_files]
    return tuple(dict.fromkeys(files[0] for files in kinds)), tuple(dict.fromkeys(f for fs in kinds for f in fs))


@dataclass(frozen=True)
class LoraCheckpoint:
    # POSIX path relative to the LoRA directory, e.g. "speaker_r64/checkpoints/step_0005000".
    id: str
    path: Path
    engine: str  # "" when neither its config nor its folder names one
    base_model: str | None = None
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
        """When its weights last changed (any of its weight files; a saved-reference voice: its file)."""
        files = [self.path / f for f in _files()[1]]
        return max((f.stat().st_mtime for f in files if f.is_file()), default=0.0)

    @property
    def step(self) -> int | None:
        """Its training step: from its folder name (step_N), else its training_state.json."""
        if m := re.search(r"step_(\d+)$", self.id):
            return int(m.group(1))
        try:
            return int(json.loads((self.path / "training_state.json").read_text(encoding="utf-8"))["step"])
        except (OSError, ValueError, KeyError, TypeError):
            return None

    @property
    def rank(self) -> int | None:
        r = (self.raw_config or {}).get("r")
        return int(r) if isinstance(r, int | float) else None

    @property
    def alpha(self) -> float | None:
        alpha = (self.raw_config or {}).get("alpha")
        return float(alpha) if isinstance(alpha, int | float) else None


def _config_file(ckpt_dir: Path) -> Path:
    config_file = ckpt_dir / CONFIG_FILE
    return config_file if config_file.is_file() else ckpt_dir / VOICE_FILE


def _load(lora_dir: Path, ckpt_dir: Path) -> LoraCheckpoint:
    ckpt_id = ckpt_dir.relative_to(lora_dir).as_posix()
    folder = ckpt_id.split("/")[0]
    ckpt = LoraCheckpoint(id=ckpt_id, path=ckpt_dir, engine=folder if folder in engine_folders() else "")
    config_file = _config_file(ckpt_dir)
    if not config_file.is_file():
        return ckpt
    try:
        info = json.loads(config_file.read_text(encoding="utf-8"))
        raw = info.get("lora_config") or None
        version = info.get("version") if isinstance(info.get("version"), str) else None
        engine = str(info.get("engine") or ckpt.engine)
        return replace(ckpt, base_model=info.get("base_model"), engine=engine, raw_config=raw, version=version)
    except (OSError, ValueError, TypeError) as e:
        logger.warning("Ignoring unreadable %s: %s", config_file, e)
        return ckpt


def scan(lora_dir: Path) -> list[LoraCheckpoint]:
    """Return every checkpoint under ``lora_dir``, newest step first."""
    lora_dir = lora_dir.resolve()
    if not lora_dir.is_dir():
        return []
    folders = {marker.parent for name in _files()[0] for marker in lora_dir.rglob(name)}
    ckpts = [_load(lora_dir, folder) for folder in folders if not folder.name.startswith(".")]
    return sorted(ckpts, key=lambda c: c.id, reverse=True)


def resolve(lora_dir: Path, lora_id: str) -> LoraCheckpoint:
    """Look up one checkpoint by id, refusing paths outside ``lora_dir``."""
    lora_dir = lora_dir.resolve()
    ckpt_dir = (lora_dir / lora_id).resolve()
    if not ckpt_dir.is_relative_to(lora_dir) or not any((ckpt_dir / m).is_file() for m in _files()[0]):
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
    """The engine of a run or voice folder lying directly in the LoRA folder (its
    train_config.yaml, else its checkpoints' lora_config.json; none recorded: from before
    engines), or None when it holds neither."""
    import yaml

    config = folder / RUN_CONFIGS[0]
    if config.is_file():
        try:
            data = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            return None
        return str(data.get("engine") or LEGACY_ENGINE) if isinstance(data, dict) else None
    for ckpt in scan(folder):
        return ckpt.engine or LEGACY_ENGINE
    return None


def migrate(lora_dir: Path) -> list[tuple[str, str]]:
    """Move runs and voices lying directly in ``lora_dir`` into their engine's
    folder, then write the engine into every run and checkpoint config in an
    engine folder that records none (stamp_engines); returns the (from, to) names moved.
    Paths into the folder that a run's train_config records (absolute, or the webui's
    ``lora/<name>/...``) follow it. A folder whose engine can't be told, or whose name is
    taken in its engine folder, stays (and is logged)."""
    lora_dir = lora_dir.resolve()
    if not lora_dir.is_dir():
        return []
    moved = _move_into_engine_folders(lora_dir)
    stamp_engines(lora_dir)
    return moved


def stamp_engines(lora_dir: Path) -> int:
    """Write ``engine`` into the run configs and checkpoint configs of each engine folder that
    lack it (the folder names their engine); returns how many files changed."""
    import yaml

    changed = 0
    for engine in sorted(engine_folders()):
        folder = engine_dir(lora_dir, engine)
        if not folder.is_dir():
            continue
        for path in [p for name in RUN_CONFIGS for p in folder.glob(f"*/{name}")]:
            try:
                data = yaml.safe_load(path.read_text(encoding="utf-8"))
            except (OSError, yaml.YAMLError) as e:
                logger.warning("Ignoring unreadable %s: %s", path, e)
                continue
            if isinstance(data, dict) and not data.get("engine"):
                text = yaml.safe_dump({"engine": engine, **data}, sort_keys=False, allow_unicode=True)
                path.write_text(text, encoding="utf-8")
                changed += 1
        for path in [p for name in (CONFIG_FILE, VOICE_FILE) for p in folder.rglob(name)]:
            try:
                info = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as e:
                logger.warning("Ignoring unreadable %s: %s", path, e)
                continue
            if isinstance(info, dict) and not info.get("engine"):
                path.write_text(json.dumps({"engine": engine, **info}, indent=2), encoding="utf-8")
                changed += 1
    if changed:
        logger.info("Recorded the engine in %d run and checkpoint configs", changed)
    return changed


def _move_into_engine_folders(lora_dir: Path) -> list[tuple[str, str]]:
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
