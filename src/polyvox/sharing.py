"""Shareable packages: export a checkpoint, import someone else's.

A package is a zip holding one folder, ``<name>/``, with the files its engine writes
(Engine.package_files): the weights, a README and a config naming the engine (lora_config.json,
or voice.json; no engine: VoxCPM). Nothing else from a run goes in: no optimizer state, local
paths, logs, manifests (transcripts) or dataset audio besides a voice's reference clip. Entries
carry no dates or permissions, so the same checkpoint always packs to the same bytes.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

from . import engines
from .lora_library import CONFIG_FILE, LEGACY_ENGINE, VOICE_FILE, LoraCheckpoint, engine_dir

ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)  # zip's earliest date: entries carry no export time
PACKAGE_NAME = r"^[^\W_][\w.-]{0,79}$"  # Unicode letters allowed, like run names
FILE_NAME = re.compile(r"[\w.-]{1,80}")
MAX_FILES = 16
MAX_PACKAGE_BYTES = 2 * 1024**3
STORED = (".safetensors", ".pth", ".ckpt", ".npy")  # weights and features don't compress


def _engine(engine_id: str) -> type[engines.Engine]:
    cls = engines.classes().get(engine_id)
    if cls is None or not cls.capabilities.package:
        raise ValueError(f"Unknown engine '{engine_id}', or one whose voices can't be shared as packages.")
    return cls


def export_package(ckpt: LoraCheckpoint, name: str, base_model: str, half: bool = True) -> Path:
    """Write a package for ``ckpt`` to a temporary zip file and return its path."""
    if not re.fullmatch(PACKAGE_NAME, name):
        raise ValueError("Package name may contain letters (any language), digits, '.', '_' and '-' only.")
    files = _engine(ckpt.engine).package_files(ckpt, name, base_model, half)
    fd, tmp = tempfile.mkstemp(prefix="polyvox_package_", suffix=".zip")
    os.close(fd)
    with zipfile.ZipFile(tmp, "w") as zf:
        for file, data in files.items():
            method = zipfile.ZIP_STORED if file.endswith(STORED) else zipfile.ZIP_DEFLATED
            info = zipfile.ZipInfo(f"{name}/{file}", date_time=ZIP_EPOCH)
            info.compress_type, info.external_attr = method, 0o644 << 16
            zf.writestr(info, data)
    return Path(tmp)


def read_package(package: Path, filename: str) -> tuple[str, dict[str, bytes]]:
    """A package's name and files. ``filename``: the uploaded file's name, which names it when the
    zip holds the files without a folder (a folder zipped again)."""
    try:
        zf = zipfile.ZipFile(package)
    except zipfile.BadZipFile as e:
        raise ValueError("Not a zip file.") from e
    with zf:
        entries = [info for info in zf.infolist() if not info.is_dir()]
        folders = set()
        for info in entries:
            parts = PurePosixPath(info.filename).parts
            if len(parts) > 2 or ".." in parts or info.filename.startswith(("/", "\\")):
                raise ValueError(f"Unexpected entry in package: {info.filename}")
            if not FILE_NAME.fullmatch(parts[-1]):
                raise ValueError(f"Unexpected file in package: {info.filename}")
            folders.add(parts[0] if len(parts) == 2 else PurePosixPath(filename.replace("\\", "/")).stem)
        if len(folders) != 1 or not entries:
            raise ValueError("A package holds exactly one voice.")
        if len(entries) > MAX_FILES or sum(i.file_size for i in entries) > MAX_PACKAGE_BYTES:
            raise ValueError("Package is too large.")
        name = folders.pop()
        if not re.fullmatch(PACKAGE_NAME, name):
            raise ValueError(f"Invalid package name: {name}")
        return name, {PurePosixPath(i.filename).name: zf.read(i) for i in entries}


def package_engine(files: dict[str, bytes]) -> str:
    """The engine its config names (lora_config.json, else voice.json). None named: a VoxCPM
    LoRA in voxcpm's own format (and our packages made before they named it)."""
    for config in (CONFIG_FILE, VOICE_FILE):
        if config in files:
            try:
                info = json.loads(files[config])
            except ValueError as e:
                raise ValueError(f"{config} is invalid: {e}") from e
            return str((info.get("engine") if isinstance(info, dict) else None) or LEGACY_ENGINE)
    raise ValueError(f"A package needs {CONFIG_FILE} or {VOICE_FILE}.")


def import_package(lora_dir: Path, package: Path, filename: str) -> tuple[str, str]:
    """Unpack a package into ``lora_dir/<engine>/<name>/``; returns the new id and its engine."""
    name, files = read_package(package, filename)
    engine = package_engine(files)
    folder = engine_dir(lora_dir.resolve(), engine)
    return f"{engine}/{_engine(engine).import_package(folder, name, files)}", engine
