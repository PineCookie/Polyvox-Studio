"""RVC voices in the LoRA folder: ``rvc/<name>/`` holds voice.json (engine, version, rate),
rvc_voice.pth (RVC's voice file, as its WebUI saves it: the WebUI and Applio load ours too) and,
optionally, rvc_index.pt (the voice's training content features, float16 [rows, 768]; a training
run keeps one in its folder for all its checkpoints).

A shareable package (sharing.py) holds rvc_voice.pth (re-saved with RVC's own header fields only),
voice.json and the index as rvc_index.npy (a plain array: faiss users build their .index from it)."""

import io
import json
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

ENGINE = "rvc"
BASE_REPO = "lj1995/VoiceConversionWebUI"
WEIGHTS = "rvc_voice.pth"
INDEX = "rvc_index.pt"
INFO = "voice.json"  # lora_library.VOICE_FILE: marks the folder as a voice
PACKAGE_INDEX = "rvc_index.npy"
PACKAGE_FILES = (WEIGHTS, INFO, PACKAGE_INDEX)
# What RVC's own voice files hold; Applio adds its own fields (author, model name, hashes,
# dataset length...), which a package leaves out.
VOICE_KEYS = ("weight", "config", "info", "sr", "f0", "version")
MAX_INDEX_ROWS = 2_000_000
RATES = (32000, 40000, 48000)  # RVC v2's (train.RATES)


def check_voice_file(path: Path) -> dict:
    """The voice file's header (sr, version, ...) when it is an RVC v2 voice with pitch, else ValueError."""
    try:
        ck = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:  # not a torch file, or one that needs code to load
        raise ValueError(f"{path.name} isn't an RVC voice file ({e.__class__.__name__}).") from e
    if not isinstance(ck, dict) or "weight" not in ck or "config" not in ck:
        raise ValueError(f"{path.name} isn't an RVC voice file (no weights / config).")
    if ck.get("version") != "v2" or not ck.get("f0", 1):
        raise ValueError(
            f"{path.name}: only RVC v2 voices with pitch are supported (this one: {ck.get('version', 'v1')})."
        )
    # Applio records these; RVC's WebUI always uses ContentVec features and the HiFi-GAN decoder.
    embedder = str(ck.get("embedder_model") or "contentvec")
    if embedder.lower() != "contentvec":
        raise ValueError(f"{path.name} was trained on {embedder} features; only ContentVec voices are supported.")
    vocoder = str(ck.get("vocoder") or "HiFi-GAN")
    if vocoder.lower() != "hifi-gan":
        raise ValueError(f"{path.name} uses the {vocoder} vocoder; only HiFi-GAN voices are supported.")
    return {"sample_rate": int(ck["config"][-1]), "version": "v2", "info": str(ck.get("info", ""))}


def write_info(folder: Path, sample_rate: int, source: str) -> None:
    info = {"engine": ENGINE, "version": "v2", "sample_rate": sample_rate, "base_model": BASE_REPO, "source": source}
    (folder / INFO).write_text(json.dumps(info, indent=1), encoding="utf-8")


def import_webui(voice_dir: Path, pth: Path, source: str = "imported", index: torch.Tensor | None = None) -> dict:
    """An RVC voice file copied into ``voice_dir`` (created; must not exist), with its index if given."""
    header = check_voice_file(pth)
    if header["sample_rate"] not in RATES:
        raise ValueError(
            f"{pth.name}: only 32, 40 and 48 kHz voices are supported (this one: {header['sample_rate']} Hz)."
        )
    voice_dir.mkdir(parents=True)
    try:
        shutil.copyfile(pth, voice_dir / WEIGHTS)
        write_info(voice_dir, header["sample_rate"], source)
        if index is not None:
            save_index(voice_dir, index)
    except BaseException:
        shutil.rmtree(voice_dir, ignore_errors=True)
        raise
    return header


def voice_root(checkpoint: Path) -> Path:
    """The folder a checkpoint belongs to: a training run's for its rvc/<run>/checkpoints/step_N."""
    return checkpoint.parent.parent if checkpoint.parent.name == "checkpoints" else checkpoint


def index_file(checkpoint: Path, voice_root: Path) -> Path | None:
    """The index a checkpoint uses: its own, else its run's / voice folder's."""
    return next((p for p in (checkpoint / INDEX, voice_root / INDEX) if p.is_file()), None)


def package_files(checkpoint: Path) -> dict[str, bytes]:
    """A voice's files for a shareable package, by file name: the voice file with RVC's header
    fields only (an "<N>epoch" info kept, any other text dropped), voice.json without its source,
    and the index as a plain float16 array."""
    header = check_voice_file(checkpoint / WEIGHTS)
    ck = torch.load(checkpoint / WEIGHTS, map_location="cpu", weights_only=True)
    voice = {k: ck[k] for k in VOICE_KEYS if k in ck}
    voice["info"] = voice["info"] if re.fullmatch(r"\d+epoch", str(voice.get("info", ""))) else ""
    buf = io.BytesIO()
    torch.save(voice, buf)
    info = {"engine": ENGINE, "version": "v2", "sample_rate": header["sample_rate"], "base_model": BASE_REPO}
    files = {WEIGHTS: buf.getvalue(), INFO: (json.dumps(info, indent=2) + "\n").encode()}
    if index := index_file(checkpoint, voice_root(checkpoint)):
        rows = torch.load(index, map_location="cpu", weights_only=True)
        buf = io.BytesIO()
        np.save(buf, rows.half().numpy(), allow_pickle=False)
        files[PACKAGE_INDEX] = buf.getvalue()
    return files


def package_index(data: bytes) -> torch.Tensor:
    """A package's index array, checked (2-D, 768 features, float, finite)."""
    try:
        array = np.load(io.BytesIO(data), allow_pickle=False)
    except Exception as e:
        raise ValueError(f"{PACKAGE_INDEX} isn't a plain array file ({e.__class__.__name__}).") from e
    if array.ndim != 2 or array.shape[1] != 768 or not 0 < array.shape[0] <= MAX_INDEX_ROWS:
        raise ValueError(f"{PACKAGE_INDEX} isn't an RVC v2 index (shape {array.shape}).")
    if array.dtype.kind != "f" or not np.isfinite(array).all():
        raise ValueError(f"{PACKAGE_INDEX} holds no usable features.")
    return torch.from_numpy(array.astype(np.float16))


def import_package(engine_dir: Path, name: str, files: dict[str, bytes]) -> str:
    """A voice from a package's files (package_files), checked as WebUI voice files are."""
    target = engine_dir / name
    if target.exists():
        raise FileExistsError(f"A voice named '{name}' already exists.")
    index = package_index(files[PACKAGE_INDEX]) if PACKAGE_INDEX in files else None
    with tempfile.TemporaryDirectory(prefix="rvc_") as tmp:
        pth = Path(tmp) / WEIGHTS
        pth.write_bytes(files[WEIGHTS])
        import_webui(target, pth, "package", index)
    return name


def save_index(folder: Path, features: torch.Tensor) -> int:
    """Rows kept (k-means centers above 200k rows, as RVC); written atomically."""
    from .convert import reduce_features

    rows = reduce_features(features).half().cpu().contiguous()
    tmp = folder / (INDEX + ".tmp")
    torch.save(rows, tmp)
    tmp.replace(folder / INDEX)
    return rows.shape[0]


def webui_features(logs_run: Path) -> torch.Tensor:
    """The features an RVC WebUI run built its index from (logs/<run>/total_fea.npy: plain array)."""
    path = logs_run / "total_fea.npy" if logs_run.is_dir() else logs_run
    array = np.load(path, allow_pickle=False)
    if array.ndim != 2 or array.shape[1] != 768:
        raise ValueError(f"{path.name} isn't an RVC v2 feature file (shape {array.shape}).")
    return torch.from_numpy(array.astype(np.float16))
