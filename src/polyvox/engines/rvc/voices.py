"""RVC voices in the LoRA folder: ``rvc/<name>/`` holds voice.json (engine, version, rate),
rvc_voice.pth (RVC's voice file, as its WebUI saves it: the WebUI and Applio load ours too) and,
optionally, rvc_index.pt (the voice's training content features, float16 [rows, 768]; a training
run keeps one in its folder for all its checkpoints)."""

import json
import shutil
from pathlib import Path

import numpy as np
import torch

ENGINE = "rvc"
WEIGHTS = "rvc_voice.pth"
INDEX = "rvc_index.pt"
INFO = "voice.json"  # lora_library.VOICE_FILE: marks the folder as a voice


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
    return {"sample_rate": int(ck["config"][-1]), "version": "v2", "info": str(ck.get("info", ""))}


def write_info(folder: Path, sample_rate: int, source: str) -> None:
    info = {
        "engine": ENGINE,
        "version": "v2",
        "sample_rate": sample_rate,
        "base_model": "lj1995/VoiceConversionWebUI",
        "source": source,
    }
    (folder / INFO).write_text(json.dumps(info, indent=1), encoding="utf-8")


def import_webui(voice_dir: Path, pth: Path) -> dict:
    """An RVC WebUI voice file copied into ``voice_dir`` (created; must not exist)."""
    header = check_voice_file(pth)
    if header["sample_rate"] != 40000:
        raise ValueError(
            f"{pth.name}: only 40 kHz voices are supported for now (this one: {header['sample_rate']} Hz)."
        )
    voice_dir.mkdir(parents=True)
    try:
        shutil.copyfile(pth, voice_dir / WEIGHTS)
        write_info(voice_dir, header["sample_rate"], "imported")
    except BaseException:
        shutil.rmtree(voice_dir, ignore_errors=True)
        raise
    return header


def index_file(checkpoint: Path, voice_root: Path) -> Path | None:
    """The index a checkpoint uses: its own, else its run's / voice folder's."""
    return next((p for p in (checkpoint / INDEX, voice_root / INDEX) if p.is_file()), None)


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
