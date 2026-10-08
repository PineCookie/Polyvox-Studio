"""Runtime settings, read from ``VOXCPM_STUDIO_*`` environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ENV_PREFIX = "VOXCPM_STUDIO_"


def _env(name: str, default: str) -> str:
    return os.environ.get(ENV_PREFIX + name, default)


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    # Local model directory or Hugging Face repo id.
    model: str = "openbmb/VoxCPM2"
    lora_dir: Path = Path("lora")
    # Where JSONL training manifests are found.
    data_dir: Path = Path("data")
    device: str = "auto"
    # None = compile with Triton whenever running on CUDA.
    optimize: bool | None = None
    load_denoiser: bool = True
    # CUDA device index for training jobs; None = the default device.
    train_gpu: int | None = None
    # Speech recognition (transformers-native Qwen3-ASR): local dir or Hugging Face id.
    asr_model: str = "Qwen/Qwen3-ASR-1.7B-hf"
    # "auto" = the inference device.
    asr_device: str = "auto"

    @classmethod
    def from_env(cls) -> Settings:
        optimize = _env("OPTIMIZE", "auto")
        train_gpu = _env("TRAIN_GPU", "").strip()
        return cls(
            model=_env("MODEL", cls.model),
            lora_dir=Path(_env("LORA_DIR", str(cls.lora_dir))),
            data_dir=Path(_env("DATA_DIR", str(cls.data_dir))),
            device=_env("DEVICE", cls.device),
            optimize=None if optimize == "auto" else _parse_bool(optimize),
            load_denoiser=_parse_bool(_env("DENOISER", "1")),
            train_gpu=int(train_gpu) if train_gpu else None,
            asr_model=_env("ASR_MODEL", cls.asr_model),
            asr_device=_env("ASR_DEVICE", cls.asr_device),
        )
