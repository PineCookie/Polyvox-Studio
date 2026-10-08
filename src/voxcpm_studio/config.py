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
    device: str = "auto"
    # None = compile with Triton whenever running on CUDA.
    optimize: bool | None = None
    load_denoiser: bool = True

    @classmethod
    def from_env(cls) -> Settings:
        optimize = _env("OPTIMIZE", "auto")
        return cls(
            model=_env("MODEL", cls.model),
            lora_dir=Path(_env("LORA_DIR", str(cls.lora_dir))),
            device=_env("DEVICE", cls.device),
            optimize=None if optimize == "auto" else _parse_bool(optimize),
            load_denoiser=_parse_bool(_env("DENOISER", "1")),
        )
