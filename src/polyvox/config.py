"""Runtime settings, read from ``POLYVOX_*`` environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

from . import models

ENV_PREFIX = "POLYVOX_"


def _env(name: str, default: str) -> str:
    return os.environ.get(ENV_PREFIX + name, default)


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "on")


def optimize_from_env() -> bool:
    """Settings.optimize as a training subprocess sees it (the server's
    environment): POLYVOX_OPTIMIZE on / off, or auto = on CUDA."""
    value = _env("OPTIMIZE", "auto").strip().lower()
    if value != "auto":
        return _parse_bool(value)
    import torch

    return torch.cuda.is_available()


@dataclass(frozen=True)
class Settings:
    # Model settings by name (models.all_specs: "model", "asr_model", ...): a local folder / file or
    # a hub id, where one is configured (from_env: POLYVOX_<SETTING>); the others use the model's
    # downloaded copy, else its default (model_path).
    model_settings: dict[str, str] = field(default_factory=dict)
    lora_dir: Path = Path("lora")
    # Where JSONL training manifests are found.
    data_dir: Path = Path("data")
    device: str = "auto"
    # None = compile with Triton whenever running on CUDA.
    optimize: bool | None = None
    load_denoiser: bool = True
    # CUDA device index for training jobs; None = the default device.
    train_gpu: int | None = None
    # "auto" = the inference device; forced aligners run there too.
    asr_device: str = "auto"
    # Downloaded models (models.py) live here; a model setting left at its
    # default uses the copy here once it is complete.
    models_dir: Path = Path("models")
    # Hugging Face or a mirror of it (e.g. https://hf-mirror.com) for downloads;
    # "" = huggingface.co.
    hf_endpoint: str = ""
    # Ask GitHub once a day whether a newer release is out (updates.py); from_env: on unless
    # POLYVOX_UPDATE_CHECK=0.
    update_check: bool = False
    # Host names the server answers to besides localhost and IP addresses (server/guard.py), e.g. a
    # machine name when it listens on the network; from_env: POLYVOX_ALLOWED_HOSTS, comma-separated.
    allowed_hosts: tuple[str, ...] = ()

    def configured(self, setting: str) -> bool:
        """A model setting was set to something other than its default (e.g. by an environment variable)."""
        spec = models.by_setting().get(setting)
        return setting in self.model_settings and self.model_settings[setting] != (spec.default if spec else "")

    def model_path(self, setting: str) -> str:
        """The model a setting names: as configured when set explicitly, else the
        downloaded copy in models_dir, else its default (a hub id that loading
        downloads to its cache; "" = not available)."""
        spec = models.by_setting().get(setting)
        default = spec.default if spec else ""
        if spec is None or self.configured(setting):
            return self.model_settings.get(setting, default)
        return str(spec.path(self.models_dir).resolve()) if spec.downloaded(self.models_dir) else default

    def with_models(self, **values: str) -> Settings:
        """These settings with some model settings changed (by name)."""
        return replace(self, model_settings={**self.model_settings, **values})

    @classmethod
    def from_env(cls) -> Settings:
        optimize = _env("OPTIMIZE", "auto")
        train_gpu = _env("TRAIN_GPU", "").strip()
        names = (spec.setting for spec in models.all_specs())
        return cls(
            model_settings={
                n: os.environ[ENV_PREFIX + n.upper()] for n in names if ENV_PREFIX + n.upper() in os.environ
            },
            lora_dir=Path(_env("LORA_DIR", str(cls.lora_dir))),
            data_dir=Path(_env("DATA_DIR", str(cls.data_dir))),
            device=_env("DEVICE", cls.device),
            optimize=None if optimize == "auto" else _parse_bool(optimize),
            load_denoiser=_parse_bool(_env("DENOISER", "1")),
            train_gpu=int(train_gpu) if train_gpu else None,
            asr_device=_env("ASR_DEVICE", cls.asr_device),
            models_dir=Path(_env("MODELS_DIR", str(cls.models_dir))),
            hf_endpoint=_env("HF_ENDPOINT", cls.hf_endpoint).rstrip("/"),
            update_check=_parse_bool(_env("UPDATE_CHECK", "1")),
            allowed_hosts=tuple(h.strip().lower() for h in _env("ALLOWED_HOSTS", "").split(",") if h.strip()),
        )
