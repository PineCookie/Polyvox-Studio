from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from voxcpm_studio.config import Settings
from voxcpm_studio.inference import InferenceService


def make_lora(lora_dir: Path, rel: str, r: int | None = 32, base_model: str = "openbmb/VoxCPM2") -> Path:
    """Create a fake checkpoint dir; r=None omits lora_config.json."""
    ckpt = lora_dir / rel
    ckpt.mkdir(parents=True)
    (ckpt / "lora_weights.safetensors").write_bytes(b"")
    if r is not None:
        cfg = {"enable_lm": True, "enable_dit": True, "r": r, "alpha": r}
        (ckpt / "lora_config.json").write_text(json.dumps({"lora_config": cfg, "base_model": base_model}))
    return ckpt


class FakeModel:
    def __init__(self, lora_config):
        self.lora_config = lora_config
        self.tts_model = SimpleNamespace(sample_rate=48000, last_successful_seed=7)
        self.loaded_paths: list[str] = []
        self.lora_enabled = False
        self.generate_calls: list[dict] = []

    def load_lora(self, path):
        self.loaded_paths.append(path)

    def set_lora_enabled(self, enabled):
        self.lora_enabled = enabled

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        return np.zeros(4800, dtype=np.float32)


class FakeLoader:
    def __init__(self):
        self.models: list[FakeModel] = []

    def __call__(self, settings, device, lora_config, optimize):
        model = FakeModel(lora_config)
        self.models.append(model)
        return model


@pytest.fixture
def lora_dir(tmp_path):
    d = tmp_path / "lora"
    d.mkdir()
    return d


@pytest.fixture
def loader():
    return FakeLoader()


@pytest.fixture
def service(lora_dir, loader):
    return InferenceService(Settings(lora_dir=lora_dir, device="cpu"), loader=loader)
