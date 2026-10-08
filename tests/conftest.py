from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from voxcpm_studio.config import Settings
from voxcpm_studio.inference import InferenceService
from voxcpm_studio.training import TrainingManager

FAKE_WORKER = Path(__file__).with_name("fake_train_worker.py")


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


def make_dataset(data_dir: Path, name: str, rates=(16000, 16000, 44100, 44100)) -> Path:
    """Write a webui-style dataset: audio paths relative to data_dir's parent."""
    wavs = data_dir / name / "wavs"
    wavs.mkdir(parents=True)
    lines = []
    for i, rate in enumerate(rates):
        sf.write(wavs / f"{i}.wav", np.zeros(rate, dtype=np.float32), rate)  # 1 s
        lines.append(json.dumps({"audio": f"{data_dir.name}/{name}/wavs/{i}.wav", "text": f"clip {i}"}))
    manifest = data_dir / name / "train.jsonl"
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def wait_idle(manager: TrainingManager, timeout: float = 30) -> None:
    deadline = time.monotonic() + timeout
    while manager.active_run is not None:
        assert time.monotonic() < deadline, "training job did not finish"
        time.sleep(0.05)


def wait_for(path: Path, text: str, timeout: float = 30) -> None:
    deadline = time.monotonic() + timeout
    while not (path.is_file() and text in path.read_text(encoding="utf-8", errors="replace")):
        assert time.monotonic() < deadline, f"{text!r} never appeared in {path}"
        time.sleep(0.05)


@pytest.fixture
def lora_dir(tmp_path):
    d = tmp_path / "lora"
    d.mkdir()
    return d


@pytest.fixture
def data_dir(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    return d


@pytest.fixture
def model_dir(tmp_path):
    d = tmp_path / "model"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"audio_vae_config": {"sample_rate": 16000, "out_sample_rate": 48000}}))
    return d


@pytest.fixture
def settings(lora_dir, data_dir, model_dir):
    return Settings(model=str(model_dir), lora_dir=lora_dir, data_dir=data_dir, device="cpu")


@pytest.fixture
def loader():
    return FakeLoader()


@pytest.fixture
def service(settings, loader):
    return InferenceService(settings, loader=loader)


@pytest.fixture
def training(settings):
    manager = TrainingManager(settings, command=lambda *paths: [sys.executable, str(FAKE_WORKER), *map(str, paths)])
    yield manager
    manager.shutdown(timeout=10)
