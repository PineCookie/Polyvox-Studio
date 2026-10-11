from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from safetensors.torch import save_file
from torch import nn
from voxcpm.modules.layers.lora import apply_lora_to_named_linear_modules

from voxcpm_studio.config import Settings
from voxcpm_studio.engines.voxcpm.inference import InferenceService
from voxcpm_studio.prep.aligner import Alignment, AlignService, Word
from voxcpm_studio.prep.asr import AsrService
from voxcpm_studio.training import TrainingManager

FAKE_WORKER = Path(__file__).with_name("fake_train_worker.py")


DIM = 8  # width of the tiny model's linear layers
LM_LAYERS = [f"base_lm.layers.{i}.self_attn.{p}" for i in range(2) for p in ("q_proj", "k_proj", "v_proj", "o_proj")]


def make_lora(
    lora_dir: Path,
    rel: str,
    r: int | None = 32,
    base_model: str = "openbmb/VoxCPM2",
    alpha: float | None = None,
    layers: list[str] = LM_LAYERS,
) -> Path:
    """Create a checkpoint with real LoRA weights for the tiny model; r=None
    omits lora_config.json (the default layout, rank 32, applies)."""
    ckpt = lora_dir / rel
    ckpt.mkdir(parents=True)
    rank = r or 32
    gen = torch.Generator().manual_seed(rank)
    tensors = {}
    for layer in layers:
        tensors[f"{layer}.lora_A"] = torch.randn(rank, DIM, generator=gen)
        tensors[f"{layer}.lora_B"] = torch.randn(DIM, rank, generator=gen)
    save_file(tensors, ckpt / "lora_weights.safetensors")
    if r is not None:
        cfg = {"enable_lm": True, "enable_dit": True, "r": r, "alpha": alpha or r}
        (ckpt / "lora_config.json").write_text(json.dumps({"lora_config": cfg, "base_model": base_model}))
    return ckpt


class TinyTTS(nn.Module):
    """Stands in for a VoxCPM model: same LoRA injection on a few small layers."""

    def __init__(self, lora_config):
        super().__init__()

        def block():
            attn = nn.Module()
            for p in ("q_proj", "k_proj", "v_proj", "o_proj"):
                setattr(attn, p, nn.Linear(DIM, DIM))
            layer = nn.Module()
            layer.self_attn = attn
            return layer

        self.base_lm = nn.Module()
        self.base_lm.layers = nn.ModuleList([block(), block()])
        self.feat_decoder = nn.Module()
        self.feat_decoder.layers = nn.ModuleList([block()])
        self.sample_rate = 48000
        self.last_successful_seed = 7
        kwargs = dict(r=lora_config.r, alpha=lora_config.alpha, dropout=lora_config.dropout)
        if lora_config.enable_lm:
            apply_lora_to_named_linear_modules(
                self.base_lm, target_submodule_names=lora_config.target_modules_lm, **kwargs
            )
        if lora_config.enable_dit:
            apply_lora_to_named_linear_modules(
                self.feat_decoder, target_submodule_names=lora_config.target_modules_dit, **kwargs
            )

    def lora_delta(self, layer: str) -> torch.Tensor:
        """Effective weight change of one layer: scaling * B @ A."""
        module = self.get_submodule(layer)
        return module.scaling * module.lora_B @ module.lora_A


class FakeModel:
    def __init__(self, lora_config):
        self.lora_config = lora_config
        self.tts_model = TinyTTS(lora_config)
        self.generate_calls: list[dict] = []

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        return np.zeros(4800, dtype=np.float32)


class FakeAsrInputs(dict):
    def to(self, *args):
        return self


class FakeAsrProcessor:
    """Mimics the transformers Qwen3ASRProcessor calls AsrService makes.
    Transcribes each clip as "clip of <N> samples"."""

    def __init__(self):
        self.requests = []
        self.forced = False

    def apply_transcription_request(self, audios, language=None, prompt=None):
        self.requests.append({"n": len(audios), "language": language, "prompt": prompt})
        lengths = torch.tensor([[len(a)] for a in audios], dtype=torch.float32)
        return FakeAsrInputs(input_ids=torch.zeros(len(audios), 2), lengths=lengths, language=language)

    def decode(self, tokens, skip_special_tokens, return_format):
        # FakeAsrModel.generate puts each clip's length after the 2 prompt tokens.
        return [
            {"transcription": f" clip of {int(row[0])} samples ", "language": None if self.forced else "Chinese"}
            for row in tokens
        ]


class FakeAsrModel:
    device = "cpu"
    dtype = torch.float32

    def __init__(self, processor):
        self.processor = processor

    def generate(self, input_ids, lengths, language, max_new_tokens):
        self.processor.forced = language is not None
        return torch.cat([input_ids, lengths], dim=1)


class FakeAsrLoader:
    def __init__(self):
        self.calls = 0

    def __call__(self, model, device):
        self.calls += 1
        processor = FakeAsrProcessor()
        return processor, FakeAsrModel(processor)


class FakeAligner:
    """Spreads the words evenly over the clip's non-silent part; a word "错"
    ("wrong") anywhere makes the score low, as a mismatched text would."""

    def __init__(self, kind):
        self.kind = kind
        self.calls = []

    def align(self, audios, words):
        self.calls.append(len(audios))
        out = []
        for audio, ws in zip(audios, words, strict=True):
            loud = np.nonzero(np.abs(audio) > 0.01)[0]
            lo, hi = (loud[0], loud[-1] + 1) if len(loud) else (0, len(audio))
            step = (hi - lo) / len(ws) / 16000
            out.append(
                Alignment(
                    [
                        Word(w, lo / 16000 + k * step, lo / 16000 + (k + 1) * step, 0.1 if w == "错" else 0.9)
                        for k, w in enumerate(ws)
                    ],
                    0.1 if "错" in ws else 0.9,
                )
            )
        return out


class FakeAlignLoader:
    def __init__(self):
        self.loaded: list[str] = []
        self.backends: list[FakeAligner] = []

    def __call__(self, kind, model, device):
        self.loaded.append(kind)
        self.backends.append(FakeAligner(kind))
        return self.backends[-1]


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


class FakeSingingModel:
    """Stands in for the AudioSet tagger: a clip "is sung" when it is loud
    throughout (a held tone), else it is speech."""

    device = "cpu"

    def __call__(self, input_values):
        logits = torch.full((input_values.shape[0], 527), -20.0)
        for i, x in enumerate(input_values):
            sung = float((x.abs() > 0.01).float().mean()) > 0.9
            logits[i, 27] = 0.0 if sung else -20.0  # Singing
            logits[i, 0] = -20.0 if sung else 3.0  # Speech
        return type("Out", (), {"logits": logits})()


@pytest.fixture(autouse=True)
def fake_singing_detector(monkeypatch):
    """No test downloads or runs the real tagger."""
    from voxcpm_studio.prep import singing

    def extractor(wavs, sampling_rate, return_tensors):
        n = max(len(w) for w in wavs)
        x = torch.zeros(len(wavs), n)
        for i, w in enumerate(wavs):
            x[i, : len(w)] = torch.from_numpy(np.asarray(w, dtype=np.float32))
        return type("Features", (), {"input_values": x})()

    monkeypatch.setattr(singing, "load_ast", lambda model, device: (extractor, FakeSingingModel()))


@pytest.fixture(autouse=True)
def fake_speaker_model(monkeypatch):
    """No test downloads or runs CAM++: a clip's embedding is its coarse spectrum."""
    from voxcpm_studio import speaker

    def embed(wav):
        spectrum = np.abs(np.fft.rfft(wav, n=16_000))[:2000]
        return spectrum.reshape(40, 50).sum(axis=1) + 1e-6

    monkeypatch.setattr(speaker, "load_campplus", lambda model: embed)


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
def settings(lora_dir, data_dir, model_dir, tmp_path):
    # An empty models folder: nothing counts as downloaded.
    models = tmp_path / "models"
    return Settings(model=str(model_dir), lora_dir=lora_dir, data_dir=data_dir, device="cpu", models_dir=models)


@pytest.fixture
def loader():
    return FakeLoader()


@pytest.fixture
def service(settings, loader):
    return InferenceService(settings, loader=loader)


@pytest.fixture
def asr_loader():
    return FakeAsrLoader()


@pytest.fixture
def asr_service(settings, asr_loader):
    return AsrService(settings, loader=asr_loader)


@pytest.fixture
def align_loader():
    return FakeAlignLoader()


@pytest.fixture
def align_service(settings, align_loader):
    return AlignService(settings, loader=align_loader)


@pytest.fixture
def training(settings):
    manager = TrainingManager(settings, command=lambda *paths: [sys.executable, str(FAKE_WORKER), *map(str, paths)])
    yield manager
    manager.shutdown(timeout=10)
