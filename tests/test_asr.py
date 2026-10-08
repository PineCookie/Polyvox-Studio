import dataclasses
import io

import numpy as np
import pytest
import soundfile as sf
import torch
from fastapi.testclient import TestClient

from voxcpm_studio import asr
from voxcpm_studio.api import create_app, free_gpu_for_training
from voxcpm_studio.asr import AsrService


class FakeInputs(dict):
    def to(self, *args):
        return self


class FakeProcessor:
    """Mimics the transformers Qwen3ASRProcessor calls AsrService makes."""

    def __init__(self):
        self.requests = []

    def apply_transcription_request(self, audios, language=None, prompt=None):
        self.requests.append({"n": len(audios), "language": language, "prompt": prompt})
        lengths = torch.tensor([[len(a)] for a in audios], dtype=torch.float32)
        return FakeInputs(input_ids=torch.zeros(len(audios), 2), lengths=lengths, language=language)

    def decode(self, tokens, skip_special_tokens, return_format):
        # FakeModel.generate puts each clip's length after the 2 prompt tokens.
        return [
            {"transcription": f" clip of {int(row[0])} samples ", "language": None if self.forced else "Chinese"}
            for row in tokens
        ]


class FakeModel:
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
        processor = FakeProcessor()
        return processor, FakeModel(processor)


@pytest.fixture
def asr_loader():
    return FakeAsrLoader()


@pytest.fixture
def asr_service(settings, asr_loader):
    return AsrService(settings, loader=asr_loader)


def test_transcribe_keeps_input_order_across_batches(asr_service, asr_loader):
    lengths = [16000 * (i % 7 + 1) for i in range(20)]

    results = asr_service.transcribe([np.zeros(n, dtype=np.float32) for n in lengths])

    assert [r.text for r in results] == [f"clip of {n} samples" for n in lengths]
    assert all(r.language == "Chinese" for r in results)
    processor = asr_service._processor
    assert [r["n"] for r in processor.requests] == [16, 4]
    assert asr_loader.calls == 1


def test_language_and_prompt_are_forwarded(asr_service):
    (result,) = asr_service.transcribe([np.zeros(100, dtype=np.float32)], language="zh", prompt="小明")

    assert result.language is None  # forced, not detected
    assert asr_service._processor.requests[-1] == {"n": 1, "language": "zh", "prompt": ["小明"]}


def test_unload_frees_and_reloads_on_demand(asr_service, asr_loader):
    asr_service.transcribe([np.zeros(10, dtype=np.float32)])
    assert asr_service.loaded

    asr_service.unload()
    assert not asr_service.loaded
    asr_service.transcribe([np.zeros(10, dtype=np.float32)])
    assert asr_loader.calls == 2


def test_load_audio_downmixes_and_resamples(tmp_path):
    path = tmp_path / "stereo.wav"
    sf.write(path, np.zeros((32000, 2), dtype=np.float32), 32000)  # 1 s, stereo, 32 kHz

    wav = asr.load_audio(path)

    assert wav.dtype == np.float32 and wav.ndim == 1 and abs(len(wav) - 16000) <= 1


@pytest.mark.parametrize("device, train_gpu, unloaded", [("cuda:0", None, True), ("cuda:1", 0, False)])
def test_training_frees_the_asr_gpu(service, settings, asr_loader, device, train_gpu, unloaded):
    asr_service = AsrService(dataclasses.replace(settings, asr_device=device), loader=asr_loader)
    asr_service.transcribe([np.zeros(10, dtype=np.float32)])

    free_gpu_for_training(service, train_gpu, asr_service)

    assert asr_service.loaded is not unloaded


def test_asr_endpoint_and_unload(service, training, asr_service):
    client = TestClient(create_app(service.settings, service, training, asr_service))
    buf = io.BytesIO()
    sf.write(buf, np.zeros(8000, dtype=np.float32), 16000, format="WAV")

    resp = client.post("/api/asr", files={"audio": ("ref.wav", buf.getvalue(), "audio/wav")}, data={"language": "zh"})

    assert resp.json() == {"text": "clip of 8000 samples", "language": None}
    assert client.get("/api/status").json()["asr_loaded"] is True
    assert client.post("/api/model/unload").json()["asr_loaded"] is False
