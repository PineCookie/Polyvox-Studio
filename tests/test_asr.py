import dataclasses
import io

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from polyvox.prep import asr
from polyvox.prep.asr import AsrService
from polyvox.server import create_app, free_gpu_for_training


def test_transcribe_keeps_input_order_across_batches(asr_service, asr_loader):
    lengths = [16000 * (i % 7 + 1) for i in range(40)]

    results = asr_service.transcribe([np.zeros(n, dtype=np.float32) for n in lengths])

    assert [r.text for r in results] == [f"clip of {n} samples" for n in lengths]
    assert all(r.language == "Chinese" for r in results)
    processor = asr_service._processor
    assert [r["n"] for r in processor.requests] == [asr.BATCH_SIZE, 40 - asr.BATCH_SIZE]
    assert asr_loader.calls == 1


def test_long_clips_make_smaller_batches(asr_service):
    seconds = asr.BATCH_SECONDS // 4  # four of them fill a batch
    asr_service.transcribe([np.zeros(16000 * seconds, dtype=np.float32) for _ in range(10)])

    assert [r["n"] for r in asr_service._processor.requests] == [4, 4, 2]


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
def test_training_frees_the_asr_gpu(service, settings, asr_loader, device, train_gpu, unloaded, monkeypatch):
    # Device names as configured, also without a GPU (e.g. CI on the CPU build).
    monkeypatch.setattr("voxcpm.model.utils.resolve_runtime_device", lambda device, default: device)
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


def test_language_lock_forces_a_language_once_enough_clips_agree():
    lock = asr.LanguageLock()
    clip, blip = np.zeros(3 * asr.SAMPLE_RATE, np.float32), np.zeros(asr.SAMPLE_RATE // 2, np.float32)
    zh, ja = asr.Transcript("示例", "Chinese"), asr.Transcript("嗯", "Japanese")
    lock.add([clip] * 15, [zh] * 15)
    lock.add([blip] * 10, [ja] * 10)  # short clips don't vote
    assert lock.language is None  # 15 votes: not enough yet
    lock.add([clip], [zh])
    assert lock.language == "Chinese"

    mixed = asr.LanguageLock()
    mixed.add([clip] * 40, [zh] * 30 + [asr.Transcript("example", "English")] * 10)
    assert mixed.language is None  # 75% agree: keep detecting per clip
