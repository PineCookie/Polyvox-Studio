import io
from pathlib import Path

import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from voxcpm_studio.api import create_app, free_gpu_for_training
from voxcpm_studio.inference import SynthesisRequest

from conftest import make_dataset, make_lora, wait_idle


@pytest.fixture
def client(service, training):
    return TestClient(create_app(service.settings, service, training))


def test_list_loras(client, lora_dir):
    make_lora(lora_dir, "voice/step_1", r=64)
    make_lora(lora_dir, "legacy", r=None)

    body = client.get("/api/loras").json()

    assert [(x["id"], x["config"] and x["config"]["r"]) for x in body] == [("voice/step_1", 64), ("legacy", None)]


def test_tts_returns_wav_and_seed(client):
    resp = client.post("/api/tts", data={"text": "hello", "seed": "7"})

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"
    assert resp.headers["x-seed"] == "7"
    audio, sr = sf.read(io.BytesIO(resp.content))
    assert sr == 48000 and len(audio) == 4800


def test_tts_reference_upload_is_passed_and_cleaned_up(client, loader):
    resp = client.post(
        "/api/tts",
        data={"text": "hello", "prompt_text": "spoken"},
        files={"reference_audio": ("ref.wav", b"RIFF", "audio/wav")},
    )

    assert resp.status_code == 200
    call = loader.models[0].generate_calls[0]
    assert call["reference_wav_path"].endswith(".wav")
    assert call["prompt_text"] == "spoken"
    assert not Path(call["reference_wav_path"]).exists()


def test_tts_errors_map_to_http_codes(client):
    assert client.post("/api/tts", data={"text": "  "}).status_code == 422
    assert client.post("/api/tts", data={"text": "hi", "lora_id": "../x"}).status_code == 404
    assert client.post("/api/tts", data={"text": "hi", "cfg_value": "99"}).status_code == 422


def test_status_reports_loaded_model(client):
    assert client.get("/api/status").json()["loaded"] is False
    client.post("/api/tts", data={"text": "hello"})
    assert client.get("/api/status").json()["loaded"] is True
    assert client.post("/api/model/unload").json()["loaded"] is False


def test_datasets_list_and_inspect(client, data_dir):
    make_dataset(data_dir, "voice")

    assert client.get("/api/datasets").json() == ["voice/train.jsonl"]
    report = client.get("/api/datasets/voice/train.jsonl").json()
    assert (report["samples"], report["valid"]) == (4, 4)
    assert report["sample_rates"] == {"16000": 2, "44100": 2}
    assert client.get("/api/datasets/../x.jsonl").status_code == 404


def test_training_run_lifecycle(client, training, data_dir):
    make_dataset(data_dir, "voice")

    resp = client.post("/api/training/runs", json={"name": "v1", "dataset": "voice/train.jsonl", "num_iters": 20})
    assert resp.status_code == 200 and resp.json()["name"] == "v1"
    wait_idle(training)

    (summary,) = client.get("/api/training/runs").json()
    assert (summary["name"], summary["status"]) == ("v1", "complete")
    detail = client.get("/api/training/runs/v1").json()
    sample = client.get(f"/api/training/runs/v1/samples/{detail['samples'][0]['file']}")
    assert sample.status_code == 200 and sample.headers["content-type"] == "audio/wav"

    assert client.post("/api/training/runs/v1/resume").status_code == 422
    assert client.post("/api/training/stop").status_code == 409
    assert client.get("/api/training/runs/nope").status_code == 404
    assert client.get("/api/training/runs/v1/samples/..%2Ftrain_config.yaml").status_code == 404
    assert client.post("/api/training/runs", json={"name": "../x", "dataset": "voice/train.jsonl"}).status_code == 422


@pytest.mark.parametrize(
    "device, train_gpu, unloaded",
    [("cuda", None, True), ("cuda:0", 0, True), ("cuda:1", None, False), ("cuda", 1, False), ("cpu", None, False)],
)
def test_training_frees_the_inference_gpu_it_needs(service, device, train_gpu, unloaded):
    service.synthesize(SynthesisRequest(text="hi"))
    service.device = device

    free_gpu_for_training(service, train_gpu)

    assert service.status()["loaded"] is not unloaded
