import io
from pathlib import Path

import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from voxcpm_studio.api import create_app

from conftest import make_lora


@pytest.fixture
def client(service):
    return TestClient(create_app(service.settings, service))


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
