"""Model downloads: where settings point, and the download job (nothing is fetched)."""

import dataclasses

from fastapi.testclient import TestClient
from test_importer import wait_job

from voxcpm_studio import models
from voxcpm_studio.api import create_app


def fake_download(spec, models_dir, endpoint=None, progress=None):
    folder = models_dir / spec.folder
    folder.mkdir(parents=True, exist_ok=True)
    for name in spec.files:
        (folder / name).write_bytes(b"x")
    if progress:
        progress(spec.size_mb, spec.size_mb)
    fake_download.calls.append((spec.setting, endpoint))


fake_download.calls = []


def test_a_setting_uses_the_downloaded_copy_once_complete(settings):
    spec = models.BY_SETTING["asr_model"]
    assert settings.model_path("asr_model") == "Qwen/Qwen3-ASR-1.7B-hf"  # not downloaded: the hub id

    folder = settings.models_dir / spec.folder
    folder.mkdir(parents=True)
    (folder / "config.json").write_text("{}")
    assert settings.model_path("asr_model") == "Qwen/Qwen3-ASR-1.7B-hf"  # incomplete: not used

    fake_download(spec, settings.models_dir)
    assert settings.model_path("asr_model") == str(folder.resolve())
    # A model set explicitly is used as configured.
    assert dataclasses.replace(settings, asr_model="/home/someone/asr").model_path("asr_model") == "/home/someone/asr"
    # Separation models point to their checkpoint file; "" until downloaded.
    vocals = models.BY_SETTING["vocals_model"]
    assert settings.model_path("vocals_model") == ""
    fake_download(vocals, settings.models_dir)
    assert settings.model_path("vocals_model").endswith(vocals.entry)


def test_models_endpoint_lists_and_downloads(service, training, asr_service, monkeypatch):
    fake_download.calls = []
    monkeypatch.setattr(models, "download", fake_download)
    client = TestClient(create_app(service.settings, service, training, asr_service))

    listed = {m["setting"]: m for m in client.get("/api/models").json()["models"]}
    assert listed["model"]["state"] == "custom"  # the tests' fake model dir
    assert listed["vocals_model"]["state"] == "missing" and listed["model"]["required"]
    assert client.get("/api/status").json()["cleanup"] == []

    body = {"settings": ["vocals_model", "asr_model"], "endpoint": "https://hf-mirror.com/"}
    job = wait_job(client, client.post("/api/models/download", json=body).json()["id"])
    assert job["state"] == "done" and job["result"] == {"downloaded": ["vocals_model", "asr_model"]}
    assert fake_download.calls == [("vocals_model", "https://hf-mirror.com"), ("asr_model", "https://hf-mirror.com")]
    listed = {m["setting"]: m for m in client.get("/api/models").json()["models"]}
    assert listed["vocals_model"]["state"] == "downloaded"
    assert client.get("/api/status").json()["cleanup"] == ["separate"]  # usable without a restart

    assert client.post("/api/models/download", json={"settings": ["nope"]}).status_code == 422
    bad_source = {"settings": ["asr_model"], "endpoint": "hf-mirror.com"}
    assert client.post("/api/models/download", json=bad_source).status_code == 422
