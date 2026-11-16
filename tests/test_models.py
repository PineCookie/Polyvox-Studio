"""Model downloads: where settings point, and the download job (nothing is fetched)."""

import dataclasses

import pytest
from fastapi.testclient import TestClient
from test_importer import wait_job

from polyvox import models
from polyvox.server import create_app


def fake_download(spec, models_dir, endpoint=None, progress=None, way="prebuilt"):
    folder = models_dir / spec.folder
    folder.mkdir(parents=True, exist_ok=True)
    for name in spec.files:
        (folder / name).write_bytes(b"x")
    if progress:
        progress(spec.size_mb, spec.size_mb)
    fake_download.calls.append((spec.setting, endpoint))
    fake_download.ways.append((spec.setting, way))


fake_download.calls = []
fake_download.ways = []


def test_download_progress_counts_received_bytes_once():
    """huggingface_hub's Xet reporter, given our bar class, sends bytes received and
    bytes written through that one bar (not two bars counted twice)."""
    import logging
    from types import SimpleNamespace

    from huggingface_hub.utils._xet_progress_reporting import XetDownloadProgressReporter

    received = models._Received()
    reporter = XetDownloadProgressReporter(
        reconstruction_desc="model.safetensors", total=100, log_level=logging.WARNING, tqdm_class=received.hf_bar()
    )

    def report(written, transferred):
        reporter.update_progress(
            SimpleNamespace(
                total_bytes_completed=written,
                total_transfer_bytes_completed=transferred,
                total_bytes_completion_rate=None,
                total_transfer_bytes_completion_rate=None,
                total_bytes=100,
            )
        )

    report(0, 30)  # received, not yet written
    assert (received.written, received.transferred, received.bytes) == (0, 30, 30)
    report(60, 55)  # written catches up (compressed transfers can stay below the size)
    assert (received.written, received.transferred, received.bytes) == (60, 55, 60)

    callback = received.modelscope_callback()("g2pw.zip", 100)  # ModelScope's per-file callback
    received.reset()
    callback.update(25)
    assert received.bytes == 25


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
    assert listed["vocals_model"]["state"] == "missing" and "required" not in listed["model"]
    assert client.get("/api/status").json()["cleanup"] == []
    assert client.get("/api/status").json()["no_speech_model"] is False  # the tests' VoxCPM is configured

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


def test_any_engines_model_counts_as_a_speech_model(settings):
    """No model is required: one engine's model, downloaded or configured, is enough."""
    names = ["model", "dots_model", "drama3_model"]
    unset = dataclasses.replace(settings, model=type(settings)().model)  # VoxCPM at its default (the hub id)
    assert not models.any_available(unset, names)
    fake_download(models.BY_SETTING["dots_model"], unset.models_dir)
    assert models.any_available(unset, names)
    assert models.any_available(settings, ["model"])  # the tests' VoxCPM is configured


def test_models_come_in_groups_and_drama3_offers_its_ways(service, training, asr_service, monkeypatch):
    fake_download.calls, fake_download.ways = [], []
    monkeypatch.setattr(models, "download", fake_download)
    client = TestClient(create_app(service.settings, service, training, asr_service))
    listed = {m["setting"]: m for m in client.get("/api/models").json()["models"]}
    assert {m["group"] for m in listed.values()} == set(models.GROUPS)
    assert listed["asr_model"]["group"] == "dataset" and listed["vocals_model"]["group"] == "cleanup"
    assert [w["id"] for w in listed["drama3_model"]["ways"]] == ["prebuilt", "convert", "convert_keep"]
    assert listed["drama3_model"]["ways"][1]["size_mb"] > listed["drama3_model"]["ways"][0]["size_mb"]
    assert listed["asr_model"]["ways"] == []  # one way only
    body = {"settings": ["drama3_model", "asr_model"], "ways": {"drama3_model": "convert_keep"}}
    job = wait_job(client, client.post("/api/models/download", json=body).json()["id"])
    assert job["state"] == "done" and fake_download.ways == [
        ("drama3_model", "convert_keep"),
        ("asr_model", "prebuilt"),
    ]
    wrong = {"settings": ["asr_model"], "ways": {"asr_model": "convert"}}  # nothing to convert from
    assert client.post("/api/models/download", json=wrong).status_code == 422


def test_converting_keeps_the_source_only_when_asked(tmp_path, monkeypatch):
    built = []

    def fake_build(spec, models_dir, endpoint, progress, source, keep=False):
        built.append((source.repo, keep))

    monkeypatch.setattr(models, "_download_and_build", fake_build)
    monkeypatch.setattr(models, "_fetch_hf", lambda *a: pytest.fail("the build isn't wanted"))
    spec = models.BY_SETTING["drama3_model"]
    models.download(spec, tmp_path, way="convert")
    models.download(spec, tmp_path, way="convert_keep")
    assert built == [("fishaudio/drama-3", False), ("fishaudio/drama-3", True)]
    with pytest.raises(ValueError, match="can't be downloaded"):
        models.download(models.BY_SETTING["asr_model"], tmp_path, way="convert")
