"""Separation plumbing, without real models (no GPU, nothing downloaded)."""

import json

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from test_importer import wait_job, write_list

from voxcpm_studio.api import create_app
from voxcpm_studio.prep import importer, separation
from voxcpm_studio.prep.separation import Chain, SeparationService


def test_config_loader_reads_msst_tuples_and_nothing_else_unsafe(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("model:\n  sizes: !!python/tuple\n  - 4096\n  - 2048\naudio:\n  chunk_size: 100\n")
    assert separation.load_config(path)["model"]["sizes"] == (4096, 2048)

    path.write_text("x: !!python/object/apply:os.system ['echo hi']\n")
    with pytest.raises(Exception):  # noqa: B017 - any yaml error: arbitrary objects are refused
        separation.load_config(path)


def test_resolve_model_finds_the_config(tmp_path):
    folder = tmp_path / "m"
    folder.mkdir()
    (folder / "a.ckpt").write_bytes(b"")
    (folder / "config.yaml").write_text("{}")
    assert separation.resolve_model(str(folder / "a.ckpt")) == (folder / "a.ckpt", folder / "config.yaml")

    (folder / "other.yaml").write_text("{}")  # now ambiguous
    with pytest.raises(ValueError, match="which config"):
        separation.resolve_model(str(folder / "a.ckpt"))
    assert separation.resolve_model(f"{folder / 'a.ckpt'}|{folder / 'other.yaml'}")[1] == folder / "other.yaml"
    with pytest.raises(FileNotFoundError):
        separation.resolve_model(str(folder / "missing.ckpt"))


@pytest.mark.parametrize("length", [500, 4321, 20_000])
def test_demix_with_an_identity_model_returns_the_input(length):
    audio = np.random.default_rng(length).normal(0, 0.3, (2, length)).astype(np.float32)

    out = separation.demix(lambda batch: batch, audio, chunk=1000, overlap=2)

    assert out.shape == audio.shape and np.allclose(out, audio, atol=1e-5)


def halving_chain(rate=16000):
    """Stands in for a separation model: "removes" half of the signal."""
    calls = []

    def process(audio):
        calls.append(audio.shape)
        return audio * 0.5

    return Chain(rate=rate, channels=2, process=process), calls


def test_import_replaces_clips_with_the_separated_voice(tmp_path, data_dir):
    src = tmp_path / "speaker_c"
    src.mkdir()
    sf.write(src / "a.flac", np.full((32000, 2), 0.4, dtype=np.float32), 32000)  # stereo, 32 kHz
    (src / "a.lab").write_text("示例", encoding="utf-8")
    chain, calls = halving_chain()

    result = importer.import_dataset(str(src), None, "clean", data_dir, separation=chain)

    assert result.imported == 1 and calls == [(2, 16000)]  # read stereo, at the model's rate
    out = data_dir / "clean" / "wavs" / "a.wav"
    info = sf.info(str(out))
    assert (info.samplerate, info.channels) == (16000, 1)
    assert np.abs(sf.read(out)[0]).mean() == pytest.approx(0.2, abs=0.01)
    manifest = (data_dir / "clean" / "train.jsonl").read_text(encoding="utf-8")
    assert json.loads(manifest)["audio"] == "wavs/a.wav"
    with pytest.raises(ValueError, match="needs copying"):
        importer.import_dataset(str(src), None, "in_place", data_dir, copy=False, separation=chain)


def test_service_loads_a_chain_once_and_runs_models_in_order():
    loads = []

    class Model:
        def __init__(self, spec):
            self.spec, self.rate, self.channels = spec, 44100, 2

        def __call__(self, audio):
            return audio + (1 if self.spec == "vocals" else 10)

    service = SeparationService(
        lambda: "cpu", loader=lambda specs, device: loads.append(specs) or [Model(s) for s in specs]
    )

    chain = service.chain(["vocals", "dereverb"])
    assert chain.process(np.zeros((2, 4)))[0, 0] == 11 and (chain.rate, chain.channels) == (44100, 2)
    service.chain(["vocals", "dereverb"])
    assert loads == [["vocals", "dereverb"]] and service.loaded
    service.unload()
    assert not service.loaded


def test_import_endpoint_refuses_clean_up_without_a_model(service, training, asr_service, data_dir, tmp_path):
    clip = tmp_path / "a.wav"
    sf.write(clip, np.zeros(16000, dtype=np.float32), 16000)
    lst = write_list(tmp_path / "speaker_a.list", [f"{clip}|speaker_a|ZH|示例"])
    client = TestClient(create_app(service.settings, service, training, asr_service))

    assert client.get("/api/status").json()["cleanup"] == []
    resp = client.post("/api/datasets/import", json={"source": str(lst), "name": "x", "separate": True})
    assert resp.status_code == 422 and "not downloaded" in resp.json()["detail"]


def test_import_endpoint_separates_with_the_configured_chain(
    service, training, asr_service, data_dir, tmp_path, monkeypatch
):
    from dataclasses import replace

    settings = replace(service.settings, vocals_model="vocals.ckpt", denoise_model="denoise.ckpt")
    chain, calls = halving_chain()
    asked = []
    monkeypatch.setattr(SeparationService, "chain", lambda self, specs: asked.append(specs) or chain)
    clip = tmp_path / "a.wav"
    sf.write(clip, np.full(16000, 0.4, dtype=np.float32), 16000)
    lst = write_list(tmp_path / "speaker_a.list", [f"{clip}|speaker_a|ZH|示例"])
    client = TestClient(create_app(settings, service, training, asr_service))

    assert client.get("/api/status").json()["cleanup"] == ["separate", "denoise"]
    body = {"source": str(lst), "name": "speaker_a", "separate": True, "denoise": True}
    job = wait_job(client, client.post("/api/datasets/import", json=body).json()["id"])

    assert job["state"] == "done", job
    assert asked == [["vocals.ckpt", "denoise.ckpt"]] and len(calls) == 1
