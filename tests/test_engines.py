import json
import sys

import numpy as np
import pytest
from fastapi.testclient import TestClient

from polyvox import engines, lora_library, training
from polyvox.engines import Capabilities, Engine, Param, SynthesisResult
from polyvox.engines.voxcpm.engine import VoxCPMEngine
from polyvox.engines.voxcpm.inference import SynthesisRequest
from polyvox.server import create_app
from polyvox.training import TrainParams

from conftest import make_dataset, make_lora, wait_idle


class OtherEngine(Engine):
    """A second engine that only records what it was asked."""

    id = "other"
    name = "Other"
    capabilities = Capabilities(24_000, 24_000, ("en",), True, False, False, False, False)
    params = (Param("steps", "int", 8, 1, 32),)
    model_setting = "model"
    worker_module = "other_worker"

    def __init__(self):
        self.requests = []
        self.is_loaded = False

    @property
    def device(self):
        return "cuda:0"

    @property
    def loaded(self):
        return self.is_loaded

    def status(self):
        return {"loaded": self.is_loaded}

    def synthesize(self, req):
        self.requests.append((req, self.options(req.options)))
        self.is_loaded = True
        return SynthesisResult(24_000, np.zeros(240, dtype=np.float32), req.seed)

    def progress(self):
        return {"phase": "idle"}

    def unload(self):
        self.is_loaded = False

    def training_config(self, params, run_dir, train_manifest, val_manifest):
        return {
            "num_iters": params.num_iters,
            "save_path": str(run_dir / "checkpoints"),
            "train_manifest": str(train_manifest),
            "val_manifest": str(val_manifest or ""),
            "lora": {"r": params.lora_rank, "alpha": params.lora_alpha},
        }


@pytest.fixture
def other():
    return OtherEngine()


@pytest.fixture
def registry(settings, service, other):
    return {"voxcpm": VoxCPMEngine(settings, service), "other": other}


@pytest.fixture
def manager(settings, registry):
    manager = training.TrainingManager(
        settings, command=lambda *paths: [sys.executable, "-c", "pass"], engines_=registry
    )
    yield manager
    manager.shutdown(timeout=10)


@pytest.fixture
def client(settings, service, registry, manager):
    return TestClient(create_app(settings, service, manager, engines_=registry))


def test_engines_are_listed_with_their_settings(client):
    body = {e["id"]: e for e in client.get("/api/engines").json()}

    assert set(body) == {"voxcpm", "other"}
    assert body["voxcpm"]["capabilities"]["style"] is True
    assert [p["name"] for p in body["voxcpm"]["params"]] == ["cfg_value", "inference_timesteps", "normalize", "denoise"]
    assert body["other"]["params"][0] == {
        "name": "steps",
        "kind": "int",
        "default": 8,
        "min": 1,
        "max": 32,
        "step": None,
        "choices": None,
        "base_only": False,
        "versions": [],
    }


def test_tts_options_reach_the_engine_and_old_fields_still_work(client, loader):
    resp = client.post("/api/tts", data={"text": "hi", "options": json.dumps({"cfg_value": 1.6, "denoise": False})})
    assert resp.status_code == 200
    call = loader.models[0].generate_calls[-1]
    assert (call["cfg_value"], call["denoise"], call["inference_timesteps"]) == (1.6, False, 10)

    client.post("/api/tts", data={"text": "hi", "options": json.dumps({"cfg_value": 1.6}), "cfg_value": "2.5"})
    assert loader.models[0].generate_calls[-1]["cfg_value"] == 2.5


def test_tts_option_errors(client):
    assert client.post("/api/tts", data={"text": "hi", "engine": "nope"}).status_code == 404
    assert client.post("/api/tts", data={"text": "hi", "options": "[1]"}).status_code == 422
    assert client.post("/api/tts", data={"text": "hi", "options": '{"steps": 4}'}).status_code == 422
    assert client.post("/api/tts", data={"text": "hi", "options": '{"cfg_value": 9}'}).status_code == 422


def test_one_tts_engine_per_gpu(client, service, other):
    service.device = "cuda:0"
    client.post("/api/tts", data={"text": "hi"})
    assert service.status()["loaded"]

    resp = client.post("/api/tts", data={"text": "hi", "engine": "other", "options": '{"steps": 4}'})

    assert resp.status_code == 200
    assert other.requests[0][1] == {"steps": 4}
    assert not service.status()["loaded"]  # made room
    assert client.get("/api/status").json()["loaded_engines"] == ["other"]
    assert client.post("/api/model/unload").json()["loaded_engines"] == []


def test_runs_record_their_engine(client, manager, data_dir, lora_dir):
    make_dataset(data_dir, "voice")

    for name, engine in (("a", "voxcpm"), ("b", "other")):
        resp = client.post("/api/training/runs", json={"name": name, "engine": engine, "dataset": "voice/train.jsonl"})
        assert resp.status_code == 200, resp.text
        assert resp.json()["engine"] == engine
        assert training.read_config(lora_dir / engine / name)["engine"] == engine
        wait_idle(manager)

    assert training.read_config(lora_dir / "other" / "b")["lora"] == {"r": 32, "alpha": 32}
    assert "pretrained_path" not in training.read_config(lora_dir / "other" / "b")  # VoxCPM's keys stay VoxCPM's
    bad = client.post("/api/training/runs", json={"name": "c", "engine": "nope", "dataset": "voice/train.jsonl"})
    assert bad.status_code == 422


def test_worker_command_runs_the_engines_worker(tmp_path):
    training.write_config(tmp_path, {"num_iters": 1})
    cmd = training.worker_command(tmp_path / training.CONFIG_FILE, tmp_path / "samples", tmp_path / "stop")
    assert cmd[1:3] == ["-m", "polyvox.engines.voxcpm.worker"]

    training.write_config(tmp_path, {"engine": "nope"})
    with pytest.raises(KeyError):
        training.worker_command(tmp_path / training.CONFIG_FILE, tmp_path / "samples", tmp_path / "stop")


def test_other_engines_checkpoints(service, loader, lora_dir):
    make_lora(lora_dir, "vox/step_1", r=16)
    ckpt = lora_dir / "dots" / "step_1"
    ckpt.mkdir(parents=True)
    (ckpt / lora_library.WEIGHTS_FILE).write_bytes(b"")
    info = {"engine": "other", "lora_config": {"r": 64, "alpha": 32, "targets": ["x"]}, "base_model": "org/other"}
    (ckpt / lora_library.CONFIG_FILE).write_text(json.dumps(info))

    found = {c.id: c for c in lora_library.scan(lora_dir)}
    assert (found["dots/step_1"].engine, found["dots/step_1"].config) == ("other", None)
    assert (found["dots/step_1"].rank, found["dots/step_1"].alpha) == (64, 32.0)
    assert (found["vox/step_1"].engine, found["vox/step_1"].rank) == ("voxcpm", 16)

    with pytest.raises(ValueError, match="another model"):
        service.synthesize(SynthesisRequest(text="hi", lora_id="dots/step_1"))
    service.synthesize(SynthesisRequest(text="hi"))
    assert loader.models[-1].lora_config.r == 16  # the other engine's rank 64 is not covered


def test_default_engine_and_params():
    assert engines.DEFAULT == "voxcpm" and TrainParams(name="x", dataset="d").engine == "voxcpm"
    p = Param("n", "int", 1, 0, 5)
    assert p.coerce("3") == 3
    with pytest.raises(ValueError):
        p.coerce(9)
    assert Param("b", "bool", True).coerce("false") is False
