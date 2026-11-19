"""RVC: the model's checkpoint handling, the conversion pipeline's parts, voices and the Convert
API, with a tiny model (same code, small sizes) and fake feature / pitch models."""

import dataclasses
import io
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from fastapi.testclient import TestClient

from polyvox import engines
from polyvox.engines.rvc import convert, model, train, voices
from polyvox.engines.rvc.engine import Runtime, RvcEngine
from polyvox.engines.voxcpm.engine import VoxCPMEngine
from polyvox.server.app import create_app

TINY = model.Config(
    spec_channels=17,
    inter_channels=8,
    hidden_channels=8,
    filter_channels=16,
    n_heads=2,
    n_layers=1,
    resblock_kernel_sizes=(3,),
    resblock_dilation_sizes=((1, 3),),
    upsample_rates=(10, 10, 4),  # 400 samples per frame: 40 kHz at 100 frames per second
    upsample_initial_channel=16,
    upsample_kernel_sizes=(20, 20, 8),
    spk_embed_dim=2,
    gin_channels=8,
)


def tiny_voice(path, version="v2", sample_rate=40000):
    torch.manual_seed(0)
    g = model.Synthesizer(dataclasses.replace(TINY, sample_rate=sample_rate))
    train.with_weight_norm(g, train.norm_layers(g))
    weights = {k: v.half() for k, v in train.plain_state(g).items() if not k.startswith("enc_q")}
    config = [getattr(g.cfg, f) for f in g.cfg.__dataclass_fields__]
    config = [[list(x) if isinstance(x, tuple) else x for x in v] if isinstance(v, tuple) else v for v in config]
    torch.save({"weight": weights, "config": config, "info": "test", "sr": "40k", "f0": 1, "version": version}, path)
    return path


class FakeContentVec:
    def __call__(self, audio16k):
        frames = max(1, (len(audio16k) - 400) // 320 + 1)
        return torch.ones(1, frames, 768) * float(np.abs(audio16k).mean())


class FakeRmvpe:
    def f0(self, audio, threshold=0.03):
        return np.full(len(audio) // 160 + 1, 220.0)


def fake_runtime(*_):
    cv, rm = FakeContentVec(), FakeRmvpe()
    return Runtime(cv, rm, convert.Converter(cv, rm, "cpu"))


def test_weight_norm_pairs_fold_into_the_weights_torch_computes():
    conv = torch.nn.utils.parametrizations.weight_norm(torch.nn.Conv1d(3, 4, 5))
    state = train.plain_state(torch.nn.Sequential(conv))
    assert {"0.weight_g", "0.weight_v", "0.bias"} <= set(state)
    folded = model.fold_weight_norm(state)
    assert torch.allclose(folded["0.weight"], conv.weight, atol=1e-6)


def test_a_saved_voice_loads_and_synthesizes(tmp_path):
    voice = model.load_voice(tiny_voice(tmp_path / "v.pth"), "cpu")
    assert voice.sample_rate == 40000 and voice.version == "v2" and not hasattr(voice.model, "enc_q")
    feats = torch.randn(1, 50, 768)
    pitch = torch.full((1, 50), 100, dtype=torch.long)
    f0 = torch.full((1, 50), 220.0)
    audio = voice.model.infer(feats, pitch, f0)
    assert audio.shape == (1, 50 * 400) and torch.isfinite(audio).all()


def test_voice_files_are_checked(tmp_path):
    assert voices.check_voice_file(tiny_voice(tmp_path / "a.pth"))["sample_rate"] == 40000
    with pytest.raises(ValueError, match="only RVC v2"):
        voices.check_voice_file(tiny_voice(tmp_path / "b.pth", version="v1"))
    (tmp_path / "c.pth").write_bytes(b"not a checkpoint")
    with pytest.raises(ValueError, match="isn't an RVC voice"):
        voices.check_voice_file(tmp_path / "c.pth")
    with pytest.raises(ValueError, match="40 kHz"):
        voices.import_webui(tmp_path / "voice", tiny_voice(tmp_path / "d.pth", sample_rate=48000))
    assert not (tmp_path / "voice").exists()


def test_pitch_is_quantized_and_octave_flips_are_folded_back():
    f0 = np.array([0.0, 50.0, 220.0, 1100.0, 5000.0])
    assert convert.coarse_pitch(f0).tolist()[:2] == [1, 1] and convert.coarse_pitch(f0)[-1] == 255
    line = np.full(60, 220.0)
    line[30] = 440.0  # one frame an octave up
    line[10:13] = 0.0  # unvoiced stays unvoiced
    cleaned = convert.clean_f0(line)
    assert np.allclose(cleaned[30], 220.0) and (cleaned[10:13] == 0).all()


def test_the_index_blends_nearest_features_and_reduces_big_ones(monkeypatch):
    features = torch.eye(768)[:20] * 3
    index = convert.Index(features)
    blended = index.blend(features[:2] + 0.001)
    assert torch.allclose(blended, features[:2], atol=1e-2)  # its own row dominates
    monkeypatch.setattr(convert, "INDEX_MAX_ROWS", 100)
    monkeypatch.setattr(convert, "INDEX_CENTERS", 10)
    many = torch.randn(500, 768)
    assert convert.reduce_features(many).shape == (10, 768)
    assert convert.reduce_features(many[:50]).shape == (50, 768)


def test_training_audio_is_cut_like_rvc():
    rate = 1000
    lengths = [len(p) for p in train.pieces(np.zeros(int(10.5 * rate)), rate)]
    assert lengths[:2] == [3700, 3700] and lengths[-1] <= 4000 and len(lengths) == 3
    assert [len(p) for p in train.pieces(np.zeros(3000), rate)] == [3000]


def test_conversion_covers_long_audio_in_pieces(tmp_path):
    voice = model.load_voice(tiny_voice(tmp_path / "v.pth"), "cpu")
    rt = fake_runtime()
    calls = []
    audio = np.sin(np.arange(16000 * 45) / 16000 * 2 * np.pi * 220).astype(np.float32) * 0.3
    out = rt.converter.convert(voice, audio, 16000, None, convert.Options(), lambda d, t: calls.append((d, t)))
    assert abs(len(out) / 40000 - 45) < 0.05 and calls[-1] == (2, 2)  # cut once near 38 s


@pytest.fixture
def rvc_app(settings, service, training):
    settings = dataclasses.replace(settings, rvc_model="rvc-models", contentvec_model="contentvec")
    registry = engines.create(settings, voxcpm=VoxCPMEngine(settings, service), rvc=RvcEngine(settings, fake_runtime))
    return TestClient(create_app(settings, service, training, engines_=registry))


def wait_job(client, job_id, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["state"] != "running":
            return job
        time.sleep(0.1)
    raise AssertionError("job didn't finish")


def test_voices_import_and_recordings_convert_through_the_api(rvc_app, tmp_path):
    pth = tiny_voice(tmp_path / "speaker_a.pth")
    r = rvc_app.post("/api/rvc/import", json={"name": "speaker_a", "path": str(pth)})
    assert r.status_code == 200 and r.json()["id"] == "rvc/speaker_a"
    assert rvc_app.post("/api/rvc/import", json={"name": "speaker_a", "path": str(pth)}).status_code == 409
    (listed,) = [v for v in rvc_app.get("/api/loras").json() if v["engine"] == "rvc"]
    assert listed["id"] == "rvc/speaker_a"
    caps = {e["id"]: e["capabilities"] for e in rvc_app.get("/api/engines").json()}
    assert caps["rvc"]["conversion"] and caps["rvc"]["full_finetune"] and not caps["voxcpm"]["conversion"]

    buf = io.BytesIO()
    sf.write(buf, np.sin(np.arange(44100 * 2) / 44100 * 2 * np.pi * 220) * 0.3, 44100, format="WAV")
    files = {"audio": ("song.wav", buf.getvalue(), "audio/wav")}
    form = {"voice": "rvc/speaker_a", "separate": "false", "dereverb": "false", "options": '{"shift": 2}'}
    r = rvc_app.post("/api/convert", data=form, files=files)
    assert r.status_code == 202
    job = wait_job(rvc_app, r.json()["id"])
    assert job["state"] == "done", job["error"]
    assert job["result"]["files"] == ["vocals_converted", "vocals"]
    wav = rvc_app.get(f"/api/convert/{job['id']}/vocals_converted")
    audio, rate = sf.read(io.BytesIO(wav.content))
    assert rate == 44100 and abs(len(audio) / rate - 2) < 0.05
    assert rvc_app.get(f"/api/convert/{job['id']}/cover").status_code == 404  # not separated: no remix

    bad = {**form, "options": '{"nope": 1}'}
    assert rvc_app.post("/api/convert", data=bad, files=files).status_code == 422
    assert rvc_app.post("/api/convert", data={**form, "voice": "rvc/nobody"}, files=files).status_code == 404
    separated = {**form, "separate": "true"}
    r = rvc_app.post("/api/convert", data=separated, files=files)
    assert r.status_code == 422 and "separation" in r.json()["detail"]


def test_an_index_comes_from_a_webui_feature_file(rvc_app, tmp_path):
    pth = tiny_voice(tmp_path / "speaker_b.pth")
    logs = tmp_path / "logs" / "speaker_b"
    logs.mkdir(parents=True)
    np.save(logs / "total_fea.npy", np.random.default_rng(0).standard_normal((300, 768)).astype(np.float32))
    r = rvc_app.post("/api/rvc/import", json={"name": "speaker_b", "path": str(pth), "index_logs": str(logs)})
    assert r.status_code == 200 and r.json()["index_rows"] == 300
    saved = torch.load(tmp_path / "lora" / "rvc" / "speaker_b" / voices.INDEX, weights_only=True)
    assert saved.dtype == torch.float16 and saved.shape == (300, 768)
    assert rvc_app.post("/api/rvc/index", json={"id": "rvc/speaker_b"}).status_code == 422


def test_piece_counts_match_the_cut():
    for seconds in (0.5, 3.7, 4.0, 4.01, 7.4, 7.41, 12.3, 30.0):
        assert train.piece_count(seconds, 1000) == len(train.pieces(np.zeros(int(round(seconds * 1000))), 1000)), (
            seconds
        )
    from polyvox.engines.rvc.worker import plan

    assert plan(1151, 16, 100) == (72, 7200)
    assert plan(5, 16, 3) == (1, 3)


def test_a_training_run_is_planned_from_the_dataset(settings, monkeypatch, tmp_path):
    from polyvox.training import TrainParams

    from conftest import make_dataset

    settings = dataclasses.replace(settings, rvc_model=str(tmp_path / "rvc"), contentvec_model=str(tmp_path / "cv"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    make_dataset(settings.data_dir, "speaker_a")
    # As TrainingManager hands it over: the run's copy of the manifest, with absolute paths.
    wavs = sorted((settings.data_dir / "speaker_a" / "wavs").glob("*.wav"))
    manifest = tmp_path / "train.jsonl"
    manifest.write_text("".join(f'{{"audio": "{w.as_posix()}", "text": "clip"}}\n' for w in wavs), encoding="utf-8")
    engine = RvcEngine(settings, fake_runtime)
    params = TrainParams(name="run_a", engine="rvc", dataset="speaker_a/train.jsonl", batch_size=2, rvc_epochs=3)
    config = engine.training_config(params, tmp_path / "run", manifest, None)
    assert config["pretrained_g"].endswith("f0G40k.pth") and config["rmvpe"].endswith("rmvpe.pt")
    assert config["epochs"] == 3 and config["save_every"] == 25 and config["save_path"].endswith("checkpoints")
    assert config["num_iters"] % 3 == 0 and config["num_iters"] >= 3
    assert config["feature_cache"].endswith(str(Path(".cache") / "rvc_v2_40k.pt"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA"):
        engine.training_config(params, tmp_path / "run", manifest, None)
