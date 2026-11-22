"""RVC: the model's checkpoint handling, the conversion pipeline's parts, voices and the Convert
API, with a tiny model (same code, small sizes) and fake feature / pitch models."""

import dataclasses
import io
import json
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
    with pytest.raises(ValueError, match="32, 40 and 48 kHz"):
        voices.import_webui(tmp_path / "voice", tiny_voice(tmp_path / "d.pth", sample_rate=44100))
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
    assert calls[-1] == (2, 2)  # cut once near 38 s
    assert len(out) == len(audio) // 160 * 400  # to the frame: no piece comes back short
    for seconds in (79.99, 80.0, 80.01, 121.37):  # more pieces, of odd and even frame counts
        longer = np.resize(audio, int(16000 * seconds))
        assert len(rt.converter.convert(voice, longer, 16000)) == len(longer) // 160 * 400


def test_long_audio_is_cut_in_a_quiet_moment():
    t = np.arange(16000 * 45) / 16000
    audio = np.sin(t * 2 * np.pi * 220) * 0.3
    audio[(t > 35.0) & (t < 35.3)] *= 0.01  # a breath between two lines
    (cut,) = convert.quiet_cuts(audio)
    assert 35.0 < cut / 16000 < 35.3
    assert convert.quiet_cuts(audio[: 16000 * 40]) == []  # fits one piece


def rvc_client(settings, service, training, **extra) -> TestClient:
    settings = dataclasses.replace(settings, rvc_model="rvc-models", contentvec_model="contentvec", **extra)
    registry = engines.create(settings, voxcpm=VoxCPMEngine(settings, service), rvc=RvcEngine(settings, fake_runtime))
    return TestClient(create_app(settings, service, training, engines_=registry))


@pytest.fixture
def rvc_app(settings, service, training):
    return rvc_client(settings, service, training)


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


class HalfBacking:
    """A karaoke network that calls half of everything backing vocals."""

    def get_inputs(self):
        return [dataclasses.make_dataclass("Input", ["name"])("input")]

    def run(self, _outputs, feed):
        return [feed["input"] * 0.5]


def test_the_split_gives_lead_and_backing_back_as_a_whole():
    from polyvox.prep import karaoke

    t = np.arange(karaoke.RATE * 7) / karaoke.RATE  # longer than one chunk
    vocals = np.stack([np.sin(2 * np.pi * 220 * t), np.sin(2 * np.pi * 330 * t)]).astype(np.float32) * 0.5
    calls = []
    lead, backing = karaoke.Karaoke(lambda _: HalfBacking()).split("kara.onnx", vocals, lambda d, n: calls.append(n))
    assert lead.shape == backing.shape == vocals.shape and calls[-1] == len(calls) == 2
    inner = slice(karaoke.N_FFT, -karaoke.N_FFT)
    assert np.allclose(backing[:, inner], vocals[:, inner] * 0.5, atol=1e-3)
    assert np.allclose(lead + karaoke.COMPENSATE * backing, vocals, atol=1e-6)


def test_a_split_converts_the_lead_and_keeps_or_converts_the_backing(
    rvc_app, settings, service, training, tmp_path, monkeypatch
):
    from polyvox.prep import karaoke

    pth = tiny_voice(tmp_path / "speaker_a.pth")
    assert rvc_app.post("/api/rvc/import", json={"name": "speaker_a", "path": str(pth)}).status_code == 200
    buf = io.BytesIO()
    sf.write(buf, np.sin(np.arange(44100 * 2) / 44100 * 2 * np.pi * 220) * 0.3, 44100, format="WAV")
    files = {"audio": ("song.wav", buf.getvalue(), "audio/wav")}
    form = {"voice": "rvc/speaker_a", "separate": "false", "dereverb": "false", "split": "true"}
    r = rvc_app.post("/api/convert", data=form, files=files)
    assert r.status_code == 422 and "karaoke" in r.json()["detail"]  # its model isn't there
    assert rvc_app.get("/api/status").json()["karaoke"] is False

    monkeypatch.setattr(karaoke, "load_session", lambda _: HalfBacking())
    client = rvc_client(settings, service, training, karaoke_model=str(tmp_path / "kara.onnx"))
    assert client.get("/api/status").json()["karaoke"] is True
    assert client.post("/api/convert", data={**form, "backing": "nope"}, files=files).status_code == 422
    for backing in ("keep", "convert"):
        r = client.post("/api/convert", data={**form, "backing": backing, "seed": "3"}, files=files)
        assert r.status_code == 202
        job = wait_job(client, r.json()["id"])
        assert job["state"] == "done", job["error"]
        assert job["result"]["files"] == ["vocals_converted", "vocals", "lead", "backing"]
        lead, rate = sf.read(io.BytesIO(client.get(f"/api/convert/{job['id']}/lead").content))
        back, _ = sf.read(io.BytesIO(client.get(f"/api/convert/{job['id']}/backing").content))
        sung, _ = sf.read(io.BytesIO(client.get(f"/api/convert/{job['id']}/vocals_converted").content))
        assert rate == 44100 and lead.shape == back.shape == sung.shape
        assert abs(np.sqrt(np.mean(back**2)) / np.sqrt(np.mean(lead**2)) - 0.5 / (1 - 0.5 * 1.065)) < 0.05
        if backing == "keep":  # the converted lead plus the backing as it was
            kept = sung
        else:
            assert not np.allclose(sung, kept, atol=1e-3)  # the backing converted too


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


def applio_voice(path, **extra):
    """A tiny voice with the fields Applio adds to its voice files."""
    ck = torch.load(tiny_voice(path), weights_only=True)
    torch.save({**ck, "author": "someone", "model_name": "speaker_c", "dataset_length": 12, **extra}, path)
    return path


def test_voices_trained_on_other_features_or_vocoders_are_refused(tmp_path):
    assert voices.check_voice_file(applio_voice(tmp_path / "a.pth", embedder_model="contentvec"))
    with pytest.raises(ValueError, match="spin features"):
        voices.check_voice_file(applio_voice(tmp_path / "b.pth", embedder_model="spin"))
    with pytest.raises(ValueError, match="RefineGAN"):
        voices.check_voice_file(applio_voice(tmp_path / "c.pth", vocoder="RefineGAN"))


def test_a_voice_travels_as_a_package_with_its_index(rvc_app, tmp_path):
    import zipfile

    pth = applio_voice(tmp_path / "speaker_c.pth")
    logs = tmp_path / "logs" / "speaker_c"
    logs.mkdir(parents=True)
    features = np.random.default_rng(1).standard_normal((300, 768)).astype(np.float32)
    np.save(logs / "total_fea.npy", features)
    body = {"name": "speaker_c", "path": str(pth), "index_logs": str(logs)}
    assert rvc_app.post("/api/rvc/import", json=body).status_code == 200
    assert {e["id"]: e for e in rvc_app.get("/api/engines").json()}["rvc"]["capabilities"]["package"] == "voice"

    r = rvc_app.get("/api/loras/export", params={"id": "rvc/speaker_c", "name": "shared_c"})
    assert r.status_code == 200
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        assert sorted(zf.namelist()) == [f"shared_c/{f}" for f in sorted(["README.md", *voices.PACKAGE_FILES])]
        voice = torch.load(io.BytesIO(zf.read("shared_c/rvc_voice.pth")), weights_only=True)
        info = json.loads(zf.read("shared_c/voice.json"))
        readme = zf.read("shared_c/README.md").decode()
    assert set(voice) == set(voices.VOICE_KEYS) and voice["info"] == ""  # Applio's fields and free text gone
    assert info == {"engine": "rvc", "version": "v2", "sample_rate": 40000, "base_model": voices.BASE_REPO}
    assert "300 rows" in readme and "faiss.IndexFlatL2" in readme

    files = {"package": ("shared_c.zip", r.content, "application/zip")}
    r = rvc_app.post("/api/loras/import", files=files)
    assert r.status_code == 200 and r.json() == {"id": "rvc/shared_c", "engine": "rvc"}
    assert rvc_app.post("/api/loras/import", files=files).status_code == 409
    folder = tmp_path / "lora" / "rvc" / "shared_c"
    index = torch.load(folder / voices.INDEX, weights_only=True)
    assert torch.equal(index, torch.from_numpy(features).half())
    original = torch.load(pth, weights_only=True)["weight"]
    imported = torch.load(folder / voices.WEIGHTS, weights_only=True)["weight"]
    assert original.keys() == imported.keys() and all(torch.equal(original[k], imported[k]) for k in original)
    assert json.loads((folder / voices.INFO).read_text(encoding="utf-8"))["source"] == "package"

    bad = io.BytesIO()
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("broken/rvc_voice.pth", b"not a voice")
        zf.writestr("broken/rvc_index.npy", b"not an array")
    r = rvc_app.post("/api/loras/import", files={"package": ("broken.zip", bad.getvalue(), "application/zip")})
    assert r.status_code == 422 and not (tmp_path / "lora" / "rvc" / "broken").exists()


def test_each_rate_fits_together_as_rvc_configures_it():
    # RVC's saved voices: segment_size is the training window in frames (40k: 12800 samples / 400).
    assert [train.model_config(r).segment_size for r in (32000, 40000, 48000)] == [40, 32, 36]
    for rate in train.RATES:
        cfg, spectro = train.model_config(rate), train.Spectrograms(rate, "cpu")
        assert cfg.hop == spectro.hop == rate // 100  # the decoder upsamples a frame to one hop
        spec = spectro.linear(torch.zeros(1, cfg.segment_size * spectro.hop))
        assert spec.shape == (1, cfg.spec_channels, cfg.segment_size)
        assert spectro.mel(torch.zeros(1, 4 * spectro.hop)).shape[1] == train.RATES[rate].n_mels


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
    with pytest.raises(ValueError, match="base models"):  # not downloaded yet
        RvcEngine(settings, fake_runtime).training_config(None, tmp_path / "run", tmp_path / "x.jsonl", None)
    settings = dataclasses.replace(settings, rvc_train_model=str(tmp_path / "rvc" / "pretrained_v2"))
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
    assert config["feature_cache"].endswith(str(Path(".cache") / "rvc_v2_40k.pt")) and config["sample_rate"] == 40000
    config = engine.training_config(
        params.model_copy(update={"rvc_sample_rate": 48000}), tmp_path / "run", manifest, None
    )
    assert config["pretrained_d"].endswith("f0D48k.pth") and config["feature_cache"].endswith("rvc_v2_48k.pt")
    assert config["sample_rate"] == 48000 and config["rmvpe"] == str(tmp_path / "rvc" / "rmvpe.pt")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA"):
        engine.training_config(params, tmp_path / "run", manifest, None)
