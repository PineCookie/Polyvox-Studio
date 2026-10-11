import io
import json
import threading
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from fastapi.testclient import TestClient

from polyvox import lora_library
from polyvox.data import manifests
from polyvox.engines import SynthesisRequest
from polyvox.engines.gpt_sovits import voices
from polyvox.engines.gpt_sovits.engine import GptSovitsEngine
from polyvox.engines.gpt_sovits.synth import split_text
from polyvox.server import create_app

RATE = 16_000


def tone(path, seconds: float, rate: int = RATE):
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(seconds * rate)) / rate
    sf.write(path, (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), rate)
    return path


def webui_weights(folder, tag=b"06"):
    """A SoVITS file tagged as the webui saves v2ProPlus weights, and a GPT checkpoint."""
    folder.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    torch.save({"weight": {"w": torch.zeros(2)}, "config": {"model": {}}}, buf)
    sovits = folder / "voice_e8_s100.pth"
    sovits.write_bytes(tag + buf.getvalue()[2:])
    gpt = folder / "voice-e15.ckpt"
    torch.save({"weight": {"model.w": torch.zeros(2)}, "config": {"data": {"max_sec": 54}}}, gpt)
    return sovits, gpt


class FakeVoice:
    sample_rate = 32_000

    def __init__(self, key):
        self.key = key


class FakePipeline:
    """Stands in for synth.Pipeline: records what it was asked to do."""

    def __init__(self):
        self.voices, self.references, self.calls = [], [], []

    def base_weights(self, version="v2ProPlus"):
        return f"s2G{version}.pth", "s1v3.ckpt"

    def load_voice(self, sovits, gpt, key=None):
        self.voices.append((str(sovits), str(gpt)))
        return FakeVoice(key)

    def reference(self, voice, wav, text, key=None):
        self.references.append((str(wav), text))
        return ("ref", str(wav), text)

    def synthesize(self, voice, ref, text, options, seed, on_token=None):
        for _ in range(4):
            on_token()
        self.calls.append({"voice": voice.key, "ref": ref, "text": text, "options": options, "seed": seed})
        return np.zeros(320, dtype=np.float32)


@pytest.fixture
def pipeline():
    return FakePipeline()


@pytest.fixture
def engine(settings, pipeline, monkeypatch):
    monkeypatch.setattr(GptSovitsEngine, "unavailable", lambda self: None)
    return GptSovitsEngine(settings, loader=lambda base, g2pw, device: pipeline)


def make_voice(lora_dir, name="voice_a", reference=True, tmp=None):
    folder = lora_dir / name
    folder.mkdir(parents=True)
    (folder / voices.SOVITS_FILE).write_bytes(b"s")
    (folder / voices.GPT_FILE).write_bytes(b"g")
    voices.write_config(folder, "v2ProPlus")
    if reference:
        voices.write_reference(folder, tone(tmp / f"{name}_ref.wav", 5.0), "示例文本。", "test")
    return folder


def test_voices_are_listed_with_loras(lora_dir, tmp_path):
    make_voice(lora_dir, tmp=tmp_path)
    (ckpt,) = lora_library.scan(lora_dir)
    assert (ckpt.id, ckpt.engine, ckpt.version, ckpt.rank) == ("voice_a", "gpt_sovits", "v2ProPlus", None)
    assert lora_library.resolve(lora_dir, "voice_a").path == ckpt.path


def test_synthesis_needs_a_reference_and_uses_the_voices_own(engine, pipeline, lora_dir, tmp_path):
    with pytest.raises(ValueError, match="reference clip"):
        engine.synthesize(SynthesisRequest(text="你好"))  # the base model has none
    make_voice(lora_dir, tmp=tmp_path)

    result = engine.synthesize(SynthesisRequest(text="你好", lora_id="voice_a", seed=5, options={"top_k": 20}))
    engine.synthesize(SynthesisRequest(text="再来一次", lora_id="voice_a"))  # voice and reference stay loaded

    assert (result.sample_rate, result.seed, len(result.audio)) == (32_000, 5, 320)
    assert pipeline.voices == [
        (str(lora_dir / "voice_a" / voices.SOVITS_FILE), str(lora_dir / "voice_a" / voices.GPT_FILE))
    ]
    assert pipeline.references == [(str(lora_dir / "voice_a" / voices.REFERENCE_WAV), "示例文本。")]
    assert pipeline.calls[0]["options"].top_k == 20 and pipeline.calls[0]["options"].temperature == 1.0
    progress = engine.progress()
    assert progress["phase"] == "idle" and progress["step"] == 4


def test_a_given_reference_and_the_base_model(engine, pipeline, tmp_path):
    clip = tone(tmp_path / "clip.wav", 4.0)
    engine.synthesize(SynthesisRequest(text="hello", reference_wav=clip, prompt_text=""))

    assert pipeline.voices == [("s2Gv2ProPlus.pth", "s1v3.ckpt")]
    assert pipeline.references == [(str(clip), "")]
    with pytest.raises(ValueError):
        engine.synthesize(SynthesisRequest(text="hi", reference_wav=clip, options={"cfg_value": 2.0}))
    with pytest.raises(ValueError, match="empty"):
        engine.synthesize(SynthesisRequest(text=" 。 ", reference_wav=clip))


def test_other_engines_checkpoints_are_refused(engine, lora_dir, tmp_path):
    folder = lora_dir / "lora_x"
    folder.mkdir()
    (folder / lora_library.WEIGHTS_FILE).write_bytes(b"x")
    with pytest.raises(ValueError, match="another model"):
        engine.synthesize(SynthesisRequest(text="hi", lora_id="lora_x", reference_wav=tone(tmp_path / "c.wav", 4)))


def test_long_texts_are_cut_into_chunks_of_four_sentences():
    # As upstream's default cut: groups of four, the rest joining the last group.
    assert split_text("一。二。三。四。五。六。七。八。九") == ["一。二。三。四。", "五。六。七。八。九。"]
    assert split_text("一。二。三。四。五。六。七") == ["一。二。三。四。五。六。七。"]
    assert split_text("  ") == []


def dataset(data_dir, clips):
    """clips: (name, seconds, text, check or None)."""
    folder = data_dir / "speaker_a"
    entries, checks = [], {}
    for name, seconds, text, check in clips:
        tone(folder / "wavs" / f"{name}.wav", seconds)
        entries.append({"audio": f"wavs/{name}.wav", "text": text})
        if check:
            checks[f"wavs/{name}.wav"] = {"text": manifests.text_key(text), **check}
    manifests.write_manifest(entries, folder / "train.jsonl")
    (folder / manifests.ALIGN_FILE).write_text(json.dumps({"aligner": "qwen3", "clips": checks}), encoding="utf-8")
    return folder / "train.jsonl"


def test_the_default_reference_is_the_best_checked_clip(data_dir):
    manifest = dataset(
        data_dir,
        [
            ("short", 2.0, "短。", {"score": 0.99, "flags": []}),
            ("flagged", 6.0, "有问题。", {"score": 0.95, "flags": ["mismatch"]}),
            ("good", 7.0, "很好。", {"score": 0.9, "flags": []}),
            ("ok", 6.0, "还行。", {"score": 0.7, "flags": []}),
            ("unchecked", 6.0, "没查过。", None),
        ],
    )
    path, text = voices.pick_reference(manifest)
    assert (path.name, text) == ("good.wav", "很好。")


def test_unchecked_datasets_give_the_clip_nearest_six_seconds(data_dir):
    manifest = dataset(data_dir, [("a", 9.0, "甲。", None), ("b", 5.5, "乙。", None), ("c", 12.0, "丙。", None)])
    assert voices.pick_reference(manifest)[0].name == "b.wav"
    with pytest.raises(ValueError):
        voices.pick_reference(dataset(data_dir, [("z", 2.0, "短。", None)]))


def test_webui_weights_import_as_a_voice(lora_dir, tmp_path):
    sovits, gpt = webui_weights(tmp_path / "webui")
    clip = tone(tmp_path / "ref.m4a.wav", 5.0)

    assert voices.import_webui(lora_dir, "声音_a", sovits, gpt, (clip, "示例。")) == "声音_a"
    (ckpt,) = lora_library.scan(lora_dir)
    assert (ckpt.engine, ckpt.version) == ("gpt_sovits", "v2ProPlus")
    assert voices.read_reference(ckpt.path)[1] == "示例。"
    assert (ckpt.path / voices.SOVITS_FILE).read_bytes() == sovits.read_bytes()  # still loadable by the webui
    with pytest.raises(FileExistsError):
        voices.import_webui(lora_dir, "声音_a", sovits, gpt, (clip, ""))
    voices.import_webui(lora_dir, "no_reference", sovits, gpt, None)  # one is added later
    assert voices.read_reference(lora_dir / "no_reference") is None


def test_unsupported_or_unsafe_weights_are_not_imported(lora_dir, tmp_path):
    v4, gpt = webui_weights(tmp_path / "v4", tag=b"04")
    clip = tone(tmp_path / "ref.wav", 5.0)
    with pytest.raises(ValueError, match="v4"):
        voices.import_webui(lora_dir, "old", v4, gpt, (clip, ""))
    sovits, _ = webui_weights(tmp_path / "ok")
    with pytest.raises(ValueError, match="3-10 s"):
        voices.import_webui(lora_dir, "short", sovits, gpt, (tone(tmp_path / "s.wav", 1.0), ""))
    assert not any(lora_dir.iterdir())  # nothing left behind


def test_voice_api_import_reference_and_list(settings, service, training, engine, data_dir, tmp_path):
    client = TestClient(create_app(settings, service, training, engines_={"gpt_sovits": engine}))
    dataset(data_dir, [("good", 6.0, "很好。", {"score": 0.9, "flags": []})])
    sovits, gpt = webui_weights(tmp_path / "webui")

    r = client.post(
        "/api/loras/import-gpt-sovits",
        json={"name": "voice_b", "sovits": f'"{sovits}"', "gpt": str(gpt), "dataset": "speaker_a/train.jsonl"},
    )
    assert r.status_code == 200 and r.json() == {"id": "voice_b", "engine": "gpt_sovits"}
    (lora,) = client.get("/api/loras").json()
    assert (lora["reference_text"], lora["version"]) == ("很好。", "v2ProPlus")
    assert client.get("/api/loras/reference", params={"id": "voice_b"}).status_code == 200

    with (tmp_path / "new.wav").open("wb") as f:
        sf.write(f, np.zeros(int(4 * RATE), dtype=np.float32), RATE, format="WAV")
    r = client.post(
        "/api/loras/reference",
        data={"id": "voice_b", "text": "新的参考。"},
        files={"audio": ("new.wav", (tmp_path / "new.wav").read_bytes(), "audio/wav")},
    )
    assert r.status_code == 200 and r.json()["reference_text"] == "新的参考。"
    assert client.post("/api/loras/reference", data={"id": "voice_b"}).status_code == 422
    assert client.post("/api/loras/import-gpt-sovits", json={"name": "x", "sovits": "a", "gpt": "b"}).status_code == 422


# --------------------------------------------------------------------------- training


def test_training_config_is_full_finetuning_in_two_stages(engine, data_dir, tmp_path):
    from polyvox.training import TrainParams

    dataset(data_dir, [("good", 6.0, "很好。", {"score": 0.9, "flags": []}), ("ok", 5.0, "还行。", None)])
    run_manifest = tmp_path / "train.jsonl"
    manifests.write_manifest([{"audio": f"wavs/{i}.wav", "text": "示例。"} for i in range(20)], run_manifest)
    params = TrainParams(name="run_a", engine="gpt_sovits", dataset="speaker_a/train.jsonl", batch_size=8)
    config = engine.training_config(params, tmp_path / "run_a", run_manifest, None)

    # SoVITS 8 epochs of ceil(20 / 8) batches; GPT 15 epochs of 4 batches of 5 (a quarter of the clips), 4 per update
    assert config["num_iters"] == 8 * 3 + 15
    assert (config["sovits_epochs"], config["gpt_epochs"], config["gpt_batch_size"]) == (8, 15, 8)
    assert "lora" not in config and "optimizer" not in config  # every weight trains, as upstream
    assert Path(config["reference"]["audio"]).name == "good.wav" and config["reference"]["text"] == "很好。"
    assert GptSovitsEngine.capabilities.training and GptSovitsEngine.capabilities.full_finetune
    with pytest.raises(ValueError, match="epochs"):
        engine.training_config(
            params.model_copy(update={"sovits_epochs": 0, "gpt_epochs": 0}), tmp_path, run_manifest, None
        )


def test_a_run_the_engine_refuses_leaves_nothing_behind(settings, engine, data_dir, lora_dir):
    from polyvox.training import TrainingManager, TrainParams

    dataset(data_dir, [("short", 2.0, "短。", None), ("long", 12.0, "长。", None)])  # no clip fits a reference
    manager = TrainingManager(settings, command=lambda *paths: ["python"], engines_={"gpt_sovits": engine})
    with pytest.raises(ValueError, match="No clip"):
        manager.start(TrainParams(name="run_x", engine="gpt_sovits", dataset="speaker_a/train.jsonl", val_holdout=0))
    assert not (lora_dir / "run_x").exists()


def test_stage_plans_and_the_gpts_clip_rules():
    from polyvox.engines.gpt_sovits import worker

    assert worker.plan_steps(20, 8, 2, 20, 8, 3) == (6, 3)
    assert worker.plan_steps(100, 8, 0, 96, 8, 1) == (0, 3)  # GPT only: 12 batches of 8
    assert worker.gpt_batch_size(10, 8) == 2 and worker.gpt_batch_size(2, 8) == 1

    def clip(seconds, phones):
        return {"semantic": torch.zeros(int(seconds * 25)), "phones": torch.zeros(phones)}

    assert worker.gpt_keeps(clip(4, 40))
    assert not worker.gpt_keeps(clip(4, 4))  # 1 phoneme per second: text missing words
    assert not worker.gpt_keeps(clip(4, 120))  # 30 per second: audio missing speech
    assert not worker.gpt_keeps(clip(60, 600))  # longer than upstream's 54 s


def test_trained_gpt_weights_are_saved_as_the_webui_saves_them(tmp_path):
    from polyvox.engines.gpt_sovits.worker import save_gpt
    from polyvox.vendor.gpt_sovits.process_ckpt import load_weights

    path = tmp_path / voices.GPT_FILE
    save_gpt(path, {"h.w": torch.ones(2)}, {"data": {"max_sec": 54}}, "GPT-e15")
    saved = load_weights(path)
    assert list(saved["weight"]) == ["model.h.w"] and saved["weight"]["model.h.w"].dtype == torch.float16
    assert (saved["config"]["data"]["max_sec"], saved["info"]) == (54, "GPT-e15")


def test_length_batches_are_whole_epochs_of_similar_lengths():
    from polyvox.engines.gpt_sovits.worker import LengthBatches

    frames = [100, 1500, 120, 1600, 110, 1550, 130, 90, 1700]
    batches = LengthBatches(frames, batch_size=4)
    epoch = batches.epoch(0)

    assert batches.per_epoch == 3 and len(epoch) == 3 and all(len(b) == 4 for b in epoch)
    assert set().union(*epoch) == set(range(len(frames)))  # every clip each epoch
    assert epoch == batches.epoch(0) and epoch != batches.epoch(1)  # a resumed run gets the same batches
    # Short and long clips apart, as upstream's buckets keep them: at most one batch straddles the gap.
    mixed = [b for b in epoch if min(frames[i] for i in b) < 300 < max(frames[i] for i in b)]
    assert len(mixed) <= 1
    assert [len(b) for b in LengthBatches([100, 120], batch_size=4).epoch(0)] == [4]


def test_trained_sovits_weights_are_saved_as_the_webui_saves_them(lora_dir, tmp_path):
    from polyvox.engines.gpt_sovits.worker import save_sovits
    from polyvox.vendor.gpt_sovits.process_ckpt import get_sovits_version_from_path_fast, load_sovits_new

    path = tmp_path / voices.SOVITS_FILE
    weights = {"dec.w": torch.ones(3), "enc_q.w": torch.ones(2)}
    save_sovits(path, weights, {"data": {"sampling_rate": 32000}, "model": {}}, "8epoch_100iteration", "v2ProPlus")

    assert get_sovits_version_from_path_fast(str(path)) == ["v2", "v2ProPlus", False]
    saved = load_sovits_new(path)  # weights only: nothing pickled but tensors and containers
    assert list(saved["weight"]) == ["dec.w"] and saved["weight"]["dec.w"].dtype == torch.float16
    assert saved["config"]["data"]["sampling_rate"] == 32000
    _, gpt = webui_weights(tmp_path / "webui")
    voices.import_webui(lora_dir, "trained", path, gpt, None)  # the same checks a webui voice passes


def test_a_checkpoint_without_gpt_weights_speaks_with_the_base_gpt(engine, pipeline, lora_dir, tmp_path):
    folder = make_voice(lora_dir, tmp=tmp_path)
    (folder / voices.GPT_FILE).unlink()
    engine.synthesize(SynthesisRequest(text="你好", lora_id="voice_a"))
    assert pipeline.voices == [(str(folder / voices.SOVITS_FILE), "s1v3.ckpt")]


class FakeFrontEnds:
    """synth.Pipeline's front ends for features.clip_features: records what HuBERT hears."""

    device, dtype = "cpu", torch.float32

    def __init__(self):
        self.heard = []
        pipeline = self

        class Hubert(torch.nn.Module):
            def model(self, x):
                pipeline.heard.append(x)
                return {"last_hidden_state": torch.zeros(1, x.shape[-1] // 320, 768)}

        class Sv:
            def compute_embedding3(self, wav):
                return torch.ones(1, 20480)

        self.hubert, self.sv = Hubert(), Sv()

    def phones_and_bert(self, text, language="zh", lead_in=True):
        assert not lead_in  # training audio has no lead-in pause
        return [5, 6, 7], torch.zeros(1024, 3)


class FakeBase:
    class sovits:
        @staticmethod
        def extract_latent(ssl):
            return torch.zeros(1, 1, ssl.shape[-1] // 2, dtype=torch.long)


def test_clip_features_follow_upstreams_levels_and_skip_unusable_clips(tmp_path):
    from polyvox.engines.gpt_sovits import features

    front = FakeFrontEnds()
    clip = tone(tmp_path / "a.wav", 2.0, rate=16_000)  # peak 0.3
    out = features.clip_features(front, FakeBase, str(clip), "示例。")

    assert len(out["wav"]) == 64_000 and out["wav"].dtype == torch.int16  # 32 kHz
    # Half normalized to a 0.95 peak, half as recorded: 0.475 + 0.15
    assert abs(int(out["wav"].abs().max()) / 32768 - 0.625) < 0.01
    assert abs(float(front.heard[0].abs().max()) - 0.625 * features.HUBERT_SCALE) < 10
    assert out["phones"].tolist() == [5, 6, 7] and out["bert"].shape == (1024, 3) and out["sv"].shape == (20480,)
    assert out["semantic"].shape[0] == out["ssl"].shape[-1] // 2

    sf.write(tmp_path / "silent.wav", np.zeros(16_000, dtype=np.float32), 16_000)
    stop = threading.Event()
    records = [{"audio": str(clip), "text": "一。"}, {"audio": str(tmp_path / "silent.wav"), "text": "二。"}]
    records.append({"audio": str(tmp_path / "missing.wav"), "text": "三。"})
    clips, skipped = features.extract(front, FakeBase, records, stop)
    assert [c["index"] for c in clips] == [0] and skipped == {"silent": 1, "unreadable": 1}


def test_features_are_cached_per_manifest(tmp_path):
    from polyvox.engines.gpt_sovits import features

    manifest = tmp_path / "train.jsonl"
    manifest.write_text('{"audio": "a.wav", "text": "一"}\n', encoding="utf-8")
    cache = tmp_path / "cache" / "features.pt"
    made = []

    def make():
        made.append(1)
        return [{"index": 0, "wav": torch.zeros(4, dtype=torch.int16)}], {"silent": 2}

    key = features.cache_key(manifest, tmp_path, "v2ProPlus")
    assert features.cached(cache, key, lambda: None) == ([], {})  # stopped: nothing saved
    assert not cache.exists()
    first = features.cached(cache, key, make)
    again = features.cached(cache, key, make)
    assert len(made) == 1 and again[1] == {"silent": 2} and again[0][0]["index"] == first[0][0]["index"]
    manifest.write_text('{"audio": "a.wav", "text": "改"}\n', encoding="utf-8")  # edited: extracted again
    features.cached(cache, features.cache_key(manifest, tmp_path, "v2ProPlus"), make)
    assert len(made) == 2
