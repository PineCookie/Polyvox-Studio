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


def test_the_default_reference_prefers_fluent_sentences_over_the_best_score(data_dir):
    manifest = dataset(
        data_dir,
        [
            ("hesitant", 6.0, "嗯，呃，今天我们一起去看看那边的情况。", {"score": 0.99, "flags": []}),
            ("short", 6.0, "好的嗯。", {"score": 0.98, "flags": []}),
            ("stutter", 6.0, "其实其实我们今天就先说到这里吧。", {"score": 0.97, "flags": []}),
            ("fluent", 6.0, "我们今天就先说到这里，下次再见吧。", {"score": 0.8, "flags": []}),
        ],
    )
    assert voices.pick_reference(manifest)[0].name == "fluent.wav"


def test_text_quality():
    assert voices.text_quality("我们今天就先说到这里，下次再见吧。") == (True, True)
    assert voices.text_quality("原来是这样啊，那我们明天再说吧。") == (True, True)  # 啊 as a particle is fine
    assert voices.text_quality("哈哈，原来是这样，那我们明天再说吧。")[0] is False  # a filler-only clause
    assert voices.text_quality("谢谢大家，慢慢看看就会明白的了。") == (True, True)  # doubled characters are words
    assert voices.text_quality("视线高起来了嗯。") == (False, False)


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
    v3, gpt = webui_weights(tmp_path / "v3", tag=b"03")
    clip = tone(tmp_path / "ref.wav", 5.0)
    with pytest.raises(ValueError, match="v3"):
        voices.import_webui(lora_dir, "old", v3, gpt, (clip, ""))
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
    assert r.status_code == 200 and r.json() == {"id": "gpt_sovits/voice_b", "engine": "gpt_sovits"}
    (lora,) = client.get("/api/loras").json()
    assert (lora["reference_text"], lora["version"]) == ("很好。", "v2ProPlus")
    assert client.get("/api/loras/reference", params={"id": "gpt_sovits/voice_b"}).status_code == 200

    with (tmp_path / "new.wav").open("wb") as f:
        sf.write(f, np.zeros(int(4 * RATE), dtype=np.float32), RATE, format="WAV")
    r = client.post(
        "/api/loras/reference",
        data={"id": "gpt_sovits/voice_b", "text": "新的参考。"},
        files={"audio": ("new.wav", (tmp_path / "new.wav").read_bytes(), "audio/wav")},
    )
    assert r.status_code == 200 and r.json()["reference_text"] == "新的参考。"
    assert client.post("/api/loras/reference", data={"id": "gpt_sovits/voice_b"}).status_code == 422
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
    assert Path(config["feature_cache"]).parent == data_dir / "speaker_a" / ".cache"  # shared by the dataset's runs
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


def test_clip_features_are_cached_per_clip_for_every_run_of_a_dataset(tmp_path):
    from polyvox.engines.gpt_sovits import features

    front, stop = FakeFrontEnds(), threading.Event()
    a, b = tone(tmp_path / "a.wav", 2.0), tone(tmp_path / "b.wav", 2.0)
    sf.write(tmp_path / "silent.wav", np.zeros(16_000, dtype=np.float32), 16_000)
    cache, meta = tmp_path / ".cache" / "gpt_sovits.pt", features.cache_meta(tmp_path, "v2ProPlus")
    run_1 = [{"audio": str(a), "text": "一。"}, {"audio": str(tmp_path / "silent.wav"), "text": "静。"}]

    clips, skipped = features.extract(front, FakeBase, run_1, stop, cache, meta)
    assert [c["index"] for c in clips] == [0] and skipped == {"silent": 1} and len(front.heard) == 1
    run_2 = [{"audio": str(b), "text": "二。"}, *run_1]  # another run (split) of the dataset
    clips, skipped = features.extract(front, FakeBase, run_2, stop, cache, meta)
    assert [c["index"] for c in clips] == [0, 1] and skipped == {"silent": 1}
    assert len(front.heard) == 2  # only the new clip computed; the skip remembered

    edited = [{"audio": str(a), "text": "改。"}]  # a changed transcript: computed again
    features.extract(front, FakeBase, edited, stop, cache, meta)
    assert len(front.heard) == 3
    features.extract(front, FakeBase, edited, stop, cache, features.cache_meta(tmp_path, "v2Pro"))
    assert len(front.heard) == 4  # other models: nothing reused

    stop.set()
    assert features.extract(front, FakeBase, [{"audio": str(b), "text": "新。"}], stop, cache, meta) is None


# --------------------------------------------------------------------------- decoding


@pytest.mark.parametrize("seed", range(5))
def test_fast_decoding_makes_upstreams_tokens(seed, monkeypatch):
    """Greedy, so sampling noise doesn't matter: the same tokens and count as
    upstream's infer_panel, with or without a prompt, stopping where it stops."""
    from polyvox.engines.gpt_sovits.decode import Decoder
    from polyvox.vendor.gpt_sovits.AR.models import t2s_model

    monkeypatch.setattr(t2s_model, "tqdm", lambda it, *a, **k: it)
    torch.manual_seed(seed)
    config = {"model": {"hidden_dim": 32, "embedding_dim": 32, "head": 4, "n_layer": 2, "vocab_size": 1025,
                        "phoneme_vocab_size": 732, "dropout": 0, "EOS": 1024}}  # fmt: skip
    gpt = t2s_model.Text2SemanticDecoder(config).eval()
    with torch.no_grad():  # sharp logits; the end of speech likelier the higher the seed
        gpt.ar_predict_layer.weight.mul_(20)
        gpt.ar_predict_layer.weight[1024].mul_(0.5 + 0.4 * seed)
    x, bert = torch.randint(0, 700, (1, 12)), torch.randn(1, 1024, 12)
    decoder = Decoder(gpt, max_len=1600)
    for prompt, penalty in ((torch.randint(0, 1024, (1, 9)), 1.35), (None, 1.0)):
        expected = gpt.infer_panel(x, torch.tensor([12]), prompt, bert, top_k=1, early_stop_num=120,
                                   repetition_penalty=penalty)  # fmt: skip
        tokens, n = decoder.generate(x, prompt, bert, top_k=1, early_stop_num=120, repetition_penalty=penalty)
        assert torch.equal(tokens, expected[0].long()) and n == expected[1]


def test_the_base_models_version_is_a_choice(engine, pipeline, lora_dir, tmp_path):
    clip = tone(tmp_path / "clip.wav", 4.0)
    engine.synthesize(SynthesisRequest(text="你好", reference_wav=clip, options={"base_version": "v2Pro"}))
    make_voice(lora_dir, tmp=tmp_path)  # a voice keeps its own version whatever is chosen
    engine.synthesize(SynthesisRequest(text="你好", lora_id="voice_a", options={"base_version": "v2Pro"}))
    assert pipeline.voices[0] == ("s2Gv2Pro.pth", "s1v3.ckpt")
    assert pipeline.voices[1][0].endswith(voices.SOVITS_FILE)
    with pytest.raises(ValueError, match="one of"):
        engine.synthesize(SynthesisRequest(text="你好", reference_wav=clip, options={"base_version": "v3"}))
    with pytest.raises(ValueError, match="v4 isn't downloaded"):
        engine.synthesize(SynthesisRequest(text="你好", reference_wav=clip, options={"base_version": "v4"}))
    (param,) = [p for p in engine.public()["params"] if p["name"] == "base_version"]
    assert (param["kind"], list(param["choices"]), param["base_only"]) == (
        "choice",
        ["v2ProPlus", "v2Pro", "v4"],
        True,
    )


# --------------------------------------------------------------------------- v4


def test_v4_webui_voices_import_and_package_with_their_lora_rank(lora_dir, tmp_path):
    from polyvox import sharing
    from polyvox.vendor.gpt_sovits.process_ckpt import get_sovits_version_from_path_fast, load_sovits_new

    _, gpt = webui_weights(tmp_path / "webui")
    buf = io.BytesIO()
    lora_key = "cfm.base_model.model.estimator.transformer_blocks.0.attn.to_q.lora_A.default.weight"
    torch.save({"weight": {lora_key: torch.zeros(2, 4).half()}, "config": {"model": {}}, "lora_rank": 32}, buf)
    sovits = tmp_path / "webui" / "voice_v4_e10_s100_l32.pth"
    sovits.write_bytes(b"04" + buf.getvalue()[2:])
    voices.import_webui(lora_dir, "voice_v4", sovits, gpt, (tone(tmp_path / "ref.wav", 5.0), "示例。"))
    (ckpt,) = lora_library.scan(lora_dir)
    assert ckpt.version == "v4"

    package = sharing.export_package(ckpt, "shared_v4", "")
    other = tmp_path / "other_lora"
    other.mkdir()
    sharing.import_package(other, package, "shared_v4.zip")
    shared = other / "gpt_sovits" / "shared_v4" / voices.SOVITS_FILE
    assert get_sovits_version_from_path_fast(str(shared))[1:] == ["v4", True]
    assert load_sovits_new(shared)["lora_rank"] == 32  # the webui needs it to load the voice


def test_v4_lora_weights_fold_into_the_dits_layers():
    from polyvox.engines.gpt_sovits.synth import merge_lora

    peft = "cfm.base_model.model.estimator.transformer_blocks.0.attn.to_q"
    a, b = torch.randn(2, 4), torch.randn(3, 2)
    weights = {
        "cfm.estimator.transformer_blocks.0.attn.to_q.weight": torch.ones(3, 4).half(),  # the base's
        "cfm.estimator.transformer_blocks.0.attn.to_q.bias": torch.zeros(3),
        f"{peft}.lora_A.default.weight": a,
        f"{peft}.lora_B.default.weight": b,
        "cfm.base_model.model.estimator.rotary_embed.inv_freq": torch.ones(2),
        "wns1.w": torch.ones(1),
    }
    merged = merge_lora(weights)
    assert sorted(merged) == [
        "cfm.estimator.rotary_embed.inv_freq",
        "cfm.estimator.transformer_blocks.0.attn.to_q.bias",
        "cfm.estimator.transformer_blocks.0.attn.to_q.weight",
        "wns1.w",
    ]
    w = merged["cfm.estimator.transformer_blocks.0.attn.to_q.weight"]
    assert w.dtype == torch.float16 and torch.allclose(w.float(), (1 + b @ a).half().float())


def test_the_dits_rotary_embedding_rotates_interleaved_pairs():
    from polyvox.vendor.gpt_sovits.f5_tts.rotary import RotaryEmbedding, apply_rotary_pos_emb

    freqs, scale = RotaryEmbedding(8).forward_from_seq_len(5)
    q = torch.randn(1, 2, 5, 8)
    rotated = apply_rotary_pos_emb(q, freqs, scale)
    # Pairs (0, 1), (2, 3), ... turn as complex numbers by position x frequency.
    pairs = torch.view_as_complex(q.unflatten(-1, (4, 2)).contiguous())
    angle = torch.arange(5)[:, None] * (1.0 / 10000 ** (torch.arange(0, 8, 2) / 8))
    expected = torch.view_as_real(pairs * torch.polar(torch.ones_like(angle), angle)).flatten(-2)
    assert torch.allclose(rotated, expected, atol=1e-5)


def test_v4s_faster_dit_makes_the_same_output():
    from types import SimpleNamespace

    from polyvox.engines.gpt_sovits.decode import fast_dit
    from polyvox.vendor.gpt_sovits.f5_tts.model import DiT

    torch.manual_seed(0)
    dit = DiT(dim=64, depth=2, heads=2, dim_head=32, ff_mult=2, mel_dim=100, text_dim=16, conv_layers=1).eval()
    x, cond, text = torch.randn(1, 100, 50), torch.randn(1, 100, 50), torch.randn(1, 16, 50)
    inputs = (x, cond, torch.tensor([50]), torch.tensor([0.3]), torch.tensor([0.125]), text)
    with torch.no_grad():
        expected = dit(*inputs)
        fast_dit(SimpleNamespace(cfm=SimpleNamespace(estimator=dit)), compiled=False)  # blocks without the mask
        assert torch.allclose(dit(*inputs), expected, atol=1e-6)


def test_v4_makes_its_mel_window_by_window_each_continuing_the_last():
    from types import SimpleNamespace

    from polyvox.engines.gpt_sovits import synth

    frames, keep = 2300, 40  # semantic frames' conditions; the reference's mel frames
    prompts = []

    class Cfm:
        def inference(self, x, lengths, prompt, steps, inference_cfg_rate):
            prompts.append(prompt[0, 0].clone())
            assert x.shape[1] <= synth.V4_WINDOW and int(lengths) == x.shape[1] and steps == 8
            return x.transpose(2, 1)[:, :100]  # "mel" = the conditions: frame i holds i

    class Sovits:
        cfm = Cfm()

        def decode_encp(self, codes, text, refer, ge, speed):
            return torch.arange(frames).float().expand(1, 512, frames), ge

    pipeline = synth.Pipeline.__new__(synth.Pipeline)
    pipeline.device = "cpu"
    pipeline.vocoder = lambda mel: mel[:, :1]  # [1, 1, T]
    ref = SimpleNamespace(spec=None, ge=None, mel=torch.full((1, 100, keep), -5.0), fea=torch.zeros(1, 512, keep))
    voice = SimpleNamespace(sovits=Sovits())
    audio = pipeline._v4_audio(voice, ref, None, [1, 2], synth.Options())

    made = (audio + 12) / 14 * 2 - 1  # undo the mel scaling: the frame numbers
    assert torch.allclose(made, torch.arange(frames).float(), atol=1e-3)
    assert len(prompts) == -(-frames // (synth.V4_WINDOW - keep))  # windows of 1000 frames, prompt included
    assert (prompts[0] == -5).all()  # the reference's mel first, then each window's last frames
    assert torch.equal(prompts[1], torch.arange(synth.V4_WINDOW - 2 * keep, synth.V4_WINDOW - keep).float())


def test_training_can_start_from_v2pro(engine, data_dir, tmp_path):
    from polyvox.training import TrainParams

    dataset(data_dir, [("good", 6.0, "我们今天就先说到这里，下次再见吧。", None)])
    run_manifest = tmp_path / "train.jsonl"
    manifests.write_manifest([{"audio": "wavs/0.wav", "text": "示例。"}], run_manifest)
    params = TrainParams(name="run_v", engine="gpt_sovits", dataset="speaker_a/train.jsonl", gsv_version="v2Pro")
    config = engine.training_config(params, tmp_path / "run_v", run_manifest, None)
    assert config["version"] == "v2Pro" and config["feature_cache"].endswith("gpt_sovits_v2Pro.pt")


# --------------------------------------------------------------------------- packages


def test_voice_packages_round_trip_without_local_details(lora_dir, tmp_path):
    import zipfile

    from polyvox import sharing
    from polyvox.vendor.gpt_sovits.process_ckpt import get_sovits_version_from_path_fast, load_sovits_new, load_weights

    folder = lora_dir / "voice_a"
    folder.mkdir()
    private = {"exp_dir": "X:/old_pc/logs/voice", "pretrained_s2G": "X:/old_pc/s2G.pth", "segment_size": 20480}
    buf = io.BytesIO()
    torch.save({"weight": {"w": torch.ones(2).half()}, "config": {"train": private, "data": {"exp_dir": "X:/d"},
                                                                  "name": "speaker_a-v2pp"}, "info": "8epoch"}, buf)  # fmt: skip
    (folder / voices.SOVITS_FILE).write_bytes(b"06" + buf.getvalue()[2:])
    gpt_config = {"data": {"max_sec": 54}, "train": {"exp_name": "speaker_a", "half_weights_save_dir": "X:/w"},
                  "output_dir": "X:/old_pc/logs_s1", "train_semantic_path": "X:/old_pc/6.tsv"}  # fmt: skip
    torch.save({"weight": {"model.w": torch.ones(2).half()}, "config": gpt_config}, folder / voices.GPT_FILE)
    voices.write_config(folder, "v2ProPlus")
    voices.write_reference(folder, tone(tmp_path / "ref.wav", 5.0), "示例文本。", "speaker_a/train.jsonl: x.wav")
    (ckpt,) = lora_library.scan(lora_dir)

    package = sharing.export_package(ckpt, "shared_voice", "")
    with zipfile.ZipFile(package) as zf:
        names = sorted(zf.namelist())
        everything = b"".join(zf.read(n) for n in names)
    assert names == [f"shared_voice/{f}" for f in sorted([voices.SOVITS_FILE, voices.GPT_FILE, voices.REFERENCE_WAV,
                                                          voices.REFERENCE_JSON, "lora_config.json", "README.md"])]  # fmt: skip
    assert b"old_pc" not in everything and b"speaker_a" not in everything  # paths, names and the reference's source

    other = tmp_path / "other_lora"
    other.mkdir()
    assert sharing.import_package(other, package, "shared_voice.zip") == ("gpt_sovits/shared_voice", "gpt_sovits")
    (imported,) = lora_library.scan(other)
    assert (imported.engine, imported.version) == ("gpt_sovits", "v2ProPlus")
    assert get_sovits_version_from_path_fast(str(imported.path / voices.SOVITS_FILE))[1] == "v2ProPlus"
    sovits = load_sovits_new(imported.path / voices.SOVITS_FILE)
    assert sovits["config"]["train"] == {"segment_size": 20480} and sovits["config"]["name"] == "shared_voice"
    assert torch.equal(sovits["weight"]["w"], torch.ones(2).half())
    assert load_weights(imported.path / voices.GPT_FILE)["config"] == {"data": {"max_sec": 54}, "train": {}}
    assert voices.read_reference(imported.path)[1] == "示例文本。"
    with pytest.raises(FileExistsError):
        sharing.import_package(other, package, "shared_voice.zip")


def test_a_voice_package_may_lack_the_gpt_and_reference(lora_dir, tmp_path):
    from polyvox import sharing

    sovits, _ = webui_weights(tmp_path / "webui")
    folder = lora_dir / "sovits_only"
    folder.mkdir()
    (folder / voices.SOVITS_FILE).write_bytes(sovits.read_bytes())
    voices.write_config(folder, "v2ProPlus")
    (ckpt,) = lora_library.scan(lora_dir)
    package = sharing.export_package(ckpt, "sovits_only_pkg", "")
    other = tmp_path / "other_lora"
    other.mkdir()
    sharing.import_package(other, package, "x.zip")
    assert sorted(p.name for p in (other / "gpt_sovits" / "sovits_only_pkg").iterdir()) == [
        "lora_config.json",
        voices.SOVITS_FILE,
    ]
