import io
import json

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
