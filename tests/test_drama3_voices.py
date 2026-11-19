"""Saved Drama 3 voices: a name and a reference (no training), listed with the other voices."""

import dataclasses
import json
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch
from fastapi.testclient import TestClient

from polyvox import engines, lora_library
from polyvox.data import manifests
from polyvox.engines import SynthesisRequest
from polyvox.engines.common import reference
from polyvox.engines.fish_drama3 import engine
from polyvox.server import create_app

RATE = 16_000


def tone(path, seconds):
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(seconds * RATE)) / RATE
    sf.write(path, (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), RATE)
    return path


def dataset(data_dir, texts=("第一句话。", "第二句话。", "第三句话。", "第四句话。")):
    folder = data_dir / "speaker_a"
    entries, checks = [], {}
    for i, text in enumerate(texts):
        tone(folder / "wavs" / f"{i}.wav", 5.0 + i)
        entries.append({"audio": f"wavs/{i}.wav", "text": text})
        checks[f"wavs/{i}.wav"] = {"text": manifests.text_key(text), "score": 0.9, "flags": []}
    manifests.write_manifest(entries, folder / "train.jsonl")
    (folder / manifests.ALIGN_FILE).write_text(json.dumps({"aligner": "qwen3", "clips": checks}), encoding="utf-8")
    return folder / "train.jsonl"


def create(target, name, **kw):
    cls = engine.FishDrama3Engine
    return reference.create_voice(target, name, cls.id, cls.base_repo(), cls.name, **kw)


def test_a_voice_is_its_reference_and_lists_like_the_others(lora_dir, tmp_path):
    target = lora_library.engine_dir(lora_dir, "fish_drama3")
    create(target, "voice_a", audio=tone(tmp_path / "clip.wav", 6.0), text="示例。", source="upload")

    (voice,) = lora_library.scan(lora_dir)
    assert (voice.id, voice.engine, voice.base_model) == ("fish_drama3/voice_a", "fish_drama3", "fishaudio/drama-3")
    assert voice.voice_path == "fish_drama3/voice_a" and voice.weights_mtime > 0
    assert lora_library.resolve(lora_dir, "fish_drama3/voice_a").engine == "fish_drama3"
    with pytest.raises(FileExistsError):
        create(target, "voice_a", audio=tmp_path / "clip.wav", text="示例。")
    with pytest.raises(ValueError, match="transcript"):
        create(target, "voice_b", audio=tmp_path / "clip.wav", text="")
    with pytest.raises(ValueError, match="name"):
        create(target, "../x", audio=tmp_path / "clip.wav", text="示例。")
    assert sorted(p.name for p in target.iterdir()) == ["voice_a"]  # nothing half-made left
    assert engines.with_capability("reference_voices") is engine.FishDrama3Engine
    assert engines.with_capability("reference_voices", "fish_s2") is None


class Letters:
    def encode(self, text):
        return SimpleNamespace(ids=[ord(c) for c in text])


class FakeModel:
    tokenizer = Letters()
    semantic_begin = 0x110000

    def __init__(self):
        self.prompts = []

    def generate(self, rows, sampling, max_frames, on_frame, head=0):
        self.prompts.append("".join(chr(r) for r in rows[:, 0].tolist() if r < self.semantic_begin))
        return torch.zeros(10, 20, dtype=torch.long)


class FakeRuntime:
    device = "cpu"

    def __init__(self, m):
        self.model, self.encoded = m, []
        self.codec = SimpleNamespace(sample_rate=44_100, decode=lambda c: torch.zeros(c.shape[1] * 2048))

    def encode(self, wav):
        self.encoded.append(wav)
        return torch.zeros(10, 5, dtype=torch.long)


def test_a_voice_speaks_with_its_reference_unless_given_another(settings, lora_dir, monkeypatch, tmp_path):
    create(
        lora_library.engine_dir(lora_dir, "fish_drama3"),
        "voice_a",
        audio=tone(tmp_path / "c.wav", 6.0),
        text="她的参考。",
    )
    monkeypatch.setattr(engine.FishDrama3Engine, "unavailable", lambda self: None)
    m = FakeModel()
    rt = FakeRuntime(m)
    tts = engine.FishDrama3Engine(dataclasses.replace(settings, drama3_model="X:/models/drama3"), lambda *a: rt)

    tts.synthesize(SynthesisRequest(text="你好。", lora_id="fish_drama3/voice_a"))
    assert rt.encoded[-1] == lora_dir.resolve() / "fish_drama3" / "voice_a" / "reference.wav"
    assert "<|speaker:0|> 她的参考。" in m.prompts[-1]

    other = tone(tmp_path / "other.wav", 5.0)
    tts.synthesize(
        SynthesisRequest(text="你好。", lora_id="fish_drama3/voice_a", prompt_wav=other, prompt_text="别的。")
    )
    assert rt.encoded[-1] == other and "<|speaker:0|> 别的。" in m.prompts[-1]
    with pytest.raises(KeyError):
        tts.synthesize(SynthesisRequest(text="你好。", lora_id="fish_drama3/nobody"))


def test_voices_are_made_and_changed_through_the_api(settings, service, training, data_dir, tmp_path):
    client = TestClient(create_app(settings, service, training))
    dataset(data_dir)

    r = client.post("/api/loras/reference-voice", data={"name": "voice_a", "dataset": "speaker_a/train.jsonl"})
    assert r.status_code == 200 and r.json() == {"id": "fish_drama3/voice_a", "engine": "fish_drama3"}
    (voice,) = [v for v in client.get("/api/loras").json() if v["engine"] == "fish_drama3"]
    assert voice["reference_text"].count("句话") == 3  # the best three clips, joined
    listed = {e["id"]: e for e in client.get("/api/engines").json()}
    caps = {i: (e["capabilities"]["package"], e["capabilities"]["voice_reference_clips"]) for i, e in listed.items()}
    assert caps == {"voxcpm": ("lora", 1), "dots_tts": ("lora", 1), "gpt_sovits": ("voice", 1),
                    "fish_s2": ("lora", 3), "fish_drama3": (None, 3), "rvc": ("voice", 1)}  # fmt: skip
    assert listed["fish_drama3"]["base_repo"] == "fishaudio/drama-3"
    assert client.get("/api/loras/reference", params={"id": "fish_drama3/voice_a"}).status_code == 200

    clip = tone(tmp_path / "up.wav", 4.0).read_bytes()
    r = client.post("/api/loras/reference", data={"id": "fish_drama3/voice_a", "text": "新的参考。"},
                    files={"audio": ("up.wav", clip, "audio/wav")})  # fmt: skip
    assert r.status_code == 200 and r.json()["reference_text"] == "新的参考。"

    r = client.post(
        "/api/loras/reference-voice",
        data={"name": "voice_b", "text": "上传的。"},
        files={"audio": ("up.wav", clip, "audio/wav")},
    )
    assert r.status_code == 200 and r.json()["id"] == "fish_drama3/voice_b"
    taken = client.post("/api/loras/reference-voice", data={"name": "voice_a", "dataset": "speaker_a/train.jsonl"})
    assert taken.status_code == 409
    assert client.post("/api/loras/reference-voice", data={"name": "voice_c"}).status_code == 422  # no clips
    refused = client.post(
        "/api/loras/reference-voice", data={"name": "x", "engine": "fish_s2", "dataset": "speaker_a/train.jsonl"}
    )
    assert refused.status_code == 422  # S2-Pro voices are trained
