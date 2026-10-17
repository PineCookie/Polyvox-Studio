"""Reference clips picked from a dataset, joined for engines that take one clip."""

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from polyvox.data import manifests, references
from polyvox.engines import Capabilities, Engine, ReferenceInput, SynthesisResult
from polyvox.server import create_app


def tone(path: Path, seconds: float, rate: int = 16_000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(seconds * rate)) / rate
    sf.write(path, (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), rate)


def dataset(data_dir: Path, clips) -> Path:
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


CLIPS = [
    ("good", 6.0, "我们今天就先说到这里，下次再见吧。", {"score": 0.9, "flags": []}),
    ("short", 2.0, "这一句太短了，不能当参考用的。", {"score": 0.99, "flags": []}),
    ("hesitant", 6.0, "嗯，呃，今天我们一起去看看那边的情况。", {"score": 0.99, "flags": []}),
    ("flagged", 6.0, "这一句有问题，所以不应该选它。", {"score": 0.95, "flags": ["mismatch"]}),
    ("fine", 7.0, "其实没有那么复杂，仔细想想就明白了。", {"score": 0.8, "flags": []}),
    ("dropped", 6.0, "这一句在审查里被去掉了，也不选。", {"score": 0.97, "flags": []}),
]


def test_pick_gives_the_best_clips_first_and_leaves_out_the_unfit(data_dir):
    manifest = dataset(data_dir, CLIPS)
    manifests.set_dropped(manifest, None, 5, True)

    picked = references.pick(manifest, 3)
    assert [c.audio.name for c in picked] == ["good.wav", "fine.wav", "hesitant.wav"]
    assert (picked[0].index, picked[0].seconds) == (0, 6.0)
    assert [c.index for c in references.pick(manifest, 3, exclude=frozenset({0}))] == [4, 2]
    with pytest.raises(ValueError):
        references.pick(manifest, 1, exclude=frozenset({0, 2, 4}))


def test_join_puts_clips_one_after_another_with_their_transcripts(tmp_path):
    tone(tmp_path / "a.wav", 1.0, 16_000)
    tone(tmp_path / "b.wav", 0.5, 32_000)
    clips = [
        references.Clip(0, tmp_path / "a.wav", "第一句", 1.0),
        references.Clip(1, tmp_path / "b.wav", "第二句！", 0.5),
    ]

    text = references.join(clips, tmp_path / "joined.wav")

    assert text == "第一句。第二句！"
    audio, rate = sf.read(tmp_path / "joined.wav")
    assert rate == 32_000 and len(audio) == int((1.0 + references.JOIN_PAUSE_S + 0.5) * 32_000)


def test_tts_clones_from_dataset_clips_joined_into_one_reference(service, training, loader, data_dir):
    manifest = dataset(data_dir, CLIPS)
    manifests.set_dropped(manifest, None, 5, True)
    client = TestClient(create_app(service.settings, service, training))
    picked = client.get("/api/dataset-references", params={"id": "speaker_a/train.jsonl", "count": 2}).json()
    assert [c["index"] for c in picked] == [0, 4] and picked[0]["seconds"] == 6.0

    chosen = json.dumps({"dataset": "speaker_a/train.jsonl", "indices": [0, 4]})
    resp = client.post("/api/tts", data={"text": "hi", "reference_clips": chosen})  # VoxCPM's reference slot only

    assert resp.status_code == 200, resp.text
    call = loader.models[0].generate_calls[-1]
    joined = Path(call["reference_wav_path"])
    assert not joined.exists() and not call.get("prompt_wav_path")  # a temporary file, removed; no prompt
    assert (manifest.parent / "wavs" / "good.wav").exists()  # the dataset's own clips stay

    resp = client.post("/api/tts", data={"text": "hi", "reference_clips": chosen, "prompt_clips": chosen})
    call = loader.models[0].generate_calls[-1]
    assert call["prompt_text"] == "我们今天就先说到这里，下次再见吧。其实没有那么复杂，仔细想想就明白了。"
    assert call["prompt_wav_path"] and call["reference_wav_path"]


class Recorder(Engine):
    """An engine with the given reference inputs that records its requests."""

    name = "Recorder"
    params = ()
    model_setting = "model"
    worker_module = "x"

    def __init__(self, id, inputs):
        self.id = id
        self.capabilities = Capabilities(
            24_000, 24_000, ("zh",), True, True, False, False, False, reference_inputs=inputs
        )
        self.requests = []

    device = "cpu"
    loaded = False

    def status(self):
        return {"loaded": False}

    def synthesize(self, req):
        self.requests.append(req)
        return SynthesisResult(24_000, np.zeros(240, dtype=np.float32), req.seed)

    def progress(self):
        return {"phase": "idle"}

    def unload(self):
        pass

    def training_config(self, params, run_dir, train_manifest, val_manifest):
        return {}


def test_each_engine_gets_its_own_inputs(service, training, data_dir):
    dataset(data_dir, CLIPS)
    one_clip = Recorder("one_clip", (ReferenceInput("clip", "prompt", "optional"),))  # as dots.tts
    two_parts = Recorder(  # as GPT-SoVITS
        "two_parts",
        (
            ReferenceInput("prompt", "prompt", "optional", max_seconds=10.0),
            ReferenceInput("timbre", "reference", "none", several=True, max_seconds=10.0),
        ),
    )
    registry = {"one_clip": one_clip, "two_parts": two_parts}
    client = TestClient(create_app(service.settings, service, training, engines_=registry))
    chosen = json.dumps({"dataset": "speaker_a/train.jsonl", "indices": [0, 4]})
    wav = (data_dir / "speaker_a" / "wavs" / "good.wav").read_bytes()

    def post(engine, files=None, **data):
        return client.post("/api/tts", data={"text": "hi", "engine": engine, **data}, files=files)

    # One clip: sent the old way it is the clip (with its transcript); several are joined into one.
    assert post("one_clip", files={"reference_audio": ("a.wav", wav)}, prompt_text="示例。").status_code == 200
    req = one_clip.requests[-1]
    assert req.reference_wav is None and req.prompt_wav is not None and req.prompt_text == "示例。"
    assert post("one_clip", prompt_clips=chosen).status_code == 200
    assert one_clip.requests[-1].prompt_text.startswith("我们今天就先说到这里") and one_clip.requests[-1].prompt_wav
    assert post("one_clip", reference_clips=chosen, prompt_clips=chosen).status_code == 422

    # Several timbre clips as they are; the prompt takes one short clip.
    assert post("two_parts", reference_clips=chosen, prompt_clips=chosen).status_code == 200
    req = two_parts.requests[-1]
    assert (req.reference_wav.name, [p.name for p in req.extra_references]) == ("good.wav", ["fine.wav"])
    assert (req.prompt_wav.name, req.prompt_text) == ("good.wav", "我们今天就先说到这里，下次再见吧。")


def test_tts_dataset_clip_errors(service, training, data_dir):
    dataset(data_dir, CLIPS)
    client = TestClient(create_app(service.settings, service, training))

    def post(value, **extra):
        return client.post("/api/tts", data={"text": "hi", "reference_clips": value}, **extra).status_code

    assert post(json.dumps({"dataset": "speaker_a/train.jsonl", "indices": [99]})) == 404
    assert post(json.dumps({"dataset": "nope/train.jsonl", "indices": [0]})) == 404
    assert post(json.dumps({"dataset": "speaker_a/train.jsonl", "indices": []})) == 422
    assert post("not json") == 422
    upload = {"reference_audio": ("a.wav", b"RIFF", "audio/wav")}
    assert post(json.dumps({"dataset": "speaker_a/train.jsonl", "indices": [0]}), files=upload) == 422
    wav = {"prompt_audio": ("a.wav", (data_dir / "speaker_a" / "wavs" / "good.wav").read_bytes())}
    no_text = client.post("/api/tts", data={"text": "hi"}, files=wav)  # VoxCPM's prompt needs its transcript
    assert no_text.status_code == 422 and "transcript" in no_text.json()["detail"]
