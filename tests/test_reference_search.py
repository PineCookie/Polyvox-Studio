"""Reference clips chosen by what the engine makes of a few candidate sets."""

import json
import time

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from polyvox import lora_library, reference_search, speaker
from polyvox.data import manifests
from polyvox.engines import SynthesisResult
from polyvox.engines.common import reference
from polyvox.engines.fish_drama3 import engine
from polyvox.server import create_app
from polyvox.speaker import SpeakerService

RATE = 16_000
TEXTS = "一二三四五六七八九十甲乙丙丁戊己庚辛壬癸子丑寅卯辰巳午未申酉"


def tone(path, seconds, hz):
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(seconds * RATE)) / RATE
    sf.write(path, (0.3 * np.sin(2 * np.pi * hz * t)).astype(np.float32), RATE)


def mixed_dataset(data_dir, clips=24, off_voice=3):
    """Clips of 220 Hz (the speaker), and ``off_voice`` of 500 Hz whose checks score best,
    so the cleanest-transcript pick takes them."""
    folder = data_dir / "speaker_a"
    entries, checks = [], {}
    for i in range(clips):
        off = i < off_voice
        tone(folder / "wavs" / f"{i}.wav", 4.0 + (i % 5), 500 if off else 220)
        text = f"这是第{TEXTS[i]}句很干净的示例句子。"
        entries.append({"audio": f"wavs/{i}.wav", "text": text})
        checks[f"wavs/{i}.wav"] = {"text": manifests.text_key(text), "score": 0.99 if off else 0.5, "flags": []}
    manifests.write_manifest(entries, folder / "train.jsonl")
    (folder / manifests.ALIGN_FILE).write_text(json.dumps({"aligner": "qwen3", "clips": checks}), encoding="utf-8")
    return folder / "train.jsonl"


def echo(spoken):
    """A fake engine: each take sounds like its reference's first second."""

    def synthesize(text, wav, transcript, seed):
        audio, rate = sf.read(wav, dtype="float32")
        spoken.append((wav.name, text, seed))
        return audio[:rate], rate

    return synthesize


def test_the_set_whose_takes_sound_most_like_the_speaker_is_kept(settings, data_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(reference_search, "TESTS", 4)
    manifest = mixed_dataset(data_dir)
    embed = SpeakerService(settings).embed_wavs
    phases = []
    p = reference_search.plan(manifest, 3, embed, progress=lambda phase, done, total: phases.append(phase))
    by_name = {c.name: {clip.index for clip in c.clips} for c in p.candidates}
    assert by_name["clean"] == {0, 1, 2}  # the off-voice clips read cleanest
    assert not by_name["alike"] & {0, 1, 2}
    assert len(p.tests) == 4 and not {t.index for t in p.tests} & set().union(*by_name.values())

    spoken = []
    tried = reference_search.search(echo(spoken), p, embed, tmp_path, lambda *a: phases.append(a[0]))
    chosen = reference_search.best(tried)
    scores = {t.name: t for t in tried}
    assert chosen.name != "clean" and scores["clean"].score < chosen.score
    assert scores["clean"].pitch == pytest.approx(12 * np.log2(500 / 220), abs=0.5)  # 500 Hz vs the speaker's 220
    assert abs(chosen.pitch) < 0.5
    # every set speaks the same sentences with the same seeds
    per_set = {}
    for wav, text, seed in spoken:
        per_set.setdefault(wav, []).append((text, seed))
    assert len(set(map(tuple, per_set.values()))) == 1 and len(per_set) == len(p.candidates)
    assert {"choosing_clips", "speaking_tests", "scoring_takes"} <= set(phases)


def test_a_small_dataset_is_refused(settings, data_dir):
    manifest = mixed_dataset(data_dir, clips=6)
    with pytest.raises(ValueError, match="too few"):
        reference_search.plan(manifest, 3, SpeakerService(settings).embed_wavs)


def test_the_quick_pick_takes_the_clips_most_like_the_voice(settings, service, training, data_dir, monkeypatch):
    manifest = mixed_dataset(data_dir)
    client = TestClient(create_app(settings, service, training))

    def picked(by, exclude=()):
        params = [("id", "speaker_a/train.jsonl"), ("count", "3"), ("by", by), *(("exclude", str(i)) for i in exclude)]
        r = client.get("/api/dataset-references", params=params)
        assert r.status_code == 200, r.text
        return [c["index"] for c in r.json()]

    assert sorted(picked("text")) == [0, 1, 2]  # the cleanest transcripts: the off-voice clips
    alike = picked("voice")
    assert len(alike) == 3 and not set(alike) & {0, 1, 2}
    assert not set(picked("voice", alike)) & (set(alike) | {0, 1, 2})

    # A checked dataset's own likeness scores are used as they are: nothing is measured.
    align = manifest.parent / manifests.ALIGN_FILE
    data = json.loads(align.read_text(encoding="utf-8"))
    for i, check in enumerate(data["clips"].values()):
        check["speaker"] = 0.1 if i < 3 else 0.5 + i / 100  # the last clips most alike
    data["voice"] = [1.0, 0.0]
    align.write_text(json.dumps(data), encoding="utf-8")

    def no_model(model):
        raise AssertionError("measured although the check had scores")

    monkeypatch.setattr(speaker, "load_campplus", no_model)
    client = TestClient(create_app(settings, service, training))  # a fresh speaker model
    assert picked("voice") == [23, 22, 21]


def finished(client, job):
    for _ in range(600):
        job = client.get(f"/api/jobs/{job['id']}").json()
        if job["state"] != "running":
            return job
        time.sleep(0.05)
    raise AssertionError("the job didn't finish")


def test_a_search_returns_clips_that_a_voice_is_then_saved_with(settings, service, training, data_dir, monkeypatch):
    monkeypatch.setattr(reference_search, "TESTS", 4)
    monkeypatch.setattr(engine.FishDrama3Engine, "unavailable", lambda self: None)
    fake = echo([])

    def synthesize(self, req):
        audio, rate = fake(req.text, req.prompt_wav, req.prompt_text, req.seed)
        return SynthesisResult(rate, audio, req.seed)

    monkeypatch.setattr(engine.FishDrama3Engine, "synthesize", synthesize)
    mixed_dataset(data_dir)
    client = TestClient(create_app(settings, service, training))
    dataset = "speaker_a/train.jsonl"

    job = finished(
        client, client.post("/api/reference-search", json={"dataset": dataset, "engine": "fish_drama3"}).json()
    )
    assert job["state"] == "done", job["error"]
    result = job["result"]
    assert result["chosen"] != "clean" and [t["name"] for t in result["tried"]][0] == "clean"
    kept = [c["index"] for c in result["clips"]]
    assert len(kept) == 3 and not set(kept) & {0, 1, 2} and all(c["text"] for c in result["clips"])
    assert not (settings.lora_dir / "fish_drama3").exists()  # a search saves nothing

    def texts(voice_id):
        return reference.read_reference(lora_library.resolve(settings.lora_dir, voice_id).path)[1]

    def said(indices):
        return [f"这是第{TEXTS[i]}句" for i in indices]

    # The clips shown are the clips saved.
    form = {"name": "voice_a", "dataset": dataset, "indices": json.dumps(kept)}
    assert client.post("/api/loras/reference-voice", data=form).status_code == 200
    assert all(t in texts("fish_drama3/voice_a") for t in said(kept))
    change = {"id": "fish_drama3/voice_a", "dataset": dataset, "indices": "[0, 1]"}
    assert client.post("/api/loras/reference", data=change).status_code == 200
    saved = texts("fish_drama3/voice_a")
    assert all(t in saved for t in said([0, 1])) and not any(t in saved for t in said(kept))
    assert client.post("/api/loras/reference", data={**change, "indices": "[999]"}).status_code == 404
    assert client.post("/api/loras/reference", data={**change, "indices": "[]"}).status_code == 422
    assert client.post("/api/loras/reference", data={**change, "indices": "x"}).status_code == 422

    other = client.post("/api/reference-search", json={"dataset": dataset, "engine": "fish_s2"})
    assert other.status_code == 422  # S2-Pro can't test sets by speaking
    assert client.post("/api/reference-search", json={"dataset": dataset}).status_code == 422  # which engine?
    listed = {e["id"]: e["capabilities"]["reference_search"] for e in client.get("/api/engines").json()}
    assert listed["fish_drama3"] and not listed["fish_s2"]
