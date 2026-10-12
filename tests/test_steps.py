"""Dataset processing steps (prep/steps.py) on mixed material: prepared clips
with and without text plus a long recording."""

import json

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient
from test_aligner import tone
from test_importer import wait_job

from polyvox.data import manifests, record
from polyvox.prep import checks, slicer, steps
from polyvox.prep.asr import load_audio
from polyvox.prep.slicer import SliceOptions
from polyvox.server import create_app

OPTS = SliceOptions(min_seconds=2.0, max_seconds=10.0, target_seconds=6.0, max_inner_gap=1.0, pad_seconds=0.1)


def no_vad(wav):
    return []


def material(folder):
    """Two prepared clips (one with text) and a 40 s recording."""
    folder.mkdir()
    sf.write(folder / "a.wav", tone([(0.3, 1.7)], 2.0), 16000)
    (folder / "a.lab").write_text("第一句", encoding="utf-8")
    sf.write(folder / "b.flac", tone([(0.3, 1.7)], 2.0), 16000)
    sf.write(folder / "talk.wav", tone([(1.0, 6.0), (8.0, 13.0)], 40.0), 16000)
    return folder


def transcriber(paths):
    return [("示例", "Chinese") for _ in paths]


def test_steps_work_on_what_still_needs_them(tmp_path, data_dir, align_service):
    src = material(tmp_path / "speaker_a")
    folder = steps.create(data_dir, "speaker_a")

    added = steps.add_source(folder, str(src), None, data_dir)
    assert (added.clips, added.recordings) == (2, 1)
    status = steps.status(folder)
    assert status["slice"] == {"recordings": 1, "seconds": 40.0}
    assert status["transcribe"] == {"clips": 1} and status["check"] == {"clips": 1}
    assert steps.remaining(folder) == ["slice", "transcribe", "check"]
    # Adding the same source again adds nothing.
    assert (steps.add_source(folder, str(src), None, data_dir).clips, len(record.read_record(folder).recordings)) == (
        0,
        1,
    )

    vad = lambda wav: [(1.0, 6.0), (8.0, 13.0)]
    assert steps.slice_recordings(folder, OPTS, vad)["clips"] == 2  # no guide: the clips have no text yet
    assert steps.status(folder)["transcribe"] == {"clips": 3}
    result = steps.transcribe(folder, transcriber, settings={"language": "zh"})
    assert result == {"transcribed": 3, "dropped": 0, "tidied": 0}
    assert set(manifests.read_sidecar(folder / manifests.ASR_FILE)) == {
        "wavs/b.flac",
        "wavs/talk_00000900_00006100.wav",
        "wavs/talk_00007900_00013100.wav",
    }

    checker = checks.make_checker(lambda w, t: align_service.align(w, t, "qwen3"), "qwen3", no_vad)
    assert steps.check(folder, checker, "qwen3", load_audio)["checked"] == 4
    assert steps.remaining(folder) == []
    # Only the edited clip is checked again.
    manifests.set_text(folder / "train.jsonl", 0, "第一句话")
    assert steps.status(folder)["check"] == {"clips": 1}
    steps.check(folder, checker, "qwen3", load_audio)
    assert steps.status(folder)["check"] == {"clips": 0}

    saved = record.read_record(folder)
    assert saved.steps["transcribe"] == {"language": "zh", "tidy_fillers": False}
    assert saved.steps["check"] == {"aligner": "qwen3"}


def test_step_endpoints(service, training, asr_service, align_service, data_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(slicer, "silero_vad_many", lambda wavs: [[] for _ in wavs])
    src = material(tmp_path / "speaker_a")
    (src / "talk.wav").unlink()  # no recording: no slicing (needs the real VAD)
    folder = steps.create(data_dir, "speaker_a")
    client = TestClient(create_app(service.settings, service, training, asr_service, align=align_service))
    ds = "speaker_a/train.jsonl"

    job = client.post("/api/datasets/add-source", json={"id": ds, "source": str(src)}).json()
    assert wait_job(client, job["id"])["result"] == {"clips": 2, "recordings": 0}
    status = client.get("/api/datasets/steps", params={"id": ds}).json()
    assert status["transcribe"] == {"clips": 1} and status["slice"] == {"recordings": 0, "seconds": 0}

    job = client.post("/api/datasets/run-remaining", json={"id": ds}).json()
    done = wait_job(client, job["id"])
    assert done["state"] == "done", done
    assert set(done["result"]) >= {"transcribe", "check"}
    status = client.get("/api/datasets/steps", params={"id": ds}).json()
    assert (status["transcribe"], status["check"]) == ({"clips": 0}, {"clips": 0})
    assert status["settings"]["transcribe"]["language"] == "auto"

    job = client.post("/api/datasets/step", json={"id": ds, "step": "tidy", "normalize_lufs": -20}).json()
    assert wait_job(client, job["id"])["state"] == "done"
    assert record.read_record(folder).steps["tidy"] == {"trim": True, "normalize_lufs": -20.0}
    assert client.post("/api/datasets/step", json={"id": ds, "step": "nope"}).status_code == 422
    entries = [json.loads(line) for line in (folder / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(e["text"] for e in entries)
    assert np.isfinite(sf.info(str(folder / "wavs" / "b.flac")).duration)
