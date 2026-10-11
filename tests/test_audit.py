"""Review: edit transcripts, drop clips, compare with ASR (pseudo data)."""

import json

import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from test_aligner import tone
from test_checks import write_dataset
from test_importer import wait_job

from voxcpm_studio.api import create_app
from voxcpm_studio.data import manifests
from voxcpm_studio.prep import checks, singing, slicer
from voxcpm_studio.training import TrainParams

ID = "speaker_a/train.jsonl"


@pytest.fixture
def client(service, training, asr_service, align_service, monkeypatch):
    monkeypatch.setattr(slicer, "silero_vad_many", lambda wavs: [[] for _ in wavs])
    return TestClient(create_app(service.settings, service, training, asr_service, align=align_service))


def test_set_text_rewrites_one_entry_and_keeps_the_original(data_dir):
    manifest = write_dataset(data_dir, ["第一句", "第二句"])
    before = manifest.read_text(encoding="utf-8")

    manifests.set_text(manifest, 1, " 改过的一句 ")

    texts = [json.loads(line)["text"] for line in manifest.read_text(encoding="utf-8").splitlines()]
    assert texts == ["第一句", "改过的一句"]
    assert (manifest.parent / "train.jsonl.orig").read_text(encoding="utf-8") == before
    with pytest.raises(ValueError, match="empty"):
        manifests.set_text(manifest, 0, "  ")
    with pytest.raises(KeyError):
        manifests.set_text(manifest, 5, "x")


def test_editing_a_flagged_clip_checks_it_again(client, data_dir):
    manifest = write_dataset(data_dir, ["第一句", "错错错句"])
    assert wait_job(client, client.post("/api/datasets/check", json={"id": ID}).json()["id"])["state"] == "done"
    assert client.get("/api/dataset-clip", params={"id": ID, "index": 1}).json()["flags"] == ["mismatch"]

    clip = client.post("/api/dataset-clip/text", json={"id": ID, "index": 1, "text": "第二句"}).json()

    assert clip["text"] == "第二句" and clip["flags"] == [] and [w["text"] for w in clip["words"]] == ["第", "二", "句"]
    assert json.loads(manifest.read_text(encoding="utf-8").splitlines()[1])["text"] == "第二句"
    rows = client.get("/api/dataset-rows", params={"id": ID}).json()
    assert rows["checks"]["checked"] == 2 and rows["checks"]["flagged"] == 0


def test_dropped_clips_stay_listed_but_leave_training_and_stats(client, training, data_dir):
    write_dataset(data_dir, ["第一句", "第二句", "第三句", "第四句"])

    assert client.post("/api/dataset-clip/drop", json={"id": ID, "index": 1}).json()["dropped"] is True

    page = client.get("/api/dataset-rows", params={"id": ID}).json()
    assert page["total"] == 4 and page["dropped"] == 1 and [r["dropped"] for r in page["rows"]] == [0, 1, 0, 0]
    only = client.get("/api/dataset-rows", params={"id": ID, "flag": "dropped"}).json()
    assert [r["index"] for r in only["rows"]] == [1]
    report = client.get("/api/dataset-report", params={"id": ID}).json()
    assert (report["samples"], report["dropped"], report["valid"]) == (3, 1, 3)

    training.start(TrainParams(name="run", dataset=ID, val_holdout=0, num_iters=1))
    lines = (training.lora_dir / "run" / "manifests" / "train.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["text"] for line in lines] == ["第一句", "第三句", "第四句"]

    client.post("/api/dataset-clip/drop", json={"id": ID, "index": 1, "dropped": False})
    assert client.get("/api/dataset-rows", params={"id": ID}).json()["dropped"] == 0
    assert client.post("/api/dataset-clip/drop", json={"id": ID, "index": 9}).status_code == 404


def test_sung_clips_are_flagged_and_can_be_dropped_at_once(client, data_dir):
    manifest = write_dataset(data_dir, ["第一句", "第二句", "第三句"])
    sf.write(manifest.parent / "wavs" / "1.wav", tone([(0.0, 2.0)], 2.0), 16000)  # held throughout: "sung"

    sing = {"id": ID, "detect_singing": True}
    assert wait_job(client, client.post("/api/datasets/check", json=sing).json()["id"])["state"] == "done"
    rows = client.get("/api/dataset-rows", params={"id": ID, "flag": "singing"}).json()["rows"]
    assert [r["index"] for r in rows] == [1]

    assert client.post("/api/datasets/drop-flagged", json={"id": ID, "flag": "singing"}).json() == {"dropped": 1}
    assert client.post("/api/datasets/drop-flagged", json={"id": ID, "flag": "singing"}).json() == {"dropped": 0}
    assert client.get("/api/dataset-rows", params={"id": ID}).json()["dropped"] == 1
    assert client.post("/api/datasets/drop-flagged", json={"id": ID, "flag": "nope"}).status_code == 422
    # Without detection asked for (the default), nothing is flagged as sung.
    body = {"id": ID}
    assert (
        wait_job(client, client.post("/api/datasets/check", json=body).json()["id"])["result"]["flags"]["singing"] == 0
    )


def test_singing_rule_needs_little_speech():
    assert singing.is_sung(0.05, 0.1)
    assert not singing.is_sung(0.05, 0.6)  # talk over music lifts the singing score
    assert not singing.is_sung(0.001, 0.0)


def test_compare_asr_counts_what_the_transcript_lacks():
    result = checks.compare_asr("今天天气真好。", "嗯，今天天气真好，我们出去走走吧")
    assert (result["extra"], result["missing"], result["changed"], result["words"]) == (7, 0, 0, 6)  # fillers ignored
    assert checks._hears_more(result)
    assert not checks._hears_more(checks.compare_asr("今天天气真好", "今天天汽真好"))  # one wrong character
    assert checks.asr_diff("今天好", "今天真好") == [
        ("equal", "今 天", "今 天"),
        ("extra", "", "真"),
        ("equal", "好", "好"),
    ]


def test_check_can_compare_with_asr_but_not_asr_made_texts(client, data_dir):
    manifest = write_dataset(data_dir, ["clip", "第二句"])  # ASR hears more than "clip"
    (manifest.parent / manifests.ASR_FILE).write_text(json.dumps({"wavs/1.wav": {"language": "zh"}}))

    job = client.post("/api/datasets/check", json={"id": ID, "compare_asr": True}).json()
    assert wait_job(client, job["id"])["result"]["flags"]["asr_more"] == 1

    first = client.get("/api/dataset-clip", params={"id": ID, "index": 0}).json()
    assert first["flags"] == ["asr_more"]  # the fake ASR hears "clip of N samples"
    assert first["heard"]["text"].startswith("clip of") and first["heard"]["extra"] >= 3
    assert {op for op, _, _ in first["heard"]["diff"]} >= {"extra"}
    second = client.get("/api/dataset-clip", params={"id": ID, "index": 1}).json()
    assert second["heard"] is None and second["flags"] == []  # its text came from ASR
