"""Alignment checks on synthetic audio: tones stand in for speech."""

import json

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from test_aligner import tone
from test_importer import wait_job, write_list

from voxcpm_studio import checks, importer, manifests
from voxcpm_studio.aligner import Alignment, Word
from voxcpm_studio.api import create_app
from voxcpm_studio.asr import load_audio


def aligned(start, end, score=0.9):
    return Alignment([Word("一", start, (start + end) / 2, score), Word("二", (start + end) / 2, end, score)], score)


def test_a_clean_clip_has_no_flags():
    wav = tone([(0.3, 2.0)], 2.4)

    check = checks.check_clip(wav, aligned(0.4, 1.8), "qwen3", [(0.3, 2.0)])

    assert check.flags == []
    assert (check.start, check.end) == pytest.approx((0.3, 2.0), abs=0.02)  # refined to the sound
    assert (check.lead, check.trail, check.duration) == pytest.approx((0.3, 0.4, 2.4), abs=0.02)


def test_flags_mismatch_extra_speech_cut_edges_and_silence():
    wav = tone([(0.0, 1.0), (1.8, 2.4)], 4.0)  # speech from the very start; a phrase the text lacks

    check = checks.check_clip(wav, aligned(0.0, 1.0, score=0.2), "qwen3", [(0.0, 1.0), (1.8, 2.4)])

    assert check.flags == ["mismatch", "extra_speech", "cut_edge", "long_silence"]
    assert check.unaligned == pytest.approx(0.6, abs=0.02)
    # Thresholds are per aligner: the same score passes for CTC's scale.
    assert "mismatch" not in checks.check_clip(wav, aligned(0.0, 1.0, score=0.2), "ctc", [(0.0, 1.0)]).flags
    assert checks.check_clip(wav, Alignment([], 0.0), "ctc", []).flags[0] == "mismatch"


def test_a_noisy_background_is_not_mistaken_for_speech_at_the_edge():
    wav = tone([(0.4, 1.6)], 2.0)
    wav += np.random.default_rng(1).normal(0, 0.012, len(wav)).astype(np.float32)  # floor ~25 dB below the voice

    check = checks.check_clip(wav, aligned(0.5, 1.5), "qwen3", [(0.4, 1.6)])

    assert "cut_edge" not in check.flags
    assert (check.start, check.end) == pytest.approx((0.4, 1.6), abs=0.03)


def test_trim_keeps_padding_around_all_speech():
    wav = tone([(1.5, 3.0)], 5.0)
    check = checks.check_clip(wav, aligned(1.6, 2.9), "qwen3", [(1.5, 3.0)])
    assert "long_silence" in check.flags

    start, end = check.trim_range()
    after = check.trimmed(start, end)

    assert (start, end) == pytest.approx((1.35, 3.15), abs=0.02)
    assert (after.lead, after.trail, after.start) == pytest.approx((0.15, 0.15, 0.15), abs=0.02)
    assert "long_silence" not in after.flags
    assert after.trim_range() is None  # nothing left worth cutting


def write_dataset(data_dir, texts):
    folder = data_dir / "speaker_a"
    (folder / "wavs").mkdir(parents=True)
    lines = []
    for i, text in enumerate(texts):
        sf.write(folder / "wavs" / f"{i}.wav", tone([(0.5, 1.5)], 2.0), 16000)
        lines.append(json.dumps({"audio": f"wavs/{i}.wav", "text": text}, ensure_ascii=False))
    (folder / "train.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return folder / "train.jsonl"


def no_vad(wav):
    return []


def test_check_dataset_writes_results_the_review_shows(data_dir, align_service):
    manifest = write_dataset(data_dir, ["第一句", "错的一句", "第三句"])
    phases = []
    checker = checks.make_checker(lambda w, t: align_service.align(w, t, "qwen3"), "qwen3", no_vad)

    summary = checks.check_dataset(manifest, None, checker, "qwen3", load_audio, lambda *p: phases.append(p))

    assert summary == {
        "aligner": "qwen3",
        "checked": 3,
        "flagged": 1,
        "flags": {"mismatch": 1, "extra_speech": 0, "cut_edge": 0, "long_silence": 0},
    }
    assert phases[-1] == ("aligning", 3, 3)
    page = manifests.read_rows(manifest, flagged=True)
    assert [(r.file, r.flags) for r in page.rows] == [("wavs/1.wav", ["mismatch"])]


def test_import_checks_and_trims_the_copies(tmp_path, data_dir, align_service):
    src = tmp_path / "recordings"
    src.mkdir()
    sf.write(src / "long_pause.flac", tone([(2.0, 3.0)], 4.5), 16000)  # trimmed, rewritten as WAV
    sf.write(src / "tight.wav", tone([(0.1, 0.9)], 1.0), 16000)
    (src / "long_pause.lab").write_text("第一句", encoding="utf-8")
    (src / "tight.lab").write_text("第二句", encoding="utf-8")
    checker = checks.make_checker(lambda w, t: align_service.align(w, t, "ctc"), "ctc", no_vad)

    result = importer.import_dataset(str(src), None, "trimmed", data_dir, checker=checker, aligner="ctc", trim=True)

    assert (result.imported, result.trimmed) == (2, 1)
    assert result.checks["flagged"] == 0  # the long silence was cut away
    target = data_dir / "trimmed"
    assert sorted(p.name for p in (target / "wavs").iterdir()) == ["long_pause.wav", "tight.wav"]
    assert sf.info(str(target / "wavs" / "long_pause.wav")).duration == pytest.approx(1.3, abs=0.03)
    assert sf.info(str(target / "wavs" / "tight.wav")).duration == pytest.approx(1.0)
    saved = json.loads((target / manifests.ALIGN_FILE).read_text(encoding="utf-8"))
    assert saved["aligner"] == "ctc" and saved["clips"]["wavs/long_pause.wav"]["lead"] == pytest.approx(0.15, abs=0.02)
    assert (src / "long_pause.flac").is_file()  # the source is untouched

    with pytest.raises(ValueError, match="needs copying and checking"):
        importer.import_dataset(str(src), None, "x", data_dir, trim=True)


def test_check_job_endpoint(service, training, asr_service, align_service, align_loader, data_dir, monkeypatch):
    monkeypatch.setattr(checks, "silero_vad", no_vad)
    write_dataset(data_dir, ["第一句", "错的一句"])
    client = TestClient(create_app(service.settings, service, training, asr_service, align=align_service))

    resp = client.post("/api/datasets/check", json={"id": "speaker_a/train.jsonl", "aligner": "qwen3"})
    assert resp.status_code == 202 and resp.json()["kind"] == "check"
    job = wait_job(client, resp.json()["id"])
    assert job["state"] == "done" and job["result"]["flagged"] == 1, job
    assert align_loader.loaded == ["qwen3"]

    page = client.get("/api/dataset-rows", params={"id": "speaker_a/train.jsonl", "order": "worst"}).json()
    assert page["checks"]["aligner"] == "qwen3" and page["rows"][0]["flags"] == ["mismatch"]
    flagged = client.get("/api/dataset-rows", params={"id": "speaker_a/train.jsonl", "flagged": True}).json()
    assert flagged["total"] == 1
    assert client.get("/api/status").json()["aligner_loaded"] is True
    assert client.post("/api/model/unload").json()["aligner_loaded"] is False

    assert client.post("/api/datasets/check", json={"id": "nope/train.jsonl"}).status_code == 404
    assert client.post("/api/datasets/check", json={"id": "speaker_a/train.jsonl", "aligner": "x"}).status_code == 422


def test_import_endpoint_with_checks(service, training, asr_service, align_service, data_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(checks, "silero_vad", no_vad)
    src = tmp_path / "speaker_b"
    src.mkdir()
    sf.write(src / "a.wav", tone([(0.3, 1.2)], 1.5), 16000)
    lst = write_list(tmp_path / "speaker_b.list", [f"{src / 'a.wav'}|speaker_b|ZH|示例"])
    client = TestClient(create_app(service.settings, service, training, asr_service, align=align_service))

    body = {"source": str(lst), "name": "speaker_b", "check": True, "trim": True}
    job = wait_job(client, client.post("/api/datasets/import", json=body).json()["id"])

    assert job["state"] == "done", job
    assert job["result"]["checks"]["checked"] == 1 and job["result"]["trimmed"] == 1  # 0.3 s silent edges
    assert (data_dir / "speaker_b" / manifests.ALIGN_FILE).is_file()
    no_check = {**body, "name": "other", "check": False}
    assert client.post("/api/datasets/import", json=no_check).status_code == 422
