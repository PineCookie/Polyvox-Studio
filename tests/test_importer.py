"""Importer tests on pseudo data: invented speakers, paths and sentences."""

import json
import time

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from voxcpm_studio import importer, manifests
from voxcpm_studio.api import create_app
from voxcpm_studio.asr import AsrService
from voxcpm_studio.jobs import JobCancelled
from voxcpm_studio.training import TrainParams

from conftest import make_dataset, wait_for

# A list made on another machine: none of these locations exist here.
STALE_WINDOWS = r"X:\old_pc\tts_project\speaker_a\clips"
STALE_POSIX = "/home/someone/tts_project/clips"


def wav(path, seconds=1.0, rate=32000):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.zeros(int(rate * seconds), dtype=np.float32), rate)
    return path


@pytest.fixture
def clips(tmp_path):
    """Clips that exist, but not where the list says."""
    folder = tmp_path / "moved" / "clips"
    return [wav(folder / f"take_{i}.wav") for i in range(3)]


def write_list(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def fake_transcriber(texts=None):
    """Transcribes a file as `texts[name]`, or "asr <name>"; records calls."""
    calls = []

    def transcribe(paths):
        calls.append([p.name for p in paths])
        return [((texts or {}).get(p.name, f"asr {p.stem}"), "Chinese") for p in paths]

    transcribe.calls = calls
    return transcribe


def wait_job(client, job_id, timeout=30):
    deadline = time.monotonic() + timeout
    while (job := client.get(f"/api/jobs/{job_id}").json())["state"] == "running":
        assert time.monotonic() < deadline, "job did not finish"
        time.sleep(0.02)
    return job


def test_gsv_list_with_stale_paths_relinks_by_file_name(tmp_path, data_dir, clips):
    lst = write_list(
        tmp_path / "lists" / "示例.list",  # a Unicode speaker name
        [
            rf"{STALE_WINDOWS}\take_0.wav|clips|ZH|第一句。",
            f"{STALE_POSIX}/take_1.wav|clips|ZH|第二句。",
            rf"{STALE_WINDOWS}\take_9.wav|clips|ZH|没有音频的一句。",
            rf"{STALE_WINDOWS}\take_2.wav|clips|ZH|",
            rf"{STALE_WINDOWS}\other\take_0.wav|clips|ZH|重复的一句。",
        ],
    )

    without = importer.preview_import(str(lst), None, data_dir)
    assert (without.kind, without.rows, without.found, without.missing) == ("gsv", 5, 0, 5)

    p = importer.preview_import(f'"{lst}"', str(clips[0].parent), data_dir)  # quoted, as Explorer copies
    assert (p.found, p.untranscribed, p.missing, p.duplicates) == (2, 1, 1, 1)
    assert p.missing_examples == [rf"{STALE_WINDOWS}\take_9.wav"]
    assert p.total_seconds == pytest.approx(3.0) and p.sample_rates == {32000: 3}
    assert p.suggested_name == "示例"


def test_csv_resolves_against_sibling_wavs_folder(tmp_path, data_dir):
    root = tmp_path / "speaker_b"
    wav(root / "wavs" / "0001.wav")
    wav(root / "wavs" / "0002.wav")
    csv = write_list(root / "metadata.csv", ["0001.wav|Hello there.", "0002|Second line."])  # second by id only

    p = importer.preview_import(str(csv), str(root / "wavs"), data_dir)

    assert (p.kind, p.found, p.suggested_name) == ("csv", 2, "speaker_b")


def test_list_next_to_its_wavs_folder_needs_no_audio_folder(tmp_path, data_dir):
    root = tmp_path / "speaker_c"
    wav(root / "wavs" / "take_0001.wav")
    lst = write_list(root / "speaker_c.list", [f"{STALE_POSIX}/take_0001.wav|clips|EN|A line."])

    assert importer.preview_import(str(lst), None, data_dir).found == 1


def test_folder_with_sidecar_transcripts(tmp_path, data_dir):
    folder = tmp_path / "recordings"
    wav(folder / "a.wav").with_suffix(".lab").write_text("第一句", encoding="utf-8")
    wav(folder / "sub" / "b.flac").with_suffix(".txt").write_text("second", encoding="utf-8")
    wav(folder / "no_text.wav")

    p = importer.preview_import(str(folder), None, data_dir)

    assert (p.kind, p.rows, p.found, p.untranscribed) == ("folder", 3, 2, 1)


def test_import_copies_audio_and_writes_relative_manifest(tmp_path, data_dir, clips):
    same_name_elsewhere = wav(tmp_path / "other" / "take_0.wav")
    lst = write_list(
        tmp_path / "speaker_a.list",
        [f"{c}|clips|EN|line {i}" for i, c in enumerate(clips)] + [f"{same_name_elsewhere}|clips|EN|same file name"],
    )

    result = importer.import_dataset(str(lst), None, "speaker_a", data_dir)

    assert (result.id, result.imported, result.transcribed) == ("speaker_a/train.jsonl", 4, 0)
    manifest = data_dir / "speaker_a" / "train.jsonl"
    entries = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    assert [e["audio"] for e in entries] == [
        "wavs/take_0.wav",
        "wavs/take_1.wav",
        "wavs/take_2.wav",
        "wavs/take_0_2.wav",
    ]
    # Self-contained: resolvable without any legacy fallback.
    resolved, errors = manifests.read_manifest(manifest)
    assert not errors and all(manifests.Path(e["audio"]).is_file() for e in resolved)
    assert not list(data_dir.glob(".import-*"))
    assert not (data_dir / "speaker_a" / importer.ASR_FILE).exists()


def test_untranscribed_clips_are_left_out_without_asr(tmp_path, data_dir):
    folder = tmp_path / "recordings"
    wav(folder / "a.wav").with_suffix(".lab").write_text("第一句", encoding="utf-8")
    wav(folder / "b.wav")

    result = importer.import_dataset(str(folder), None, "partial", data_dir)

    assert (result.imported, result.transcribed) == (1, 0)


def test_transcribe_on_import_fills_missing_texts_in_order(tmp_path, data_dir):
    folder = tmp_path / "recordings"
    for name in ("a", "b", "c", "d"):
        wav(folder / f"{name}.wav")
    (folder / "b.lab").write_text("已有文本", encoding="utf-8")
    transcriber = fake_transcriber({"d.wav": "  "})  # ASR hears nothing in d
    phases = []

    result = importer.import_dataset(
        str(folder), None, "asr_set", data_dir, transcriber=transcriber, progress=lambda *p: phases.append(p)
    )

    assert (result.imported, result.transcribed) == (3, 2)
    assert transcriber.calls == [["a.wav", "c.wav", "d.wav"]]  # only clips without text
    entries = [
        json.loads(line) for line in (data_dir / "asr_set" / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert entries == [
        {"audio": "wavs/a.wav", "text": "asr a"},
        {"audio": "wavs/b.wav", "text": "已有文本"},
        {"audio": "wavs/c.wav", "text": "asr c"},
    ]
    asr_rows = json.loads((data_dir / "asr_set" / importer.ASR_FILE).read_text(encoding="utf-8"))
    assert asr_rows == {"wavs/a.wav": {"language": "Chinese"}, "wavs/c.wav": {"language": "Chinese"}}
    assert ("transcribing", 3, 3) in phases and any(p[0] == "copying" for p in phases)


def test_cancelled_import_leaves_nothing(tmp_path, data_dir, clips):
    lst = write_list(tmp_path / "speaker_a.list", [f"{c}|clips|EN|" for c in clips])

    def cancel_while_copying(phase, done, total):
        if phase == "copying":
            raise JobCancelled()

    with pytest.raises(JobCancelled):
        importer.import_dataset(
            str(lst), None, "cancelled", data_dir, transcriber=fake_transcriber(), progress=cancel_while_copying
        )
    assert not (data_dir / "cancelled").exists() and not list(data_dir.glob(".import-*"))


def test_import_in_place_and_conflicts(tmp_path, data_dir, clips):
    lst = write_list(tmp_path / "speaker_a.list", [f"{c}|clips|EN|line" for c in clips])

    importer.import_dataset(str(lst), None, "in_place", data_dir, copy=False)
    entries = (data_dir / "in_place" / "train.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(entries[0])["audio"] == str(clips[0].resolve())
    assert not (data_dir / "in_place" / "wavs").exists()

    with pytest.raises(FileExistsError):
        importer.import_dataset(str(lst), None, "in_place", data_dir)
    with pytest.raises(ValueError):
        importer.import_dataset(str(lst), None, "../x", data_dir)
    missing = write_list(tmp_path / "no_audio.list", [rf"{STALE_WINDOWS}\a.wav|clips|EN|line"])
    with pytest.raises(ValueError, match="nothing to import"):
        importer.import_dataset(str(missing), None, "no_audio", data_dir)
    with pytest.raises(FileNotFoundError):
        importer.preview_import(str(tmp_path / "nope.list"), None, data_dir)


def test_import_job_endpoints(service, training, asr_service, data_dir, tmp_path, clips):
    lst = write_list(
        tmp_path / "speaker_a.list",
        [f"{c}|clips|EN|" if i == 0 else f"{c}|clips|EN|line {i}" for i, c in enumerate(clips)],
    )
    client = TestClient(create_app(service.settings, service, training, asr_service))

    preview = client.post("/api/datasets/import/preview", json={"source": str(lst)}).json()
    assert (preview["found"], preview["untranscribed"], preview["suggested_name"]) == (2, 1, "speaker_a")

    resp = client.post(
        "/api/datasets/import", json={"source": str(lst), "name": "speaker_a", "transcribe": True, "language": "en"}
    )
    assert resp.status_code == 202 and resp.json()["kind"] == "import"
    job = wait_job(client, resp.json()["id"])
    assert job["state"] == "done", job
    assert job["result"] == {"id": "speaker_a/train.jsonl", "imported": 3, "transcribed": 1, "sliced": 0}
    first = json.loads((data_dir / "speaker_a" / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert first["text"] == "clip of 16000 samples"  # the fake ASR; 1 s at 16 kHz
    assert asr_service._processor.requests[-1]["language"] == "en"
    # A forced language is recorded as forced (the model reports none then).
    asr_rows = json.loads((data_dir / "speaker_a" / importer.ASR_FILE).read_text(encoding="utf-8"))
    assert list(asr_rows.values()) == [{"language": "en"}]

    assert "speaker_a/train.jsonl" in client.get("/api/datasets").json()
    assert client.post("/api/datasets/import", json={"source": str(lst), "name": "speaker_a"}).status_code == 409
    assert client.post("/api/datasets/import", json={"source": str(tmp_path / "x.mp4"), "name": "x"}).status_code == 422
    assert client.post("/api/datasets/import/preview", json={"source": str(tmp_path / "x.mp4")}).status_code == 422
    assert client.get("/api/jobs/nope").status_code == 404
    no_asr = {"source": str(lst), "name": "other", "slice": True, "transcribe": False}
    assert client.post("/api/datasets/import", json=no_asr).status_code == 422


def test_transcribing_import_refused_while_training_shares_the_gpu(
    service, training, asr_service, data_dir, tmp_path, clips, monkeypatch
):
    make_dataset(data_dir, "voice")
    training.start(TrainParams(name="busy", dataset="voice/train.jsonl", num_iters=1000))
    wait_for(training.lora_dir / "busy" / "console.log", "waiting for stop")
    monkeypatch.setattr(AsrService, "device", property(lambda self: "cuda:0"))
    lst = write_list(tmp_path / "speaker_a.list", [f"{c}|clips|EN|" for c in clips])
    client = TestClient(create_app(service.settings, service, training, asr_service))

    body = {"source": str(lst), "name": "speaker_a", "transcribe": True}
    resp = client.post("/api/datasets/import", json=body)
    assert resp.status_code == 409 and "training" in resp.json()["detail"]
    # Without transcription the import does not need the GPU.
    body["transcribe"] = False
    assert client.post("/api/datasets/import", json=body).status_code == 202


def test_failed_import_job_reports_the_error(service, training, data_dir, tmp_path):
    lst = write_list(tmp_path / "no_audio.list", [rf"{STALE_WINDOWS}\a.wav|clips|EN|line"])
    client = TestClient(create_app(service.settings, service, training))

    job = wait_job(client, client.post("/api/datasets/import", json={"source": str(lst), "name": "x"}).json()["id"])

    assert job["state"] == "failed" and "nothing to import" in job["error"]
