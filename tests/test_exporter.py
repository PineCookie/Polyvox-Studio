import json
import time

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from voxcpm_studio.api import create_app
from voxcpm_studio.prep import exporter

RATE = 44_100


@pytest.fixture
def client(service, training):
    return TestClient(create_app(service.settings, service, training))


def make_dataset(tmp_path, clips):
    """clips: [(seconds, text)]; one dropped in review (excluded.json) is added."""
    folder = tmp_path / "data" / "speaker_a"
    (folder / "wavs").mkdir(parents=True)
    lines = []
    for i, (seconds, text) in enumerate([*clips, (2.0, "被丢弃的")]):
        wav = folder / "wavs" / f"clip {i}.wav"
        sf.write(wav, np.full(int(RATE * seconds), 0.1, dtype=np.float32), RATE)
        lines.append(json.dumps({"audio": f"wavs/clip {i}.wav", "text": text}, ensure_ascii=False))
    manifest = folder / "train.jsonl"
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (folder / "excluded.json").write_text(json.dumps({"clips": [f"wavs/clip {len(clips)}.wav"]}), encoding="utf-8")
    return manifest


CLIPS = [
    (2.0, "示例，第一句|还有一行\n换行"),
    (0.5, "太短"),
    (3.0, "Hello there."),
    (20.0, "太长的一段"),
    (2.5, "例えば"),
]


def test_gpt_sovits_list(tmp_path):
    manifest = make_dataset(tmp_path, CLIPS)
    out = tmp_path / "exports" / "speaker_a_gpt_sovits"
    result = exporter.export_dataset(manifest, None, out, "gpt_sovits", "speaker_a", min_seconds=1, max_seconds=15)

    assert result["clips"] == 3 and result["skipped"] == {"short": 1, "long": 1, "unreadable": 0}
    rows = [line.split("|") for line in (out / "speaker_a.list").read_text(encoding="utf-8").splitlines()]
    assert [(r[1], r[2], r[3]) for r in rows] == [
        ("speaker_a", "zh", "示例，第一句 还有一行 换行"),
        ("speaker_a", "en", "Hello there."),
        ("speaker_a", "ja", "例えば"),
    ]
    first = rows[0][0]
    assert first == str(out / "wavs" / "speaker_a_00001.wav") and sf.info(first).samplerate == RATE
    assert not list(tmp_path.glob("exports/.*"))  # no staging folder left


def test_resampled_kaldi_and_forced_language(tmp_path):
    manifest = make_dataset(tmp_path, CLIPS[:1] + CLIPS[2:3])
    out = tmp_path / "kaldi"
    exporter.export_dataset(manifest, None, out, "kaldi", "speaker_a", sample_rate=16_000)

    assert (out / "wav.scp").read_text(encoding="utf-8").splitlines()[
        1
    ] == f"speaker_a_00002 {out / 'wavs' / 'speaker_a_00002.wav'}"
    assert (out / "utt2spk").read_text(encoding="utf-8").split() == [
        "speaker_a_00001",
        "speaker_a",
        "speaker_a_00002",
        "speaker_a",
    ]
    assert (out / "spk2utt").read_text(encoding="utf-8").strip() == "speaker_a speaker_a_00001 speaker_a_00002"
    info = sf.info(out / "wavs" / "speaker_a_00001.wav")
    assert info.samplerate == 16_000 and abs(info.duration - 2.0) < 0.01

    forced = tmp_path / "forced"
    exporter.export_dataset(manifest, None, forced, "gpt_sovits", "speaker_a", language="yue")
    assert {line.split("|")[2] for line in (forced / "speaker_a.list").read_text(encoding="utf-8").splitlines()} == {
        "yue"
    }


@pytest.mark.parametrize(
    ("fmt", "index", "first_line"),
    [
        ("ljspeech", "metadata.csv", "speaker_a_00001|Hello there.|Hello there."),
        ("csv", "metadata.csv", "audio_file|text"),
        ("lab", "wavs/speaker_a_00001.lab", "Hello there."),
    ],
)
def test_other_layouts(tmp_path, fmt, index, first_line):
    manifest = make_dataset(tmp_path, [(2.0, "Hello there.")])
    out = tmp_path / fmt
    exporter.export_dataset(manifest, None, out, fmt, "speaker_a")
    assert (out / index).read_text(encoding="utf-8").splitlines()[0] == first_line
    if fmt == "csv":
        assert (out / index).read_text(encoding="utf-8").splitlines()[1] == "wavs/speaker_a_00001.wav|Hello there."


def test_refuses_a_folder_with_files_and_cleans_up_on_failure(tmp_path):
    manifest = make_dataset(tmp_path, [(2.0, "示例")])
    taken = tmp_path / "taken"
    taken.mkdir()
    (taken / "x.txt").write_text("x")
    with pytest.raises(FileExistsError):
        exporter.export_dataset(manifest, None, taken, "lab", "speaker_a")

    def cancel(phase, done, total):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        exporter.export_dataset(manifest, None, tmp_path / "cancelled", "lab", "speaker_a", progress=cancel)
    assert not (tmp_path / "cancelled").exists() and not (tmp_path / ".cancelled.export-tmp").exists()


def test_export_api(client, data_dir):
    folder = data_dir / "voice" / "wavs"
    folder.mkdir(parents=True)
    sf.write(folder / "a.wav", np.zeros(RATE * 2, dtype=np.float32), RATE)
    (data_dir / "voice" / "train.jsonl").write_text(
        json.dumps({"audio": "wavs/a.wav", "text": "示例"}) + "\n", encoding="utf-8"
    )

    formats = client.get("/api/export-formats").json()
    assert [f["id"] for f in formats] == [f.id for f in exporter.FORMATS]
    default = client.get("/api/datasets/export-dir", params={"id": "voice/train.jsonl", "format": "lab"}).json()["dir"]
    assert default.endswith("voice_lab")

    job = client.post("/api/datasets/export", json={"id": "voice/train.jsonl", "format": "lab"}).json()
    deadline = time.monotonic() + 30
    while (job := client.get(f"/api/jobs/{job['id']}").json())["state"] == "running":
        assert time.monotonic() < deadline, "export did not finish"
        time.sleep(0.02)
    assert job["state"] == "done" and job["result"]["clips"] == 1
    assert client.post("/api/datasets/export", json={"id": "voice/train.jsonl", "format": "lab"}).status_code == 409
    assert client.post("/api/datasets/export", json={"id": "voice/train.jsonl", "format": "nope"}).status_code == 422
