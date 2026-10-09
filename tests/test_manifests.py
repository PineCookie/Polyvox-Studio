import json

import numpy as np
import pytest
import soundfile as sf

from voxcpm_studio import manifests

from conftest import make_dataset


def test_scan_and_resolve_stay_inside_data_dir(data_dir):
    make_dataset(data_dir, "a")
    (data_dir.parent / "outside.jsonl").write_text("")

    assert manifests.scan(data_dir) == ["a/train.jsonl"]
    assert manifests.resolve(data_dir, "a/train.jsonl").is_file()
    for bad in ("../outside.jsonl", "a/wavs/0.wav", "missing.jsonl"):
        with pytest.raises(KeyError):
            manifests.resolve(data_dir, bad)


def test_read_manifest_resolves_manifest_relative_and_legacy_paths(data_dir):
    manifest = make_dataset(data_dir, "a")  # legacy: relative to data_dir's parent
    with manifest.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"audio": "wavs/0.wav", "text": "manifest-relative"}) + "\n")
        f.write("not json\n")
        f.write(json.dumps({"audio": "x.wav", "text": " "}) + "\n")

    entries, errors = manifests.read_manifest(manifest, manifests.legacy_root(data_dir))

    assert len(entries) == 5
    assert all(manifests.Path(e["audio"]).is_file() for e in entries)
    assert [e.split(":")[0] for e in errors] == ["Line 6", "Line 7"]


def test_inspect_reports_rates_durations_and_missing_files(data_dir):
    manifest = make_dataset(data_dir, "a")
    with manifest.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"audio": "gone.wav", "text": "missing"}) + "\n")
        f.write(json.dumps({"audio": "x.wav", "text": ""}) + "\n")

    report = manifests.inspect(manifest, manifests.legacy_root(data_dir))

    assert (report.samples, report.valid) == (5, 4)
    assert report.sample_rates == {16000: 2, 44100: 2}
    assert report.total_seconds == pytest.approx(4.0)
    assert len(report.errors) == 1 and "file not found" in report.errors[0]
    assert report.warnings == ["Skipped: Line 6: missing 'text'"]


def test_scan_skips_work_in_progress_folders(data_dir):
    make_dataset(data_dir, "a")
    (data_dir / ".import-b").mkdir()
    (data_dir / ".import-b" / "train.jsonl").write_text("")

    assert manifests.scan(data_dir) == ["a/train.jsonl"]


def test_read_rows_pages_filters_and_flags_asr(data_dir):
    manifest = data_dir / "speaker_a" / "train.jsonl"
    wavs = data_dir / "speaker_a" / "wavs"
    wavs.mkdir(parents=True)
    lines = []
    for i in range(5):
        sf.write(wavs / f"take_{i}.wav", np.zeros(16000 * (i + 1), dtype=np.float32), 16000)
        lines.append(json.dumps({"audio": f"wavs/take_{i}.wav", "text": f"line {i}" + (" hello" if i % 2 else "")}))
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (manifest.parent / manifests.ASR_FILE).write_text(json.dumps({"wavs/take_3.wav": {"language": "en"}}))

    total, rows = manifests.read_rows(manifest, offset=1, limit=2)
    assert total == 5 and [(r.index, r.file, r.seconds) for r in rows] == [
        (1, "wavs/take_1.wav", 2.0),
        (2, "wavs/take_2.wav", 3.0),
    ]

    total, rows = manifests.read_rows(manifest, query="HELLO")
    assert total == 2 and [(r.index, r.asr) for r in rows] == [(1, False), (3, True)]

    assert manifests.clip_audio(manifest, None, 4).name == "take_4.wav"
    with pytest.raises(KeyError):
        manifests.clip_audio(manifest, None, 5)


def test_split_holdout_is_reproducible_and_keeps_training_data():
    entries = [{"audio": str(i), "text": str(i)} for i in range(10)]

    train, val = manifests.split_holdout(entries, 2)

    assert len(train) == 8 and len(val) == 2
    assert manifests.split_holdout(entries, 2) == (train, val)
    assert manifests.split_holdout(entries[:1], 2) == (entries[:1], [])
