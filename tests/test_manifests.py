import json

import pytest

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


def test_split_holdout_is_reproducible_and_keeps_training_data():
    entries = [{"audio": str(i), "text": str(i)} for i in range(10)]

    train, val = manifests.split_holdout(entries, 2)

    assert len(train) == 8 and len(val) == 2
    assert manifests.split_holdout(entries, 2) == (train, val)
    assert manifests.split_holdout(entries[:1], 2) == (entries[:1], [])
