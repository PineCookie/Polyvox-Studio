import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from voxcpm_studio.data import manifests

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


def five_clips(data_dir):
    manifest = data_dir / "speaker_a" / "train.jsonl"
    wavs = data_dir / "speaker_a" / "wavs"
    wavs.mkdir(parents=True)
    lines = []
    for i in range(5):
        sf.write(wavs / f"take_{i}.wav", np.zeros(16000 * (i + 1), dtype=np.float32), 16000)
        lines.append(json.dumps({"audio": f"wavs/take_{i}.wav", "text": f"line {i}" + (" hello" if i % 2 else "")}))
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def test_read_rows_pages_filters_and_flags_asr(data_dir):
    manifest = five_clips(data_dir)
    (manifest.parent / manifests.ASR_FILE).write_text(json.dumps({"wavs/take_3.wav": {"language": "en"}}))

    page = manifests.read_rows(manifest, offset=1, limit=2)
    assert page.total == 5 and page.checks is None
    assert [(r.index, r.file, r.seconds) for r in page.rows] == [
        (1, "wavs/take_1.wav", 2.0),
        (2, "wavs/take_2.wav", 3.0),
    ]

    page = manifests.read_rows(manifest, query="HELLO")
    assert page.total == 2 and [(r.index, r.asr) for r in page.rows] == [(1, False), (3, True)]

    assert manifests.clip_audio(manifest, None, 4).name == "take_4.wav"
    with pytest.raises(KeyError):
        manifests.clip_audio(manifest, None, 5)


def test_rows_carry_a_waveform_outline(data_dir):
    manifest = five_clips(data_dir)
    loud = np.zeros(16000, dtype=np.float32)
    loud[8000:12000] = 0.5  # the third quarter
    sf.write(data_dir / "speaker_a" / "wavs" / "take_0.wav", loud, 16000)

    peaks = manifests.read_rows(manifest, limit=1).rows[0].peaks

    assert len(peaks) == manifests.PEAK_BINS
    quarter = manifests.PEAK_BINS // 4
    assert max(peaks[: 2 * quarter]) == 0 and min(peaks[2 * quarter : 3 * quarter]) == pytest.approx(0.5, abs=0.01)
    (data_dir / "speaker_a" / "wavs" / "take_1.wav").unlink()
    assert manifests.read_rows(manifest, offset=1, limit=1).rows[0].peaks is None


def test_read_rows_shows_checks_for_the_text_they_were_made_for(data_dir):
    manifest = five_clips(data_dir)

    def check(text, score, flags):
        return {"text": manifests.text_key(text), "score": score, "flags": flags, "start": 0.2, "end": 0.8}

    clips = {
        "wavs/take_0.wav": check("line 0", 0.9, []),
        "wavs/take_1.wav": check("line 1 hello", 0.2, ["mismatch"]),
        "wavs/take_2.wav": check("an older transcript", 0.1, ["mismatch"]),  # stale: text changed since
        "wavs/take_3.wav": check("line 3 hello", 0.7, ["cut_edge", "long_silence"]),
    }
    (manifest.parent / manifests.ALIGN_FILE).write_text(json.dumps({"aligner": "ctc", "clips": clips}))

    page = manifests.read_rows(manifest)
    assert page.rows[0].speech == (0.2, 0.8) and page.rows[2].speech is None
    assert [(r.score, r.flags) for r in page.rows] == [
        (0.9, []),
        (0.2, ["mismatch"]),
        (None, []),
        (0.7, ["cut_edge", "long_silence"]),
        (None, []),
    ]
    assert page.checks == {
        "aligner": "ctc",
        "checked": 3,
        "flagged": 2,
        "flags": {"mismatch": 1, "cut_edge": 1, "long_silence": 1},
    }

    flagged = manifests.read_rows(manifest, flag="any")
    assert flagged.total == 2 and [r.index for r in flagged.rows] == [1, 3]
    worst = manifests.read_rows(manifest, worst_first=True)
    assert [r.index for r in worst.rows] == [1, 3, 0, 2, 4]  # unchecked last


def test_split_holdout_is_reproducible_and_keeps_training_data():
    entries = [{"audio": str(i), "text": str(i)} for i in range(10)]

    train, val = manifests.split_holdout(entries, 2)

    assert len(train) == 8 and len(val) == 2
    assert manifests.split_holdout(entries, 2) == (train, val)
    assert manifests.split_holdout(entries[:1], 2) == (entries[:1], [])


def test_backslash_paths_from_windows_files_work_everywhere():
    assert manifests.local_path("lora\\legacy\\logs").as_posix() == "lora/legacy/logs"
    assert manifests.local_path("data/voice/wavs/a.wav").as_posix() == "data/voice/wavs/a.wav"
    assert manifests.local_path("X:\\old_pc\\a.wav") == Path("X:\\old_pc\\a.wav")  # a drive path stays as written
