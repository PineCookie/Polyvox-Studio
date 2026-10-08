"""Importer tests on pseudo data: invented speakers, paths and sentences."""

import json

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from voxcpm_studio import importer, manifests
from voxcpm_studio.api import create_app

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
    assert (without.kind, without.rows, without.found, without.missing) == ("gsv", 5, 0, 4)

    p = importer.preview_import(f'"{lst}"', str(clips[0].parent), data_dir)  # quoted, as Explorer copies
    assert (p.found, p.missing, p.skipped_empty, p.duplicates) == (2, 1, 1, 1)
    assert p.missing_examples == [rf"{STALE_WINDOWS}\take_9.wav"]
    assert p.total_seconds == pytest.approx(2.0) and p.sample_rates == {32000: 2}
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

    assert (p.kind, p.rows, p.found, p.skipped_empty) == ("folder", 3, 2, 1)


def test_import_copies_audio_and_writes_relative_manifest(tmp_path, data_dir, clips):
    same_name_elsewhere = wav(tmp_path / "other" / "take_0.wav")
    lst = write_list(
        tmp_path / "speaker_a.list",
        [f"{c}|clips|EN|line {i}" for i, c in enumerate(clips)] + [f"{same_name_elsewhere}|clips|EN|same file name"],
    )

    dataset_id, p = importer.import_dataset(str(lst), None, "speaker_a", data_dir)

    assert dataset_id == "speaker_a/train.jsonl" and p.found == 4
    manifest = data_dir / "speaker_a" / "train.jsonl"
    entries = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    assert [e["audio"] for e in entries] == ["wavs/take_0.wav", "wavs/take_1.wav", "wavs/take_2.wav", "wavs/take_0_2.wav"]
    # Self-contained: resolvable without any legacy fallback.
    resolved, errors = manifests.read_manifest(manifest)
    assert not errors and all(manifests.Path(e["audio"]).is_file() for e in resolved)
    assert not list(data_dir.glob(".import-*"))


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
    with pytest.raises(ValueError, match="No audio"):
        importer.import_dataset(str(missing), None, "no_audio", data_dir)
    with pytest.raises(FileNotFoundError):
        importer.preview_import(str(tmp_path / "nope.list"), None, data_dir)


def test_import_endpoints(service, training, data_dir, tmp_path, clips):
    lst = write_list(tmp_path / "speaker_a.list", [f"{c}|clips|EN|line {i}" for i, c in enumerate(clips)])
    client = TestClient(create_app(service.settings, service, training))

    preview = client.post("/api/datasets/import/preview", json={"source": str(lst)}).json()
    assert (preview["found"], preview["suggested_name"]) == (3, "speaker_a")

    resp = client.post("/api/datasets/import", json={"source": str(lst), "name": "speaker_a"})
    assert resp.status_code == 200 and resp.json()["id"] == "speaker_a/train.jsonl"
    assert "speaker_a/train.jsonl" in client.get("/api/datasets").json()
    assert client.post("/api/datasets/import", json={"source": str(lst), "name": "speaker_a"}).status_code == 409
    assert client.post("/api/datasets/import/preview", json={"source": str(tmp_path / "x.mp4")}).status_code == 422
