import json
import math

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from test_importer import wait_job, write_list

from voxcpm_studio import importer, loudness
from voxcpm_studio.api import create_app

RATE = 44100


def sine(amplitude, seconds=3.0, rate=RATE, freq=997.0):
    t = np.arange(int(seconds * rate)) / rate
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_integrated_loudness_of_a_sine():
    # BS.1770: a full-scale 997 Hz sine on one channel reads -3.01 LUFS.
    assert loudness.integrated(sine(1.0), RATE) == pytest.approx(-3.01, abs=0.05)
    assert loudness.integrated(sine(0.1), RATE) == pytest.approx(-23.01, abs=0.05)
    stereo = np.stack([sine(0.1), sine(0.1)])  # (channels, frames): both channels count
    assert loudness.integrated(stereo, RATE) == pytest.approx(-20.0, abs=0.05)


def test_unmeasurable_audio_keeps_its_level():
    assert loudness.integrated(np.zeros(RATE, dtype=np.float32), RATE) == -math.inf
    short = sine(0.1, seconds=0.2)  # shorter than one 400 ms block
    assert loudness.integrated(short, RATE) == -math.inf
    assert loudness.gain_db(short, RATE) == 0.0


def test_normalize_reaches_the_target_unless_the_peak_would_pass_the_ceiling():
    out = loudness.normalize(sine(0.02), RATE, -20.0)
    assert loudness.integrated(out, RATE) == pytest.approx(-20.0, abs=0.05)
    spiky = sine(0.01)
    spiky[RATE] = 0.5  # a click far above the tone
    out = loudness.normalize(spiky, RATE, -20.0)
    assert loudness.peak_db(out) == pytest.approx(loudness.PEAK_CEILING_DB, abs=0.01)
    assert loudness.integrated(out, RATE) < -21


def test_normalize_dataset_in_place_keeps_originals_and_checks(tmp_path):
    root = tmp_path / "voice"
    (root / "wavs").mkdir(parents=True)
    sf.write(root / "wavs" / "quiet.wav", sine(0.01), RATE, subtype="PCM_16")
    sf.write(root / "wavs" / "there.wav", sine(0.1 * 10 ** (3 / 20)), RATE, subtype="PCM_16")  # ~-20 LUFS
    sf.write(root / "wavs" / "stereo.flac", np.stack([sine(0.3), sine(0.3)]).T, RATE)
    lines = [{"audio": f"wavs/{n}", "text": "示例"} for n in ("quiet.wav", "there.wav", "stereo.flac")]
    manifest = root / "train.jsonl"
    manifest.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in lines), encoding="utf-8")
    (root / "align.json").write_text('{"aligner": "qwen3", "clips": {}}', encoding="utf-8")

    result = loudness.normalize_dataset(manifest, None, -20.0)

    assert (result["normalized"], result["unchanged"], result["skipped"]) == (2, 1, 0)
    assert result["after"][1] == pytest.approx(-20.0, abs=0.2)
    for name in ("quiet.wav", "stereo.flac"):
        audio, rate = sf.read(root / "wavs" / name, always_2d=True)
        assert loudness.integrated(audio.T, rate) == pytest.approx(-20.0, abs=0.1)
        assert (root / "originals" / "wavs" / name).exists()
    assert sf.info(root / "wavs" / "stereo.flac").channels == 2  # own format and channels
    assert not (root / "originals" / "wavs" / "there.wav").exists()
    assert (root / "align.json").read_text(encoding="utf-8") == '{"aligner": "qwen3", "clips": {}}'
    # Again: nothing left to do, and the first originals stay.
    original = (root / "originals" / "wavs" / "quiet.wav").read_bytes()
    assert loudness.normalize_dataset(manifest, None, -20.0)["normalized"] == 0
    assert (root / "originals" / "wavs" / "quiet.wav").read_bytes() == original


def test_import_normalizes_the_copies(tmp_path, data_dir):
    src = tmp_path / "recordings"
    src.mkdir()
    sf.write(src / "loud.flac", sine(0.5), RATE)
    (src / "loud.lab").write_text("示例", encoding="utf-8")

    importer.import_dataset(str(src), None, "normalized", data_dir, normalize=-23.0)

    audio, rate = sf.read(data_dir / "normalized" / "wavs" / "loud.wav")
    assert loudness.integrated(audio, rate) == pytest.approx(-23.0, abs=0.1)
    with pytest.raises(ValueError, match="copying"):
        importer.import_dataset(str(src), None, "linked", data_dir, copy=False, normalize=-23.0)


def test_normalize_endpoints(service, training, data_dir, tmp_path):
    clip = tmp_path / "a.wav"
    sf.write(clip, sine(0.5), RATE)
    lst = write_list(tmp_path / "speaker_a.list", [f"{clip}|speaker_a|ZH|示例"])
    client = TestClient(create_app(service.settings, service, training))

    body = {"source": str(lst), "name": "a", "normalize_lufs": -20}
    assert wait_job(client, client.post("/api/datasets/import", json=body).json()["id"])["state"] == "done"
    audio, rate = sf.read(data_dir / "a" / "wavs" / "a.wav")
    assert loudness.integrated(audio, rate) == pytest.approx(-20.0, abs=0.1)

    job = wait_job(
        client, client.post("/api/datasets/normalize", json={"id": "a/train.jsonl", "target_lufs": -26}).json()["id"]
    )
    assert (job["state"], job["result"]["normalized"]) == ("done", 1), job
    audio, rate = sf.read(data_dir / "a" / "wavs" / "a.wav")
    assert loudness.integrated(audio, rate) == pytest.approx(-26.0, abs=0.1)
    assert client.post("/api/datasets/normalize", json={"id": "nope/train.jsonl"}).status_code == 404
    assert client.post("/api/datasets/normalize", json={"id": "a/train.jsonl", "target_lufs": 0}).status_code == 422
