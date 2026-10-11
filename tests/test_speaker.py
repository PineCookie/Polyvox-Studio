import json

import numpy as np
import soundfile as sf

from polyvox import speaker
from polyvox.config import Settings

RATE = 16_000


def tone(path, hz, seconds=1.0, seed=0):
    t = np.arange(int(RATE * seconds)) / RATE
    noise = np.random.default_rng(seed).normal(0, 0.01, t.size)
    sf.write(path, (0.3 * np.sin(2 * np.pi * hz * t) + noise).astype(np.float32), RATE)
    return path


class PitchEmbedder:
    """A stand-in for CAM++: the spectrum in coarse bands, so equal tones match."""

    def __init__(self):
        self.calls = 0

    def __call__(self, wav):
        self.calls += 1
        spectrum = np.abs(np.fft.rfft(wav, n=RATE))[:2000]
        return spectrum.reshape(40, 50).sum(axis=1)


def make_run(tmp_path, voice_hz=220):
    data = tmp_path / "data" / "wavs"
    data.mkdir(parents=True)
    lines = [json.dumps({"audio": str(tone(data / f"{i}.wav", voice_hz, seed=i)), "text": "示例"}) for i in range(12)]
    manifest = tmp_path / "data" / "train.jsonl"
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    run = tmp_path / "run"
    (run / "samples").mkdir(parents=True)
    return run, manifest


def service(embedder, tmp_path):
    return speaker.SpeakerService(Settings(models_dir=tmp_path / "models"), loader=lambda model: embedder)


def test_samples_in_the_voice_score_higher(tmp_path):
    run, manifest = make_run(tmp_path)
    tone(run / "samples" / "step_0000010_0.wav", 220, seed=99)  # the voice
    tone(run / "samples" / "step_0000010_1.wav", 660, seed=98)  # someone else
    tone(run / "samples" / "reference_0.wav", 220, seed=97)

    result = service(PitchEmbedder(), tmp_path).score_run(run, manifest)

    scores = result["scores"]
    assert set(scores) == {"step_0000010_0.wav", "step_0000010_1.wav", "reference_0.wav"}
    assert scores["step_0000010_0.wav"] > 0.9 > 0.5 > scores["step_0000010_1.wav"]
    assert result["target"] > 0.9  # other real clips of the set


def test_scores_are_cached_until_files_change(tmp_path):
    run, manifest = make_run(tmp_path)
    tone(run / "samples" / "step_0000010_0.wav", 220)
    embedder = PitchEmbedder()
    svc = service(embedder, tmp_path)
    svc.score_run(run, manifest)
    first = embedder.calls

    svc.score_run(run, manifest)
    assert embedder.calls == first  # voice and sample from the cache

    tone(run / "samples" / "step_0000020_0.wav", 660)
    (run / "samples" / "step_0000010_0.wav").unlink()
    result = service(embedder, tmp_path).score_run(run, manifest)  # a new service reads the file cache
    assert embedder.calls == first + 1
    assert set(result["scores"]) == {"step_0000020_0.wav"}


def test_run_without_samples(tmp_path):
    run, manifest = make_run(tmp_path)
    assert service(PitchEmbedder(), tmp_path).score_run(run, manifest)["scores"] == {}
