"""Reading audio without a system FFmpeg (soundfile, else PyAV)."""

import numpy as np
import pytest
import soundfile as sf
import torchaudio
from test_slicer import bursts, write_m4a

from voxcpm_studio import audio_io


def test_m4a_reference_becomes_wav(tmp_path):
    m4a = tmp_path / "ref.m4a"
    write_m4a(m4a, bursts([(0.2, 0.8)], 1.0, 24000)[None], 24000)

    wav = audio_io.to_wav(m4a)

    assert wav.suffix == ".wav" and not m4a.exists()
    assert sf.info(str(wav)).duration == pytest.approx(1.0, abs=0.05)
    plain = tmp_path / "plain.wav"
    sf.write(plain, np.zeros(100, dtype=np.float32), 16000)
    assert audio_io.to_wav(plain) == plain  # soundfile reads it: unchanged
    junk = tmp_path / "junk.wav"
    junk.write_bytes(b"not audio")
    assert audio_io.to_wav(junk) == junk  # left for the caller to report


def test_torchaudio_load_reads_through_audio_io(tmp_path, monkeypatch):
    monkeypatch.setattr(torchaudio, "load", torchaudio.load)  # restored after the test
    monkeypatch.setattr(torchaudio, "save", torchaudio.save)
    audio = bursts([(0.2, 0.8)], 1.0, 16000)
    sf.write(tmp_path / "a.wav", np.stack([audio, audio], axis=1), 16000)
    write_m4a(tmp_path / "a.m4a", audio[None], 16000)

    audio_io.patch_torchaudio()

    stereo, rate = torchaudio.load(str(tmp_path / "a.wav"))
    assert stereo.shape == (2, 16000) and rate == 16000
    part, _ = torchaudio.load(str(tmp_path / "a.wav"), frame_offset=100, num_frames=50)
    assert part.shape == (2, 50)
    decoded, rate = torchaudio.load(str(tmp_path / "a.m4a"))
    assert decoded.shape[0] == 1 and rate == 16000 and abs(decoded.shape[1] - 16000) < 2048

    torchaudio.save(str(tmp_path / "out.wav"), stereo, 16000)
    assert sf.info(str(tmp_path / "out.wav")).channels == 2


def test_damaged_m4a_reads_with_the_damage_silent(tmp_path):
    rate = 16000
    t = np.arange(8 * rate) / rate
    audio = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    m4a = tmp_path / "rec.m4a"
    write_m4a(m4a, audio[None], rate)
    clean, _ = audio_io.read(m4a)
    # Garbage over a stretch of the encoded audio in the middle of the file.
    data = bytearray(m4a.read_bytes())
    mid = len(data) // 2
    data[mid : mid + 600] = bytes((i * 37 + 11) % 256 for i in range(600))
    m4a.write_bytes(bytes(data))

    damaged, got_rate = audio_io.read(m4a)

    assert got_rate == rate and damaged.shape == clean.shape  # the full length, in place
    same = np.isclose(damaged[0], clean[0], atol=1e-4)
    assert same[: 2 * rate].all() and same[-2 * rate :].all()  # untouched before and after
    assert not same.all()  # the damage didn't decode to the original audio
    later, _ = audio_io.read(m4a, 6.0, 7.0)  # a range read past the damage is in place too
    assert np.allclose(later[0], clean[0, 6 * rate : 7 * rate], atol=1e-4)
