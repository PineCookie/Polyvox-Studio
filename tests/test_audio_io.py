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
