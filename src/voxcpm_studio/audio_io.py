"""Reading audio without a system FFmpeg.

soundfile reads WAV / FLAC / OGG / MP3 with exact seeking; anything else (m4a,
video containers) is decoded with PyAV, whose wheels bundle FFmpeg. Since
torchaudio 2.9, ``torchaudio.load`` / ``save`` go through torchcodec, which needs
FFmpeg's shared libraries installed on the system; torchcodec is not installed
(pyproject.toml), and ``patch_torchaudio`` points both here for voxcpm's own
calls (the reference denoiser).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def info(path: Path | str) -> tuple[float, int]:
    """(duration in seconds, sample rate)."""
    try:
        i = sf.info(str(path))
        return i.duration, i.samplerate
    except Exception:
        pass
    import av

    try:
        with av.open(str(path)) as container:
            if not container.streams.audio:
                raise ValueError("no audio stream")
            stream = container.streams.audio[0]
            rate = stream.codec_context.sample_rate
            if stream.duration is not None and stream.time_base:
                duration = float(stream.duration * stream.time_base)
            else:
                duration = (container.duration or 0) / av.time_base
    except (av.FFmpegError, ValueError, OSError) as e:
        raise ValueError(f"Unreadable audio: {Path(path).name}") from e
    if not duration or not rate:
        raise ValueError(f"Unreadable audio: {Path(path).name}")
    return duration, rate


def read(path: Path | str, start: float = 0.0, end: float | None = None) -> tuple[np.ndarray, int]:
    """(float32 samples as (channels, frames), sample rate) of [start, end)
    seconds (end None: to the end), at the file's own rate and channels."""
    try:
        with sf.SoundFile(str(path)) as f:
            rate = f.samplerate
            f.seek(int(start * rate))
            frames = -1 if end is None else max(0, int((end - start) * rate))
            return f.read(frames, dtype="float32", always_2d=True).T, rate
    except Exception:  # not a format libsndfile reads
        pass
    return _read_av(path, start, end)


def _read_av(path: Path | str, start: float, end: float | None) -> tuple[np.ndarray, int]:
    import av

    try:
        with av.open(str(path)) as container:
            stream = container.streams.audio[0]
            rate = stream.codec_context.sample_rate
            if start > 0:  # lands on a frame at or before `start`; trimmed below
                container.seek(int(start / stream.time_base), stream=stream, backward=True)
            resampler = av.AudioResampler(format="fltp")  # float, planar; own layout and rate
            chunks: list[np.ndarray] = []
            first: float | None = None  # time of the first decoded sample
            for frame in container.decode(stream):
                if first is None:
                    first = float(frame.time) if frame.time is not None else 0.0
                for out in resampler.resample(frame):
                    chunks.append(out.to_ndarray())
                if end is not None and frame.time is not None and frame.time >= end:
                    break
            for out in resampler.resample(None):
                chunks.append(out.to_ndarray())
    except (av.FFmpegError, IndexError, OSError) as e:
        raise ValueError(f"Unreadable audio: {Path(path).name}") from e
    if not chunks:
        return np.zeros((1, 0), dtype=np.float32), rate
    audio = np.concatenate(chunks, axis=1).astype(np.float32)
    skip = max(0, round((start - (first or 0.0)) * rate))
    stop = None if end is None else skip + max(0, round((end - start) * rate))
    return audio[:, skip:stop], rate


def resample(audio: np.ndarray, rate: int, target: int) -> np.ndarray:
    """(channels, frames) float32 at ``rate`` -> at ``target``."""
    if rate == target:
        return audio
    import torch
    import torchaudio.functional as F

    return F.resample(torch.from_numpy(np.ascontiguousarray(audio)), rate, target).numpy()


def load(uri, frame_offset: int = 0, num_frames: int = -1, normalize: bool = True, channels_first: bool = True, **_):
    """``torchaudio.load`` with soundfile / PyAV: (float32 tensor, sample rate)."""
    import torch

    audio, rate = read(uri)
    audio = audio[:, frame_offset : None if num_frames < 0 else frame_offset + num_frames]
    tensor = torch.from_numpy(np.ascontiguousarray(audio))
    return (tensor if channels_first else tensor.T), rate


def save(uri, src, sample_rate: int, channels_first: bool = True, **_) -> None:
    """``torchaudio.save`` with soundfile (the format follows the file name, e.g. WAV)."""
    audio = src.detach().cpu().float().numpy()
    sf.write(str(uri), audio.T if channels_first else audio, sample_rate)


def patch_torchaudio() -> None:
    """Make ``torchaudio.load`` / ``save`` go through this module (no system FFmpeg
    needed). voxcpm and its reference denoiser call only these two."""
    import torchaudio

    torchaudio.load = load
    torchaudio.save = save


def to_wav(path: Path) -> Path:
    """``path`` if soundfile reads it, else a WAV copy next to it (the original
    is removed): voxcpm's reference loading only reads such files directly.
    A file that can't be decoded at all is returned as is, for voxcpm to report."""
    try:
        sf.info(str(path))
        return path
    except Exception:
        pass
    try:
        audio, rate = read(path)
    except ValueError:
        return path
    wav = path.with_suffix(".wav") if path.suffix.lower() != ".wav" else path.with_name(f"{path.stem}_pcm.wav")
    sf.write(str(wav), audio.T, rate, subtype="PCM_16")
    path.unlink(missing_ok=True)
    return wav
