"""Reading audio without a system FFmpeg.

soundfile reads WAV / FLAC / OGG / MP3 with exact seeking; anything else (m4a,
video containers) is decoded with PyAV, whose wheels bundle FFmpeg. Since
torchaudio 2.9, ``torchaudio.load`` / ``save`` go through torchcodec, which needs
FFmpeg's shared libraries installed on the system; torchcodec is not installed
(pyproject.toml), and ``patch_torchaudio`` points both here for voxcpm's own
calls (the reference denoiser).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


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


RECOVER_JUMPS_S = (1.0, 5.0, 15.0, 30.0, 60.0, 120.0, 300.0, 600.0)  # tried after a stream stops early
SEEK_SLACK_S = 30.0  # a seek landing further before its target hit a broken index: give up on it
PREROLL_S = 0.1  # decode from a little before a seek target: AAC's first frame after a seek is off


def _frame_array(frame) -> np.ndarray:
    """A decoded frame as float32 (channels, samples), whatever its sample format."""
    data = frame.to_ndarray()
    if not frame.format.is_planar:  # interleaved: (1, samples * channels)
        data = data.reshape(-1, len(frame.layout.channels)).T
    if data.dtype.kind == "i":
        data = data.astype(np.float32) / float(np.iinfo(data.dtype).max)
    elif data.dtype.kind == "u":  # 8-bit audio is offset binary
        data = (data.astype(np.float32) - 128.0) / 128.0
    return data.astype(np.float32, copy=False)


def _read_av(path: Path | str, start: float, end: float | None) -> tuple[np.ndarray, int]:
    """Decode [start, end) with PyAV, placing each frame by its own timestamp.

    Damaged recordings decode anyway, as FFmpeg's tools do: a packet that fails
    to decode stays silent, and when the stream stops before its end (a broken
    index can hide the rest of a file from a straight read), reading resumes
    after a seek past the damage. Either way the audio after keeps its time.
    """
    import av

    name = Path(path).name
    try:
        with av.open(str(path)) as container:
            stream = container.streams.audio[0]
            rate = stream.codec_context.sample_rate
            tb = stream.time_base
            if stream.duration is not None and tb:
                duration = float(stream.duration * tb)
            else:
                duration = (container.duration or 0) / av.time_base or None
            limit = end if end is not None else duration
            pieces: list[tuple[int, np.ndarray]] = []  # (first sample, relative to start; samples)
            bad = 0  # packets that failed to decode
            reached = start  # end of the audio decoded so far, seconds

            def decode_from(at: float) -> bool:
                """Decode from a seek to `at`; True if it got audio past `reached`."""
                nonlocal bad, reached
                if at > 0:
                    container.seek(int(max(0.0, at - PREROLL_S) / tb), stream=stream, backward=True)
                progressed = False
                for packet in container.demux(stream):
                    try:
                        frames = packet.decode()
                    except av.error.InvalidDataError:
                        bad += 1
                        continue
                    for frame in frames:
                        t = float(frame.time) if frame.time is not None else reached
                        if t < at - SEEK_SLACK_S:
                            return progressed  # landed far behind (seen: at 5 s for 4980 s)
                        data = _frame_array(frame)
                        frame_end = t + data.shape[1] / rate
                        if frame_end <= start or frame_end <= reached - 0.5:
                            continue  # before the range, or a seek that landed behind
                        pieces.append((round((t - start) * rate), data))
                        progressed = progressed or frame_end > reached
                        reached = max(reached, frame_end)
                    if limit is not None and reached >= limit:
                        break
                return progressed

            decode_from(start)
            # The stream ended early: look further on for audio past the damage.
            if limit is not None:
                for jump in RECOVER_JUMPS_S:
                    if reached >= limit - 0.5:
                        break
                    broke, target = reached, min(reached + jump, limit - 0.5)
                    if decode_from(target):
                        logger.warning("%s: the audio broke off at %.1f s; resumed at %.1f s", name, broke, target)
    except (av.FFmpegError, IndexError, OSError) as e:
        raise ValueError(f"Unreadable audio: {name}") from e
    if bad:
        logger.warning("%s: %d damaged audio packets decoded as silence", name, bad)
    if not pieces:
        return np.zeros((1, 0), dtype=np.float32), rate
    channels = max(p.shape[0] for _, p in pieces)
    total = round((limit - start) * rate) if limit is not None else max(i + p.shape[1] for i, p in pieces)
    audio = np.zeros((channels, max(total, 0)), dtype=np.float32)
    for i, p in pieces:
        a, b = max(i, 0), min(i + p.shape[1], audio.shape[1])
        if b > a:
            audio[: p.shape[0], a:b] = p[:, a - i : b - i]
    return audio, rate


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
