"""Slice long recordings into training clips.

Recordings are read in windows of about WINDOW_S (ending at a pause), so hours
of compressed audio (m4a, mp3, ...) never sit in memory at once.

With a transcript guide (ASR + forced aligner), clips are cut between words:

1. Voice activity detection (Silero VAD) finds the speech; it is grouped into
   chunks of up to CHUNK_MAX_S at the longest pauses (what ASR handles well).
2. Each chunk is transcribed (with punctuation) and its words aligned.
3. Cut points are chosen between words by dynamic programming: sentence ends
   are rewarded most, then clause punctuation, then long pauses; clips far from
   the target length are penalized and never exceed the maximum. A cut sits at
   the quietest moment between the two words.
4. Each clip's text is its exact slice of the chunk's transcript.

Without a guide, cuts follow pauses alone (VAD segments grouped the same way),
and the clips get no text.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from .. import timing
from ..data import audio_io

VAD_RATE = 16_000
VAD_MIN_SILENCE_S = 0.1
ENERGY_FRAME_S = 0.02
CUT_REWARD_CAP_S = 1.0  # pauses beyond this are all "clearly a boundary"
CUT_WEIGHT = 2.0
WINDOW_S = 600.0  # audio decoded at a time
WINDOW_SLACK_S = 60.0  # a window ends at the longest pause in its last minute
CHUNK_MAX_S = 28.0  # per ASR call (Qwen3-ASR's features cover 30 s)
# Silence kept around an ASR chunk (at most halfway to the next). Generous:
# Silero can mark a phrase's onset ~0.5 s late, and a clip can't start before
# its chunk, so a short pad cut clips into their first word.
CHUNK_PAD_S = 1.0
SENTENCE_END = set("。！？!?…")
WORD_CUT_COST = 1.0  # a clause mark alone does not justify a short clip
MIN_PAUSE_S = 0.08  # quieter stretch between words that counts as a pause
EDGE_QUIET_DB = 25.0  # a clip edge must be this far below the speech (else moved inward)
NO_PAUSE_PENALTY = 3.0  # cutting where the sound goes on: only if a clip would be too long
CHUNK_CUT_COST = 3.0  # ASR chunks break only at long pauses (more context)
CLAUSE_END = set("，,、；;：:")

# 16 kHz mono audio -> speech segments as (start, end) in seconds.
Vad = Callable[[np.ndarray], list[tuple[float, float]]]


@dataclass(frozen=True)
class SliceOptions:
    min_seconds: float = 2.0  # shorter clips are dropped
    max_seconds: float = 15.0
    target_seconds: float = 8.0
    max_inner_gap: float = 1.0  # a longer pause always ends a clip
    pad_seconds: float = 0.15


@dataclass
class Guide:
    """What the transcript-guided slicer needs, on 16 kHz mono chunks."""

    # chunks -> (text, language) each
    transcribe: Callable[[list[np.ndarray]], list[tuple[str, str | None]]]
    # (chunks, texts) -> aligner.Alignment each
    align: Callable[[list[np.ndarray], list[str]], list]


@dataclass
class Slice:
    path: Path
    text: str  # "" when sliced without a guide
    language: str | None = None
    start: float = 0.0  # seconds into the recording
    end: float = 0.0


_vad_model = None
_vad_lock = threading.Lock()
_vad_models: dict[str, object] = {}  # device -> Silero, for long audio (silero_vad_long)
VAD_WINDOW = 512  # samples per Silero step at 16 kHz
VAD_BATCH = 64
# Long audio is run as pieces side by side, each after VAD_CONTEXT_S of the audio
# before it: Silero's state takes ~40-60 s to forget where it started, so with
# less context a piece's probabilities differ from one continuous pass.
VAD_PIECE_S = 60.0
VAD_CONTEXT_S = 60.0
vad_device: Callable[[], str] = lambda: "cpu"  # noqa: E731  (the API points it at the ASR device)


def silero_vad(wav16k: np.ndarray) -> list[tuple[float, float]]:
    """Speech segments with Silero VAD (bundled model). Audio longer than a
    couple of minutes goes through silero_vad_long."""
    global _vad_model
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad

    if len(wav16k) > (VAD_PIECE_S + VAD_CONTEXT_S) * VAD_RATE:
        return silero_vad_long(wav16k)
    with _vad_lock:  # the model is stateful
        if _vad_model is None:
            _vad_model = load_silero_vad()
        stamps = get_speech_timestamps(
            torch.from_numpy(np.ascontiguousarray(wav16k)),
            _vad_model,
            sampling_rate=VAD_RATE,
            min_silence_duration_ms=int(VAD_MIN_SILENCE_S * 1000),
            speech_pad_ms=0,
        )
    return [(s["start"] / VAD_RATE, s["end"] / VAD_RATE) for s in stamps]


def silero_vad_long(wav16k: np.ndarray) -> list[tuple[float, float]]:
    """``silero_vad`` for long audio: VAD_PIECE_S pieces in one batch (each
    warmed up on the VAD_CONTEXT_S before it), on ``vad_device()``. Per step
    Silero makes one small model call, so a 10 min window was ~19k calls in a
    row; this makes it ~3.7k, nearly the same segments (a few boundaries move by
    a step), ~3.5x faster on a GPU."""
    import torch
    from silero_vad import load_silero_vad
    from silero_vad.utils_vad import get_speech_timestamps_from_probs

    device = vad_device()
    steps = -(-len(wav16k) // VAD_WINDOW)
    piece, context = int(VAD_PIECE_S * VAD_RATE) // VAD_WINDOW, int(VAD_CONTEXT_S * VAD_RATE) // VAD_WINDOW
    span = piece + context
    padded = torch.zeros((context + steps) * VAD_WINDOW)  # silence before the start: the first piece's context
    padded[context * VAD_WINDOW : context * VAD_WINDOW + len(wav16k)] = torch.from_numpy(np.ascontiguousarray(wav16k))
    starts = range(0, steps, piece)
    batch = torch.zeros(len(starts), span * VAD_WINDOW)
    for row, s in enumerate(starts):  # windows [s - context, s + piece) of the audio
        part = padded[s * VAD_WINDOW : (s + span) * VAD_WINDOW]
        batch[row, : len(part)] = part
    with _vad_lock, torch.no_grad():
        model = _vad_models.get(device)
        if model is None:
            model = _vad_models[device] = load_silero_vad().to(device)
        batch = batch.to(device)
        model.reset_states()
        probs = torch.stack(
            [model(batch[:, k * VAD_WINDOW : (k + 1) * VAD_WINDOW], VAD_RATE)[:, 0] for k in range(span)], dim=1
        )
    stamps = get_speech_timestamps_from_probs(
        probs[:, context:].reshape(-1)[:steps].tolist(),
        sampling_rate=VAD_RATE,
        min_silence_duration_ms=int(VAD_MIN_SILENCE_S * 1000),
        speech_pad_ms=0,
        audio_length_samples=len(wav16k),
    )
    return [(s["start"] / VAD_RATE, s["end"] / VAD_RATE) for s in stamps]


def silero_vad_many(wavs: list[np.ndarray]) -> list[list[tuple[float, float]]]:
    """``silero_vad`` for many clips at once: the clips' windows go through the
    model as one batch per step (each keeps its own state), then Silero's own
    rules turn each clip's probabilities into segments. Per clip, Silero steps
    through 32 ms windows one model call at a time, which made VAD the slowest
    part of a dataset check."""
    global _vad_model
    import torch
    from silero_vad import load_silero_vad
    from silero_vad.utils_vad import get_speech_timestamps_from_probs

    out: list[list[tuple[float, float]] | None] = [None] * len(wavs)
    order = sorted(range(len(wavs)), key=lambda i: len(wavs[i]))  # similar lengths per batch
    with _vad_lock, torch.no_grad():
        if _vad_model is None:
            _vad_model = load_silero_vad()
        for b in range(0, len(order), VAD_BATCH):
            idx = order[b : b + VAD_BATCH]
            steps = max(1, -(-max(len(wavs[i]) for i in idx) // VAD_WINDOW))
            batch = torch.zeros(len(idx), steps * VAD_WINDOW)
            for row, i in enumerate(idx):
                batch[row, : len(wavs[i])] = torch.from_numpy(np.ascontiguousarray(wavs[i]))
            _vad_model.reset_states()
            probs = torch.stack(
                [_vad_model(batch[:, s * VAD_WINDOW : (s + 1) * VAD_WINDOW], VAD_RATE)[:, 0] for s in range(steps)],
                dim=1,
            ).tolist()
            for row, i in enumerate(idx):
                n = max(1, -(-len(wavs[i]) // VAD_WINDOW))
                stamps = get_speech_timestamps_from_probs(
                    probs[row][:n],
                    sampling_rate=VAD_RATE,
                    min_silence_duration_ms=int(VAD_MIN_SILENCE_S * 1000),
                    speech_pad_ms=0,
                    audio_length_samples=len(wavs[i]),
                )
                out[i] = [(s["start"] / VAD_RATE, s["end"] / VAD_RATE) for s in stamps]
    return out  # type: ignore[return-value]


# Reading


class Source:
    """An audio file read by time range, mono, at its own or a given rate.

    Read through audio_io: soundfile (WAV / FLAC / OGG / MP3, exact seeking),
    else PyAV (m4a, video containers).
    """

    def __init__(self, path: Path):
        self.path = path
        self.duration, self.rate = audio_io.info(path)

    def read(self, start: float, end: float, rate: int | None = None) -> np.ndarray:
        """Mono float32 samples of [start, end) seconds, at ``rate`` (default: the file's)."""
        return self.read_channels(start, end, rate, 1)[0]

    def read_channels(self, start: float, end: float, rate: int | None = None, channels: int = 1) -> np.ndarray:
        """(channels, samples) float32 of [start, end) seconds: ``channels`` = 1
        downmixes, 2 gives stereo (a mono file is duplicated)."""
        start, end = max(0.0, start), min(self.duration, end)
        if end <= start:
            return np.zeros((channels, 0), dtype=np.float32)
        # The file's own channels, mixed here: FFmpeg's downmix to mono sums the
        # channels at -3 dB each, i.e. +3 dB for ordinary centered audio, which
        # clipped loud moments once written as PCM16.
        with timing.stage("decoding"):
            audio, file_rate = audio_io.read(self.path, start, end)
            audio = audio.mean(axis=0, keepdims=True) if channels == 1 else audio[:channels]
            if audio.shape[0] != channels:
                audio = np.repeat(audio[:1], channels, axis=0)
            if rate:
                audio = audio_io.resample(audio, file_rate, rate)
            return audio.astype(np.float32)


def probe(path: Path) -> tuple[float, int]:
    """(duration in seconds, sample rate) of any readable audio file."""
    src = Source(path)
    return src.duration, src.rate


# Planning cuts from pauses


def split_long(
    segments: list[tuple[float, float]], wav16k: np.ndarray, max_seconds: float
) -> list[tuple[float, float]]:
    """Split segments longer than ``max_seconds`` at their quietest frame
    (searched in the middle half, so both parts stay reasonably long)."""
    frame = int(ENERGY_FRAME_S * VAD_RATE)
    out: list[tuple[float, float]] = []
    todo = list(segments)
    while todo:
        start, end = todo.pop(0)
        if end - start <= max_seconds:
            out.append((start, end))
            continue
        lo = int((start + (end - start) * 0.25) * VAD_RATE)
        hi = int((start + (end - start) * 0.75) * VAD_RATE)
        window = wav16k[lo:hi]
        n = max(1, len(window) // frame)
        rms = np.sqrt(np.mean(window[: n * frame].reshape(n, frame) ** 2, axis=1)) if len(window) >= frame else [0]
        cut = (lo + int(np.argmin(rms)) * frame + frame // 2) / VAD_RATE
        todo[:0] = [(start, cut), (cut, end)]
    return sorted(out)


def plan_clips(
    segments: list[tuple[float, float]], opts: SliceOptions, cut_cost: float = 0.0
) -> list[tuple[float, float]]:
    """Group consecutive speech segments into clips (start, end) in seconds;
    `cut_cost` makes every cut cost that much (fewer, longer pieces)."""
    n = len(segments)
    if n == 0:
        return []
    gaps = [segments[k + 1][0] - segments[k][1] for k in range(n - 1)]
    inf = float("inf")
    best = [inf] * (n + 1)  # best[j + 1]: cost of segments[0..j]
    back = [0] * (n + 1)
    best[0] = 0.0
    for j in range(n):
        for i in range(j, -1, -1):
            if i < j and gaps[i] > opts.max_inner_gap:
                break  # this pause must end a clip
            dur = segments[j][1] - segments[i][0]
            if dur > opts.max_seconds and i < j:
                break
            cost = ((dur - opts.target_seconds) / opts.target_seconds) ** 2
            if dur < opts.min_seconds:
                cost += 10.0 * (opts.min_seconds - dur)
            if j < n - 1:
                cost += cut_cost - CUT_WEIGHT * min(gaps[j], CUT_REWARD_CAP_S)
            if best[i] + cost < best[j + 1]:
                best[j + 1], back[j + 1] = best[i] + cost, i
    clips = []
    j = n
    while j > 0:
        i = back[j]
        clips.append((segments[i][0], segments[j - 1][1]))
        j = i
    return clips[::-1]


def pad_clips(clips: list[tuple[float, float]], pad: float, duration: float) -> list[tuple[float, float]]:
    """Extend clips into the silence around them, at most halfway to a neighbor."""
    out = []
    for k, (start, end) in enumerate(clips):
        lo = (clips[k - 1][1] + start) / 2 if k > 0 else 0.0
        hi = (end + clips[k + 1][0]) / 2 if k + 1 < len(clips) else duration
        out.append((max(start - pad, lo), min(end + pad, hi)))
    return out


def find_clips(
    wav16k: np.ndarray, opts: SliceOptions, vad: Vad = silero_vad, segments: list[tuple[float, float]] | None = None
) -> list[tuple[float, float]]:
    duration = len(wav16k) / VAD_RATE
    segments = split_long(vad(wav16k) if segments is None else segments, wav16k, opts.max_seconds)
    clips = pad_clips(plan_clips(segments, opts), opts.pad_seconds, duration)
    return [(s, e) for s, e in clips if e - s >= opts.min_seconds]


# Planning cuts from words


def _break_score(text: str, pause: float) -> float:
    """How good a place to end a clip the stretch between two words is: by the
    pause the audio really has there (measured on the waveform: ASR often puts
    a full stop where the speaker never paused) and the punctuation of the
    transcript, which only counts when there is a pause."""
    if pause < MIN_PAUSE_S:
        return -NO_PAUSE_PENALTY
    stripped = text.strip()
    punct = 1.5 if any(c in SENTENCE_END for c in stripped) else 0.5 if any(c in CLAUSE_END for c in stripped) else 0.0
    return punct + 1.5 * min(pause, CUT_REWARD_CAP_S)


def plan_word_clips(
    times: list[tuple[float, float]], breaks: list[float], opts: SliceOptions, pauses: list[float] | None = None
) -> list[tuple[int, int]]:
    """Group words (start, end) into clips [(first, last)], cutting after word k
    rewarded by ``breaks[k]``. Pads count toward the length."""
    n = len(times)
    if n == 0:
        return []
    pad2 = 2 * opts.pad_seconds
    if pauses is None:
        pauses = [times[k + 1][0] - times[k][1] for k in range(n - 1)]
    inf = float("inf")
    best = [inf] * (n + 1)
    back = [0] * (n + 1)
    best[0] = 0.0
    for j in range(n):
        for i in range(j, -1, -1):
            if i < j and pauses[i] > opts.max_inner_gap:
                break  # this pause must end a clip
            dur = times[j][1] - times[i][0] + pad2
            if dur > opts.max_seconds and i < j:
                break
            cost = ((dur - opts.target_seconds) / opts.target_seconds) ** 2
            if dur < opts.min_seconds:
                cost += 10.0 * (opts.min_seconds - dur)
            if j < n - 1:
                cost += WORD_CUT_COST - breaks[j]
            if best[i] + cost < best[j + 1]:
                best[j + 1], back[j + 1] = best[i] + cost, i
    clips = []
    j = n
    while j > 0:
        clips.append((back[j], j - 1))
        j = back[j]
    return clips[::-1]


def _quietest(wav16k: np.ndarray, lo: float, hi: float) -> float:
    """The quietest 10 ms in [lo, hi] seconds (the middle when too short)."""
    frame = VAD_RATE // 100
    b = min(int(hi * VAD_RATE), len(wav16k))
    a = min(int(lo * VAD_RATE), b)
    if b - a < 2 * frame:
        return (a + b) / 2 / VAD_RATE
    n = (b - a) // frame
    rms = np.sqrt(np.mean(wav16k[a : a + n * frame].reshape(n, frame) ** 2, axis=1))
    # Of the (near-)quietest frames, the one nearest the middle: in digital
    # silence every frame ties, and the middle leaves both words room.
    quiet = np.flatnonzero(rms <= rms.min() * 1.5 + 1e-6)
    k = int(quiet[np.argmin(np.abs(quiet - (n - 1) / 2))])
    return (a + k * frame + frame // 2) / VAD_RATE


def _pauses(loud: np.ndarray, times: list[tuple[float, float]]) -> list[tuple[float, float, float]]:
    """(length, start, end) in seconds of the longest quiet stretch between each two
    consecutive words, on 10 ms frames (``loud`` from aligner.loud_frames). The
    search reaches a little into the words: aligner edges are approximate."""
    out = []
    for (s0, e0), (s1, e1) in zip(times[:-1], times[1:], strict=True):
        lo = max(s0, e0 - 0.1)
        hi = min(e1, max(e0, s1) + 0.1)
        a, b = int(lo * 100), min(len(loud), int(np.ceil(hi * 100)))
        best, best_end, run = 0, 0, 0
        for f in range(a, b):
            run = run + 1 if not loud[f] else 0
            if run > best:
                best, best_end = run, f + 1
        out.append((best / 100, (best_end - best) / 100, best_end / 100))
    return out


def _quiet_edges(db: np.ndarray, start: float, end: float, speech_start: float, speech_end: float):
    """Move clip edges that sit on sound inward to the first quiet 10 ms frame
    (``db`` per frame relative to the loud frames), never past the speech.

    Without this, clips can begin or end in sound: the tail of an earlier word
    where an ASR chunk began, or an untranscribed filler inside a "pause" that
    was quiet relative to the chunk but not to the clip.
    """
    f, limit = int(start * 100), int(speech_start * 100)
    while f < min(limit, len(db)) and db[f] > -EDGE_QUIET_DB:
        f += 1
    g, limit = int(np.ceil(end * 100)) - 1, int(np.ceil(speech_end * 100)) - 1
    while g > max(limit, 0) and g < len(db) and db[g] > -EDGE_QUIET_DB:
        g -= 1
    return max(start, f / 100), min(end, (g + 1) / 100)


@dataclass
class WordClip:
    start: float  # seconds, within the chunk
    end: float
    text: str


def word_clips(wav16k: np.ndarray, text: str, alignment, opts: SliceOptions) -> list[WordClip]:
    """Clips of one transcribed, aligned chunk, cut between words."""
    from .aligner import _frame_db, loud_frames, refine_edges, word_spans

    spans = word_spans(text)
    words = alignment.words
    if not words or len(words) != len(spans):
        return []
    duration = len(wav16k) / VAD_RATE
    # The aligner can place trailing words past the audio (text ASR heard beyond
    # the chunk's speech); such words end at the chunk's end.
    times = [(min(w.start, duration), min(max(w.end, w.start), duration)) for w in words]
    pauses = _pauses(loud_frames(wav16k), times)
    breaks = [_break_score(text[spans[k][1] : spans[k + 1][0]], pauses[k][0]) for k in range(len(words) - 1)]
    plan = plan_word_clips(times, breaks, opts, [p[0] for p in pauses])
    # Cut at the quietest moment of the pause after a clip's last word: a
    # pause found while the previous word still fades has its middle too early,
    # and the next clip would open with that word's tail. Words that run
    # together are cut at their quietest moment.
    cuts = [
        _quietest(wav16k, pauses[j][1], pauses[j][2])
        if pauses[j][0] > 0
        else _quietest(wav16k, times[j][1], max(times[j][1], times[j + 1][0]))
        for (_, j) in plan[:-1]
    ]
    db = _frame_db(wav16k)
    db -= np.percentile(db, 95)
    out = []
    for k, (i, j) in enumerate(plan):
        lo = cuts[k - 1] if k > 0 else 0.0
        hi = cuts[k] if k < len(cuts) else duration
        speech_start, speech_end = refine_edges(wav16k, times[i][0], times[j][1], inner=(times[i][1], times[j][0]))
        start = max(lo, min(speech_start, times[i][0]) - opts.pad_seconds)
        end = min(hi, max(speech_end, times[j][1]) + opts.pad_seconds)
        start, end = _quiet_edges(db, start, end, speech_start, speech_end)
        clip_text = text[spans[i][0] : spans[j + 1][0] if j + 1 < len(spans) else len(text)]
        clip_text = clip_text.strip().lstrip("".join(SENTENCE_END | CLAUSE_END)).strip()
        if end - start >= opts.min_seconds and clip_text:
            out.append(WordClip(start, end, clip_text))
    return out


# Slicing


def _resample(audio: np.ndarray, rate: int, to: int) -> np.ndarray:
    if rate == to:
        return audio
    import torch
    import torchaudio.functional as F

    return F.resample(torch.from_numpy(np.ascontiguousarray(audio)), rate, to).numpy()


@dataclass
class Window:
    start: float  # seconds into the recording
    wav16k: np.ndarray  # mono at VAD_RATE, for VAD / ASR / alignment
    audio: np.ndarray  # mono at `rate`, what clips are cut from
    rate: int
    segments: list[tuple[float, float]]  # speech, seconds into the window


def windows(src: Source, vad: Vad, length: float = WINDOW_S, chain=None) -> Iterator[Window]:
    """Consecutive windows of a recording, each ending in the longest pause of
    its last minute so no speech is split. With a separation ``chain``
    (separation.Chain), each window is separated first: everything after works
    on the voice alone, and clips are cut from it."""
    t0 = 0.0
    while t0 < src.duration - 0.05:
        end = min(src.duration, t0 + length + WINDOW_SLACK_S)
        if chain is not None:
            audio, rate = chain.process(src.read_channels(t0, end, chain.rate, chain.channels)).mean(axis=0), chain.rate
        else:
            audio, rate = src.read(t0, end), src.rate
        audio = audio.astype(np.float32)
        with timing.stage("decoding"):
            wav = _resample(audio, rate, VAD_RATE)
        with timing.stage("vad"):
            segments = vad(wav)
        cut = len(wav) / VAD_RATE
        if end < src.duration:  # not the last window: end at a pause
            gaps = [
                (segments[k + 1][0] - segments[k][1], (segments[k][1] + segments[k + 1][0]) / 2)
                for k in range(len(segments) - 1)
                if length - WINDOW_SLACK_S <= segments[k][1] <= length + WINDOW_SLACK_S
            ]
            cut = max(gaps)[1] if gaps else length
            segments = [(s, min(e, cut)) for s, e in segments if s < cut]
        yield Window(t0, wav[: int(cut * VAD_RATE)], audio[: int(cut * rate)], rate, segments)
        t0 += cut


def _chunks(wav16k: np.ndarray, segments: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Speech grouped into ASR-sized pieces at the longest pauses."""
    opts = SliceOptions(
        min_seconds=0.0, max_seconds=CHUNK_MAX_S - 2 * CHUNK_PAD_S, target_seconds=20.0, max_inner_gap=3.0
    )
    planned = plan_clips(split_long(segments, wav16k, opts.max_seconds), opts, CHUNK_CUT_COST)
    return pad_clips(planned, CHUNK_PAD_S, len(wav16k) / VAD_RATE)


# (phase, done, total); may raise to cancel.
Progress = Callable[[str, int, int], None]


def slice_recording(
    path: Path,
    out_dir: Path,
    opts: SliceOptions,
    vad: Vad = silero_vad,
    guide: Guide | None = None,
    progress: Callable[[float], None] | None = None,
    chain=None,
) -> list[Slice]:
    """Write ``<stem>_<start ms>_<end ms>.wav`` clips (mono PCM16, source rate,
    or the separation models' rate with a ``chain``) into ``out_dir``. With a
    ``guide``, clips are cut between words and carry their transcript.
    ``progress`` gets the seconds processed so far."""
    src = Source(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    out: list[Slice] = []
    for window in windows(src, vad, chain=chain):
        t0, wav, segments = window.start, window.wav16k, window.segments
        if guide is None:
            pieces = [(s, e, "", None) for s, e in find_clips(wav, opts, vad, segments)]
        else:
            pieces = []
            spans = _chunks(wav, segments)
            chunk_wavs = [wav[int(s * VAD_RATE) : int(e * VAD_RATE)] for s, e in spans]
            heard = guide.transcribe(chunk_wavs)
            keep = [k for k, (text, _) in enumerate(heard) if text.strip()]
            alignments = guide.align([chunk_wavs[k] for k in keep], [heard[k][0] for k in keep]) if keep else []
            for k, alignment in zip(keep, alignments, strict=True):
                text, language = heard[k]
                for c in word_clips(chunk_wavs[k], text, alignment, opts):
                    pieces.append((spans[k][0] + c.start, spans[k][0] + c.end, c.text, language))
        audio, rate = window.audio, window.rate
        with timing.stage("writing clips"):
            for start, end, text, language in pieces:
                file = out_dir / f"{path.stem}_{int((t0 + start) * 1000):08d}_{int((t0 + end) * 1000):08d}.wav"
                sf.write(file, audio[int(start * rate) : int(end * rate)], rate, subtype="PCM_16")
                out.append(Slice(file, text, language, round(t0 + start, 3), round(t0 + end, 3)))
        if progress:
            progress(t0 + len(wav) / VAD_RATE)
    return out
