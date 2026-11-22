"""A voice's reference clips chosen by what the engine makes of them (capabilities.reference_search).

references.pick ranks a dataset's clips by how clean their transcripts read, but those aren't
always the clips a model clones a voice best from: on a mixed dataset the cleanest clips can be
among the least like the speaker, and a fast, pause-free clip makes the voice speak that way. On
Drama 3, clips chosen for likeness and typical delivery lifted one voice's likeness by 0.13 and
another's by 0.03, and put the pitch on the speaker's (it was 1-2.6 semitones high), but cost a
third voice 0.02, whose clean clips were already typical. So the sets are tried, not trusted: each
candidate speaks the same dataset sentences with the same seeds, and the one whose takes sound
most like the speaker is kept.

Candidates, each ``count`` clips references.pick would accept:
    clean    references.pick (the cleanest transcripts)
    alike    the clips most like the voice (CAM++ against the dataset's mean)
    typical  among the LIKE_SHARE most like the voice, the nearest the dataset's median pace
             and pause share
Score: the takes' mean CAM++ likeness to the voice, less PITCH_COST per semitone their pitch sits
off the real recordings of the same sentences.

That takes about a minute. pick_alike is the quick version: the "alike" set alone, measured but
not spoken (it was the set kept in each real search so far).
"""

from __future__ import annotations

import random
import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import audio_io, manifests, references
from .speaker import read_16k

POOL = 200  # candidate clips measured (a random sample of those references.pick accepts)
TESTS = 12  # dataset sentences each candidate speaks
MIN_TESTS = 4
TEST_SECONDS = (3.0, 9.0)  # their clips' lengths: the real recordings the takes are held against
VOICE_CLIPS = 60  # averaged into the voice
VOICE_SECONDS = (2.0, 15.0)
LIKE_SHARE = 0.3  # "typical" picks among this share of the pool most like the voice
PITCH_COST = 0.01  # likeness per semitone the takes' pitch sits off the speaker's
NOISE_FLOOR_DB = -72.0  # under every clip scored: pauses of gated (digitally silent) data would skew CAM++
SEED = 1000  # sentence i is spoken with seed SEED + i by every candidate
RATE = 16_000
QUIET_DB = 35.0  # frames this far under the loud ones are quiet (pauses, silent edges)
EMBED_BATCH = 32
KEPT_DATASETS = 8  # datasets whose clip likeness stays in memory (likeness)

Progress = Callable[[str, int, int], None]  # (phase, done, total); may raise to cancel

_ranked: OrderedDict[tuple, list[tuple[float, references.Clip]]] = OrderedDict()
_ranked_lock = threading.Lock()


def _mtime(path: Path) -> int:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return 0


def ranked(
    manifest: Path, embed: Callable[[list[np.ndarray]], np.ndarray], root: Path | None = None, seed: int = 0
) -> list[tuple[float, references.Clip]]:
    """(likeness to the dataset's voice, clip) of the clips references.pick accepts with clean
    transcripts, most alike first: the "alike" candidate without speaking anything, so in
    seconds. Uses the dataset check's own scores where it measured them (align.json "speaker",
    against its "voice"); the rest, at most POOL clips, are measured against that voice (or,
    unchecked, the mean of VOICE_CLIPS clips). Kept while the manifest and its checks are unchanged."""
    manifest = manifest.resolve()
    align = manifest.parent / manifests.ALIGN_FILE
    key = (str(manifest), _mtime(manifest), _mtime(align))
    with _ranked_lock:
        if key in _ranked:
            _ranked.move_to_end(key)
            return _ranked[key]
    rng = random.Random(seed)
    clips = references.usable(manifest, root)
    lo, hi = references.PICK_RANGE_S
    fine = [
        u for u in clips if not u.flagged and lo <= u.clip.seconds <= hi and all(references.text_quality(u.clip.text))
    ]
    out = [(float(u.speaker), u.clip) for u in fine if u.speaker is not None]
    missing = [u.clip for u in fine if u.speaker is None]
    rng.shuffle(missing)
    missing = missing[: max(0, POOL - len(out))]
    if missing:
        saved = manifests.read_sidecar(align).get("voice")
        if saved:
            voice = np.asarray(saved, dtype=np.float64)
        else:
            measured = {c.index for c in missing}
            others = [u.clip for u in clips if not u.flagged and u.clip.index not in measured
                      and VOICE_SECONDS[0] <= u.clip.seconds <= VOICE_SECONDS[1]]  # fmt: skip
            rng.shuffle(others)
            voice = embed([read_16k(c.audio) for c in (others or missing)[:VOICE_CLIPS]]).mean(axis=0)
        voice /= np.linalg.norm(voice)
        vectors = embed([read_16k(c.audio) for c in missing])
        out += [(float(v @ voice), c) for c, v in zip(missing, vectors, strict=True)]
    out.sort(key=lambda sc: -sc[0])
    with _ranked_lock:
        _ranked[key] = out
        while len(_ranked) > KEPT_DATASETS:
            _ranked.popitem(last=False)
    return out


def pick_alike(
    manifest: Path,
    count: int,
    embed: Callable[[list[np.ndarray]], np.ndarray],
    root: Path | None = None,
    exclude: frozenset[int] = frozenset(),
) -> list[references.Clip]:
    """The ``count`` clips most like the dataset's voice (ranked), leaving out ``exclude``."""
    chosen = [c for _, c in ranked(manifest, embed, root) if c.index not in exclude][:count]
    if not chosen:
        raise ValueError("No clip of 4-10 s with a clean transcript and no flags in this dataset.")
    return chosen


@dataclass(frozen=True)
class Candidate:
    name: str  # clean | alike | typical
    clips: tuple[references.Clip, ...]


@dataclass(frozen=True)
class Plan:
    candidates: tuple[Candidate, ...]
    tests: tuple[references.Clip, ...]  # the sentences, with the speaker's own recordings of them
    voice: np.ndarray  # unit-length mean CAM++ embedding of the dataset


@dataclass(frozen=True)
class Tried:
    name: str
    clips: tuple[references.Clip, ...]
    likeness: float  # mean over the takes
    pitch: float  # semitones the takes sit off the speaker's (+ = higher); nan: no pitch found

    @property
    def score(self) -> float:
        return self.likeness - (0.0 if np.isnan(self.pitch) else PITCH_COST * abs(self.pitch))

    def public(self) -> dict:
        pitch = None if np.isnan(self.pitch) else round(self.pitch, 2)
        return {"name": self.name, "indices": [c.index for c in self.clips], "likeness": round(self.likeness, 4),
                "pitch": pitch, "score": round(self.score, 4)}  # fmt: skip


def floored(wave: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return wave + rng.standard_normal(len(wave)).astype(np.float32) * np.float32(10 ** (NOISE_FLOOR_DB / 20))


def delivery(wave: np.ndarray, text: str) -> tuple[float, float]:
    """(spoken characters per second, share of quiet frames) of a 16 kHz clip, its silent edges left out."""
    frame, hop = 512, 160
    if len(wave) < frame:
        return 0.0, 0.0
    starts = np.arange(0, len(wave) - frame + 1, hop)
    rms = np.sqrt(np.square(wave[starts[:, None] + np.arange(frame)]).mean(axis=1)) + 1e-9
    loud = np.flatnonzero(rms > rms.max() * 10 ** (-QUIET_DB / 20))
    span = rms[loud[0] : loud[-1] + 1]
    seconds = len(span) * hop / RATE
    quiet = float((span < np.percentile(span, 95) * 10 ** (-QUIET_DB / 20)).mean())
    return sum(c.isalnum() for c in text) / max(seconds, 0.1), quiet


def median_pitch(wave: np.ndarray) -> float:
    """Median pitch of the voiced frames, semitones above 100 Hz (nan: none)."""
    import librosa

    f0, voiced, _ = librosa.pyin(wave, fmin=70, fmax=600, sr=RATE, frame_length=1024, hop_length=160)
    f0 = f0[voiced & ~np.isnan(f0)]
    return float(np.median(12 * np.log2(f0 / 100.0))) if len(f0) else float("nan")


def plan(
    manifest: Path,
    count: int,
    embed: Callable[[list[np.ndarray]], np.ndarray],
    root: Path | None = None,
    progress: Progress | None = None,
    seed: int = 0,
) -> Plan:
    """The candidate sets of ``count`` clips, the test sentences and the voice of a dataset.
    ``embed``: unit-length CAM++ embeddings of 16 kHz clips. ValueError when the dataset is too
    small to compare sets on."""
    rng, noise = random.Random(seed), np.random.default_rng(seed)
    clips = references.usable(manifest, root)
    clean = references.pick(manifest, count, root, seed)
    taken = {c.index for c in clean}
    fine = [u.clip for u in clips if not u.flagged and all(references.text_quality(u.clip.text))]
    rng.shuffle(fine)
    tests = [c for c in fine if c.index not in taken and TEST_SECONDS[0] <= c.seconds <= TEST_SECONDS[1]][:TESTS]
    taken |= {c.index for c in tests}
    lo, hi = references.PICK_RANGE_S
    pool = [c for c in fine if c.index not in taken and lo <= c.seconds <= hi][:POOL]
    if len(tests) < MIN_TESTS or len(pool) < count:
        need = MIN_TESTS + count + len(clean)
        raise ValueError(f"This dataset has too few clean clips to compare reference sets on (about {need} needed).")
    in_pool = {c.index for c in pool}
    others = [u.clip for u in clips if not u.flagged and u.clip.index not in taken]
    rng.shuffle(others)
    others.sort(key=lambda c: c.index in in_pool)  # clips outside the pool first (stable: still shuffled)
    voice_clips = [c for c in others if VOICE_SECONDS[0] <= c.seconds <= VOICE_SECONDS[1]][:VOICE_CLIPS]

    total = len(voice_clips) + len(pool)
    vectors, measured = [], []
    for start in range(0, total, EMBED_BATCH):
        part = (voice_clips + pool)[start : start + EMBED_BATCH]
        if progress:
            progress("choosing_clips", start, total)
        waves = [read_16k(c.audio) for c in part]
        vectors.append(embed([floored(w, noise) for w in waves]))
        measured += [delivery(w, c.text) for w, c in zip(waves, part, strict=True)]
    vectors = np.concatenate(vectors)
    voice = vectors[: len(voice_clips)].mean(axis=0)
    voice /= np.linalg.norm(voice)
    like = vectors[len(voice_clips) :] @ voice
    pace_quiet = np.array(measured[len(voice_clips) :])

    alike = [pool[k] for k in np.argsort(-like, kind="stable")[:count]]
    close = np.flatnonzero(like >= np.quantile(like, 1 - LIKE_SHARE))
    if len(close) < count:
        close = np.argsort(-like, kind="stable")[:count]
    median = np.median(pace_quiet, axis=0)
    spread = np.subtract(*np.percentile(pace_quiet, [75, 25], axis=0)).clip(1e-6)
    off = (((pace_quiet[close] - median) / spread) ** 2).sum(axis=1)
    typical = [pool[k] for k in close[np.argsort(off, kind="stable")[:count]]]

    candidates, seen = [], set()
    for name, chosen in (("clean", clean), ("alike", alike), ("typical", typical)):
        key = frozenset(c.index for c in chosen)
        if key not in seen:
            seen.add(key)
            candidates.append(Candidate(name, tuple(chosen)))
    return Plan(tuple(candidates), tuple(tests), voice)


def search(
    synthesize: Callable[[str, Path, str, int], tuple[np.ndarray, int]],
    p: Plan,
    embed: Callable[[list[np.ndarray]], np.ndarray],
    workdir: Path,
    progress: Progress | None = None,
) -> list[Tried]:
    """Each candidate's takes of the test sentences and their scores (best: max by score).
    ``synthesize(text, reference wav, its transcript, seed)`` -> (audio, sample rate)."""
    noise = np.random.default_rng(0)
    takes: dict[str, list[np.ndarray]] = {}
    total = len(p.candidates) * len(p.tests)
    for n, cand in enumerate(p.candidates):
        wav = workdir / f"{cand.name}.wav"
        transcript = references.join(list(cand.clips), wav)
        takes[cand.name] = []
        for i, test in enumerate(p.tests):
            if progress:
                progress("speaking_tests", n * len(p.tests) + i, total)
            audio, rate = synthesize(test.text, wav, transcript, SEED + i)
            mono = np.asarray(audio, dtype=np.float32)
            mono = mono.mean(axis=0) if mono.ndim == 2 else mono
            takes[cand.name].append(np.ascontiguousarray(audio_io.resample(mono[None], rate, RATE)[0]))
    steps = len(p.tests) + total
    if progress:
        progress("scoring_takes", 0, steps)
    real = []
    for i, test in enumerate(p.tests):
        real.append(median_pitch(read_16k(test.audio)))
        if progress:
            progress("scoring_takes", i + 1, steps)
    tried, done = [], len(p.tests)
    for cand in p.candidates:
        waves = takes[cand.name]
        like = embed([floored(w, noise) for w in waves]) @ p.voice
        offsets = []
        for w, r in zip(waves, real, strict=True):
            offsets.append(median_pitch(w) - r)
            done += 1
            if progress:
                progress("scoring_takes", done, steps)
        found = [o for o in offsets if not np.isnan(o)]
        tried.append(Tried(cand.name, cand.clips, float(like.mean()), float(np.mean(found)) if found else float("nan")))
    return tried


def best(tried: list[Tried]) -> Tried:
    return max(tried, key=lambda t: t.score)  # ties: the earlier candidate (clean first)
