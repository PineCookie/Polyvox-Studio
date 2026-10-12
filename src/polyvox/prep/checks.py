"""Per-clip checks from forced alignment, stored next to a manifest.

For each clip with a transcript:

- ``mismatch``: part of the text does not match the audio (aligner score over
  the words that are not fillers below MISMATCH_BELOW for that aligner).
- ``filler``: a filler or laugh (嗯, 哈哈, 呵呵, ...) the aligner doubts: it may
  not be in the audio, or not as written. Aligners are unsure of fillers in
  general, so these do not count as a mismatch.
- ``edge_sound``: voice activity before the first or after the last aligned
  word, with a pause between (an untranscribed 嗯 or laugh, or words the text
  misses). Trimming (``trim_range``) can cut it off cleanly.
- ``extra_speech``: other voice activity outside the aligned words, too close
  to them to cut off. The aligner score cannot see either: the remaining words
  still align well.
- ``cut_edge``: the sound is still loud at the start or end of the file (within
  CUT_EDGE_DB of the speech); a word may be cut. Measured on the waveform:
  Silero VAD keeps a segment open to the end of a clip that ends shortly after
  speech.
- ``long_silence``: over LEAD_SILENCE_S of silence before the speech, or over
  TRAIL_SILENCE_S after it.
- ``other_speaker``: the voice is unlike the dataset's (CAM++ embedding vs the
  dataset's voice, ``voice_of``), e.g. a guest or played-back audio. Short clips
  give unsteady embeddings, so they need a lower score (SPEAKER_BELOW).
- ``clipping``: flat tops at the clip's peak (loudness.clipped_runs).

``align.json`` (manifests.ALIGN_FILE) holds the results by audio path as in
the manifest, with a hash of the text each was made for, so editing a
transcript makes its check stale instead of wrong.
"""

from __future__ import annotations

import json
import shutil
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields, replace
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import soundfile as sf

from .. import timing
from ..data import record
from ..data.manifests import ALIGN_FILE, ASR_FILE, clip_entries, read_manifest, read_sidecar, row_key, text_key
from . import slicer
from .aligner import (
    SAMPLE_RATE,
    Alignment,
    edge_levels,
    loud_frames,
    refine_edges,
    sound_spans,
    split_words,
    words_score,
)
from .fillers import is_filler
from .singing import is_sung
from .slicer import Vad

# Set to flag a few percent of correct transcripts while catching most with
# part of the text wrong. The backends' scores are on different scales.
MISMATCH_BELOW = {"qwen3": 0.45, "ctc": 1e-5}
# A single word this unsure is marked in the clip view: rarely a correct word,
# mostly words that are not what was said.
WEAK_WORD_BELOW = {"qwen3": 0.55, "ctc": 6e-5}
EXTRA_SPEECH_S = 0.3
EXTRA_MARGIN_S = 0.25  # voice this close to the aligned words is theirs (breath, release)
LEAD_SILENCE_S = 1.0
# VoxCPM's fine-tuning guide asks for less than this after the speech: the model
# learns when to stop from where the clips end.
TRAIL_SILENCE_S = 0.5
# A loud edge counts as cut only within this many dB of the clip's speech:
# background music and room noise sit lower and are not a cut word.
CUT_EDGE_DB = 20.0
TRIM_PAD_S = 0.15  # silence kept around the speech when trimming (as the slicer pads)
MIN_TRIM_S = 0.1  # trimming less than this is not worth rewriting a file
CHECK_CHUNK = 32  # clips per aligner call; progress and cancel granularity
EDGE_SOUND_GAP_S = 0.15  # a pause this long lets trimming cut off untranscribed edge sound
# A filler the audio lacks: mostly quiet over its span (at least FILLER_PROBE_S
# from its start). Qwen3 often squeezes a filler that is clearly there to a span
# shorter than FILLER_SQUEEZED_S: such a span says nothing about the audio, so it
# is not counted.
FILLER_SQUEEZED_S = 0.04
FILLER_PROBE_S = 0.1
FILLER_LOUD_SHARE = 0.3
# ASR hears more than the transcript has (words left out): at least this many
# words, and this share of the transcript's words. Catches most left-out words
# while rarely flagging a correct transcript.
ASR_MORE_MIN = 2
ASR_MORE_SHARE = 0.1
FLAGS = (
    "singing",
    "other_speaker",
    "clipping",
    "mismatch",
    "asr_more",
    "filler",
    "edge_sound",
    "extra_speech",
    "cut_edge",
    "long_silence",
)
# (clips shorter than this many seconds, flagged below this similarity to the
# voice): set from other people's clips scored against a dataset's voice, which
# this catches ~96% of from 2.5 s (~60% under 1.5 s), flagging under 0.5% of
# the right speaker's clips (~4% under 1.5 s).
SPEAKER_BELOW = ((1.5, 0.3), (2.5, 0.5), (float("inf"), 0.55))
SPEAKER_MIN_S = 0.5  # shorter clips get no speaker score
VOICE_MIN_CLIPS = 10  # fewer scored clips make no voice to compare with
VOICE_KEEP = 0.8  # the voice is the mean of the clips closest to it (twice), leaving other speakers out
EMBED_EVERY = 512  # clips held for speaker embedding at once (batched by length)
# Trimming many clips at once leaves these for review: the "sound" at an edge
# may be words of the transcript the aligner misplaced, or words it misses.
TRIM_REVIEW_FIRST = ("mismatch", "asr_more")
# Not raised for texts ASR made from the clip itself: there a low alignment
# score mostly marks untranscribed laughs or fast speech, not wrong words. The
# score is still saved (worst-first order).
ASR_MADE_UNFLAGGED = ("mismatch",)
ORIGINALS_DIR = "originals"  # next to a manifest: clips as they were before trimming in place
TRIM_IN_PLACE_FORMATS = (".wav", ".flac", ".ogg")  # soundfile writes these back as they were


@dataclass
class ClipCheck:
    score: float  # over the words that are not fillers (see fillers.py)
    start: float  # aligned speech, seconds (edges refined on the waveform)
    end: float
    lead: float  # silence before any speech (aligned words or voice activity)
    trail: float  # silence after it
    duration: float
    unaligned: float  # seconds of voice activity outside the aligned words
    flags: list[str] = field(default_factory=list)
    # [text, start, end, confidence] per aligned word, as the aligner placed it.
    words: list[list] = field(default_factory=list)
    # Untranscribed sound before the words / after them that a pause separates
    # from them: (where a cut would exclude it, seconds of it), else None.
    sound_before: tuple[float, float] | None = None
    sound_after: tuple[float, float] | None = None
    # compare_asr(): what ASR heard, when the check compared with it.
    asr: dict | None = None
    # (sung, speech) scores of the singing detector, when the check ran it.
    singing: tuple[float, float] | None = None
    # Similarity of the voice to the dataset's (with_speaker), when scored.
    speaker: float | None = None
    # Flat tops at the peak (loudness.clipped_runs), when measured.
    clipped: int | None = None

    def trim_range(self, pad: float = TRIM_PAD_S) -> tuple[float, float] | None:
        """(start, end) to keep: ``pad`` of silence around the speech, and
        without untranscribed sound at the edges when a pause separates it
        (so audio and text agree). None when that would remove less than
        MIN_TRIM_S."""
        if self.sound_before:
            cut_start = max(self.sound_before[0], self.start - pad)
        else:
            cut_start = max(0.0, self.lead - pad)
        if self.sound_after:
            keep_end = min(self.sound_after[0], self.end + pad)
        else:
            keep_end = min(self.duration, self.duration - self.trail + pad)
        if cut_start + (self.duration - keep_end) < MIN_TRIM_S:
            return None
        return cut_start, keep_end

    def trimmed(self, start: float, end: float) -> ClipCheck:
        """This check for the clip cut to [start, end]."""
        cut_before = self.sound_before is not None and start >= self.sound_before[0] - 1e-6
        cut_after = self.sound_after is not None and end <= self.sound_after[0] + 1e-6
        lead = (self.start if cut_before else self.lead) - start
        trail = end - (self.end if cut_after else self.duration - self.trail)
        unaligned = self.unaligned
        unaligned -= self.sound_before[1] if cut_before else 0.0
        unaligned -= self.sound_after[1] if cut_after else 0.0
        sound_before = None if cut_before else _shift(self.sound_before, start)
        sound_after = None if cut_after else _shift(self.sound_after, start)
        flags = [f for f in self.flags if f not in ("long_silence", "edge_sound", "extra_speech")]
        flags += _sound_flags(unaligned, sound_before, sound_after)
        if _long_silence(lead, trail):
            flags.append("long_silence")
        return replace(
            self,
            start=self.start - start,
            end=self.end - start,
            lead=max(0.0, lead),
            trail=max(0.0, trail),
            duration=end - start,
            unaligned=round(max(0.0, unaligned), 3),
            flags=[f for f in FLAGS if f in flags],
            words=[[w, round(s - start, 3), round(e - start, 3), c] for w, s, e, c in self.words],
            sound_before=sound_before,
            sound_after=sound_after,
        )

    @classmethod
    def from_saved(cls, saved: dict) -> ClipCheck:
        """A check as align.json holds it (older entries lack the newer fields)."""
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in saved.items() if k in names})


def _units(text: str) -> list[str]:
    """Words to compare a transcript with ASR on: no punctuation, no fillers, any case."""
    return [w.lower() for w in split_words(text) if not is_filler(w)]


def asr_diff(text: str, heard: str) -> list[tuple[str, str, str]]:
    """(op, transcript part, ASR part) per stretch: op is equal, missing (only
    in the transcript), extra (only heard) or changed."""
    a, b = _units(text), _units(heard)
    out = []
    for op, i1, i2, j1, j2 in SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        name = {"equal": "equal", "delete": "missing", "insert": "extra", "replace": "changed"}[op]
        out.append((name, " ".join(a[i1:i2]), " ".join(b[j1:j2])))
    return out


def compare_asr(text: str, heard: str) -> dict:
    """What ASR heard against the transcript: counts of words only ASR heard
    (extra), only in the transcript (missing) and substituted (changed)."""
    a, b = _units(text), _units(heard)
    extra = missing = changed = 0
    for op, i1, i2, j1, j2 in SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        n, m = i2 - i1, j2 - j1
        if op == "insert":
            extra += m
        elif op == "delete":
            missing += n
        elif op == "replace":
            changed += min(n, m)
            extra += max(0, m - n)
            missing += max(0, n - m)
    return {"text": heard, "extra": extra, "missing": missing, "changed": changed, "words": len(a)}


def _hears_more(asr: dict) -> bool:
    return asr["extra"] >= max(ASR_MORE_MIN, ASR_MORE_SHARE * asr["words"])


def _sound_flags(unaligned: float, before: tuple | None, after: tuple | None) -> list[str]:
    """edge_sound for sound a pause separates from the words, extra_speech for the rest."""
    separable = sum(s[1] for s in (before, after) if s)
    return (["edge_sound"] if separable else []) + (["extra_speech"] if unaligned - separable >= EXTRA_SPEECH_S else [])


def _long_silence(lead: float, trail: float) -> bool:
    return lead > LEAD_SILENCE_S or trail > TRAIL_SILENCE_S


def _shift(sound: tuple[float, float] | None, by: float) -> tuple[float, float] | None:
    return (sound[0] - by, sound[1]) if sound else None


def _edge_sound(
    segments: list[tuple[float, float]], start: float, end: float
) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    """Untranscribed voice activity before ``start`` / after ``end`` that a
    pause of at least EDGE_SOUND_GAP_S separates from the words: (cut point in
    the middle of that pause, seconds of sound), for each side."""
    before = [(s, min(e, start)) for s, e in segments if s < start - EXTRA_MARGIN_S]
    after = [(max(s, end), e) for s, e in segments if e > end + EXTRA_MARGIN_S]
    out_before = out_after = None
    if before:
        last = max(e for _, e in before)
        if start - last >= EDGE_SOUND_GAP_S:
            out_before = (round((last + start) / 2, 3), round(sum(e - s for s, e in before), 3))
    if after:
        first = min(s for s, _ in after)
        if first - end >= EDGE_SOUND_GAP_S:
            out_after = (round((end + first) / 2, 3), round(sum(e - s for s, e in after), 3))
    return out_before, out_after


def _merge(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for s, e in sorted(spans):
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def _overlap_outside(segments: list[tuple[float, float]], lo: float, hi: float) -> float:
    return sum(max(0.0, min(e, lo) - s) + max(0.0, e - max(s, hi)) for s, e in segments)


def check_clip(wav: np.ndarray, alignment: Alignment, kind: str, segments: list[tuple[float, float]]) -> ClipCheck:
    """Check one 16 kHz clip given its alignment and voice activity segments."""
    duration = len(wav) / SAMPLE_RATE
    span = alignment.span
    # A text that starts / ends with a filler (哈哈, 啊) has that sound: aligners
    # often squeeze the filler against the next word, leaving the laugh itself
    # outside the words. Sound on that side is not untranscribed, and trimming
    # must keep it.
    placed = [w for w in alignment.words if w.aligned]
    filler_first = bool(placed) and is_filler(placed[0].text)
    filler_last = bool(placed) and is_filler(placed[-1].text)
    if span is None:  # nothing alignable (e.g. only punctuation or unknown characters)
        start, end, unaligned = 0.0, duration, 0.0
    else:
        start, end = refine_edges(wav, *span, inner=alignment.inner)
        lo = -np.inf if filler_first else start - EXTRA_MARGIN_S
        hi = np.inf if filler_last else end + EXTRA_MARGIN_S
        unaligned = _overlap_outside(segments, lo, hi)
    # Silence is measured to any sound, aligned or not, speech to VAD or just
    # audible (a laugh, a sigh VAD passes over): trimming must keep all of it,
    # except edge sound a pause apart, which is flagged for review.
    around = _merge(segments + sound_spans(wav))
    voice_start = min([start] + [s for s, _ in around])
    voice_end = max([end] + [e for _, e in around])
    lead, trail = voice_start, duration - voice_end
    loud = loud_frames(wav)
    sound_before, sound_after = _edge_sound(around, start, end) if span else (None, None)
    sound_before = None if filler_first else sound_before
    sound_after = None if filler_last else sound_after

    # Fillers (嗯, 哈哈, ...) are uncertain for any aligner: the mismatch score
    # covers the other words. A filler placed on quiet audio (e.g. one 呵 too
    # many) gets a soft flag.
    score = words_score(alignment.words, kind, skip=is_filler)
    if score is None:  # nothing but fillers
        score = alignment.score

    def absent(w) -> bool:
        if kind == "qwen3" and w.end - w.start < FILLER_SQUEEZED_S:
            return False  # squeezed: no telling where it is (CTC spans are onsets only, so always short)
        a = int(w.start * 100)
        b = max(a + 1, int(np.ceil(max(w.end, w.start + FILLER_PROBE_S) * 100)))
        loud_share = float(loud[a:b].mean()) if a < len(loud) else 0.0
        return loud_share < FILLER_LOUD_SHARE

    doubtful_filler = any(w.aligned and is_filler(w.text) and absent(w) for w in alignment.words)

    flags = []
    if span is None or score < MISMATCH_BELOW[kind]:
        flags.append("mismatch")
    if doubtful_filler:
        flags.append("filler")
    flags += _sound_flags(unaligned, sound_before, sound_after)
    first, last = edge_levels(wav)
    if (loud[0] and first > -CUT_EDGE_DB) or (loud[-1] and last > -CUT_EDGE_DB):
        flags.append("cut_edge")
    if _long_silence(lead, trail):
        flags.append("long_silence")
    return ClipCheck(
        sound_before=sound_before,
        sound_after=sound_after,
        score=round(score, 6),
        start=round(start, 3),
        end=round(end, 3),
        lead=round(lead, 3),
        trail=round(trail, 3),
        duration=round(duration, 3),
        unaligned=round(unaligned, 3),
        flags=flags,
        words=[
            [w.text, round(w.start, 3), round(w.end, 3), float(f"{w.confidence:.3g}")]
            for w in alignment.words
            if w.aligned
        ],
    )


def with_asr(check: ClipCheck, text: str, heard: str) -> ClipCheck:
    """``check`` with what ASR heard compared to ``text`` (flag asr_more)."""
    asr = compare_asr(text, heard)
    flags = set(check.flags) | ({"asr_more"} if _hears_more(asr) else set())
    return replace(check, asr=asr, flags=[f for f in FLAGS if f in flags])


def other_speaker(score: float, seconds: float) -> bool:
    return score < next(below for shorter, below in SPEAKER_BELOW if seconds < shorter)


def with_speaker(check: ClipCheck, score: float) -> ClipCheck:
    """``check`` with its similarity to the dataset's voice (flag other_speaker)."""
    flags = set(check.flags) - {"other_speaker"} | (
        {"other_speaker"} if other_speaker(score, check.duration) else set()
    )
    return replace(check, speaker=round(score, 4), flags=[f for f in FLAGS if f in flags])


def with_clipping(check: ClipCheck, runs: int | None) -> ClipCheck:
    flags = set(check.flags) - {"clipping"} | ({"clipping"} if runs else set())
    return replace(check, clipped=runs, flags=[f for f in FLAGS if f in flags])


def voice_of(vectors: np.ndarray) -> np.ndarray | None:
    """A dataset's voice from unit speaker embeddings: their mean, then twice the
    mean of the VOICE_KEEP share closest to it (a minority of other speakers
    stays out). None with fewer than VOICE_MIN_CLIPS."""
    if len(vectors) < VOICE_MIN_CLIPS:
        return None
    voice = vectors.mean(axis=0)
    for _ in range(2):
        voice = voice / np.linalg.norm(voice).clip(1e-9)
        scores = vectors @ voice
        voice = vectors[scores >= np.quantile(scores, 1 - VOICE_KEEP)].mean(axis=0)
    return voice / np.linalg.norm(voice).clip(1e-9)


def clipped_in(path: Path) -> int | None:
    """loudness.clipped_runs of an audio file as stored; None if unreadable."""
    from ..data import audio_io
    from .loudness import clipped_runs

    try:
        audio, _ = audio_io.read(path)
    except Exception:
        return None
    return clipped_runs(audio)


# (16 kHz clips, transcripts[, compare with ASR per clip]) -> checks
Checker = Callable[..., list[ClipCheck]]
# (16 kHz clips, transcripts) -> alignments
AlignFn = Callable[[list[np.ndarray], list[str]], list[Alignment]]
# 16 kHz clips -> what ASR hears in each
Transcribe = Callable[[list[np.ndarray]], list[str]]
# 16 kHz clips -> (sung, speech) scores each (singing.SingingService.scores)
SingingScores = Callable[[list[np.ndarray]], list[tuple[float, float]]]
# 16 kHz clips -> unit speaker embeddings, one row each (speaker.SpeakerService.embed_wavs)
Embed = Callable[[list[np.ndarray]], np.ndarray]


def make_checker(
    align: AlignFn,
    kind: str,
    vad: Vad | None = None,
    transcribe: Transcribe | None = None,
    singing: SingingScores | None = None,
) -> Checker:
    """``vad`` defaults to Silero VAD, batched over the clips (looked up per
    call). With ``transcribe``, clips are also compared with what ASR hears,
    except where ``compare`` (third argument, per clip) is False: a text made
    by the same ASR would only agree with itself. Clips whose text ASR made
    (``asr_made``, fourth argument) get no mismatch flag (see ASR_MADE_UNFLAGGED).
    With `singing`, sung clips are flagged (singing.is_sung)."""

    def check(
        wavs: list[np.ndarray],
        texts: list[str],
        compare: list[bool] | None = None,
        asr_made: list[bool] | None = None,
    ) -> list[ClipCheck]:
        alignments = align(wavs, texts)
        with timing.stage("vad"):
            segments = [vad(w) for w in wavs] if vad else slicer.silero_vad_many(wavs)
        out = [check_clip(w, a, kind, s) for w, a, s in zip(wavs, alignments, segments, strict=True)]
        for i, made in enumerate(asr_made or []):
            if made:
                out[i] = replace(out[i], flags=[f for f in out[i].flags if f not in ASR_MADE_UNFLAGGED])
        if transcribe:
            todo = [i for i in range(len(wavs)) if compare is None or compare[i]]
            heard = transcribe([wavs[i] for i in todo]) if todo else []
            for i, h in zip(todo, heard, strict=True):
                out[i] = with_asr(out[i], texts[i], h)
        if singing:
            for i, (sung, speech) in enumerate(singing(wavs)):
                flags = set(out[i].flags) | ({"singing"} if is_sung(sung, speech) else set())
                out[i] = replace(
                    out[i], singing=(round(sung, 4), round(speech, 4)), flags=[f for f in FLAGS if f in flags]
                )
        return out

    return check


def check_clips(
    paths: list[Path],
    texts: list[str],
    asr_made: list[bool],
    checker: Checker,
    load: Callable[[Path], np.ndarray],
    embed: Embed | None = None,
    progress: Progress | None = None,
    voice: np.ndarray | None = None,
) -> tuple[list[ClipCheck | None], np.ndarray | None]:
    """Check clips in chunks of CHECK_CHUNK (texts ASR made are not compared
    with ASR), measure clipping, and with ``embed`` score each voice against
    ``voice`` (by default the clips' common voice; embedded EMBED_EVERY clips
    at a time: batches of similar lengths). Returns the checks (None for clips
    ``load`` can't read) and the voice."""
    progress = progress or (lambda phase, done, total: None)
    out: list[ClipCheck | None] = [None] * len(paths)
    vectors: dict[int, np.ndarray] = {}
    to_embed: dict[int, np.ndarray] = {}

    def embed_pending() -> None:
        if to_embed:
            with timing.stage("speaker"):
                vectors.update(zip(to_embed, embed(list(to_embed.values())), strict=True))
            to_embed.clear()

    for start in range(0, len(paths), CHECK_CHUNK):
        progress("aligning", start, len(paths))
        index, wavs = [], []
        for i in range(start, min(start + CHECK_CHUNK, len(paths))):
            try:
                wavs.append(load(paths[i]))
                index.append(i)
            except Exception:
                continue
        if not index:
            continue
        made = [asr_made[i] for i in index]
        checked = checker(wavs, [texts[i] for i in index], [not m for m in made], made)
        for i, wav, check in zip(index, wavs, checked, strict=True):
            with timing.stage("clipping"):
                out[i] = with_clipping(check, clipped_in(paths[i]))
            if embed and len(wav) >= SPEAKER_MIN_S * SAMPLE_RATE:
                to_embed[i] = wav
        if len(to_embed) >= EMBED_EVERY:
            embed_pending()
    embed_pending()
    progress("aligning", len(paths), len(paths))
    if voice is None and vectors:
        voice = voice_of(np.stack(list(vectors.values())))
    if voice is not None:
        for i, vector in vectors.items():
            out[i] = with_speaker(out[i], float(vector @ voice))
    return out, voice


def write_checks(
    manifest_dir: Path, kind: str, checks: dict[str, tuple[str, ClipCheck]], voice: np.ndarray | None = None
) -> None:
    """Save ``{audio key: (text, check)}`` as align.json (atomically), with the
    dataset's voice that ``check_one`` scores single clips against."""
    clips = {key: {"text": text_key(text), **asdict(check)} for key, (text, check) in checks.items()}
    data = {"aligner": kind, "clips": clips} | ({"voice": voice.round(6).tolist()} if voice is not None else {})
    _write_align(manifest_dir, data)


def summarize(checks: list[ClipCheck]) -> dict:
    counts = Counter(f for c in checks for f in c.flags)
    return {
        "checked": len(checks),
        "flagged": sum(bool(c.flags) for c in checks),
        "flags": {f: counts[f] for f in FLAGS},
    }


# (phase, done, total); may raise to cancel.
Progress = Callable[[str, int, int], None]


def check_dataset(
    manifest: Path,
    root: Path | None,
    checker: Checker,
    kind: str,
    load: Callable[[Path], np.ndarray],
    progress: Progress | None = None,
    embed: Embed | None = None,
    only_needed: bool = False,
) -> dict:
    """Check every clip of a manifest (``check_clips``) and write its
    align.json; returns a summary over the dataset's current checks.
    ``only_needed`` checks just the clips without a current check (new clips,
    edited texts) and keeps the others, scoring voices against the saved
    voice when there is one. Unreadable clips are skipped. Nothing is written
    when cancelled."""
    entries, _ = read_manifest(manifest, root)
    manifest_dir = manifest.resolve().parent
    asr_made = set(read_sidecar(manifest_dir / ASR_FILE))
    saved = read_sidecar(manifest_dir / ALIGN_FILE) if only_needed else {}
    kept = {}
    for e in entries:
        key = row_key(e["audio"], manifest_dir)
        old = saved.get("clips", {}).get(key)
        if old and old.get("text") == text_key(e["text"]):
            kept[key] = old
    todo = [e for e in entries if row_key(e["audio"], manifest_dir) not in kept]
    keys = [row_key(e["audio"], manifest_dir) for e in todo]
    saved_voice = np.asarray(saved["voice"]) if kept and saved.get("voice") else None
    checked, voice = check_clips(
        [Path(e["audio"]) for e in todo],
        [e["text"] for e in todo],
        [k in asr_made for k in keys],
        checker,
        load,
        embed,
        progress,
        saved_voice,
    )
    clips = dict(kept)
    for key, entry, check in zip(keys, todo, checked, strict=True):
        if check is not None:
            clips[key] = {"text": text_key(entry["text"]), **asdict(check)}
    data = {"aligner": saved.get("aligner", kind) if kept else kind, "clips": clips}
    if voice is not None:
        data["voice"] = voice.round(6).tolist()
    _write_align(manifest_dir, data)
    return {"aligner": data["aligner"], **summarize([ClipCheck.from_saved(c) for c in clips.values()])}


def check_one(
    manifest: Path,
    root: Path | None,
    index: int,
    checker: Checker,
    kind: str,
    load: Callable[[Path], np.ndarray],
    compare: bool | None = None,
    embed: Embed | None = None,
) -> ClipCheck:
    """Check one clip again (after its text was edited or it was trimmed) and
    update its entry in align.json; the other entries stay. Uses ``kind`` for
    the whole file only when there were no checks yet. ``compare`` with ASR
    (when the checker can): by default unless ASR made the text. With
    ``embed``, the voice is scored against the dataset's saved one."""
    entries = clip_entries(manifest, root)
    if not 0 <= index < len(entries):
        raise KeyError(f"No clip {index} in this dataset.")
    entry = entries[index]
    if not entry["text"]:
        raise ValueError("This clip has no text yet: transcribe it before checking.")
    manifest_dir = manifest.resolve().parent
    key = row_key(entry["audio"], manifest_dir)
    made = key in set(read_sidecar(manifest_dir / ASR_FILE))
    if compare is None:
        compare = not made
    audio = Path(entry["audio"])
    wav = load(audio)
    (check,) = checker([wav], [entry["text"]], [compare], [made])
    check = with_clipping(check, clipped_in(audio))
    saved = read_sidecar(manifest_dir / ALIGN_FILE)
    if embed and saved.get("voice") and len(wav) >= SPEAKER_MIN_S * SAMPLE_RATE:
        check = with_speaker(check, float(embed([wav])[0] @ np.asarray(saved["voice"])))
    clips = dict(saved.get("clips", {}))
    clips[key] = {"text": text_key(entry["text"]), **asdict(check)}
    _write_align(manifest_dir, {**saved, "aligner": saved.get("aligner", kind), "clips": clips})
    return check


def _write_align(manifest_dir: Path, data: dict) -> None:
    tmp = manifest_dir / f".{ALIGN_FILE}.tmp"
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    tmp.replace(manifest_dir / ALIGN_FILE)


def _trim_file(audio: Path, backup: Path, start: float, end: float) -> None:
    """Cut ``audio`` to [start, end) seconds in place, in its own format and
    channels; the first time, the original is copied to ``backup``."""
    info = sf.info(str(audio))
    data, rate = sf.read(
        str(audio), start=round(start * info.samplerate), stop=round(end * info.samplerate), always_2d=True
    )
    if not backup.exists():
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(audio, backup)
    tmp = audio.with_name(f".{audio.name}.tmp")
    sf.write(str(tmp), data, rate, subtype=info.subtype, format=info.format)
    tmp.replace(audio)


# (index, the clip's current check or None) -> (start, end) seconds to keep, or None
CutFor = Callable[[int, ClipCheck | None], tuple[float, float] | None]


def trim_clips(manifest: Path, root: Path | None, cut_for: CutFor) -> dict:
    """Trim clips in place to the range ``cut_for`` gives (e.g. the checked
    edges, ClipCheck.trim_range), and update their checks. Originals are kept
    under ORIGINALS_DIR next to the manifest (the first version only). Clips
    outside the manifest's folder or in a format soundfile can't write back are
    skipped. Clips cut from a recording keep their origin (record.py) in step.
    Returns {"trimmed", "skipped"}."""
    entries = clip_entries(manifest, root)
    manifest_dir = manifest.resolve().parent
    saved = read_sidecar(manifest_dir / ALIGN_FILE)
    clips = dict(saved.get("clips", {}))
    cuts: dict[str, tuple[float, float]] = {}
    trimmed = skipped = 0
    for index, entry in enumerate(entries):
        audio = Path(entry["audio"])
        key = row_key(str(audio), manifest_dir)
        stored = clips.get(key)
        current = stored is not None and stored.get("text") == text_key(entry["text"])
        check = ClipCheck.from_saved(stored) if current else None
        if (cut := cut_for(index, check)) is None:
            continue
        if not audio.is_relative_to(manifest_dir) or audio.suffix.lower() not in TRIM_IN_PLACE_FORMATS:
            skipped += 1
            continue
        _trim_file(audio, manifest_dir / ORIGINALS_DIR / key, *cut)
        if check:
            clips[key] = {"text": stored["text"], **asdict(check.trimmed(*cut))}
        cuts[key] = cut
        trimmed += 1
    if trimmed:
        _write_align(manifest_dir, {**saved, "clips": clips})
        record.trimmed(manifest_dir, cuts)
    return {"trimmed": trimmed, "skipped": skipped}


def squeezed_edge_filler(check: ClipCheck) -> bool:
    """The first or last word is a filler the aligner squeezed to nothing: where
    the sound really is (maybe a quiet blob trimming would cut) is unknown."""
    edges = check.words[:1] + check.words[-1:]
    return any(is_filler(w[0]) and w[2] - w[1] < FILLER_SQUEEZED_S for w in edges)


def trim_flagged(manifest: Path, root: Path | None, flag: str) -> dict:
    """``trim_clips`` for every clip with ``flag``, except those also flagged
    TRIM_REVIEW_FIRST or with a squeezed filler at an edge (counted as "review")."""
    review = 0

    def cut_for(_: int, check: ClipCheck | None) -> tuple[float, float] | None:
        nonlocal review
        if check is None or flag not in check.flags:
            return None
        if any(f in check.flags for f in TRIM_REVIEW_FIRST) or squeezed_edge_filler(check):
            review += 1
            return None
        return check.trim_range()

    return {**trim_clips(manifest, root, cut_for), "review": review}
