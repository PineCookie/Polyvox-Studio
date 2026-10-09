"""Per-clip checks from forced alignment, stored next to a manifest.

For each clip with a transcript:

- ``mismatch``: part of the text does not match the audio (aligner score
  below MISMATCH_BELOW for that aligner).
- ``extra_speech``: voice activity outside the aligned words, i.e. the text
  misses something said (a dropped phrase, an untranscribed filler or laugh).
  The aligner score cannot see this: the remaining words still align well.
- ``cut_edge``: the sound is still loud at the start or end of the file (within
  CUT_EDGE_DB of the speech); a word may be cut. (Measured on the waveform: Silero VAD keeps a segment open to the
  end of a clip whenever the clip ends shortly after speech.)
- ``long_silence``: over LONG_SILENCE_S of silence before or after the speech.

``align.json`` (manifests.ALIGN_FILE) holds the results by audio path as in
the manifest, with a hash of the text each was made for, so editing a
transcript makes its check stale instead of wrong.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

from .aligner import SAMPLE_RATE, Alignment, edge_levels, loud_frames, refine_edges
from .manifests import ALIGN_FILE, read_manifest, row_key, text_key
from .slicer import Vad, silero_vad

# Calibrated on ~600 real clips to flag ~4% of correct transcripts, catching
# ~65% (qwen3) / ~83% (ctc) of transcripts with a third of the words wrong.
MISMATCH_BELOW = {"qwen3": 0.45, "ctc": 1e-5}
# A single word this unsure is marked in the clip view: 5% of correct words, and
# 76% (qwen3) / 88% (ctc) of words replaced by wrong ones.
WEAK_WORD_BELOW = {"qwen3": 0.55, "ctc": 6e-5}
EXTRA_SPEECH_S = 0.3
EXTRA_MARGIN_S = 0.25  # voice this close to the aligned words is theirs (breath, release)
LONG_SILENCE_S = 1.0
# A loud edge counts as cut only within this many dB of the clip's speech: in
# live recordings, background music and room noise sit 20-25 dB down and are
# not a cut word (cut words measured -1 to -5 dB, word tails -10 to -15 dB).
CUT_EDGE_DB = 20.0
TRIM_PAD_S = 0.15  # silence kept around the speech when trimming (as the slicer pads)
MIN_TRIM_S = 0.1  # trimming less than this is not worth rewriting a file
CHECK_CHUNK = 32  # clips per aligner call; progress and cancel granularity
FLAGS = ("mismatch", "extra_speech", "cut_edge", "long_silence")


@dataclass
class ClipCheck:
    score: float
    start: float  # aligned speech, seconds (edges refined on the waveform)
    end: float
    lead: float  # silence before any speech (aligned words or voice activity)
    trail: float  # silence after it
    duration: float
    unaligned: float  # seconds of voice activity outside the aligned words
    flags: list[str] = field(default_factory=list)
    # [text, start, end, confidence] per aligned word, as the aligner placed it.
    words: list[list] = field(default_factory=list)

    def trim_range(self, pad: float = TRIM_PAD_S) -> tuple[float, float] | None:
        """(start, end) to keep, leaving ``pad`` of silence around the speech;
        None when that would remove less than MIN_TRIM_S."""
        cut_start = max(0.0, self.lead - pad)
        cut_end = max(0.0, self.trail - pad)
        if cut_start + cut_end < MIN_TRIM_S:
            return None
        return cut_start, self.duration - cut_end

    def trimmed(self, start: float, end: float) -> ClipCheck:
        """This check for the clip cut to [start, end]."""
        lead, trail = self.lead - start, self.trail - (self.duration - end)
        flags = [f for f in self.flags if f != "long_silence"]
        if max(lead, trail) > LONG_SILENCE_S:
            flags.append("long_silence")
        return replace(
            self,
            start=self.start - start,
            end=self.end - start,
            lead=lead,
            trail=trail,
            duration=end - start,
            flags=flags,
            words=[[w, round(s - start, 3), round(e - start, 3), c] for w, s, e, c in self.words],
        )


def _overlap_outside(segments: list[tuple[float, float]], lo: float, hi: float) -> float:
    return sum(max(0.0, min(e, lo) - s) + max(0.0, e - max(s, hi)) for s, e in segments)


def check_clip(wav: np.ndarray, alignment: Alignment, kind: str, segments: list[tuple[float, float]]) -> ClipCheck:
    """Check one 16 kHz clip given its alignment and voice activity segments."""
    duration = len(wav) / SAMPLE_RATE
    span = alignment.span
    if span is None:  # nothing alignable (e.g. only punctuation or unknown characters)
        start, end, unaligned = 0.0, duration, 0.0
    else:
        start, end = refine_edges(wav, *span, inner=alignment.inner)
        unaligned = _overlap_outside(segments, start - EXTRA_MARGIN_S, end + EXTRA_MARGIN_S)
    # Silence around all speech, aligned or not: trimming must keep untranscribed speech.
    voice_start = min([start] + [s for s, _ in segments[:1]])
    voice_end = max([end] + [e for _, e in segments[-1:]])
    lead, trail = voice_start, duration - voice_end
    loud = loud_frames(wav)

    flags = []
    if span is None or alignment.score < MISMATCH_BELOW[kind]:
        flags.append("mismatch")
    if unaligned >= EXTRA_SPEECH_S:
        flags.append("extra_speech")
    first, last = edge_levels(wav)
    if (loud[0] and first > -CUT_EDGE_DB) or (loud[-1] and last > -CUT_EDGE_DB):
        flags.append("cut_edge")
    if max(lead, trail) > LONG_SILENCE_S:
        flags.append("long_silence")
    return ClipCheck(
        score=round(alignment.score, 6),
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


# (16 kHz clips, transcripts) -> checks
Checker = Callable[[list[np.ndarray], list[str]], list[ClipCheck]]
# (16 kHz clips, transcripts) -> alignments
AlignFn = Callable[[list[np.ndarray], list[str]], list[Alignment]]


def make_checker(align: AlignFn, kind: str, vad: Vad | None = None) -> Checker:
    """``vad`` defaults to Silero VAD (looked up per call)."""

    def check(wavs: list[np.ndarray], texts: list[str]) -> list[ClipCheck]:
        alignments = align(wavs, texts)
        detect = vad or silero_vad
        return [check_clip(w, a, kind, detect(w)) for w, a in zip(wavs, alignments, strict=True)]

    return check


def write_checks(manifest_dir: Path, kind: str, checks: dict[str, tuple[str, ClipCheck]]) -> None:
    """Save ``{audio key: (text, check)}`` as align.json (atomically)."""
    clips = {key: {"text": text_key(text), **asdict(check)} for key, (text, check) in checks.items()}
    tmp = manifest_dir / f".{ALIGN_FILE}.tmp"
    tmp.write_text(json.dumps({"aligner": kind, "clips": clips}, ensure_ascii=False), encoding="utf-8")
    tmp.replace(manifest_dir / ALIGN_FILE)


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
) -> dict:
    """Check every clip of a manifest and write its align.json; returns a summary.
    Unreadable clips are skipped. Nothing is written when cancelled."""
    progress = progress or (lambda phase, done, total: None)
    entries, _ = read_manifest(manifest, root)
    manifest_dir = manifest.resolve().parent
    results: dict[str, tuple[str, ClipCheck]] = {}
    for start in range(0, len(entries), CHECK_CHUNK):
        progress("aligning", start, len(entries))
        chunk, wavs = [], []
        for entry in entries[start : start + CHECK_CHUNK]:
            try:
                wavs.append(load(Path(entry["audio"])))
                chunk.append(entry)
            except Exception:
                continue
        for entry, check in zip(chunk, checker(wavs, [e["text"] for e in chunk]) if chunk else [], strict=True):
            results[row_key(entry["audio"], manifest_dir)] = (entry["text"], check)
    progress("aligning", len(entries), len(entries))
    write_checks(manifest_dir, kind, results)
    return {"aligner": kind, **summarize([c for _, c in results.values()])}
