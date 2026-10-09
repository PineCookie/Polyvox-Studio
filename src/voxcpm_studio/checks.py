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

from . import slicer
from .aligner import SAMPLE_RATE, Alignment, edge_levels, loud_frames, refine_edges, split_words, words_score
from .fillers import is_filler
from .manifests import ALIGN_FILE, ASR_FILE, read_manifest, read_sidecar, row_key, text_key
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
# A filler the audio lacks: squeezed shorter than this (qwen3), or mostly quiet
# over its span (at least FILLER_PROBE_S from its start).
FILLER_SQUEEZED_S = 0.04
FILLER_PROBE_S = 0.1
FILLER_LOUD_SHARE = 0.3
# ASR hears more than the transcript has (words left out): at least this many
# words, and this share of the transcript's words. Catches most left-out words
# while rarely flagging a correct transcript.
ASR_MORE_MIN = 2
ASR_MORE_SHARE = 0.1
FLAGS = ("singing", "mismatch", "asr_more", "filler", "edge_sound", "extra_speech", "cut_edge", "long_silence")
# Trimming many clips at once leaves these for review: the "sound" at an edge
# may be words of the transcript the aligner misplaced, or words it misses.
TRIM_REVIEW_FIRST = ("mismatch", "asr_more")
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
    sound_before, sound_after = _edge_sound(segments, start, end) if span else (None, None)

    # Fillers (嗯, 哈哈, ...) are uncertain for any aligner: the mismatch score
    # covers the other words. A filler the audio seems to lack (squeezed to a
    # sliver, or placed on quiet audio: e.g. one 呵 too many) gets a soft flag.
    score = words_score(alignment.words, kind, skip=is_filler)
    if score is None:  # nothing but fillers
        score = alignment.score

    def absent(w) -> bool:
        a = int(w.start * 100)
        b = max(a + 1, int(np.ceil(max(w.end, w.start + FILLER_PROBE_S) * 100)))
        loud_share = float(loud[a:b].mean()) if a < len(loud) else 0.0
        squeezed = kind == "qwen3" and w.end - w.start < FILLER_SQUEEZED_S  # CTC spans are onsets only
        return squeezed or loud_share < FILLER_LOUD_SHARE

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


# (16 kHz clips, transcripts[, compare with ASR per clip]) -> checks
Checker = Callable[..., list[ClipCheck]]
# (16 kHz clips, transcripts) -> alignments
AlignFn = Callable[[list[np.ndarray], list[str]], list[Alignment]]
# 16 kHz clips -> what ASR hears in each
Transcribe = Callable[[list[np.ndarray]], list[str]]
# 16 kHz clips -> (sung, speech) scores each (singing.SingingService.scores)
SingingScores = Callable[[list[np.ndarray]], list[tuple[float, float]]]


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
    by the same ASR would only agree with itself. With `singing`, sung clips
    are flagged (singing.is_sung)."""

    def check(wavs: list[np.ndarray], texts: list[str], compare: list[bool] | None = None) -> list[ClipCheck]:
        alignments = align(wavs, texts)
        segments = [vad(w) for w in wavs] if vad else slicer.silero_vad_many(wavs)
        out = [check_clip(w, a, kind, s) for w, a, s in zip(wavs, alignments, segments, strict=True)]
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
    asr_made = set(read_sidecar(manifest_dir / ASR_FILE))
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
        if not chunk:
            continue
        keys = [row_key(e["audio"], manifest_dir) for e in chunk]
        checked = checker(wavs, [e["text"] for e in chunk], [k not in asr_made for k in keys])
        for key, entry, check in zip(keys, chunk, checked, strict=True):
            results[key] = (entry["text"], check)
    progress("aligning", len(entries), len(entries))
    write_checks(manifest_dir, kind, results)
    return {"aligner": kind, **summarize([c for _, c in results.values()])}


def check_one(
    manifest: Path, root: Path | None, index: int, checker: Checker, kind: str, load: Callable[[Path], np.ndarray]
) -> ClipCheck:
    """Check one clip again (after its text was edited) and update its entry
    in align.json; the other entries stay. Uses ``kind`` for the whole file
    only when there were no checks yet."""
    entries, _ = read_manifest(manifest, root)
    if not 0 <= index < len(entries):
        raise KeyError(f"No clip {index} in this dataset.")
    entry = entries[index]
    manifest_dir = manifest.resolve().parent
    key = row_key(entry["audio"], manifest_dir)
    compare = key not in set(read_sidecar(manifest_dir / ASR_FILE))
    (check,) = checker([load(Path(entry["audio"]))], [entry["text"]], [compare])
    saved = read_sidecar(manifest_dir / ALIGN_FILE)
    clips = dict(saved.get("clips", {}))
    clips[key] = {"text": text_key(entry["text"]), **asdict(check)}
    _write_align(manifest_dir, {"aligner": saved.get("aligner", kind), "clips": clips})
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


def trim_clips(manifest: Path, root: Path | None, pick: Callable[[int, ClipCheck], bool]) -> dict:
    """Trim the checked clips that ``pick(index, check)`` selects at their
    checked edges (ClipCheck.trim_range), in place, and update their checks.
    Originals are kept under ORIGINALS_DIR next to the manifest (the first
    version only). Clips outside the manifest's folder or in a format soundfile
    can't write back are skipped. Returns {"trimmed", "skipped"}."""
    entries, _ = read_manifest(manifest, root)
    manifest_dir = manifest.resolve().parent
    saved = read_sidecar(manifest_dir / ALIGN_FILE)
    clips = dict(saved.get("clips", {}))
    trimmed = skipped = 0
    for index, entry in enumerate(entries):
        audio = Path(entry["audio"])
        key = row_key(str(audio), manifest_dir)
        stored = clips.get(key)
        if not stored or stored.get("text") != text_key(entry["text"]):
            continue  # not checked, or checked for an earlier transcript
        check = ClipCheck.from_saved(stored)
        if not pick(index, check) or (cut := check.trim_range()) is None:
            continue
        if not audio.is_relative_to(manifest_dir) or audio.suffix.lower() not in TRIM_IN_PLACE_FORMATS:
            skipped += 1
            continue
        _trim_file(audio, manifest_dir / ORIGINALS_DIR / key, *cut)
        clips[key] = {"text": stored["text"], **asdict(check.trimmed(*cut))}
        trimmed += 1
    if trimmed:
        _write_align(manifest_dir, {**saved, "clips": clips})
    return {"trimmed": trimmed, "skipped": skipped}


def trim_flagged(manifest: Path, root: Path | None, flag: str) -> dict:
    """``trim_clips`` for every clip with ``flag``, except those also flagged
    TRIM_REVIEW_FIRST (counted as "review")."""
    review = 0

    def pick(_: int, check: ClipCheck) -> bool:
        nonlocal review
        if flag not in check.flags:
            return False
        if any(f in check.flags for f in TRIM_REVIEW_FIRST):
            review += 1
            return False
        return True

    return {**trim_clips(manifest, root, pick), "review": review}
