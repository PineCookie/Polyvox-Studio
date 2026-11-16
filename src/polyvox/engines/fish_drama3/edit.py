"""Editing speech by infilling (the Drama 3 technical report's Fig. 14): the audio before and
after an edit stays, the model speaks the span between them from the new text.

The span is found from the texts: the words that differ between the clip's transcript and the
new text, widened to whole words (jieba's for Chinese). An insertion or a deletion also takes the
word after it, which the model speaks again: with the edge right at the new words (or, as in the
report, the word before them) Chinese edits tended to run on into the words that follow. The
span's edges lie in the middle of the pauses between aligned words. The new audio is pasted into
the original with short crossfades, so the rest of the clip stays as it was.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

import numpy as np

from ...prep.aligner import split_words

FRAME = 2048  # samples per codec frame at 44.1 kHz
RATE = 44_100
FADE_S = 0.02
MIN_KEPT_S = 1.0  # of the original audio: the model takes the voice from it
TAGS = re.compile(r"\[[^\]]*\]|<[^>]*>")


def untagged(text: str) -> str:
    """The spoken words: [tags] and <tags> taken out."""
    return re.sub(r"\s+", " ", TAGS.sub(" ", text)).strip()


@dataclass
class Span:
    """Frames [start, end) of the clip to speak again, and the words there before and after."""

    start: int
    end: int
    old: str
    new: str


def word_edges(text: str) -> list[int]:
    """Positions in split_words(text) where words begin or end: jieba's words for Chinese, the
    split_words units elsewhere."""
    import jieba

    edges, i = [0], 0
    for token in jieba.lcut(text):
        n = len(split_words(token))
        if n:
            i += n
            edges.append(i)
    return edges


def plan(text: str, new_text: str, spans: list[tuple[float, float]], frames: int, wider: int = 0) -> Span | None:
    """The frames to speak again for ``new_text``; ``spans`` = (start, end) in seconds of each
    split_words(text) unit (the aligner's words), ``frames`` = the clip's codec frames;
    ``wider``: that many more words after the span (as far as there are). None if the words
    are the same."""
    old, new = split_words(text), split_words(new_text)
    if len(spans) != len(old):
        raise ValueError("The word times don't match the transcript.")
    ops = [op for op in SequenceMatcher(None, old, new, autojunk=False).get_opcodes() if op[0] != "equal"]
    if not ops:
        return None
    a, b, ta, tb = ops[0][1], ops[-1][2], ops[0][3], ops[-1][4]
    edges = word_edges(text)
    wa, wb = max(e for e in edges if e <= a), min(e for e in edges if e >= b)
    ta, tb = ta - (a - wa), tb + (wb - b)  # the units outside the change are the same in both
    if wa == wb or ta == tb:  # an insertion between words or a deletion: also the next word
        if wb < len(old):
            nxt = min(e for e in edges if e > wb)
            tb, wb = tb + nxt - wb, nxt
        elif wa > 0:  # at the end: the word before
            prev = max(e for e in edges if e < wa)
            ta, wa = ta - (wa - prev), prev
    for _ in range(wider):
        if wb < len(old):
            nxt = min(e for e in edges if e > wb)
            tb, wb = tb + nxt - wb, nxt

    def edge(i: int) -> float:  # seconds between unit i - 1 and unit i
        if i <= 0:
            return 0.0
        if i >= len(spans):
            return frames * FRAME / RATE
        return (spans[i - 1][1] + spans[i][0]) / 2

    start, end = round(edge(wa) * RATE / FRAME), round(edge(wb) * RATE / FRAME)
    if (start + frames - end) * FRAME / RATE < MIN_KEPT_S:
        raise ValueError("Too much of the clip changes: generate the new text instead.")
    glue = lambda units: "".join(u if not u.isascii() else f" {u} " for u in units).strip().replace("  ", " ")  # noqa: E731
    return Span(start, end, glue(old[wa:wb]), glue(new[ta:tb]))


def splice(
    original: np.ndarray, decoded: np.ndarray, span: Span, new_frames: int, fade_s: float = FADE_S
) -> np.ndarray:
    """``original`` (44.1 kHz mono) with the span replaced by the new audio: ``decoded`` is the
    whole edited clip as the codec decodes it (before + new frames + after), so the new part is
    decoded in its context. The joins lie just outside the span, where the decoded clip and the
    original say the same: the original fades into the decoded audio before the span, the
    decoded audio back into the original after it (``fade_s`` each; equal gain, as the two sides
    are the same audio: equal power would be 3 dB louder in the middle)."""
    s0, s1, e0 = span.start * FRAME, (span.start + new_frames) * FRAME, span.end * FRAME
    n = int(fade_s * RATE)

    def crossfade(out: np.ndarray, into: np.ndarray) -> np.ndarray:
        w = (0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, len(out)))).astype(np.float32)
        return out * (1 - w) + into * w

    parts = []
    a = max(0, min(s0, len(original)) - n)
    parts += [original[:a], crossfade(original[a:s0], decoded[a:s0]), decoded[s0:s1]]
    if e0 < len(original):
        w = min(n, len(decoded) - s1, len(original) - e0)
        parts += [crossfade(decoded[s1 : s1 + w], original[e0 : e0 + w]), original[e0 + w :]]
    return np.concatenate(parts).astype(np.float32)
