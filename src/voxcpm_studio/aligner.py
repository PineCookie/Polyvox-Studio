"""Forced alignment: where each word of a transcript is in its audio.

Two backends, both native to transformers / torchaudio (no extra packages):

- ``qwen3``: Qwen3-ForcedAligner (``Qwen3ASRForTokenClassification``) predicts a
  start and end per word in 80 ms steps; up to ~6 minutes of audio, 11
  languages, mixed scripts. Its confidence separates a clip's own transcript
  from a partly wrong one well.
- ``ctc``: a wav2vec2 CTC model with a character vocabulary (e.g. one per
  language, as WhisperX does) and ``torchaudio.functional.forced_align``. Sharp
  word starts, but CTC marks only a character's onset, so ends are early; the
  energy refinement below fixes the speech edges.

After that refinement both place speech edges about equally well.
"""

from __future__ import annotations

import gc
import logging
import threading
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .config import Settings

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
BATCH_SIZE = 16
BATCH_SECONDS = 160  # padded audio per model call
ALIGNERS = ("qwen3", "ctc")

# Energy refinement of the speech edges (10 ms frames).
FRAME = 160
EDGE_DROP_DB = 35.0  # "speech" = within this many dB of the clip's loud frames
EDGE_FLOOR_DB = 10.0  # ... and this far above its noise floor
EDGE_BRIDGE_FRAMES = 6  # quieter stretches up to 60 ms do not end the speech


@dataclass
class Word:
    text: str
    start: float  # seconds
    end: float
    confidence: float = 0.0  # model confidence for this word, 0..1
    aligned: bool = True  # False: no model token for it; placed at the previous word's end


@dataclass
class Alignment:
    words: list[Word]
    # The worst stretch of a few words (worst_window), 0..1: low when part of the
    # text does not match the audio. Scales differ between backends.
    score: float

    @property
    def span(self) -> tuple[float, float] | None:
        """(start of the first, end of the last) aligned word."""
        aligned = [w for w in self.words if w.aligned]
        return (aligned[0].start, aligned[-1].end) if aligned else None

    @property
    def inner(self) -> tuple[float, float] | None:
        """(end of the first, start of the last) aligned word."""
        aligned = [w for w in self.words if w.aligned]
        return (aligned[0].end, aligned[-1].start) if aligned else None


def worst_window(confidences: list[float], size: int, geometric: bool = False) -> float:
    """Lowest mean confidence over ``size`` consecutive words.

    A wrong phrase inside a long, otherwise right transcript barely moves the
    clip's mean, but sinks the windows over it. CTC confidences of wrong
    characters are near zero, so for CTC the geometric mean resolves them.
    """
    if not confidences:
        return 0.0
    c = np.asarray(confidences, dtype=np.float64)
    if geometric:
        c = np.log(np.maximum(c, 1e-12))
    size = min(size, len(c))
    sums = np.convolve(c, np.ones(size), mode="valid") / size
    worst = float(sums.min())
    return float(np.exp(worst)) if geometric else worst


# How each backend's score is taken (worst_window): window size, geometric mean.
SCORE_WINDOW = {"qwen3": (5, False), "ctc": (3, True)}


def words_score(words: list[Word], kind: str, skip: Callable[[str], bool] = lambda w: False) -> float | None:
    """The backend's worst-window score over the aligned words that ``skip``
    does not exclude; None when none are left."""
    confidences = [w.confidence for w in words if w.aligned and not skip(w.text)]
    if not confidences:
        return None
    size, geometric = SCORE_WINDOW[kind]
    return worst_window(confidences, size, geometric)


def _is_cjk(char: str) -> bool:
    o = ord(char)
    return (
        0x4E00 <= o <= 0x9FFF  # CJK ideographs
        or 0x3400 <= o <= 0x4DBF
        or 0x20000 <= o <= 0x2EBEF
        or 0xF900 <= o <= 0xFAFF
        or 0x3040 <= o <= 0x30FF  # kana: each one a syllable
    )


def word_spans(text: str) -> list[tuple[int, int]]:
    """(start, end) character offsets of the alignment units in ``text``: each
    CJK character on its own, other scripts by spaces; punctuation dropped (as
    Qwen3-ForcedAligner's own splitter, plus kana)."""
    spans: list[tuple[int, int]] = []
    start = None
    for i, char in enumerate(text):
        kept = char == "'" or unicodedata.category(char)[0] in "LN"
        if _is_cjk(char) or not kept:
            if start is not None:
                spans.append((start, i))
                start = None
            if _is_cjk(char):
                spans.append((i, i + 1))
        elif start is None:
            start = i
    if start is not None:
        spans.append((start, len(text)))
    return spans


def split_words(text: str) -> list[str]:
    """Alignment units of ``text`` (see word_spans)."""
    return [text[s:e] for s, e in word_spans(text)]


class Backend(Protocol):
    def align(self, audios: list[np.ndarray], words: list[list[str]]) -> list[Alignment]: ...


class QwenAligner:
    def __init__(self, model: str, device: str):
        import torch
        from transformers import AutoProcessor, Qwen3ASRForTokenClassification

        self.processor = AutoProcessor.from_pretrained(model)
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self.model = Qwen3ASRForTokenClassification.from_pretrained(model, dtype=dtype).to(device).eval()

    def align(self, audios: list[np.ndarray], words: list[list[str]]) -> list[Alignment]:
        import torch

        model, processor = self.model, self.processor
        conversations = [
            [{"role": "user", "content": [{"type": "audio", "audio": a}] + [{"type": "text", "text": w} for w in ws]}]
            for a, ws in zip(audios, words, strict=True)
        ]
        inputs = processor.apply_chat_template(conversations, tokenize=True, return_dict=True).to(
            model.device, model.dtype
        )
        with torch.inference_mode():
            logits = model(**inputs).logits.float()
        ts = model.config.timestamp_token_id
        stamps = processor.decode_forced_alignment(logits, inputs["input_ids"], words, ts)
        out = []
        for i, items in enumerate(stamps):
            mask = inputs["input_ids"][i] == ts
            # Two timestamps (start, end) per word.
            confidence = logits[i][mask].softmax(-1).max(-1).values.reshape(-1, 2).mean(-1).tolist()
            out.append(
                Alignment(
                    words=[
                        Word(it["text"], it["start_time"], it["end_time"], c)
                        for it, c in zip(items, confidence, strict=True)
                    ],
                    score=worst_window(confidence, *SCORE_WINDOW["qwen3"]),
                )
            )
        return out


class CtcAligner:
    def __init__(self, model: str, device: str):
        from transformers import AutoModelForCTC, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(model)
        self.model = AutoModelForCTC.from_pretrained(model).to(device).eval()
        self.vocab = self.processor.tokenizer.get_vocab()
        self.blank = self.processor.tokenizer.pad_token_id

    def tokens(self, word: str) -> list[int]:
        out = []
        for char in word:
            for variant in (char, char.upper(), char.lower()):
                if variant in self.vocab:
                    out.append(self.vocab[variant])
                    break
        return out

    def align(self, audios: list[np.ndarray], words: list[list[str]]) -> list[Alignment]:
        """One padded batch (the service groups similar lengths); each clip's
        frames are cut back out by the model's own output-length rule."""
        import torch

        inputs = self.processor(
            audios, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True, return_attention_mask=True
        )
        mask = inputs.attention_mask.to(self.model.device)
        # fp16 on CUDA: much faster, and the alignments match fp32's.
        cuda = self.model.device.type == "cuda"
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=cuda):
            logits = self.model(inputs.input_values.to(self.model.device), attention_mask=mask).logits
            frames = self.model._get_feat_extract_output_lengths(mask.sum(-1)).tolist()
        out = []
        for i, (audio, ws) in enumerate(zip(audios, words, strict=True)):
            emission = logits[i, : frames[i]].float().log_softmax(-1)
            frame_s = len(audio) / SAMPLE_RATE / max(1, frames[i])
            out.append(ctc_alignment(emission, [self.tokens(w) for w in ws], ws, self.blank, frame_s))
        return out


def ctc_alignment(emission, tokens: list[list[int]], words: list[str], blank: int, frame_s: float) -> Alignment:
    """Align per-word token ids to a (frames, classes) log-probability matrix."""
    import torch
    import torchaudio.functional as F

    flat = [t for ts in tokens for t in ts]
    if not flat:
        return Alignment([Word(w, 0.0, 0.0, aligned=False) for w in words], 0.0)
    targets = torch.tensor([flat], dtype=torch.int32, device=emission.device)
    try:
        path, scores = F.forced_align(emission[None], targets, blank=blank)
    except RuntimeError:  # more tokens than frames: the text cannot be this audio
        return Alignment([Word(w, 0.0, 0.0, aligned=False) for w in words], 0.0)
    spans = F.merge_tokens(path[0].cpu(), scores[0].exp().cpu(), blank=blank)
    out, k, previous_end = [], 0, 0.0
    for word, ts in zip(words, tokens, strict=True):
        if not ts:
            out.append(Word(word, previous_end, previous_end, aligned=False))
            continue
        mine = spans[k : k + len(ts)]
        confidence = float(np.mean([s.score for s in mine]))
        out.append(Word(word, mine[0].start * frame_s, mine[-1].end * frame_s, confidence))
        previous_end = out[-1].end
        k += len(ts)
    return Alignment(out, worst_window([w.confidence for w in out if w.aligned], *SCORE_WINDOW["ctc"]))


def _batches(order: list[int], lengths: list[int]) -> list[list[int]]:
    """Consecutive runs of ``order`` (sorted by length) with at most BATCH_SIZE
    clips and BATCH_SECONDS of padded audio: a fixed count of long clips ran
    out of GPU memory (16 x 30 s in wav2vec2's attention)."""
    out: list[list[int]] = []
    for i in order:
        batch = out[-1] if out else None
        longest = max([lengths[j] for j in batch] + [lengths[i]]) if batch else lengths[i]
        if batch and len(batch) < BATCH_SIZE and (len(batch) + 1) * longest <= BATCH_SECONDS * SAMPLE_RATE:
            batch.append(i)
        else:
            out.append([i])
    return out


# (kind, model, device) -> backend
AlignLoader = Callable[[str, str, str], Backend]


def load_aligner(kind: str, model: str, device: str) -> Backend:
    return QwenAligner(model, device) if kind == "qwen3" else CtcAligner(model, device)


class AlignService:
    """Loads one backend at a time, on the ASR device, on first use."""

    def __init__(self, settings: Settings, loader: AlignLoader = load_aligner):
        self.settings = settings
        self._loader = loader
        self._lock = threading.Lock()
        self._kind: str | None = None
        self._backend: Backend | None = None

    @property
    def device(self) -> str:
        from voxcpm.model.utils import resolve_runtime_device

        device = self.settings.device if self.settings.asr_device == "auto" else self.settings.asr_device
        return resolve_runtime_device(device, "cuda")

    @property
    def loaded(self) -> bool:
        return self._backend is not None

    def model_for(self, kind: str) -> str:
        if kind not in ALIGNERS:
            raise ValueError(f"Unknown aligner '{kind}' (use {', '.join(ALIGNERS)}).")
        return self.settings.model_path("align_model" if kind == "qwen3" else "ctc_align_model")

    def align(self, audios: list[np.ndarray], texts: list[str], kind: str = "qwen3") -> list[Alignment]:
        """Align 16 kHz mono clips with their transcripts."""
        model = self.model_for(kind)
        words = [split_words(t) for t in texts]
        results: list[Alignment | None] = [None] * len(audios)
        order = sorted(range(len(audios)), key=lambda i: len(audios[i]))  # similar lengths per batch
        with self._lock:
            backend = self._ensure_loaded(kind, model)
            todo = [i for i in order if words[i]]
            for i in order:
                if not words[i]:
                    results[i] = Alignment([], 0.0)
            for idx in _batches(todo, [len(a) for a in audios]):
                aligned = backend.align([audios[i] for i in idx], [words[i] for i in idx])
                for i, a in zip(idx, aligned, strict=True):
                    results[i] = a
        return results  # type: ignore[return-value]

    def unload(self) -> None:
        with self._lock:
            self._unload()

    def _unload(self) -> None:
        if self._backend is None:
            return
        self._backend, self._kind = None, None
        gc.collect()
        if self.device.startswith("cuda"):
            import torch

            torch.cuda.empty_cache()

    def _ensure_loaded(self, kind: str, model: str) -> Backend:
        if self._kind != kind:
            self._unload()
            logger.info("Loading %s aligner %s on %s", kind, model, self.device)
            self._backend, self._kind = self._loader(kind, model, self.device), kind
        return self._backend  # type: ignore[return-value]


def _frame_db(wav: np.ndarray) -> np.ndarray:
    n = len(wav) // FRAME
    if n == 0:
        return np.full(1, -120.0)
    rms = np.sqrt(np.mean(wav[: n * FRAME].reshape(n, FRAME) ** 2, axis=1) + 1e-12)
    return 20 * np.log10(rms)


def edge_levels(wav: np.ndarray) -> tuple[float, float]:
    """dB of the first and last 10 ms relative to the clip's loud frames (p95)."""
    db = _frame_db(wav)
    loud = np.percentile(db, 95)
    return float(db[0] - loud), float(db[-1] - loud)


def loud_frames(wav: np.ndarray) -> np.ndarray:
    """Per 10 ms frame: within EDGE_DROP_DB of the clip's loud frames, and at
    least EDGE_FLOOR_DB above its noise floor (recordings with background noise
    or music have a floor not far below the voice)."""
    db = _frame_db(wav)
    loud, floor = np.percentile(db, 95), np.percentile(db, 5)
    threshold = max(loud - EDGE_DROP_DB, min(floor + EDGE_FLOOR_DB, loud - EDGE_FLOOR_DB))
    return db > threshold


# Audible sound around the speech (laughs, sighs, a quiet word VAD misses): within
# this many dB of the clip's loud frames, at least SOUND_MIN_FRAMES long (shorter
# blips are clicks). Quieter than this: breaths and room noise.
SOUND_DROP_DB = 30.0
SOUND_MIN_FRAMES = 5


def sound_spans(wav: np.ndarray) -> list[tuple[float, float]]:
    """(start, end) seconds of each stretch of audible sound, bridging quiet gaps
    shorter than EDGE_BRIDGE_FRAMES."""
    db = _frame_db(wav)
    loud, floor = np.percentile(db, 95), np.percentile(db, 5)
    audible = db > max(loud - SOUND_DROP_DB, min(floor + EDGE_FLOOR_DB, loud - EDGE_FLOOR_DB))
    to_s = FRAME / SAMPLE_RATE
    spans: list[tuple[float, float]] = []
    start = last = None
    for i in np.flatnonzero(audible):
        if start is not None and i - last > EDGE_BRIDGE_FRAMES + 1:
            if last + 1 - start >= SOUND_MIN_FRAMES:
                spans.append((start * to_s, (last + 1) * to_s))
            start = None
        if start is None:
            start = i
        last = i
    if start is not None and last + 1 - start >= SOUND_MIN_FRAMES:
        spans.append((start * to_s, (last + 1) * to_s))
    return spans


def refine_edges(
    wav: np.ndarray, start: float, end: float, inner: tuple[float, float] | None = None
) -> tuple[float, float]:
    """Snap a speech span's edges to where the sound starts and stops.

    Aligners place edges at token onsets (CTC: trailing vowels are left out) or
    on an 80 ms grid (qwen3: the first word sometimes "starts" at 0 in a
    silent lead-in). An edge in silence moves inward to the nearest loud frame,
    but not past ``inner`` (end of the first word, start of the last); then
    both edges move outward while the audio stays loud, bridging quiet
    stretches shorter than 60 ms. Resolution: 10 ms.
    """
    loud = loud_frames(wav)
    n = len(loud)
    to_s = FRAME / SAMPLE_RATE
    inner_start, inner_end = inner or (end, start)

    first = min(n - 1, max(0, int(start / to_s)))
    last = min(n - 1, max(0, int(np.ceil(end / to_s)) - 1))
    if not loud[first]:
        ahead = np.flatnonzero(loud[first : max(first + 1, int(np.ceil(inner_start / to_s)))])
        first += int(ahead[0]) if len(ahead) else 0
    if not loud[last]:
        lo = min(last, max(0, int(inner_end / to_s)))
        behind = np.flatnonzero(loud[lo : last + 1])
        last = lo + int(behind[-1]) if len(behind) else last

    f, quiet = first, 0
    while f > 0 and quiet < EDGE_BRIDGE_FRAMES:
        f -= 1
        first, quiet = (f, 0) if loud[f] else (first, quiet + 1)
    f, quiet = last, 0
    while f < n - 1 and quiet < EDGE_BRIDGE_FRAMES:
        f += 1
        last, quiet = (f, 0) if loud[f] else (last, quiet + 1)
    duration = len(wav) / SAMPLE_RATE
    return first * to_s, min(duration, (last + 1) * to_s)
