"""Fish Audio Drama 3 as an engine: zero-shot cloning from a reference clip and its transcript.

model.py and prompt.py follow sglang-omni's implementation (Apache-2.0); the weights are the
official release (fishaudio/drama-3), converted to 4 bits after the download (convert.py), under
the Fish Audio Community License: free for research, non-commercial use and for businesses below
USD 500k yearly revenue; no hosted services.

The model is a 35B mixture of experts with 3B active per token: everything but its routed
experts (about 6 GB with the codec) stays on the GPU, the experts (16.6 GB) in pinned CPU
memory with as many of them cached on the GPU as fit. It runs on a 12 GB card, faster on larger
ones; the computer needs about 32 GB of RAM.
"""

from __future__ import annotations

import gc
import logging
import random
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ... import vram
from ...config import Settings
from ..base import Capabilities, Engine, Param, ReferenceInput, SynthesisRequest, SynthesisResult, TagGroup

logger = logging.getLogger(__name__)

ENGINE = "fish_drama3"
MIN_GB = 11.0  # a 12 GB card (fewer experts cached: slower)
MIN_RAM_GB = 30.0  # the experts in pinned memory, the rest while loading (a 32 GB computer)
RESERVE_GB = 2.5  # left free after the expert caches: prompt pass, codec decode
FRAMES_PER_CHAR = 4.5  # first guess (21.5 frames per second of audio); then learned
MIN_FRAMES = 10
LEARN_MIN_CHARS = 10
MAX_FRAMES = 1024  # per part of the text (~48 s)
CAP_FACTOR = 3.0  # a take longer than this x the expected frames has run away
CAP_MIN = 96  # frames (~4.5 s): short texts vary most
PART_BYTES = 300  # longer texts are spoken in parts of about this many UTF-8 bytes (~100 Chinese characters)
SENTENCE_ENDS = frozenset(".!?;。！？；…\n")
CLAUSE_MARKS = frozenset(",:，：、")
# Never cut inside: special tokens, a tag pair and what it holds, [bracketed] tags.
PROTECTED = re.compile(r"<\|[^|>]*\|>|<([A-Za-z][\w-]*)\b[^>]*>.*?</\1>|\[[^\[\]]*\]", re.S)
KEPT_CLIPS = 4  # reference clips whose codes are kept
FRAMES_PER_SECOND = 44_100 / 2048  # the codec's
# The report's examples (Table 10) and common sounds; any words in brackets work.
TAG_GROUPS = (
    TagGroup("sounds", ("[sighs]", "[laughs]", "[gasps]", "[clears throat]", "[pause]", "[hesitates]", "[crying]", "[whispering]")),
    TagGroup("scoped", ("<whisper></whisper>", "<slow></slow>", "<emphasis></emphasis>")),
)  # fmt: skip
LANGUAGES = ("auto", "zh", "en", "ja", "ko", "yue", "de", "fr", "es", "it", "pt", "ru")


def utf8_bytes(text: str) -> int:
    return len(text.encode("utf-8"))


def _units(text: str) -> list[tuple[str, bool]]:
    """The text as (piece, protected): tags, spans in a tag pair and special tokens are protected."""
    units, last = [], 0
    for match in PROTECTED.finditer(text):
        if match.start() > last:
            units.append((text[last : match.start()], False))
        units.append((match.group(0), True))
        last = match.end()
    if last < len(text):
        units.append((text[last:], False))
    return units


def _cut(units: list[tuple[str, bool]], marks: frozenset[str], longer_than: int) -> list[tuple[str, bool]]:
    """Free pieces longer than ``longer_than`` bytes cut after each mark (kept with the piece
    before; a point between digits isn't a sentence end)."""
    out = []
    for piece, protected in units:
        if protected or utf8_bytes(piece) <= longer_than:
            out.append((piece, protected))
            continue
        start = 0
        for i, char in enumerate(piece):
            decimal = char == "." and 0 < i < len(piece) - 1 and piece[i - 1].isdigit() and piece[i + 1].isdigit()
            if char in marks and not decimal:
                out.append((piece[start : i + 1], False))
                start = i + 1
        if start < len(piece):
            out.append((piece[start:], False))
    return out


def spoken_parts(text: str, limit: int = PART_BYTES) -> list[str]:
    """The parts a text is spoken in, one take each, as Fish's own server cuts them: up to about
    ``limit`` UTF-8 bytes, cut at sentence ends (a line break too), a longer sentence at clause
    marks, then spaces, then anywhere. Tags, tagged spans ("<whisper>...</whisper>", kept whole
    even when longer) and decimals are never cut."""
    text = re.sub(r"[^\S\n]+", " ", text.replace("\r\n", "\n")).strip()
    if not text:
        return []
    if utf8_bytes(text) <= limit:
        return [text]
    units = _cut(_units(text), SENTENCE_ENDS, 0)
    for marks in (CLAUSE_MARKS, frozenset(" ")):
        units = _cut(units, marks, limit)
    pieces = []
    for piece, protected in units:  # what is still too long: on character boundaries
        while not protected and utf8_bytes(piece) > limit:
            cut = len(piece.encode("utf-8")[:limit].decode("utf-8", "ignore"))
            pieces.append(piece[:cut])
            piece = piece[cut:]
        pieces.append(piece)
    parts, current = [], ""
    for piece in pieces:
        if current and utf8_bytes(current + piece) > limit:
            parts.append(current)
            current = ""
        current += piece
    parts.append(current)
    return [p.strip() for p in parts if p.strip()]


@dataclass
class Progress:
    phase: str = "idle"  # idle | loading | preparing | compiling | generating
    step: int = 0  # audio frames so far
    expected: int | None = None
    attempt: int = 0
    started: float = 0.0


def ram_gb() -> float | None:
    """The computer's memory in GB (None if unknown)."""
    import os
    import sys

    try:
        if sys.platform == "win32":
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                            ("avail", ctypes.c_ulonglong), ("page_total", ctypes.c_ulonglong),
                            ("page_avail", ctypes.c_ulonglong), ("virtual_total", ctypes.c_ulonglong),
                            ("virtual_avail", ctypes.c_ulonglong), ("extended", ctypes.c_ulonglong)]  # fmt: skip

            status = MemoryStatus(length=ctypes.sizeof(MemoryStatus))
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return None
            return status.total / 2**30
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    except (AttributeError, OSError, ValueError):
        return None


def frame_cap(expected: int) -> int:
    return min(MAX_FRAMES, max(CAP_MIN, round(expected * CAP_FACTOR)))


class Runtime:
    """The loaded model and codec."""

    def __init__(self, model_dir: Path, device: str, graphs: bool = True):
        import torch

        from ..fish_s2 import codec
        from . import model

        t0 = time.time()
        self.device = device
        self.codec = codec.load(model_dir / "codec.pth", device)

        def budget(per_expert: int) -> int:
            limit = ((vram.total_gb(device) or 0) - vram.CONTEXT_GB) * 2**30 - torch.cuda.memory_reserved(device)
            free = min(torch.cuda.mem_get_info(device)[0], limit)
            return int((free - RESERVE_GB * 2**30) // per_expert)

        self.model = model.load(model_dir, device, budget, graphs)
        self._codes: dict[str, object] = {}  # recent reference clips' codes, by content
        logger.info("Drama 3 ready in %.1f s", time.time() - t0)

    def encode(self, wav: Path):
        """The clip's codes; the same audio again (each request's upload is a new file) is not
        encoded again, and so keeps the model's kept states for it (model.prefill)."""
        import hashlib

        import torch
        import torchaudio

        from ...data import audio_io

        key = hashlib.sha1(Path(wav).read_bytes()).hexdigest()
        if key in self._codes:
            return self._codes[key]
        audio, rate = audio_io.read(wav)
        mono = torch.from_numpy(np.asarray(audio.mean(axis=0) if audio.ndim == 2 else audio, dtype=np.float32))
        mono = torchaudio.functional.resample(mono.to(self.device), rate, self.codec.sample_rate)
        codes = self.codec.encode(mono).cpu()
        self._codes = {**dict(list(self._codes.items())[-(KEPT_CLIPS - 1) :]), key: codes}
        return codes


class FishDrama3Engine(Engine):
    id = ENGINE
    name = "Fish Audio Drama 3"
    capabilities = Capabilities(
        sample_rate=44_100,
        training_rate=44_100,
        languages=("zh", "en", "ja", "ko"),
        reference_audio=True,
        reference_transcript=True,
        style=True,  # a style prefix on each line: "[with bright eager excitement] ..."
        tags=True,
        # Point tags at a word, scoped tags around a span; any wording works (the report's Table 10).
        tag_groups=TAG_GROUPS,
        free_tags=True,
        lora_hot_swap=False,
        training=False,
        # A clip with its transcript (several are joined into one).
        reference_inputs=(ReferenceInput("reference", "prompt", "required"),),
    )
    params = (
        Param("temperature", "float", 1.0, 0.1, 1.5, 0.05),
        Param("top_p", "float", 0.9, 0.1, 1.0, 0.05),
        Param("top_k", "int", 30, 1, 100, 1),
        Param("language", "choice", "auto", choices=LANGUAGES),  # a hint: the main language
        Param("duration", "float", 0.0, 0.0, 60.0, 0.5),  # seconds, for a one-line text; 0 = its own
    )
    model_setting = "drama3_model"

    def __init__(self, settings: Settings, loader=Runtime):
        self.settings = settings
        self._loader = loader
        self._lock = threading.Lock()
        self._runtime: Runtime | None = None
        self._progress = Progress()
        self._frames_per_char = FRAMES_PER_CHAR

    @property
    def device(self) -> str:
        return "cuda"

    @property
    def loaded(self) -> bool:
        return self._runtime is not None

    def unavailable(self) -> str | None:
        if not self.settings.model_path(self.model_setting):
            return "download it first (Models: Fish Audio Drama 3)."
        gb = vram.total_gb(self.device)
        if gb is None:
            return "it needs a CUDA GPU."
        if gb < MIN_GB:
            return f"it needs a 12 GB GPU (this one has {gb:.0f} GB)."
        ram = ram_gb()
        if ram is not None and ram < MIN_RAM_GB:
            return f"it needs 32 GB of memory (this computer has {ram:.0f} GB)."
        return None

    def status(self) -> dict:
        return {"device": self.device, "model": self.settings.model_path(self.model_setting), "loaded": self.loaded}

    def progress(self) -> dict:
        p = self._progress
        return {**p.__dict__, "elapsed": round(time.time() - p.started, 1) if p.started else 0.0}

    def synthesize(self, req: SynthesisRequest) -> SynthesisResult:
        import torch

        from . import model, prompt

        parts = spoken_parts(req.text)
        if not parts:
            raise ValueError("Text to synthesize is empty.")
        for part in parts:
            prompt.check(part)
        style = re.sub(r"\s+", " ", req.style or "").strip().strip("[]")
        if style:  # a style prefix on every part
            parts = [f"[{style}] {part}" for part in parts]
        if req.lora_id:
            raise ValueError("Fish Audio Drama 3 has no trained voices yet.")
        if reason := self.unavailable():
            raise ValueError(f"Fish Audio Drama 3 can't run: {reason}")
        clip = req.prompt_wav or req.reference_wav
        transcript = (req.prompt_text or "").strip()
        if clip is not None and not transcript:
            raise ValueError("Fish Audio Drama 3 needs the reference clip's transcript.")
        options = self.options(req.options)
        sampling = model.Sampling(float(options["temperature"]), float(options["top_p"]), int(options["top_k"]))
        language = str(options["language"])
        seconds = float(options["duration"])
        timed = seconds > 0 and len(parts) == 1  # a length only for a text spoken in one take
        hints = f"[language: {language}]" if language != "auto" else ""
        if timed:
            hints += f" [audio tokens: {round(seconds * FRAMES_PER_SECOND)}]"

        with self._lock:
            self._progress = Progress(phase="loading" if self._runtime is None else "preparing", started=time.time())
            try:
                rt = self._ensure_runtime()
                m = rt.model
                references = [(transcript, rt.encode(clip))] if clip is not None else []
                seed = req.seed if req.seed is not None else random.randrange(2**31)
                # Characters that are spoken (tags and directions aren't).
                chars = [len(re.sub(r"\W", "", re.sub(r"\[[^\]]*\]|<[^>]*>", "", part))) for part in parts]
                expected = [max(MIN_FRAMES, round(c * self._frames_per_char)) for c in chars]
                if timed:
                    expected = [max(MIN_FRAMES, round(seconds * FRAMES_PER_SECOND))]
                self._progress.expected = sum(expected)
                # The first frames of a process wait for the frame graph to be recorded (Triton compiles).
                decoder = getattr(m, "decoder", None)
                self._progress.phase = "generating" if decoder is None or decoder.warm else "compiling"
                takes = []
                # Each part is cloned from the references alone: with the earlier parts' takes in
                # the prompt as well (as Fish's server does), the voice drifted part by part.
                for part, want in zip(parts, expected, strict=True):
                    cap = frame_cap(want)
                    head, tail = prompt.parts(m.tokenizer, part, references, m.semantic_begin, hints)
                    rows = torch.cat((head, tail))
                    for attempt in range(2):  # a take that runs to its cap is drawn again once
                        self._progress.attempt = attempt
                        start = self._progress.step
                        torch.manual_seed(seed + attempt)
                        codes = m.generate(rows, sampling, cap, self._count, head=len(head))
                        if codes.shape[1] < cap:
                            break
                        logger.warning("Drama 3: a take ran to its cap (%d frames for ~%d); drawing again", cap, want)
                        self._progress.step = start
                    takes.append(codes)
                codes = torch.cat(takes, 1)
                if codes.shape[1] == 0:
                    raise ValueError("Fish Audio Drama 3 made no audio for this text.")
                audio = rt.codec.decode(codes.to(rt.device)).float().cpu().numpy()
                if sum(chars) >= LEARN_MIN_CHARS and not timed:
                    self._frames_per_char = 0.7 * self._frames_per_char + 0.3 * codes.shape[1] / sum(chars)
                return SynthesisResult(rt.codec.sample_rate, audio, seed)
            finally:
                self._progress.phase = "idle"

    def _count(self) -> None:
        self._progress.phase = "generating"
        self._progress.step += 1

    def _ensure_runtime(self) -> Runtime:
        if self._runtime is None:
            graphs = self.settings.optimize is not False
            self._runtime = self._loader(Path(self.settings.model_path(self.model_setting)), self.device, graphs)
        return self._runtime

    def training_config(self, params, run_dir, train_manifest, val_manifest) -> dict:
        raise ValueError("Fish Audio Drama 3 can't be trained here yet.")

    def unload(self) -> None:
        with self._lock:
            if self._runtime is None:
                return
            self._runtime = None
            gc.collect()
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
