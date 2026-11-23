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
import sys
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from ... import vram
from ...config import Settings
from ...lora_library import VOICE_FILE
from ..base import (
    Capabilities,
    EditRequest,
    Engine,
    Param,
    ReferenceInput,
    SynthesisRequest,
    SynthesisResult,
    TagGroup,
)

logger = logging.getLogger(__name__)

ENGINE = "fish_drama3"
MIN_GB = 11.0  # a 12 GB card (fewer experts cached: slower)
MIN_RAM_GB = 30.0  # the experts in pinned memory, the rest while loading (a 32 GB computer)
RESERVE_GB = 2.5  # left free after the expert caches: prompt pass, codec decode
GUIDANCE = 0.0  # off: a little closer to the voice by a speaker model's score, not in blind listening
FRAMES_PER_CHAR = 4.5  # first guess (21.5 frames per second of audio); then learned
MIN_FRAMES = 10
LEARN_MIN_CHARS = 10
MAX_FRAMES = 1024  # per part of the text (~48 s)
CAP_FACTOR = 3.0  # a take longer than this x the expected frames has run away
CAP_MIN = 96  # frames (~4.5 s): short texts vary most
# A long text is spoken in parts, each a take of its own from the references: within a take the
# voice drifts from them as the take grows (most on voices the model clones less well).
PART_BYTES = 300  # a part's most UTF-8 bytes (~100 Chinese characters); longer sentences are cut
TARGET_BYTES = 150  # what parts aim for (~50 Chinese characters, ~10 s), in whole sentences
SENTENCE_ENDS = frozenset(".!?;。！？；…\n")
CLAUSE_MARKS = frozenset(",:，：、")
BLANK_LINE = re.compile(r"\n(?: ?\n)+")
# Never cut inside: special tokens, a tag pair and what it holds, [bracketed] tags.
PROTECTED = re.compile(r"<\|[^|>]*\|>|<([A-Za-z][\w-]*)\b[^>]*>.*?</\1>|\[[^\[\]]*\]", re.S)
KEPT_CLIPS = 4  # reference clips whose codes are kept
EDIT_SAMPLING = (0.7, 0.7, 30)  # temperature, top-p, top-k: the report's evaluation settings
EDIT_CAP_MIN = 48  # frames an edit's new part may run to at least (~2.2 s)
# A new part this many times longer than expected has run on: in test edits right ones came out
# at 0.6-1.4x, run-ons at 1.8x and more. Whether a span runs on depends on the clip (its pauses,
# the aligned word edges), not on which model made it.
EDIT_RUNON = 1.6
EDIT_TRIES = ((0, 0), (1, 0), (1, 1))  # (more words after the span, seed offset) per try
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


def spoken_parts(text: str, limit: int = PART_BYTES, target: int = TARGET_BYTES) -> list[str]:
    """The parts a text is spoken in, one take each. A blank line always starts a new part (the
    writer's own cut); between blank lines: whole sentences (a line break ends one too), in parts
    of about ``target`` UTF-8 bytes and of even length (no short last part). Only a sentence over
    ``limit`` bytes is cut: at clause marks, then spaces, then anywhere. Tags, tagged spans
    ("<whisper>...</whisper>", kept whole even when longer) and decimals are never cut."""
    text = re.sub(r"[^\S\n]+", " ", text.replace("\r\n", "\n")).strip()
    protected = [match.span() for match in PROTECTED.finditer(text)]
    blocks, last = [], 0
    for match in BLANK_LINE.finditer(text):
        if not any(start < match.end() and match.start() < end for start, end in protected):
            blocks.append(text[last : match.start()])
            last = match.end()
    blocks.append(text[last:])
    return [part for block in blocks for part in _block_parts(block.strip(), limit, target)]


def speakable(text: str) -> str:
    """``text`` without the symbols the model can't read (ⓘ, ★, emoji, private-use characters): a
    styled or enclosed letter or digit (ⓘ, ①, 𝐀) becomes that letter, any other is left out. With
    such a symbol in its text, a take could speak the reference's transcript instead."""
    out = []
    for char in text:
        kind = unicodedata.category(char)
        if kind in ("So", "Co", "Cn", "No"):
            plain = unicodedata.normalize("NFKC", char)
            if plain != char and plain.isalnum():
                out.append(plain)
            elif kind == "No":  # (other numbers such as ½ stay as they are)
                out.append(char)
            continue
        out.append(char)
    return re.sub(r"[^\S\n]{2,}", " ", "".join(out)).strip()


def part_ends(text: str, parts: list[str]) -> list[int]:
    """Where each of ``spoken_parts(text)`` ends in ``text`` itself (the parts hold its characters
    in order, with other white space between them)."""
    ends, at = [], 0
    for part in parts:
        left = sum(not char.isspace() for char in part)
        while left and at < len(text):
            left -= not text[at].isspace()
            at += 1
        ends.append(at)
    return ends


def _block_parts(text: str, limit: int, target: int) -> list[str]:
    if not text:
        return []
    target = min(target, limit)
    count = max(1, round(utf8_bytes(text) / target))
    if count == 1 and utf8_bytes(text) <= limit:
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
    # Parts end at the piece ends nearest to even shares of the whole text.
    share = sum(utf8_bytes(piece) for piece in pieces) / count
    parts, current, done = [], "", 0
    for piece in pieces:
        more, end = utf8_bytes(piece), share * (len(parts) + 1)
        if current and (utf8_bytes(current) + more > limit or done + more - end > end - done):
            parts.append(current)
            current = ""
        current += piece
        done += more
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


def release_cache() -> None:
    """The GPU blocks a request freed back to the driver: the expert slots fill the card, and
    blocks PyTorch keeps cached between requests left new sizes no room ("allocation failed with
    OOM", then a retry after emptying the cache, mid-request)."""
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def frame_cap(expected: int) -> int:
    return min(MAX_FRAMES, max(CAP_MIN, round(expected * CAP_FACTOR)))


class Runtime:
    """The loaded model and codec."""

    def __init__(self, model_dir: Path, device: str, graphs: bool = True):
        from ..common import fish_codec as codec
        from . import model

        t0 = time.time()
        self.device = device
        # The experts (most of the bytes) and the codec load in the background of the rest.
        experts = model.read_experts(model_dir, device)
        with ThreadPoolExecutor(1, thread_name_prefix="drama3-codec") as pool:
            codec_job = pool.submit(codec.load, model_dir / "codec.pth", device)

            def budget(per_expert: int) -> int:
                codec_job.result()  # (on the GPU before its free memory is shared out)
                return self._expert_slots(per_expert)

            self.model = model.load(model_dir, device, budget, graphs, experts=experts)
            self.codec = codec_job.result()
        self._codes: dict[str, object] = {}  # recent reference clips' codes, by content
        logger.info("Drama 3 ready in %.1f s", time.time() - t0)

    def _expert_slots(self, per_expert: int) -> int:
        """GPU slots per layer for the experts: what the card has left, less the reserve."""
        import torch

        device = self.device
        limit = ((vram.total_gb(device) or 0) - vram.CONTEXT_GB) * 2**30 - torch.cuda.memory_reserved(device)
        free = min(torch.cuda.mem_get_info(device)[0], limit)
        return int((free - RESERVE_GB * 2**30) // per_expert)

    def encode(self, wav: Path):
        """The clip's codes; the same audio again (each request's upload is a new file) is not
        encoded again, and so keeps the model's kept states for it (model.prefill)."""
        import hashlib

        key = hashlib.sha1(Path(wav).read_bytes()).hexdigest()
        if key in self._codes:
            return self._codes[key]
        codes = self.codec.encode(self.mono(wav)).cpu()
        self._codes = {**dict(list(self._codes.items())[-(KEPT_CLIPS - 1) :]), key: codes}
        return codes

    def mono(self, wav: Path):
        """The clip as mono at the codec's rate, on the GPU."""
        from ...data import audio_io

        return audio_io.mono_tensor(wav, self.codec.sample_rate, self.device)


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
        voice_reference=True,  # its voices: saved references (common.reference.create_voice)
        voice_reference_clips=3,
        reference_voices=True,
        # Its references carry the voice's pitch and pace: chosen by output, they cloned far
        # better on a mixed dataset than the cleanest transcripts.
        reference_search=True,
        # A clip with its transcript (several are joined into one).
        reference_inputs=(ReferenceInput("reference", "prompt", "required"),),
        speech_edit=True,  # edit.py: changed words spoken again by infilling
        takes=True,
    )
    params = (
        Param("temperature", "float", 1.0, 0.1, 1.5, 0.05),
        Param("top_p", "float", 0.9, 0.1, 1.0, 0.05),
        Param("top_k", "int", 30, 1, 100, 1),
        # How much harder a take leans on its reference (model.guided); 0: off. Above 0 each
        # frame takes the backbone twice.
        Param("guidance", "float", GUIDANCE, 0.0, 1.0, 0.05),
        Param("language", "choice", "auto", choices=LANGUAGES),  # a hint: the main language
        Param("duration", "float", 0.0, 0.0, 60.0, 0.5),  # seconds, for a one-line text; 0 = its own
    )
    model_setting = "drama3_model"
    checkpoint_files = (VOICE_FILE,)  # its voices are saved references (common.reference.create_voice)

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

    @staticmethod
    def _parts(text: str, cloning: bool) -> list[str]:
        # Without a reference there is no voice to stay close to, and every take picks its own:
        # then only what is too long for one take is cut.
        return spoken_parts(text, target=TARGET_BYTES if cloning else PART_BYTES)

    def takes(self, text: str, cloning: bool) -> list[int]:
        return part_ends(text, self._parts(text, cloning))

    def synthesize(self, req: SynthesisRequest) -> SynthesisResult:
        import torch

        from . import model, prompt

        cloning = bool(req.prompt_wav or req.reference_wav or req.lora_id)
        parts = [spoken for part in self._parts(req.text, cloning) if (spoken := speakable(part))]
        if not parts:
            raise ValueError("Text to synthesize is empty (or only symbols that aren't spoken).")
        for part in parts:
            prompt.check(part)
        style = re.sub(r"\s+", " ", req.style or "").strip().strip("[]")
        if style:  # a style prefix on every part
            parts = [f"[{style}] {part}" for part in parts]
        if reason := self.unavailable():
            raise ValueError(f"Fish Audio Drama 3 can't run: {reason}")
        clip = req.prompt_wav or req.reference_wav
        transcript = (req.prompt_text or "").strip()
        if req.lora_id:  # a saved voice: its reference, unless the request gives a clip
            from ... import lora_library
            from ..common.reference import read_reference

            voice = lora_library.resolve(self.settings.lora_dir, req.lora_id)
            if voice.engine != ENGINE:
                raise ValueError(f"'{req.lora_id}' is not a Fish Audio Drama 3 voice.")
            if clip is None and (own := read_reference(voice.path)):
                clip, transcript = own[0], own[1].strip()
            if clip is None:
                raise ValueError("This voice has no reference clip: add one in the Voices panel.")
        if clip is not None and not transcript:
            raise ValueError("Fish Audio Drama 3 needs the reference clip's transcript.")
        options = self.options(req.options)
        guidance = float(options["guidance"]) if clip is not None else 0.0  # nothing to lean on without one
        sampling = model.Sampling(
            float(options["temperature"]), float(options["top_p"]), int(options["top_k"]), guidance
        )
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
                    # (guidance: the same turn as the model reads it with no reference)
                    more = {"plain": prompt.build(m.tokenizer, part, [], m.semantic_begin, hints)} if guidance else {}
                    for attempt in range(2):  # a take that runs to its cap is drawn again once
                        self._progress.attempt = attempt
                        start = self._progress.step
                        torch.manual_seed(seed + attempt)
                        codes = m.generate(rows, sampling, cap, self._count, head=len(head), **more)
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
                release_cache()

    def edit(self, req: EditRequest) -> SynthesisResult:
        import torch

        from . import edit, model, prompt

        text, new_text = edit.untagged(req.text), req.new_text.strip()
        if not text or not new_text:
            raise ValueError("Give the clip's transcript and the new text.")
        prompt.check(new_text)
        if reason := self.unavailable():
            raise ValueError(f"Fish Audio Drama 3 can't run: {reason}")
        words = req.align(req.audio, text)  # before taking the model: the aligner runs elsewhere
        with self._lock:
            self._progress = Progress(phase="loading" if self._runtime is None else "preparing", started=time.time())
            try:
                rt = self._ensure_runtime()
                m = rt.model
                codes = rt.encode(req.audio)
                if edit.plan(text, edit.untagged(new_text), words, codes.shape[1]) is None:
                    raise ValueError("The new text has the same words as the clip.")
                seed = req.seed if req.seed is not None else random.randrange(2**31)
                decoder = getattr(m, "decoder", None)
                self._progress.phase = "generating" if decoder is None or decoder.warm else "compiling"
                # A fill that runs on (says the words after the gap as well) comes out much longer
                # than expected: then again with one more word in the span, which mostly fixes it.
                for attempt, (wider, draw) in enumerate(EDIT_TRIES):
                    span = edit.plan(text, edit.untagged(new_text), words, codes.shape[1], wider)
                    assert span is not None
                    before, after = codes[:, : span.start], codes[:, span.end :]
                    rows = prompt.infill(m.tokenizer, new_text, before, after, m.semantic_begin)
                    old, new = (len(re.sub(r"\W", "", s)) for s in (span.old, span.new))
                    frames = span.end - span.start
                    want = round(frames * new / old) if old and frames else round(new * self._frames_per_char)
                    cap = max(EDIT_CAP_MIN, round(want * CAP_FACTOR))
                    self._progress.expected, self._progress.attempt, self._progress.step = want, attempt, 0
                    torch.manual_seed(seed + draw)
                    middle = m.generate(rows, model.Sampling(*EDIT_SAMPLING), cap, self._count)
                    if middle.shape[1] <= EDIT_RUNON * max(1, want) and middle.shape[1] < cap:
                        break
                    logger.warning("Drama 3: an edit ran on (%d frames for ~%d)", middle.shape[1], want)
                else:
                    raise ValueError(
                        "Fish Audio Drama 3 kept speaking past the edit: try changing a few more words around it."
                    )
                decoded = rt.codec.decode(torch.cat([before, middle, after], 1).to(rt.device)).float().cpu().numpy()
                original = rt.mono(req.audio).float().cpu().numpy()
                audio = edit.splice(original, decoded, span, middle.shape[1])
                seconds = (span.start * edit.FRAME / edit.RATE, (span.start + middle.shape[1]) * edit.FRAME / edit.RATE)
                return SynthesisResult(rt.codec.sample_rate, audio, seed, span=seconds)
            finally:
                self._progress.phase = "idle"
                release_cache()

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
            # Module-level buffers (a layer's streamed experts on the GPU, the pinned staging), there
            # once the real model ran.
            if model := sys.modules.get(f"{__package__}.model"):
                model.release_buffers()
            gc.collect()
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
