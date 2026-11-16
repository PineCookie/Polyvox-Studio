"""Drama 3's text-to-speech prompt ("reference-bracket-v1", as sglang-omni builds it).

Rows of 11 ids: the token, then the frame's 10 codes (zeros on text rows). A reference clip is
"[Reference N]\\nText: <|speaker:0|> transcript\\nSpeech: " and its frames; each frame's token is
its semantic token (semantic_begin + code 0).

Directions and hints are plain text in the line (the Drama 3 technical report's Table 10): point
tags "[sighs]", scoped tags "<whisper>...</whisper>", a style prefix "[with bright eager
excitement] line", "[language: en]" and "[audio tokens: N]" (a length in frames) at the end.
(The report's line-separated turns, lines joined by <|audio_separator|>, don't work with the
released weights: they never emit the separator, and such takes tend not to end.)

Speech editing (infill) follows the report's Fig. 14, without its optional duration hint: with it,
Chinese edits tended to run on past the gap.
"""

from __future__ import annotations

TTS = "convert the provided text to speech"
TTS_REF = "convert the provided text to speech reference to the following:"
INFILL = (
    "fill in the missing part of the speech; the suffix follows <|fim_suffix|> and the prefix follows <|fim_prefix|>"
)
SPEAKER = "<|speaker:0|>"
# Written by the builder only: text can't hold them.
STRUCTURAL = ("<|im_start|>", "<|im_end|>", "<|voice|>", "<|text|>", "<|interleave|>", "<|semantic:",
              "<|audio_separator|>", "<|fim_", "<|endofturn|>")  # fmt: skip


def speaker_tagged(text: str) -> str:
    text = text.strip()
    return text if text.startswith("<|speaker:") else f"{SPEAKER} {text}"


def check(text: str) -> str:
    """``text`` if it holds none of the prompt's own markers."""
    for token in STRUCTURAL:
        if token in text:
            raise ValueError(f"The text can't contain {token}.")
    return text


def build(tokenizer, text: str, references, semantic_begin: int, hints: str = ""):
    """[rows, 11] int64 for ``text`` cloned from ``references`` [(transcript, codes [10, T])]."""
    import torch

    return torch.cat(parts(tokenizer, text, references, semantic_begin, hints))


def user_text(text: str, hints: str = "") -> str:
    """A user turn's text: speaker-tagged, ``hints`` after it."""
    return speaker_tagged(check(text)) + (f" {hints.strip()}" if hints.strip() else "")


def parts(tokenizer, text: str, references, semantic_begin: int, hints: str = ""):
    """build's rows in two: the part that depends only on the references (the system turn and
    the user turn's start: the same for every text a voice speaks), then the text's (``hints``
    after it)."""
    import torch

    head: list = ["<|im_start|>system\n", TTS_REF if references else TTS]
    for i, (transcript, codes) in enumerate(references):
        head += [f"\n\n[Reference {i + 1}]\nText: {speaker_tagged(check(transcript))}\nSpeech: ", codes]
    head += ["<|im_end|>\n", "<|im_start|>user\n"]
    tail = [user_text(text, hints), "<|im_end|>\n", "<|im_start|>assistant\n<|voice|>"]
    return torch.cat(_rows(tokenizer, head, semantic_begin)), torch.cat(_rows(tokenizer, tail, semantic_begin))


def infill(tokenizer, text: str, before, after, semantic_begin: int):
    """[rows, 11] for speech editing (the report's Fig. 14): ``text`` is the clip's whole new
    transcript, ``before`` / ``after`` [10, T] the audio kept on each side; the model speaks what
    goes between them. The suffix comes first, so the prefix sits next to the new audio."""
    import torch

    segments = ["<|im_start|>system\n", INFILL, "<|im_end|>\n", "<|im_start|>user\n", speaker_tagged(check(text)) + "\n",
                "<|fim_suffix|>", after, "<|fim_prefix|>", before, "<|im_end|>\n", "<|im_start|>assistant\n<|voice|>"]  # fmt: skip
    return torch.cat(_rows(tokenizer, segments, semantic_begin))


def _rows(tokenizer, segments, semantic_begin: int):
    import torch

    rows = []
    for s in segments:
        if isinstance(s, str):
            ids = torch.tensor(tokenizer.encode(s).ids, dtype=torch.long)
            r = torch.zeros(len(ids), 11, dtype=torch.long)
            r[:, 0] = ids
        else:
            c = s.long()
            r = torch.cat([c[:1] + semantic_begin, c]).T
        rows.append(r)
    return rows
