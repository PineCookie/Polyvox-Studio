"""S2-Pro's prompt: a chat of system (instruction, the reference's transcript and
codes), user (the text) and an open assistant turn in voice mode.

Each text piece is tokenized on its own (as the model saw it in training); audio
frames sit in the token row as their semantic token, with the frame's codes in
the rows below.
"""

from __future__ import annotations

import torch
from torch import Tensor

INSTRUCTION = "convert the provided text to speech"
INSTRUCTION_WITH_REFERENCE = "convert the provided text to speech reference to the following:\n\nText:\n"
BEFORE_SPEECH = "\n\nSpeech:\n"
SPEAKER = "<|speaker:0|>"


class Prompt:
    """Builds the token rows [codebooks + 1, T] for one generation."""

    def __init__(self, tokenizer, codebooks: int, semantic_begin: int):
        self.tokenizer = tokenizer
        self.codebooks = codebooks
        self.semantic_begin = semantic_begin
        self.rows: list[Tensor] = []

    def text(self, value: str) -> Prompt:
        ids = self.tokenizer.encode(value, add_special_tokens=False)
        rows = torch.zeros(self.codebooks + 1, len(ids), dtype=torch.long)
        rows[0] = torch.tensor(ids, dtype=torch.long)
        self.rows.append(rows)
        return self

    def audio(self, codes: Tensor) -> Prompt:
        """codes: [codebooks, T] (the first row the semantic codes)."""
        codes = codes.long().cpu()
        self.rows.append(torch.cat([codes[:1] + self.semantic_begin, codes]))
        return self

    def turn(self, role: str, voice: bool = False) -> Prompt:
        return self.text(f"<|im_start|>{role}\n" + ("<|voice|>" if voice else ""))

    def end(self) -> Prompt:
        return self.text("<|im_end|>\n")

    def build(self) -> Tensor:
        return torch.cat(self.rows, 1)


def build(
    tokenizer,
    codebooks: int,
    semantic_begin: int,
    text: str,
    reference: tuple[Tensor, str] | None = None,
    history: list[tuple[str, Tensor]] = (),
) -> Tensor:
    """Rows for speaking ``text``; ``reference``: (codes, transcript) of the voice to
    clone; ``history``: earlier (text, codes) pairs of a long text, so its parts
    sound alike."""
    p = Prompt(tokenizer, codebooks, semantic_begin).turn("system")
    if reference is not None:
        codes, transcript = reference
        p.text(INSTRUCTION_WITH_REFERENCE).text(SPEAKER + transcript.strip()).text(BEFORE_SPEECH).audio(codes)
    else:
        p.text(INSTRUCTION)
    p.end()
    for said, codes in history:
        p.turn("user").text(said).end().turn("assistant", voice=True).audio(codes).end()
    return p.turn("user").text(text).end().turn("assistant", voice=True).build()
