"""Drama 3's text-to-speech prompt ("reference-bracket-v1", as sglang-omni builds it).

Rows of 11 ids: the token, then the frame's 10 codes (zeros on text rows). A reference clip is
"[Reference N]\\nText: <|speaker:0|> transcript\\nSpeech: " and its frames; each frame's token is
its semantic token (semantic_begin + code 0).
"""

from __future__ import annotations

TTS = "convert the provided text to speech"
TTS_REF = "convert the provided text to speech reference to the following:"
SPEAKER = "<|speaker:0|>"


def speaker_tagged(text: str) -> str:
    text = text.strip()
    return text if text.startswith("<|speaker:") else f"{SPEAKER} {text}"


def build(tokenizer, text: str, references, semantic_begin: int):
    """[rows, 11] int64 for ``text`` cloned from ``references`` [(transcript, codes [10, T])]."""
    import torch

    segments: list = ["<|im_start|>system\n", TTS_REF if references else TTS]
    for i, (transcript, codes) in enumerate(references):
        segments += [f"\n\n[Reference {i + 1}]\nText: {speaker_tagged(transcript)}\nSpeech: ", codes]
    segments += [
        "<|im_end|>\n",
        "<|im_start|>user\n",
        speaker_tagged(text),
        "<|im_end|>\n",
        "<|im_start|>assistant\n<|voice|>",
    ]
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
    return torch.cat(rows)
