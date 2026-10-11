"""Fillers and laughs in transcripts (嗯, 哼, 哈哈, 呵呵, ...).

ASR is unsure how many of them it heard (呵呵呵 or 呵呵呵呵), and aligners,
trained on read speech, are unsure where they are. For TTS training the exact
count does not matter (the audio carries the length) but consistency does, so
long runs are tidied to two. Different fillers stay different: 哈哈 is a real
laugh, 呵呵 a "he he", 嗯 and 嗯嗯 mean different things.
"""

from __future__ import annotations

import re
from pathlib import Path

# Non-lexical syllables: laughs, hesitations, interjections.
FILLER_CHARS = "哈呵嘿嘻嗯哼啊呃哦噢喔唔欸诶哎唉嗷"
MAX_RUN = 2

_RUN = re.compile(f"([{FILLER_CHARS}])\\1{{{MAX_RUN},}}")


def is_filler(word: str) -> bool:
    """An alignment unit that is a filler (one CJK character each)."""
    return len(word) == 1 and word in FILLER_CHARS


def tidy(text: str) -> str:
    """Runs of one filler character longer than MAX_RUN become MAX_RUN of it."""
    return _RUN.sub(lambda m: m.group(1) * MAX_RUN, text)


def tidy_manifest(path: Path) -> int:
    """Tidy the ``text`` of every entry of a manifest in place (see
    manifests.rewrite_manifest). Returns the number of entries changed."""
    from ..data.manifests import rewrite_manifest

    def change(entry: dict) -> dict | None:
        text = tidy(entry["text"])
        return {**entry, "text": text} if text != entry["text"] else None

    return rewrite_manifest(path, change)
