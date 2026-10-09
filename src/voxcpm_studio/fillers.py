"""Fillers and laughs in transcripts (嗯, 哼, 哈哈, 呵呵, ...).

ASR is unsure how many of them it heard (呵呵呵 or 呵呵呵呵), and aligners,
trained on read speech, are unsure where they are. For TTS training the exact
count does not matter (the audio carries the length) but consistency does, so
long runs are tidied to two. Different fillers stay different: 哈哈 is a real
laugh, 呵呵 a "he he", 嗯 and 嗯嗯 mean different things.
"""

from __future__ import annotations

import json
import re
import shutil
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
    """Tidy the ``text`` of every entry of a manifest in place, keeping all
    else (paths, other keys) as written; the first time, the original is kept
    as ``<name>.orig``. Returns the number of entries changed."""
    lines = path.read_text(encoding="utf-8").splitlines()
    out, changed = [], 0
    for line in lines:
        try:
            entry = json.loads(line) if line.strip() else None
        except json.JSONDecodeError:
            entry = None
        if isinstance(entry, dict) and isinstance(entry.get("text"), str):
            text = tidy(entry["text"])
            if text != entry["text"]:
                entry["text"], changed = text, changed + 1
                line = json.dumps(entry, ensure_ascii=False)
        out.append(line)
    if changed:
        backup = path.with_name(path.name + ".orig")
        if not backup.exists():
            shutil.copy2(path, backup)
        tmp = path.with_name(f".{path.name}.tmp")
        tmp.write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")
        tmp.replace(path)
    return changed
