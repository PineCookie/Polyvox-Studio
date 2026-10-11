"""Chinese / English splitting by script, replacing upstream's split_lang +
fast_langdetect version (which downloads a 125 MB detector model): the studio
offers Chinese and English only. Latin letters are English, other letters
Chinese; digits, spaces and punctuation stay with the run before them (or the
first run)."""

import unicodedata


def _lang(ch: str) -> str | None:
    if not unicodedata.category(ch).startswith("L"):
        return None
    return "en" if ch.isascii() else "zh"


class LangSegmenter:
    @staticmethod
    def getTexts(text: str, default_lang: str = "") -> list[dict]:
        runs: list[dict] = []
        pending = ""
        for ch in text:
            lang = _lang(ch)
            if lang is None:
                if runs:
                    runs[-1]["text"] += ch
                else:
                    pending += ch
                continue
            if lang == "zh" and default_lang:
                lang = default_lang
            if runs and runs[-1]["lang"] == lang:
                runs[-1]["text"] += ch
            else:
                runs.append({"lang": lang, "text": pending + ch})
                pending = ""
        if pending:
            runs.append({"lang": default_lang or "zh", "text": pending})
        return runs
