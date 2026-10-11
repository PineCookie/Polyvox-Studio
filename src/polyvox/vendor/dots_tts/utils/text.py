"""Text helpers of dots.tts, rewritten for this project (see the package docstring):
same functions and results for the languages used here, without lingua,
langcodes or WeTextProcessing."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Literal

TextLanguage = Literal["zh", "en", "unknown"]

_WHITESPACE_PATTERN = re.compile(r"\s+")

# Names and codes -> the tag code dots.tts was trained with ([ZH], [EN], ...).
_LANGUAGE_CODES = {
    "zh": "ZH", "zho": "ZH", "chi": "ZH", "cmn": "ZH", "chinese": "ZH", "mandarin": "ZH",
    "en": "EN", "eng": "EN", "english": "EN",
    "ja": "JA", "jpn": "JA", "japanese": "JA",
    "ko": "KO", "kor": "KO", "korean": "KO",
    "yue": "YUE", "cantonese": "YUE",
    "fr": "FR", "french": "FR", "de": "DE", "german": "DE", "es": "ES", "spanish": "ES",
    "ru": "RU", "russian": "RU", "it": "IT", "italian": "IT", "pt": "PT", "portuguese": "PT",
}  # fmt: skip


class _Wetext:
    def __init__(self, lang: str):
        from wetext import Normalizer

        self._n = Normalizer(lang=lang, operator="tn")

    def normalize(self, text: str) -> str:
        return self._n.normalize(text)


@lru_cache(maxsize=1)
def get_chinese_text_normalizer() -> _Wetext:
    return _Wetext("zh")


@lru_cache(maxsize=1)
def get_english_text_normalizer() -> _Wetext:
    return _Wetext("en")


def detect(text: str) -> str | None:
    """The language the script points to: kana ja, hangul ko, Han zh, Latin en."""
    stripped = text.strip()
    if not stripped:
        return None
    if re.search(r"[぀-ヿ]", stripped):
        return "ja"
    if re.search(r"[가-힯]", stripped):
        return "ko"
    if re.search(r"[一-鿿]", stripped):
        return "zh"
    if re.search(r"[A-Za-z]", stripped):
        return "en"
    return None


def normalize_language_code(language: str | None) -> str | None:
    if language is None:
        return None

    stripped = language.strip()
    if not stripped or stripped.lower() in {"none", "unknown"}:
        return None
    if stripped.startswith("口音:"):
        return stripped

    key = stripped.lower().replace("_", "-").split("-")[0]
    if key in _LANGUAGE_CODES:
        return _LANGUAGE_CODES[key]
    if key.isalpha() and len(key) in (2, 3):
        return key.upper()
    return None


def attach_language_tag(text: str, language: str | None) -> str:
    if not text:
        return text

    language_code = normalize_language_code(language)
    if language_code is None:
        return text

    if language_code == "YUE":
        language_code = "口音:粤语"

    language_tag = f"[{language_code}]"
    if text.startswith(language_tag):
        return text
    return f"{language_tag}{text}"


def detect_text_language(text: str) -> TextLanguage:
    language_code = detect(text)
    if language_code == "zh":
        return "zh"
    if language_code == "en":
        return "en"
    return "unknown"


def _normalize_with(normalizer, text: str) -> str:
    normalized = normalizer.normalize(text)
    return _WHITESPACE_PATTERN.sub(" ", normalized).strip()


def normalize_chinese_text(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    return _normalize_with(get_chinese_text_normalizer(), stripped)


def normalize_english_text(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    return _normalize_with(get_english_text_normalizer(), stripped)


def normalize_text(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""

    language = detect_text_language(stripped)
    if language == "zh":
        return _normalize_with(get_chinese_text_normalizer(), stripped)
    if language == "en":
        return _normalize_with(get_english_text_normalizer(), stripped)
    return stripped
