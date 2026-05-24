from __future__ import annotations

import re
from typing import Sequence


TEXT_SIGNALS_SCHEMA_VERSION = "text_signals_v1_2026_05_24"

_WORD_CHARS = "0-9a-zа-я"


def normalize_signal_text(value: object) -> str:
    return " ".join(str(value or "").casefold().replace("ё", "е").replace("\u00a0", " ").split())


def has_marker(text: object, marker: str) -> bool:
    value = normalize_signal_text(text)
    needle = normalize_signal_text(marker)
    if not needle:
        return False
    if re.search(rf"[^{_WORD_CHARS}]", needle):
        return needle in value
    return bool(re.search(rf"(?<![{_WORD_CHARS}]){re.escape(needle)}[{_WORD_CHARS}]*", value))


def has_any_marker(text: object, markers: Sequence[str]) -> bool:
    return any(has_marker(text, marker) for marker in markers)


def has_exact_word(text: object, word: str) -> bool:
    value = normalize_signal_text(text)
    needle = normalize_signal_text(word)
    if not needle or re.search(rf"[^{_WORD_CHARS}]", needle):
        return False
    return bool(re.search(rf"(?<![{_WORD_CHARS}]){re.escape(needle)}(?![{_WORD_CHARS}])", value))
