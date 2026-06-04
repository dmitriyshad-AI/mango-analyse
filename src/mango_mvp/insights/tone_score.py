from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


SCHEMA_VERSION = "tone_score_v1_2026_06_05"

_BUREAUCRATIC_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("current_center_frame", re.compile(r"\bв\s+рамках\s+текущего\s+учебного\s+центра\b", re.I)),
    ("formal_appeal", re.compile(r"\bпо\s+вашему\s+обращению\b", re.I)),
    ("passive_provided", re.compile(r"\bпредоставля\w+\b", re.I)),
    ("passive_carried_out", re.compile(r"\bосуществля\w+\b", re.I)),
    ("clarify_details", re.compile(r"\bуточн\w+\s+детал\w+\b", re.I)),
    ("manager_clarifies_next_step", re.compile(r"\bменеджер\s+уточн\w+\s+ближайш\w+\s+шаг\b", re.I)),
    ("impersonal_required", re.compile(r"\b(?:необходимо|требуется|следует)\b", re.I)),
    ("format_class_dependency_only", re.compile(r"\bзависит\s+от\s+(?:класса|формата|предмета)\b", re.I)),
)

_DIRECT_FIRST_RE = re.compile(
    r"^\s*(?:"
    r"да\b|нет\b|по\s+\d{1,2}\s+класс\w*|по\s+[а-яёa-z -]{3,40}\s*[:—-]|"
    r"стоимост\w*\b|цена\b|пробн\w+\b|онлайн\b|очно\b|адрес\b|"
    r"смена\b|заняти\w+\b|запис\w+\b|можно\b"
    r")",
    re.I,
)
_ACKNOWLEDGEMENT_RE = re.compile(
    r"\b(?:понимаю|вижу|хороший\s+вопрос|важно\s+сориентироваться|хочется\s+понять|"
    r"логично\s+уточнить|спасибо\s+за\s+вопрос)\b",
    re.I,
)
_BOT_STEP_RE = re.compile(
    r"\b(?:помогу|подскажу|сориентирую|подберу|могу\s+(?:подобрать|прислать|сориентировать|уточнить)|"
    r"давайте\s+(?:подбер[её]м|посмотрим|сориентируемся))\b",
    re.I,
)
_LIVE_WORD_RE = re.compile(
    r"\b(?:коротко|по\s+сути|без\s+гадания|на\s+конкретике|ориентировочно|примерно|под\s+вашу\s+задачу)\b",
    re.I,
)


@dataclass(frozen=True)
class ToneScore:
    tone_canc: int
    tone_warm: int
    tone_score: int
    flags: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "tone_canc": self.tone_canc,
            "tone_warm": self.tone_warm,
            "tone_score": self.tone_score,
            "flags": list(self.flags),
        }


def score_tone(text: Any) -> ToneScore:
    value = _clean_text(text)
    if not value:
        return ToneScore(tone_canc=0, tone_warm=0, tone_score=0, flags=("empty",))

    canc_flags = _bureaucratic_flags(value)
    warm_flags = _warm_flags(value)
    tone_canc = len(canc_flags)
    tone_warm = len(warm_flags)
    score = max(0, min(100, 55 + 14 * tone_warm - 18 * tone_canc))
    return ToneScore(tone_canc=tone_canc, tone_warm=tone_warm, tone_score=score, flags=tuple([*warm_flags, *canc_flags]))


def summarize_tone_scores(transcripts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    turns: list[dict[str, Any]] = []
    for dialog in transcripts:
        dialog_id = str(dialog.get("dialog_id") or "")
        brand = str(dialog.get("brand") or "")
        for turn in dialog.get("turns") or []:
            if not isinstance(turn, Mapping):
                continue
            scored = score_tone(turn.get("bot_text") or "")
            turns.append(
                {
                    "dialog_id": dialog_id,
                    "brand": brand,
                    "turn": turn.get("turn"),
                    **scored.as_dict(),
                }
            )

    count = len(turns)
    tone_canc = sum(int(item["tone_canc"]) for item in turns)
    tone_warm = sum(int(item["tone_warm"]) for item in turns)
    tone_score = round(sum(int(item["tone_score"]) for item in turns) / count, 1) if count else None
    return {
        "schema_version": SCHEMA_VERSION,
        "tone_canc": tone_canc,
        "tone_warm": tone_warm,
        "tone_score": tone_score,
        "turns_count": count,
        "avg_tone_canc_per_turn": round(tone_canc / count, 2) if count else None,
        "avg_tone_warm_per_turn": round(tone_warm / count, 2) if count else None,
        "turns": turns,
    }


def _bureaucratic_flags(text: str) -> list[str]:
    flags: list[str] = []
    for name, pattern in _BUREAUCRATIC_PATTERNS:
        if pattern.search(text):
            flags.append(f"canc:{name}")
    return flags


def _warm_flags(text: str) -> list[str]:
    flags: list[str] = []
    first_sentence = re.split(r"[.!?]\s+", text.strip(), maxsplit=1)[0]
    if _DIRECT_FIRST_RE.search(first_sentence):
        flags.append("warm:direct_first")
    if _ACKNOWLEDGEMENT_RE.search(text):
        flags.append("warm:acknowledgement")
    if _BOT_STEP_RE.search(text):
        flags.append("warm:bot_step")
    if _LIVE_WORD_RE.search(text):
        flags.append("warm:live_words")
    return flags


def _clean_text(text: Any) -> str:
    if text is None:
        return ""
    value = str(text).replace("\u00a0", " ")
    return " ".join(value.split())

