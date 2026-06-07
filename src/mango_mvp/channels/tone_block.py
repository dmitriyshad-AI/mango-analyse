from __future__ import annotations

import os
import re
from typing import Any, Mapping, Sequence


TONE_WARM_FRAME_ENV = "TELEGRAM_TONE_WARM_FRAME"
TONE_CLOSE_DETECT_ENV = "TELEGRAM_TONE_CLOSE_DETECT"
TONE_SELL_PROMPT_ENV = "TELEGRAM_TONE_SELL_PROMPT"
TONE_RICH_FORMAT_ENV = "TELEGRAM_TONE_RICH_FORMAT"

_WARM_PREFIXES: tuple[str, ...] = (
    "Конечно! Вот как это устроено у нас: ",
    "Да, подскажу: ",
    "Смотрите, что есть для вас: ",
)
_FORMAT_ONLINE_PREFIXES: tuple[str, ...] = (
    "Конечно! Вот как это устроено у нас: ",
    "Да, подскажу: ",
    "Смотрите, что есть для вас: ",
)
_FORMAT_OFFLINE_PREFIXES: tuple[str, ...] = (
    "Конечно! Вот как это устроено у нас: ",
    "Да, подскажу: ",
    "Смотрите, что есть для вас: ",
)
_SCHEDULE_PREFIXES: tuple[str, ...] = (
    "Подобрала для вас вариант: ",
    "Есть такая группа: ",
)


def truthy_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().casefold() in {"1", "true", "yes", "y", "да"}


def warm_frame_enabled(context: Mapping[str, Any] | None = None) -> bool:
    if isinstance(context, Mapping):
        for key in (TONE_WARM_FRAME_ENV, "tone_warm_frame_enabled"):
            if key in context:
                return truthy_value(context.get(key))
    return truthy_value(os.getenv(TONE_WARM_FRAME_ENV))


def close_detect_enabled(context: Mapping[str, Any] | None = None) -> bool:
    if isinstance(context, Mapping):
        for key in (TONE_CLOSE_DETECT_ENV, "tone_close_detect_enabled"):
            if key in context:
                return truthy_value(context.get(key))
    return truthy_value(os.getenv(TONE_CLOSE_DETECT_ENV))


def sell_prompt_enabled(context: Mapping[str, Any] | None = None) -> bool:
    if isinstance(context, Mapping):
        for key in (TONE_SELL_PROMPT_ENV, "tone_sell_prompt_enabled"):
            if key in context:
                return truthy_value(context.get(key))
    return truthy_value(os.getenv(TONE_SELL_PROMPT_ENV))


def tone_rich_format_enabled(context: Mapping[str, Any] | None = None) -> bool:
    if isinstance(context, Mapping):
        for key in (TONE_RICH_FORMAT_ENV, "tone_rich_format_enabled"):
            if key in context:
                return truthy_value(context.get(key))
    return truthy_value(os.getenv(TONE_RICH_FORMAT_ENV))


def apply_warm_frame(text: str, *, context: Mapping[str, Any] | None = None, kind: str = "") -> str:
    if not warm_frame_enabled(context):
        return str(text or "")
    value = str(text or "").strip()
    if not value:
        return value
    replacements: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("По подтверждённым данным: ", _WARM_PREFIXES),
        ("По подтвержденным данным: ", _WARM_PREFIXES),
        ("Онлайн-формат подтверждён: ", _FORMAT_ONLINE_PREFIXES),
        ("Онлайн-формат подтвержден: ", _FORMAT_ONLINE_PREFIXES),
        ("Очный формат подтверждён: ", _FORMAT_OFFLINE_PREFIXES),
        ("Очный формат подтвержден: ", _FORMAT_OFFLINE_PREFIXES),
        ("Нашёл такую группу: ", _SCHEDULE_PREFIXES),
        ("Нашел такую группу: ", _SCHEDULE_PREFIXES),
    )
    for prefix, pool in replacements:
        if not value.startswith(prefix):
            continue
        body = value[len(prefix) :].strip()
        if not body:
            return value
        return _choose_unused_prefix(pool, context=context, kind=kind) + body
    return value


def _choose_unused_prefix(
    pool: Sequence[str],
    *,
    context: Mapping[str, Any] | None,
    kind: str = "",
) -> str:
    used_text = "\n".join(_previous_bot_texts(context)).casefold().replace("ё", "е")
    for prefix in pool:
        if prefix.casefold().replace("ё", "е").strip() not in used_text:
            return prefix
    return pool[0] if pool else ""


def _previous_bot_texts(context: Mapping[str, Any] | None) -> tuple[str, ...]:
    if not isinstance(context, Mapping):
        return ()
    texts: list[str] = []
    memory = context.get("dialogue_memory_view")
    if isinstance(memory, Mapping):
        for turn in memory.get("recent_turns") or ():
            if isinstance(turn, Mapping) and str(turn.get("role") or "").casefold() in {"bot", "assistant"}:
                text = str(turn.get("text") or "").strip()
                if text:
                    texts.append(text)
    for item in context.get("recent_messages") or ():
        text = str(item or "").strip()
        low = text.casefold()
        if low.startswith(("бот:", "bot:", "assistant:", "ответ:")):
            texts.append(re.sub(r"^[^:]{1,24}:\s*", "", text).strip())
    return tuple(texts[-8:])
