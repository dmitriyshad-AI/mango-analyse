"""Линтер человечности (детерминированный, без вызова модели).

Назначение: дешёвый пред-фильтр перед отправкой черновика И триггер для дорогого
LLM-рерайта (X2). Линтер НЕ блокирует и НЕ переписывает — только поднимает флаги.
Поэтому ложное срабатывание стоит максимум один лишний вызов рерайта/взгляд менеджера,
но НЕ ломает корректный ответ.

Главный принцип против "тупости": детекторы смотрят на текст + СОСТОЯНИЕ хода
(маршрут, P0, наличие факта, предыдущий ход), а не только на ключевые слова.
"""

from __future__ import annotations

import difflib
import re
from typing import Mapping, Sequence

# H4: служебные/мета-фразы, недопустимые клиенту ВСЕГДА (тут ключевое слово безопасно).
_META_MARKERS: tuple[str, ...] = (
    "без служебных пометок",
    "автономный ответ не требуется",
    "безопасный вариант",
    "не оформляю как жалобу",
    "не оформляю как заявление",
    "дополнительный ответ клиенту сейчас не нужен",
)
_INTERNAL_ID_RE = re.compile(r"\b(?:fact_id|source_id|trace_id)\s*[:=]|fact:v3:", re.I)

# H2: фразы хендоффа к менеджеру.
_HANDOFF_RE = re.compile(
    r"переда[мя]\s+(?:вопрос\s+|запрос\s+|это\s+|его\s+|контекст\s+)?(?:менеджер|ответственн)"
    r"|уточн\w+\s+у\s+менеджер|менеджер\w*\s+уточн|проверит\s+менеджер|менеджер\s+вернется"
    r"|свяж\w+\s+менеджер|ответственн\w+\s+сотрудник",
    re.I,
)

# H1: канцелярские штампы-зачины — ТОЛЬКО явно роботизированный полный зачин.
# Узкий список нарочно: «по проверенным данным» (частый хедж) и «не буду подставлять»
# (честная фраза) убраны — они давали ложные срабатывания против судьи.
_CRUTCH_OPENERS: tuple[str, ...] = (
    "сориентирую по проверенным данным",
)


def _norm(t: object) -> str:
    return re.sub(r"\s+", " ", str(t or "").strip().lower())


def is_p0(route: str, safety_flags: str) -> bool:
    """P0/manager_only: сухой хендофф/повтор тут НОРМА, не флагать."""
    blob = f"{safety_flags or ''} {route or ''}".lower()
    return "manager_only" in blob or "p0" in blob or "high_risk" in blob


def detect_meta_leak(text: str) -> list[str]:
    low = _norm(text)
    hits = [m for m in _META_MARKERS if m in low]
    if _INTERNAL_ID_RE.search(str(text or "")):
        hits.append("internal_id")
    return hits


def detect_over_handoff(text: str, route: str, safety_flags: str) -> bool:
    # Только на НЕ-P0 отвечаемом ходе: на P0 хендофф обязателен.
    # ВАЖНО (проверено на 290 ходах): доп. гейты «факт не missing» и «есть цифра»
    # снижали согласие с судьёй (57%→50%/20%), потому что «over-handoff с фактом» —
    # это семантика (бот назвал цену, но увильнул от вопроса), её детерминированно не поймать.
    # Поэтому детектор намеренно простой; точный разбор over-handoff — задача LLM-слоя (X2).
    if is_p0(route, safety_flags):
        return False
    return bool(_HANDOFF_RE.search(str(text or "")))


def detect_stock_opener(
    text: str, prior_openers: Sequence[str], route: str = "", safety_flags: str = ""
) -> tuple[str, str] | None:
    # Сухой P0-зачин/повтор («Приняли обращение…») — норма, не флагать.
    if is_p0(route, safety_flags):
        return None
    low = _norm(text)
    for c in _CRUTCH_OPENERS:
        if low.startswith(c):
            return ("crutch", c)
    opener = " ".join(low.split()[:4])
    if opener and opener in prior_openers:
        return ("reused_in_dialog", opener)
    return None


def detect_repeat(text: str, prev_text: str, route: str, safety_flags: str) -> float | None:
    # Сухой P0-повтор допустим — исключаем.
    if is_p0(route, safety_flags):
        return None
    a, b = _norm(prev_text), _norm(text)
    if len(a) < 25 or len(b) < 25:
        return None
    sim = difflib.SequenceMatcher(None, a, b).ratio()
    return sim if sim > 0.8 else None


def lint_turn(turn: Mapping[str, object], *, prev_bot_text: str = "", prior_openers: Sequence[str] = ()) -> dict:
    """Вернуть флаги человечности для одного хода бота.

    turn: ожидает поля bot_text, bot_route, bot_safety_flags.
    """
    text = str(turn.get("bot_text", "") or "")
    route = str(turn.get("bot_route", "") or "")
    flags = str(turn.get("bot_safety_flags", "") or "")
    found: dict = {}
    meta = detect_meta_leak(text)
    if meta:
        found["meta_leak"] = meta
    if detect_over_handoff(text, route, flags):
        found["over_handoff"] = True
    op = detect_stock_opener(text, prior_openers, route=route, safety_flags=flags)
    if op:
        found["stock_opener"] = op
    sim = detect_repeat(text, prev_bot_text, route, flags)
    if sim is not None:
        found["near_repeat"] = round(sim, 2)
    return found
