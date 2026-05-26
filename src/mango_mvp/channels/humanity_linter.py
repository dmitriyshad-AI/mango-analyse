"""Cheap deterministic linter for customer-facing bot drafts.

The linter is intentionally non-authoritative: it only raises form-quality
flags and never rewrites or blocks a correct answer by itself.
"""

from __future__ import annotations

import difflib
import re
from typing import Mapping, Optional, Sequence, Tuple


_META_MARKERS: tuple[str, ...] = (
    "без служебных пометок",
    "автономный ответ не требуется",
    "безопасный вариант",
    "не оформляю как жалобу",
    "не оформляю как заявление",
    "дополнительный ответ клиенту сейчас не нужен",
)
_INTERNAL_ID_RE = re.compile(r"\b(?:fact_id|source_id|trace_id)\s*[:=]|fact:v3:", re.I)

_HANDOFF_RE = re.compile(
    r"переда[мя]\s+(?:вопрос\s+|запрос\s+|это\s+|его\s+|контекст\s+)?(?:менеджер|ответственн)"
    r"|уточн\w+\s+у\s+менеджер|менеджер\w*\s+уточн|проверит\s+менеджер|менеджер\s+вернется"
    r"|свяж\w+\s+менеджер|ответственн\w+\s+сотрудник",
    re.I,
)

_CRUTCH_OPENERS: tuple[str, ...] = (
    "сориентирую по проверенным данным",
)


def _norm(text: object) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().casefold())


def is_p0(route: str, safety_flags: str) -> bool:
    blob = f"{safety_flags or ''} {route or ''}".casefold()
    return "manager_only" in blob or "p0" in blob or "high_risk" in blob


def detect_meta_leak(text: str) -> list[str]:
    lowered = _norm(text)
    hits = [marker for marker in _META_MARKERS if marker in lowered]
    if _INTERNAL_ID_RE.search(str(text or "")):
        hits.append("internal_id")
    return hits


def detect_over_handoff(text: str, route: str, safety_flags: str) -> bool:
    if is_p0(route, safety_flags):
        return False
    return bool(_HANDOFF_RE.search(str(text or "")))


def detect_stock_opener(
    text: str,
    prior_openers: Sequence[str],
    route: str = "",
    safety_flags: str = "",
) -> Optional[Tuple[str, str]]:
    if is_p0(route, safety_flags):
        return None
    lowered = _norm(text)
    for opener in _CRUTCH_OPENERS:
        if lowered.startswith(opener):
            return ("crutch", opener)
    current_opener = " ".join(lowered.split()[:4])
    if current_opener and current_opener in prior_openers:
        return ("reused_in_dialog", current_opener)
    return None


def detect_repeat(text: str, prev_text: str, route: str, safety_flags: str) -> Optional[float]:
    if is_p0(route, safety_flags):
        return None
    prev_normalized = _norm(prev_text)
    current_normalized = _norm(text)
    if len(prev_normalized) < 25 or len(current_normalized) < 25:
        return None
    similarity = difflib.SequenceMatcher(None, prev_normalized, current_normalized).ratio()
    return similarity if similarity > 0.8 else None


def lint_turn(
    turn: Mapping[str, object],
    *,
    prev_bot_text: str = "",
    prior_openers: Sequence[str] = (),
) -> dict[str, object]:
    text = str(turn.get("bot_text", "") or "")
    route = str(turn.get("bot_route", "") or "")
    flags = str(turn.get("bot_safety_flags", "") or "")
    found: dict[str, object] = {}

    meta = detect_meta_leak(text)
    if meta:
        found["meta_leak"] = meta
    if detect_over_handoff(text, route, flags):
        found["over_handoff"] = True
    opener = detect_stock_opener(text, prior_openers, route=route, safety_flags=flags)
    if opener:
        found["stock_opener"] = opener
    repeat = detect_repeat(text, prev_bot_text, route, flags)
    if repeat is not None:
        found["near_repeat"] = round(repeat, 2)
    return found


__all__ = [
    "detect_meta_leak",
    "detect_over_handoff",
    "detect_repeat",
    "detect_stock_opener",
    "is_p0",
    "lint_turn",
]
