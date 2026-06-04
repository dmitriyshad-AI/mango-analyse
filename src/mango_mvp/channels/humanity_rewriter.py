"""X2 form-only rewrite layer for customer-facing bot drafts.

The rewriter is untrusted. It may improve warmth, directness, and variety, but
it is not allowed to introduce facts or weaken route/safety decisions.
"""

from __future__ import annotations

import re
from typing import Callable, Mapping, Optional, Sequence, Union

from mango_mvp.channels.humanity_linter import detect_meta_leak, is_p0


_FOREIGN_BRAND_TOKENS: dict[str, tuple[str, ...]] = {
    "foton": ("унпк", "мфти", "unpk"),
    "unpk": ("фотон", "цдпо", "црдо", "cdpofoton", "foton"),
}
_NO_FACT_FLAGS: tuple[str, ...] = ("missing_facts", "unverified_fact", "topic_not_allowed")


def _norm_brand(brand: str) -> str:
    value = (brand or "").strip().casefold()
    if value in {"foton", "фотон"}:
        return "foton"
    if value in {"unpk", "унпк", "унпк мфти"}:
        return "unpk"
    return value


def _numbers(text: str) -> set[str]:
    normalized = re.sub(r"(?<=\d)[\s\u00a0](?=\d)", "", str(text or ""))
    return set(re.findall(r"\d+", normalized))


ConfirmedFacts = Optional[Union[Mapping, Sequence, str]]


def _facts_blob(confirmed_facts: ConfirmedFacts) -> str:
    if confirmed_facts is None:
        return ""
    if isinstance(confirmed_facts, str):
        return confirmed_facts
    if isinstance(confirmed_facts, Mapping):
        return " ".join(_facts_blob(value) for value in confirmed_facts.values())
    if isinstance(confirmed_facts, (list, tuple)):
        return " ".join(_facts_blob(value) for value in confirmed_facts)
    return str(confirmed_facts)


def fact_drift(rewrite: str, original_draft: str, confirmed_facts: ConfirmedFacts) -> list[str]:
    backed_numbers = _numbers(original_draft) | _numbers(_facts_blob(confirmed_facts))
    return sorted(_numbers(rewrite) - backed_numbers)


def brand_leak(rewrite: str, active_brand: str) -> bool:
    lowered = str(rewrite or "").casefold()
    return any(token in lowered for token in _FOREIGN_BRAND_TOKENS.get(_norm_brand(active_brand), ()))


def should_rewrite(turn: Mapping[str, object], linter_flags: Mapping[str, object], *, mode: str = "linter") -> bool:
    route = str(turn.get("bot_route", "") or "")
    safety_flags = str(turn.get("bot_safety_flags", "") or "")
    if route == "manager_only" or is_p0(route, safety_flags):
        return False
    lowered_flags = safety_flags.casefold()
    if any(flag in lowered_flags for flag in _NO_FACT_FLAGS):
        return False
    if mode == "all_eligible":
        return True
    return bool(linter_flags)


def build_rewrite_prompt(
    original_draft: str,
    *,
    client_message: str,
    confirmed_facts: ConfirmedFacts,
    active_brand: str,
    linter_flags: Mapping[str, object],
) -> str:
    issues = ", ".join(sorted(str(key) for key in linter_flags)) if linter_flags else "тон/прямота"
    brand = _norm_brand(active_brand)
    return (
        "Перепиши черновик ответа живее, теплее и прямее, сохранив смысл и маршрут.\n"
        f"Активный бренд: {brand}. Последний вопрос клиента: {client_message}\n"
        f"Что улучшить: {issues}.\n"
        "ЖЁСТКИЕ ПРАВИЛА:\n"
        "- Меняй только форму. Никаких новых чисел, дат, цен, имён, адресов или фактов.\n"
        "- Конкретика допустима только если она уже есть в черновике или подтверждённых фактах ниже.\n"
        "- Не упоминай другой бренд. Не раскрывай, что ты ИИ/бот/GPT. Не добавляй служебных пометок.\n"
        "- Не меняй маршрут, P0-решение и передачу менеджеру.\n"
        "- Сначала прямой ответ на вопрос, потом 1-2 коротких пояснения и один следующий шаг.\n"
        "- Playbook менеджера: признай сомнение клиента одной короткой фразой, затем опирайся на факт и предложи мягкий шаг.\n"
        "- Срочность допустима только честная: если дедлайн или ограничение уже есть в фактах. Иначе не создавай её.\n"
        "- Не дави, не создавай срочность, не обещай результат и не используй однотипные зачины вроде «Понимаю» каждый раз.\n"
        "- Если черновик звучит роботски («Сориентирую», «По фактам», «Передам менеджеру»), сделай формулировку естественнее, но не меняй смысл.\n"
        "- Держи ответ компактным: 1-3 коротких предложения; абзацы допустимы, если это улучшает читаемость.\n"
        "- Верни только текст ответа клиенту, без JSON, комментариев и заголовков.\n\n"
        f"ЧЕРНОВИК:\n{original_draft}\n\n"
        f"ПОДТВЕРЖДЁННЫЕ ФАКТЫ (единственный источник конкретики):\n{_facts_blob(confirmed_facts)[:1500]}\n"
    )


def apply_rewrite(
    turn: Mapping[str, object],
    *,
    rewrite_fn: Optional[Callable[[str], str]],
    confirmed_facts: ConfirmedFacts = None,
    active_brand: str = "",
    client_message: str = "",
    linter_flags: Optional[Mapping[str, object]] = None,
    sanitize_fn: Optional[Callable[[str], str]] = None,
    validate_fn: Optional[Callable[[str], Optional[str]]] = None,
    mode: str = "linter",
) -> dict[str, object]:
    original = str(turn.get("bot_text", "") or "")
    linter_flags = linter_flags or {}
    output: dict[str, object] = {"draft_text": original, "rewritten": False, "fallback_reason": None}

    if rewrite_fn is None:
        output["fallback_reason"] = "rewriter_disabled"
        return output
    if not should_rewrite(turn, linter_flags, mode=mode):
        output["fallback_reason"] = "not_triggered"
        return output

    prompt = build_rewrite_prompt(
        original,
        client_message=client_message,
        confirmed_facts=confirmed_facts,
        active_brand=active_brand,
        linter_flags=linter_flags,
    )
    try:
        candidate = str(rewrite_fn(prompt) or "").strip()
    except Exception:
        output["fallback_reason"] = "rewriter_error"
        return output

    if not candidate:
        output["fallback_reason"] = "empty_candidate"
        return output
    if sanitize_fn is not None:
        candidate = str(sanitize_fn(candidate) or "").strip()

    drift = fact_drift(candidate, original, confirmed_facts)
    if drift:
        output["fallback_reason"] = f"fact_drift:{','.join(drift)}"
        return output
    if brand_leak(candidate, active_brand):
        output["fallback_reason"] = "brand_leak"
        return output
    if detect_meta_leak(candidate):
        output["fallback_reason"] = "meta_leak"
        return output
    if validate_fn is not None:
        violation = validate_fn(candidate)
        if violation:
            output["fallback_reason"] = violation
            return output

    output["draft_text"] = candidate
    output["rewritten"] = True
    return output


__all__ = [
    "apply_rewrite",
    "brand_leak",
    "build_rewrite_prompt",
    "fact_drift",
    "should_rewrite",
]
