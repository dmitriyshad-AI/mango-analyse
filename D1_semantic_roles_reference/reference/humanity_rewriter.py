"""Каркас X2 — второй LLM-проход «живее по форме» с безопасностью по построению.

Идея: рерайтер НЕДОВЕРЕННЫЙ. Он меняет только форму ответа (тон, прямота, без повторов),
НЕ источник фактов. Его выход прогоняется через те же проверки, и при ЛЮБОМ нарушении или
дрейфе факта — откат на исходный черновик. То есть рерайт может только улучшить тон или быть
отброшен; сделать ответ менее безопасным он не может.

Здесь НЕТ живого вызова модели — он инъектируется как `rewrite_fn(prompt)->str`, чтобы тесты
гоняли логику на моке. Кодекс при внедрении подключает реальный (маленький, дешёвый) вызов модели
и ДОПОЛНИТЕЛЬНО прогоняет штатные репо-гейты (answer_safety_classifier, sanitize_answer) на кандидате.
"""

from __future__ import annotations

import re
from typing import Callable, Mapping, Sequence

from humanity_linter import detect_meta_leak, is_p0

# Чужой бренд в активном бренде = утечка. Для unpk «мфти» допустимо (УНПК МФТИ), для foton — нет.
_FOREIGN_BRAND_TOKENS: dict[str, tuple[str, ...]] = {
    "foton": ("унпк", "мфти", "unpk"),
    "unpk": ("фотон", "цдпо", "црдо", "cdpofoton", "foton"),
}
# Флаги, при которых рерайт бессмысленен/опасен: факта нет → нечем заземлить «лучший» ответ.
_NO_FACT_FLAGS: tuple[str, ...] = ("missing_facts", "unverified_fact", "topic_not_allowed")


def _norm_brand(b: str) -> str:
    b = (b or "").strip().lower()
    if b in {"foton", "фотон"}:
        return "foton"
    if b in {"unpk", "унпк", "унпк мфти"}:
        return "unpk"
    return b


def _numbers(text: str) -> set[str]:
    """Числа из текста с нормализацией пробелов внутри числа: '29 750' -> '29750'."""
    t = re.sub(r"(?<=\d)[\s ](?=\d)", "", str(text or ""))
    return set(re.findall(r"\d+", t))


def _facts_blob(confirmed_facts: Mapping | Sequence | str | None) -> str:
    if confirmed_facts is None:
        return ""
    if isinstance(confirmed_facts, str):
        return confirmed_facts
    if isinstance(confirmed_facts, Mapping):
        return " ".join(_facts_blob(v) for v in confirmed_facts.values())
    if isinstance(confirmed_facts, (list, tuple)):
        return " ".join(_facts_blob(v) for v in confirmed_facts)
    return str(confirmed_facts)


# ---- проверки безопасности кандидата (любая «правда» -> откат) ----

def fact_drift(rewrite: str, original_draft: str, confirmed_facts) -> list[str]:
    """Числа, которые рерайт ВВЁЛ, но их нет ни в исходном черновике, ни в подтверждённых фактах."""
    backed = _numbers(original_draft) | _numbers(_facts_blob(confirmed_facts))
    return sorted(_numbers(rewrite) - backed)


def brand_leak(rewrite: str, active_brand: str) -> bool:
    low = str(rewrite or "").lower()
    return any(tok in low for tok in _FOREIGN_BRAND_TOKENS.get(_norm_brand(active_brand), ()))


# ---- триггер ----

def should_rewrite(turn: Mapping, linter_flags: Mapping, *, mode: str = "linter") -> bool:
    """Звать ли рерайт. P0 не трогаем НИКОГДА; без факта — тоже (нечем заземлить).

    mode="linter" — только когда линтер поднял флаг; mode="all_eligible" — на всех подходящих ходах.
    """
    route = str(turn.get("bot_route", "") or "")
    flags = str(turn.get("bot_safety_flags", "") or "")
    if is_p0(route, flags):
        return False
    low = flags.lower()
    if any(f in low for f in _NO_FACT_FLAGS):
        return False
    if mode == "all_eligible":
        return True
    return bool(linter_flags)


# ---- промпт рерайтера (только форма) ----

def build_rewrite_prompt(
    original_draft: str, *, client_message: str, confirmed_facts, active_brand: str, linter_flags: Mapping
) -> str:
    issues = ", ".join(sorted(linter_flags)) if linter_flags else "тон/прямота"
    return (
        "Перепиши черновик ответа ЖИВЕЕ и прямее, СОХРАНив весь смысл.\n"
        f"Активный бренд: {_norm_brand(active_brand)}. Последний вопрос клиента: {client_message}\n"
        f"Что улучшить: {issues}.\n"
        "ЖЁСТКИЕ ПРАВИЛА:\n"
        "- Меняй ТОЛЬКО форму. Никаких новых чисел, дат, цен, имён, адресов — только то, что уже есть в черновике/фактах.\n"
        "- Не упоминай другой бренд. Не раскрывай, что ты ИИ/бот. Не добавляй служебных пометок.\n"
        "- Сначала прямой ответ на вопрос, потом 1-2 пояснения, один следующий шаг. Не повторяй прошлый зачин.\n"
        "- Маршрут и решение о передаче менеджеру НЕ меняй.\n"
        f"ЧЕРНОВИК:\n{original_draft}\n"
        f"ПОДТВЕРЖДЁННЫЕ ФАКТЫ (единственный источник конкретики):\n{_facts_blob(confirmed_facts)[:1500]}\n"
        "Верни только переписанный текст ответа."
    )


# ---- главная обёртка ----

def apply_rewrite(
    turn: Mapping,
    *,
    rewrite_fn: Callable[[str], str] | None,
    confirmed_facts=None,
    active_brand: str = "",
    client_message: str = "",
    linter_flags: Mapping | None = None,
    sanitize_fn: Callable[[str], str] | None = None,
    mode: str = "linter",
) -> dict:
    """Вернуть {'draft_text', 'rewritten', 'fallback_reason'}.

    Гарантия безопасности: при любом нарушении возвращается ИСХОДНЫЙ черновик без изменений.
    rewrite_fn — единственная точка живого вызова модели (в тестах мокается). None = рерайт выключен.
    """
    original = str(turn.get("bot_text", "") or "")
    linter_flags = linter_flags or {}
    result = {"draft_text": original, "rewritten": False, "fallback_reason": None}

    if rewrite_fn is None:
        result["fallback_reason"] = "rewriter_disabled"
        return result
    if not should_rewrite(turn, linter_flags, mode=mode):
        result["fallback_reason"] = "not_triggered"
        return result

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
        result["fallback_reason"] = "rewriter_error"
        return result

    if not candidate:
        result["fallback_reason"] = "empty_candidate"
        return result
    if sanitize_fn is not None:
        candidate = str(sanitize_fn(candidate) or "").strip()

    # Проверки безопасности кандидата — любая срабатывает → откат на исходный.
    drift = fact_drift(candidate, original, confirmed_facts)
    if drift:
        result["fallback_reason"] = f"fact_drift:{','.join(drift)}"
        return result
    if brand_leak(candidate, active_brand):
        result["fallback_reason"] = "brand_leak"
        return result
    if detect_meta_leak(candidate):
        result["fallback_reason"] = "meta_leak"
        return result

    result["draft_text"] = candidate
    result["rewritten"] = True
    return result
