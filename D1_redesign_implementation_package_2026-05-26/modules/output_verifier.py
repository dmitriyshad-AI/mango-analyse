"""Шаг [4]: детерминированный ВЫХОДНОЙ верификатор черновика.

Это правильное место детерминизма: правила КРИСП и бинарны (есть чужой бренд? число вне фактов? раскрытие ИИ?).
Тут детерминизм быстрее, бесплатен и НАДЁЖНЕЕ LLM. (Понимание — нечёткое, его делает LLM; верификация — чёткая, её делает код.)

verify() возвращает список findings; пустой список = PASS. Каждый finding несёт code + человекочитаемую деталь,
которую шаг [5] (X2-ремонт) использует как КОНКРЕТНУЮ инструкцию правки.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping, Sequence

_FOREIGN_BRAND_TOKENS: dict[str, tuple[str, ...]] = {
    "foton": ("унпк", "мфти", "unpk"),
    "unpk": ("фотон", "цдпо", "црдо", "cdpofoton", "foton", "долями"),
}
_META_MARKERS: tuple[str, ...] = (
    "без служебных пометок", "автономный ответ не требуется", "безопасный вариант",
    "не оформляю как жалобу", "fact_id", "source_id", "trace_id", "fact:v3",
)
_AI_SELF_DISCLOSURE = re.compile(r"\bя\s+(?:бот|ии|gpt|нейросеть|искусственн\w+\s+интеллект)\b", re.I)
# Обещания, недопустимые в автономном черновике (P0-в-тексте):
_P0_PROMISE = re.compile(
    r"верн[её]м\s+деньг|гаранти\w+\s+результат|обязательно\s+поступит|точно\s+вернём|оформим\s+возврат",
    re.I,
)


def _norm_brand(b: str) -> str:
    b = (b or "").strip().lower()
    return "foton" if b in {"foton", "фотон"} else "unpk" if b in {"unpk", "унпк", "унпк мфти"} else b


def _numbers(text: str) -> set[str]:
    t = re.sub(r"(?<=\d)[\s ](?=\d)", "", str(text or ""))  # «29 750» -> «29750»
    return set(re.findall(r"\d+", t))


@dataclass
class Finding:
    code: str
    detail: str


def verify(
    draft_text: str,
    *,
    facts: Mapping[str, str],
    active_brand: str,
    denied_topics: Sequence[str] = (),
    forbidden_substitutions: Sequence[str] = (),
) -> list[Finding]:
    """Вернуть findings (пусто = PASS). draft проверяется на жёсткие правила безопасности."""
    findings: list[Finding] = []
    text = str(draft_text or "")
    low = text.lower()

    # 1. brand_leak — чужой бренд в активном бренде
    brand = _norm_brand(active_brand)
    for tok in _FOREIGN_BRAND_TOKENS.get(brand, ()):
        if tok in low:
            findings.append(Finding("brand_leak", f"в ответе активного бренда {brand} упомянут чужой бренд/токен «{tok}» — убрать"))
            break

    # 2. fact_grounding — каждое число в ответе должно быть в facts[]
    backed = set()
    for v in (facts or {}).values():
        backed |= _numbers(v)
    introduced = _numbers(text) - backed
    # годы/классы (1-2 значные ≤ 11 и 2026/2027) часто не «факт-значения» — оставляем как мягкую эвристику примера:
    introduced = {n for n in introduced if not (len(n) <= 2 and int(n) <= 11) and n not in {"2026", "2027"}}
    if introduced:
        findings.append(Finding("fact_grounding", f"в ответе есть числа вне подтверждённых фактов: {sorted(introduced)} — заменить на факт или убрать конкретику"))

    # 3. forbidden_scope — ответ не должен опираться на отрицённые/запрещённые темы
    for topic in tuple(denied_topics) + tuple(forbidden_substitutions):
        topic = str(topic).strip().lower()
        if topic and topic in low:
            findings.append(Finding("forbidden_scope", f"ответ затрагивает запрещённую/отрицённую тему «{topic}» — убрать, отвечать строго на текущий вопрос"))
            break

    # 4. meta_leak / ai_disclosure
    if any(m in low for m in _META_MARKERS):
        findings.append(Finding("meta_leak", "в клиентском тексте служебная/внутренняя пометка — убрать"))
    if _AI_SELF_DISCLOSURE.search(text):
        findings.append(Finding("ai_disclosure", "бот сам раскрывает природу ИИ без запроса — переформулировать"))

    # 5. p0_in_output — недопустимые обещания
    if _P0_PROMISE.search(text):
        findings.append(Finding("p0_in_output", "обещание возврата/гарантии — недопустимо автономно, передать менеджеру"))

    return findings


def passed(findings: Sequence[Finding]) -> bool:
    return not findings
