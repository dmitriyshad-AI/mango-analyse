"""Шаг [0]+[1] v2: P0 пре-гейт + смысловой КОНТРАКТ-ПЛАН (полный).

v2-усиления (ТЗ §11/§12):
- контракт = ПЛАН ПО ПОДВОПРОСАМ: каждый под-вопрос → {answerable self|manager, нужные ключи, следующий шаг};
- known_slots с ИСТОЧНИКОМ: {value, source}; слот без источника НЕЛЬЗЯ утверждать (анти-выдумка контекста);
- client_state: ситуация/тон клиента (для регистра ответа, не для озвучивания эмоции).

LLM #1 инъектируется (understand_fn). Значения фактов модель НЕ выдаёт — только ключи. История подаётся поролевая.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

# ---------- [0] детерминированный P0 пре-гейт (backstop) ----------
_P0_REFUND_DEMAND = re.compile(
    r"верн[иуеё]те?\s+деньг|оспор\w+\s+(?:операци|плат|списан)|чарджбек|chargeback|"
    r"аннулир\w*\s+договор|расторг\w+\s+договор|деньги\s+ушл|уже\s+оплат\w+.*(?:нет|не\s+было)",
    re.I,
)
_P0_COMPLAINT_LEGAL = re.compile(
    r"возмутительн|ужасн\w+\s+(?:вед[её]т|преподав)|жалоб\w*|в\s+суд|прокуратур|нарушен\w+\s+прав|роспотребнадзор",
    re.I,
)
_VALUE_LIKE = re.compile(r"(?:₽|руб|%)|^\d+$")  # ключ не должен быть значением


def p0_pre_gate(text: str) -> str | None:
    t = str(text or "")
    if _P0_REFUND_DEMAND.search(t):
        return "refund_or_payment_dispute"
    if _P0_COMPLAINT_LEGAL.search(t):
        return "complaint_or_legal"
    return None


# ---------- [1] Контракт-план ----------
@dataclass(frozen=True)
class Subquestion:
    text: str
    answerable: str = "manager"          # "self" | "manager"
    needed_fact_keys: tuple[str, ...] = ()
    next_step: str = ""


@dataclass(frozen=True)
class Slot:
    value: str
    source: str = ""                      # "client_turn_N" | "fact:<key>" | "" => НЕ утверждать


@dataclass
class AnswerContract:
    active_brand: str = ""
    current_question: str = ""
    subquestions: tuple[Subquestion, ...] = ()
    continued_topics: tuple[str, ...] = ()
    denied_topics: tuple[str, ...] = ()
    switched_topics: tuple[str, ...] = ()
    known_slots: Mapping[str, Slot] = field(default_factory=dict)
    forbidden_substitutions: tuple[str, ...] = ()
    client_state: str = ""                # «тревожная мама», «торопится», «сравнивает цену», «сомневается»...
    answerability: str = "manager_only"   # overall
    is_p0: bool = False
    p0_reason: str = ""
    confidence: float = 0.0

    def manager_only(self) -> bool:
        return self.is_p0 or self.answerability != "answer_self"

    def all_needed_fact_keys(self) -> tuple[str, ...]:
        keys: list[str] = []
        for sq in self.subquestions:
            keys.extend(sq.needed_fact_keys)
        return tuple(dict.fromkeys(k for k in keys if k))

    def assertable_slots(self) -> dict[str, str]:
        """Слоты, у которых ЕСТЬ источник — только их можно утверждать."""
        return {name: s.value for name, s in self.known_slots.items() if s.source and s.value}

    def unsourced_slots(self) -> tuple[str, ...]:
        return tuple(name for name, s in self.known_slots.items() if not s.source)


def build_understanding_prompt(
    *, conversation: Sequence[Mapping[str, str]], active_brand: str, fact_key_catalog: Sequence[str]
) -> str:
    hist = "\n".join(f"{m.get('role','?')}: {m.get('text','')}" for m in conversation)
    catalog = ", ".join(fact_key_catalog)
    return (
        f"Ты разбираешь диалог продаж учебного центра. Активный бренд: {active_brand} (только он; другой бренд не упоминать).\n"
        "Верни СТРОГО JSON-КОНТРАКТ-ПЛАН:\n"
        "{ current_question, client_state, denied_topics[], continued_topics[], switched_topics[], forbidden_substitutions[],\n"
        "  known_slots: { имя: {value, source} },   // source = откуда известно: 'client_turn_N' или 'fact:<key>'; если ниоткуда — не указывай слот\n"
        "  subquestions: [ {text, answerable: 'self'|'manager', needed_fact_keys[], next_step} ],\n"
        "  answerability: 'answer_self'|'manager_only', is_p0: bool, p0_reason, confidence: 0..1 }\n"
        "ПРАВИЛА:\n"
        "- Пойми, что клиент хочет СЕЙЧАС; учти отрицания («не про X»→denied_topics), поправки, составные вопросы (разложи в subquestions).\n"
        "- known_slots ТОЛЬКО с источником (что клиент сам сказал или факт). Не додумывай класс/предмет/формат/цель без источника.\n"
        "- client_state: ситуация/тон клиента (тревожная мама/торопится/сравнивает цену/сомневается/раздражён/просто уточняет).\n"
        "- needed_fact_keys ТОЛЬКО из каталога. Значения фактов НЕ придумывай и НЕ пиши.\n"
        "- Спор/жалоба/требование возврата/юр.угроза → is_p0=true, answerability=manager_only.\n"
        f"Каталог fact_keys: {catalog}\n"
        f"ДИАЛОГ (поролевая история):\n{hist}\n"
        "Только JSON."
    )


def _slot(value: object) -> Slot | None:
    if isinstance(value, Mapping):
        v = str(value.get("value") or "").strip()
        s = str(value.get("source") or "").strip()
        if v:
            return Slot(value=v, source=s)
    elif isinstance(value, str) and value.strip():
        return Slot(value=value.strip(), source="")  # без источника
    return None


def parse_contract(raw: object, *, active_brand: str, p0_reason_pregate: str | None = None) -> AnswerContract:
    data: Mapping = {}
    if isinstance(raw, Mapping):
        data = raw
    elif isinstance(raw, str):
        try:
            data = json.loads(raw)
        except Exception:
            data = {}

    def _seq(k):
        v = data.get(k)
        return tuple(str(x).strip() for x in v if str(x).strip()) if isinstance(v, (list, tuple)) else ()

    is_p0 = bool(data.get("is_p0")) or bool(p0_reason_pregate)
    answerability = "manager_only" if is_p0 else str(data.get("answerability") or "manager_only")
    if not data:
        return AnswerContract(active_brand=active_brand, is_p0=bool(p0_reason_pregate),
                              p0_reason=p0_reason_pregate or "", answerability="manager_only")

    subs: list[Subquestion] = []
    for item in (data.get("subquestions") or []):
        if not isinstance(item, Mapping):
            continue
        keys = tuple(str(k).strip() for k in (item.get("needed_fact_keys") or [])
                     if str(k).strip() and not _VALUE_LIKE.search(str(k)))
        subs.append(Subquestion(
            text=str(item.get("text") or "").strip(),
            answerable=str(item.get("answerable") or "manager"),
            needed_fact_keys=keys,
            next_step=str(item.get("next_step") or "").strip(),
        ))

    slots: dict[str, Slot] = {}
    for name, val in (data.get("known_slots") or {}).items():
        s = _slot(val)
        if s:
            slots[str(name)] = s

    try:
        confidence = max(0.0, min(1.0, float(data.get("confidence", 0.0))))
    except Exception:
        confidence = 0.0

    return AnswerContract(
        active_brand=active_brand,
        current_question=str(data.get("current_question") or ""),
        subquestions=tuple(subs),
        continued_topics=_seq("continued_topics"),
        denied_topics=_seq("denied_topics"),
        switched_topics=_seq("switched_topics"),
        known_slots=slots,
        forbidden_substitutions=_seq("forbidden_substitutions"),
        client_state=str(data.get("client_state") or ""),
        answerability=answerability,
        is_p0=is_p0,
        p0_reason=str(data.get("p0_reason") or p0_reason_pregate or ""),
        confidence=confidence,
    )


def understand(
    *,
    conversation: Sequence[Mapping[str, str]],
    active_brand: str,
    fact_key_catalog: Sequence[str],
    understand_fn: Callable[[str], object] | None,
) -> AnswerContract:
    last_text = conversation[-1].get("text", "") if conversation else ""
    pregate = p0_pre_gate(last_text)
    if understand_fn is None:
        return AnswerContract(active_brand=active_brand, is_p0=bool(pregate),
                              p0_reason=pregate or "", answerability="manager_only")
    prompt = build_understanding_prompt(conversation=conversation, active_brand=active_brand, fact_key_catalog=fact_key_catalog)
    try:
        raw = understand_fn(prompt)
    except Exception:
        raw = {}
    return parse_contract(raw, active_brand=active_brand, p0_reason_pregate=pregate)
