from __future__ import annotations

"""Parallel dialogue-contract draft pipeline, full v2.

This module is an opt-in LLM-first path behind TELEGRAM_DIALOGUE_CONTRACT_PIPELINE.
The default Telegram draft path is not changed.

v2 shape:
1. deterministic P0 pre-gate;
2. LLM understanding returns a contract plan with subquestions, sourced slots,
   client state, and fact keys, not fact values;
3. deterministic active-brand fact retrieval by key;
4. LLM draft from contract, facts, sourced slots, and style examples;
5. hard output verification plus semantic faithfulness check that is fail-closed;
6. optional repair and optional X2 warmth, both re-verified.
"""

import json
import os
import re
from collections.abc import Mapping as MappingABC, Sequence as SequenceABC
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from mango_mvp.channels.answer_safety_classifier import classify_answer_safety
from mango_mvp.channels.fact_retrieval import key_matches
from mango_mvp.channels.humanity_guards import has_meta_leak
from mango_mvp.channels.p0_recall_spec import codes_from_text
from mango_mvp.insights.sanitizers import sanitize_answer


DIALOGUE_CONTRACT_PIPELINE_ENV = "TELEGRAM_DIALOGUE_CONTRACT_PIPELINE"
DIALOGUE_CONTRACT_SCHEMA_VERSION = "dialogue_contract_v2_2026_05_26"
DEFAULT_KB_SNAPSHOT_PATH = Path(
    "product_data/knowledge_base/kb_release_20260520_v6_3_team_answers/kb_release_v3_snapshot.json"
)
MAX_CATALOG_KEYS = 240
MAX_REPAIR_ATTEMPTS = 2

_MONEY_OR_VALUE_RE = re.compile(
    r"(?:₽|руб(?:\.|лей|ля|ль)?|%)|\b\d[\d\s\u00a0]{2,}\s*(?:р\.|руб|₽)\b",
    re.I,
)
_NUMBER_RE = re.compile(r"\d+")
_AI_SELF_DISCLOSURE_RE = re.compile(
    r"\bя\s+(?:бот|gpt|нейросеть|искусственн\w+\s+интеллект)\b",
    re.I,
)
_P0_PROMISE_RE = re.compile(
    r"верн[её]м\s+деньг|оформим\s+возврат|гаранти\w+\s+(?:результат|поступлен)|"
    r"обязательно\s+(?:поступит|сдаст)|точно\s+верн[её]м",
    re.I,
)
_FACTUAL_CLAIM_RE = re.compile(
    r"(?:₽|руб(?:\.|лей|ля|ль)?|%|\b\d{1,3}\s*балл\w*|\b\d{1,2}\s+"
    r"(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)"
    r"(?:\s+\d{4})?\b|\b\d{1,2}[./-]\d{1,2}(?:[./-]\d{2,4})?\b)",
    re.I,
)
_BRAND_TOKENS: dict[str, tuple[str, ...]] = {
    "foton": ("унпк", "унпк мфти", "мфти", "kmipt", "@unpk", "ноу унпк", "ано дпо"),
    "unpk": ("фотон", "цдпо", "црдо", "cdpofoton", "foton", "долями", "т-банк"),
}
_META_MARKERS: tuple[str, ...] = (
    "без служебных пометок",
    "автономный ответ не требуется",
    "безопасный вариант",
    "не оформляю как жалобу",
    "fact_id",
    "source_id",
    "trace_id",
    "fact:v3",
)
_STOCK_OPENERS = ("сориентирую по проверенным данным", "по проверенным данным")
_CLERICAL = ("осуществляется", "в рамках", "по вопросу о", "данный", "необходимо отметить", "вышеуказанн")
_DRY_P0_TEXTS: tuple[str, ...] = (
    "Приняли обращение. Передам его ответственному сотруднику, он вернётся с ответом.",
    "Обращение принято. Передам ответственному сотруднику, он вернётся с ответом.",
    "Приняли. Передам обращение ответственному сотруднику, он вернётся с ответом.",
    "Зафиксировали обращение. Передам его ответственному сотруднику, он вернётся с ответом.",
)


@dataclass(frozen=True)
class Subquestion:
    text: str
    answerable: str = "manager"  # self | manager
    needed_fact_keys: tuple[str, ...] = ()
    next_step: str = ""
    question_type: str = ""
    existence_target: str = ""

    def to_json_dict(self) -> Mapping[str, Any]:
        return {
            "text": self.text,
            "answerable": self.answerable,
            "needed_fact_keys": list(self.needed_fact_keys),
            "next_step": self.next_step,
            "question_type": self.question_type,
            "existence_target": self.existence_target,
        }


@dataclass(frozen=True)
class Slot:
    value: str
    source: str = ""

    def to_json_dict(self) -> Mapping[str, str]:
        return {"value": self.value, "source": self.source}


@dataclass(frozen=True)
class AnswerContract:
    active_brand: str
    current_question: str = ""
    subquestions: tuple[Subquestion, ...] = ()
    continued_topics: tuple[str, ...] = ()
    denied_topics: tuple[str, ...] = ()
    switched_topics: tuple[str, ...] = ()
    known_slots: Mapping[str, Slot] = field(default_factory=dict)
    forbidden_substitutions: tuple[str, ...] = ()
    client_state: str = ""
    answerability: str = "manager_only"
    question_type: str = ""
    existence_target: str = ""
    is_p0: bool = False
    p0_reason: str = ""
    confidence: float = 0.0

    @property
    def composite_subquestions(self) -> tuple[str, ...]:
        return tuple(item.text for item in self.subquestions if item.text)

    @property
    def needed_fact_keys(self) -> tuple[str, ...]:
        return self.all_needed_fact_keys()

    def manager_only(self) -> bool:
        return self.is_p0 or self.answerability != "answer_self"

    def all_needed_fact_keys(self) -> tuple[str, ...]:
        keys: list[str] = []
        for subquestion in self.subquestions:
            keys.extend(subquestion.needed_fact_keys)
        return tuple(dict.fromkeys(key for key in keys if key))

    def assertable_slots(self) -> dict[str, str]:
        return {name: slot.value for name, slot in self.known_slots.items() if slot.value and slot.source}

    def unsourced_slots(self) -> tuple[str, ...]:
        return tuple(name for name, slot in self.known_slots.items() if slot.value and not slot.source)

    def to_json_dict(self) -> Mapping[str, Any]:
        return {
            "schema_version": DIALOGUE_CONTRACT_SCHEMA_VERSION,
            "active_brand": self.active_brand,
            "current_question": self.current_question,
            "subquestions": [item.to_json_dict() for item in self.subquestions],
            "composite_subquestions": list(self.composite_subquestions),
            "continued_topics": list(self.continued_topics),
            "denied_topics": list(self.denied_topics),
            "switched_topics": list(self.switched_topics),
            "known_slots": {name: slot.to_json_dict() for name, slot in self.known_slots.items()},
            "assertable_slots": self.assertable_slots(),
            "unsourced_slots": list(self.unsourced_slots()),
            "needed_fact_keys": list(self.needed_fact_keys),
            "forbidden_substitutions": list(self.forbidden_substitutions),
            "client_state": self.client_state,
            "answerability": self.answerability,
            "question_type": self.question_type,
            "existence_target": self.existence_target,
            "is_p0": self.is_p0,
            "p0_reason": self.p0_reason,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class FactStore:
    catalog: tuple[str, ...]
    store: Mapping[str, Mapping[str, str]]
    fact_records: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True)
class RetrievalResult:
    facts: Mapping[str, str]
    missing: tuple[str, ...]
    matched_keys: Mapping[str, tuple[str, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class VerificationFinding:
    code: str
    detail: str


@dataclass(frozen=True)
class FaithfulnessClaim:
    claim: str
    evidence_fact_key: str = ""
    verdict: str = ""
    reason: str = ""

    def to_json_dict(self) -> Mapping[str, str]:
        return {
            "claim": self.claim,
            "evidence_fact_key": self.evidence_fact_key,
            "verdict": self.verdict,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class FaithfulnessResult:
    unsupported: tuple[str, ...] = ()
    claims: tuple[FaithfulnessClaim, ...] = ()
    available: bool = True


@dataclass(frozen=True)
class FormFinding:
    code: str
    detail: str


@dataclass(frozen=True)
class DialogueContractPipelineResult:
    draft_text: str
    route: str
    manager_only: bool
    contract: AnswerContract
    facts: Mapping[str, str] = field(default_factory=dict)
    missing: tuple[str, ...] = ()
    findings: tuple[VerificationFinding, ...] = ()
    unsupported_claims: tuple[str, ...] = ()
    form_findings: tuple[FormFinding, ...] = ()
    fallback_reason: str = ""
    warmed: bool = False
    warmth_attempted: bool = False
    warmth_mode: str = ""
    warmth_rejected_reason: str = ""
    warmth_rejected_findings: tuple[VerificationFinding, ...] = ()
    warmth_rejected_unsupported: tuple[str, ...] = ()
    warmth_semantic_available: bool = True
    repaired: bool = False


@dataclass(frozen=True)
class Toggles:
    enforce_slot_evidence: bool = True
    semantic_faithfulness: bool = True
    form_warmth: bool = True
    warmth_mode: str = "linter"


def pipeline_enabled(context: Mapping[str, Any] | None = None) -> bool:
    if isinstance(context, MappingABC) and context.get(DIALOGUE_CONTRACT_PIPELINE_ENV) is not None:
        return _truthy(context.get(DIALOGUE_CONTRACT_PIPELINE_ENV))
    return _truthy(os.getenv(DIALOGUE_CONTRACT_PIPELINE_ENV))


def _normalize_warmth_mode(mode: object) -> str:
    value = str(mode or "").strip().casefold()
    return value if value in {"linter", "all_eligible"} else "linter"


def build_conversation(
    current_message: str,
    *,
    context: Mapping[str, Any] | None = None,
) -> tuple[Mapping[str, str], ...]:
    messages: list[Mapping[str, str]] = []
    if isinstance(context, MappingABC):
        raw_dialogue = context.get("dialogue_turns") or context.get("role_messages") or context.get("conversation_messages")
        if isinstance(raw_dialogue, SequenceABC) and not isinstance(raw_dialogue, (str, bytes, bytearray)):
            for item in raw_dialogue[-14:]:
                if not isinstance(item, MappingABC):
                    continue
                role = str(item.get("role") or item.get("speaker") or "").strip().lower()
                text = str(item.get("text") or item.get("message") or "").strip()
                if not text:
                    continue
                normalized_role = "bot" if role in {"bot", "assistant", "manager"} else "client"
                messages.append({"role": normalized_role, "text": text[:1200]})
        if not messages:
            raw_recent = context.get("recent_messages")
            if isinstance(raw_recent, SequenceABC) and not isinstance(raw_recent, (str, bytes, bytearray)):
                for idx, text in enumerate(str(item or "").strip() for item in raw_recent if str(item or "").strip()):
                    role = "client" if idx % 2 == 0 else "bot"
                    messages.append({"role": role, "text": text[:900]})
    messages.append({"role": "client", "text": str(current_message or "").strip()[:1200]})
    return tuple(messages[-15:])


def build_understanding_prompt(
    *,
    conversation: Sequence[Mapping[str, str]],
    active_brand: str,
    fact_key_catalog: Sequence[str],
    context: Mapping[str, Any] | None = None,
) -> str:
    hist = "\n".join(f"{item.get('role', '?')}: {item.get('text', '')}" for item in conversation)
    catalog = ", ".join(str(item) for item in fact_key_catalog[:MAX_CATALOG_KEYS])
    known_slots: Mapping[str, Any] = {}
    if isinstance(context, MappingABC):
        memory = context.get("dialogue_memory_view") if isinstance(context.get("dialogue_memory_view"), MappingABC) else {}
        known_slots = memory.get("known_slots") if isinstance(memory.get("known_slots"), MappingABC) else {}
    return (
        "Ты разбираешь диалог с родителем о курсах учебного центра.\n"
        f"Активный бренд: {_normalize_brand(active_brand)}. Клиентский ответ потом будет только по этому бренду.\n"
        "Верни строго JSON без пояснений:\n"
        "{ current_question, client_state, continued_topics[], denied_topics[], switched_topics[], forbidden_substitutions[],\n"
        "  known_slots: { имя: {value, source} },\n"
        "  subquestions: [ {text, answerable:'self'|'manager', question_type:'existence_yes_no'|'', existence_target, needed_fact_keys[], next_step} ],\n"
        "  answerability:'answer_self'|'manager_only', question_type:'existence_yes_no'|'', existence_target, is_p0:bool, p0_reason, confidence:0..1 }\n"
        "Правила:\n"
        "- Пойми последний вопрос клиента в контексте всей истории.\n"
        "- Если клиент говорит «не про X», X должен попасть в denied_topics и не должен стать темой ответа.\n"
        "- Составной вопрос разложи на subquestions, чтобы ответить на каждую безопасную часть.\n"
        "- Если клиент спрашивает «есть ли X / можно ли X / доступен ли X», ставь question_type='existence_yes_no' и existence_target=X.\n"
        "- Если клиент спрашивает про конкретный способ оплаты (прямой перевод/по счёту, рассрочка через банк, Долями), "
        "не подменяй его соседним способом оплаты; в current_question и subquestion.text сохрани именно спрошенный способ.\n"
        "- Гипотетический вопрос до оплаты «если передумаю / если не понравится, вернут ли деньги?» — это refund_policy, не P0; "
        "попроси ключ refund_policy.current и отвечай из факта. Реальная просьба «верните деньги», спор оплаты или жалоба — P0 manager_only.\n"
        "- known_slots указывай ТОЛЬКО с источником: 'client_turn_N' или 'fact:<key>'. Без источника слот не указывай.\n"
        "- client_state — ситуация/тон клиента для выбора регистра; не нужно потом произносить эмоцию вслух.\n"
        "- needed_fact_keys: только ключи или смысловые ключи из каталога; значения, суммы, даты и проценты не пиши.\n"
        "- Если нужен спорный возврат, жалоба, юридическая угроза или спорная оплата: is_p0=true, answerability=manager_only.\n"
        "- Если факта нет или уверенность низкая: answerability=manager_only, но current_question всё равно заполни.\n"
        f"Уже известные данные: {json.dumps(dict(known_slots), ensure_ascii=False)}\n"
        f"Каталог ключей фактов: {catalog}\n"
        f"Диалог:\n{hist}\n"
        "Только JSON."
    )


def parse_contract(
    raw: object,
    *,
    active_brand: str,
    fact_key_catalog: Sequence[str] = (),
    p0_reason_pregate: str | None = None,
) -> AnswerContract:
    data: Mapping[str, Any] = {}
    if isinstance(raw, MappingABC):
        data = raw
    elif isinstance(raw, str):
        try:
            data = _extract_json_object(raw)
        except Exception:
            data = {}
    if not data:
        return AnswerContract(
            active_brand=_normalize_brand(active_brand),
            answerability="manager_only",
            is_p0=bool(p0_reason_pregate),
            p0_reason=p0_reason_pregate or "",
        )

    catalog = tuple(str(item or "").strip() for item in fact_key_catalog if str(item or "").strip())
    subquestions = _parse_subquestions(data, catalog)
    flat_keys = tuple(_valid_contract_key(key, catalog) for key in _seq(data.get("needed_fact_keys")))
    flat_keys = tuple(item for item in dict.fromkeys(flat_keys) if item)
    raw_answerability = str(data.get("answerability") or "manager_only").strip()
    question_type = _normalize_question_type(data.get("question_type"), fallback_text=str(data.get("current_question") or ""))
    existence_target = str(data.get("existence_target") or "").strip()[:180]
    if not subquestions and (data.get("current_question") or flat_keys):
        subquestions = (
            Subquestion(
                text=str(data.get("current_question") or "").strip()[:300],
                answerable="self" if flat_keys and raw_answerability == "answer_self" else "manager",
                needed_fact_keys=flat_keys,
                question_type=question_type,
                existence_target=existence_target,
            ),
        )

    is_p0 = bool(data.get("is_p0")) or bool(p0_reason_pregate)
    answerability = raw_answerability
    if answerability not in {"answer_self", "manager_only"}:
        answerability = "manager_only"
    if is_p0:
        answerability = "manager_only"
    return AnswerContract(
        active_brand=_normalize_brand(active_brand),
        current_question=str(data.get("current_question") or "").strip()[:300],
        subquestions=subquestions,
        continued_topics=tuple(_seq(data.get("continued_topics"))),
        denied_topics=tuple(_seq(data.get("denied_topics"))),
        switched_topics=tuple(_seq(data.get("switched_topics"))),
        known_slots=_clean_slots(data.get("known_slots")),
        forbidden_substitutions=tuple(_seq(data.get("forbidden_substitutions"))),
        client_state=str(data.get("client_state") or "").strip()[:180],
        answerability=answerability,
        question_type=question_type,
        existence_target=existence_target,
        is_p0=is_p0,
        p0_reason=str(data.get("p0_reason") or p0_reason_pregate or "").strip()[:200],
        confidence=_clamp_float(data.get("confidence")),
    )


def understand(
    *,
    conversation: Sequence[Mapping[str, str]],
    active_brand: str,
    fact_key_catalog: Sequence[str],
    understand_fn: Callable[[str], object] | None,
    context: Mapping[str, Any] | None = None,
) -> AnswerContract:
    last_text = str(conversation[-1].get("text") or "") if conversation else ""
    pregate = p0_pre_gate(last_text, context=context)
    if understand_fn is None:
        return AnswerContract(
            active_brand=_normalize_brand(active_brand),
            answerability="manager_only",
            is_p0=bool(pregate),
            p0_reason=pregate or "",
        )
    prompt = build_understanding_prompt(
        conversation=conversation,
        active_brand=active_brand,
        fact_key_catalog=fact_key_catalog,
        context=context,
    )
    try:
        raw = understand_fn(prompt)
    except Exception:
        raw = {}
    return parse_contract(
        raw,
        active_brand=active_brand,
        fact_key_catalog=fact_key_catalog,
        p0_reason_pregate=pregate,
    )


def p0_pre_gate(text: str, *, context: Mapping[str, Any] | None = None) -> str | None:
    codes = codes_from_text(text)
    if codes:
        return ",".join(codes)
    decision = classify_answer_safety(client_message=text, context=context)
    if decision.p0_required:
        return ",".join(decision.risk_codes or (decision.primary_risk or "p0",))
    return None


def build_fact_store(
    *,
    active_brand: str,
    context: Mapping[str, Any] | None = None,
    snapshot_path: str | Path | None = None,
) -> FactStore:
    brand = _normalize_brand(active_brand)
    snapshot = _load_snapshot(snapshot_path or _snapshot_path_from_context(context))
    records: list[Mapping[str, Any]] = []
    store: dict[str, dict[str, str]] = {"foton": {}, "unpk": {}}
    for fact in _snapshot_facts(snapshot):
        fact_brand = _normalize_brand(str(fact.get("brand") or ""))
        if fact_brand not in {"foton", "unpk"}:
            continue
        if not _client_safe_fact(fact):
            continue
        key = str(fact.get("fact_key") or fact.get("fact_id") or "").strip()
        text = str(fact.get("client_safe_text") or fact.get("fact_text") or fact.get("manager_display_text") or "").strip()
        if not key or not text:
            continue
        previous = store.setdefault(fact_brand, {}).get(key)
        store[fact_brand][key] = text if not previous else _join_fact_text(previous, text)
        records.append(fact)

    if isinstance(context, MappingABC):
        confirmed = context.get("confirmed_facts")
        if isinstance(confirmed, MappingABC):
            for key, value in confirmed.items():
                text = _fact_value_text(value)
                if key and text:
                    store.setdefault(brand, {})[str(key)] = text

    catalog = _prioritize_catalog(tuple(store.get(brand, {}).keys()), context=context)
    return FactStore(catalog=catalog, store=store, fact_records=tuple(records))


def retrieve_facts(
    *,
    needed_fact_keys: Sequence[str],
    active_brand: str,
    fact_store: FactStore,
    k: int = 12,
) -> RetrievalResult:
    brand = _normalize_brand(active_brand)
    store = fact_store.store.get(brand, {})
    facts: dict[str, str] = {}
    missing: list[str] = []
    matched: dict[str, tuple[str, ...]] = {}
    for required in tuple(dict.fromkeys(str(item or "").strip() for item in needed_fact_keys if str(item or "").strip())):
        candidate_keys = _matched_fact_keys(required, store)
        if not candidate_keys:
            missing.append(required)
            continue
        matched[required] = tuple(candidate_keys[:k])
        for key in candidate_keys[:k]:
            facts.setdefault(key, store[key])
    return RetrievalResult(facts=facts, missing=tuple(missing), matched_keys=matched)


def build_draft_prompt(
    *,
    conversation: Sequence[Mapping[str, str]],
    contract: AnswerContract,
    facts: Mapping[str, str],
    missing: Sequence[str],
    tone_guide: str = "",
    style_examples: Sequence[str] = (),
    toggles: Toggles | None = None,
) -> str:
    toggles = toggles or Toggles()
    hist = "\n".join(f"{item.get('role', '?')}: {item.get('text', '')}" for item in conversation)
    facts_block = "\n".join(f"- {key}: {value}" for key, value in facts.items()) or "(нет подтверждённых фактов под этот вопрос)"
    subquestions = "\n".join(
        f"- {item.text or contract.current_question} [{item.answerable}]"
        + (f"; тип: {item.question_type}" if item.question_type else "")
        + (f"; X: {item.existence_target}" if item.existence_target else "")
        + (f"; следующий шаг: {item.next_step}" if item.next_step else "")
        for item in contract.subquestions
    ) or f"- {contract.current_question}"
    assertable_slots = contract.assertable_slots() if toggles.enforce_slot_evidence else {
        name: slot.value for name, slot in contract.known_slots.items() if slot.value
    }
    examples = "\n".join(f"  • {item}" for item in style_examples if str(item).strip())
    return (
        f"Активный бренд: {contract.active_brand}. Не упоминай и не сравнивай с другим брендом.\n"
        "Задача: написать клиентский ответ живо, но только из фактов ниже.\n"
        f"Текущий вопрос: {contract.current_question}\n"
        f"Под-вопросы, ответь на каждый по сути:\n{subquestions}\n"
        + (f"Ситуация клиента: {contract.client_state} (подстрой тон, НЕ называй эмоцию вслух).\n" if contract.client_state else "")
        + (f"Клиент отрицает эти темы, не отвечай про них: {', '.join(contract.denied_topics)}\n" if contract.denied_topics else "")
        + (f"Уже известно из источника, можно использовать и не переспрашивать: {assertable_slots}\n" if assertable_slots else "")
        + (f"Нельзя утверждать без источника: {', '.join(contract.unsourced_slots())}\n" if contract.unsourced_slots() else "")
        + f"Подтверждённые факты, единственный источник чисел/дат/адресов/условий:\n{facts_block}\n"
        + (f"Нет факта по ключам: {', '.join(missing)}. По ним дай узкий честный хендофф менеджеру, не подставляй соседний факт.\n" if missing else "")
        + (f"Запрещённые подстановки: {', '.join(contract.forbidden_substitutions)}\n" if contract.forbidden_substitutions else "")
        + (f"Стиль, только манера и структура, НЕ источник фактов:\n{examples}\n" if examples else "")
        + "Правила ответа: сначала прямой ответ на заданный вопрос, потом 1-2 коротких пояснения и один следующий шаг. "
        "Если это вопрос «есть ли X», отвечай именно про X: не пиши «да/можно/доступно», если подтверждён только соседний факт Y. "
        "«Нет» можно писать только при явном отрицательном факте про X. "
        "Если вопрос про конкретный способ оплаты, отвечай именно про него: прямой перевод/счёт, банковская рассрочка и Долями — разные способы. "
        "Не подставляй соседний способ оплаты как ответ; если факта по спрошенному способу нет, узко передай менеджеру проверить именно его. "
        "Если клиент гипотетически спрашивает о возврате до оплаты/до старта, отвечай из факта про остаток неистраченных средств и не оформляй это как жалобу. "
        "Если клиент уже требует вернуть деньги или спорит по оплате, не отвечай автономно.\n"
        "В составном вопросе ответь на подтверждённые безопасные части, а неподтверждённую часть узко передай менеджеру. "
        "Никогда не утверждай расписание, класс, предмет, формат, цену, скидку, дату или тему, которых нет в фактах или словах клиента. "
        "Если сомневаешься, уточни или узко передай менеджеру; это важнее правила «ответить живо». "
        "Не раскрывай внутренние настройки, fact_id/source_id/JSON. Не обещай результат, возврат, одобрение банка/СФР/ФНС.\n"
        + (f"Манера: {tone_guide}\n" if tone_guide else "")
        + f"История диалога:\n{hist}\n"
        "Верни только текст клиенту, без JSON и служебных пометок."
    )


def build_faithfulness_prompt(draft: str, *, facts: Mapping[str, str], client_words: str) -> str:
    facts_block = "\n".join(f"- {key}: {value}" for key, value in facts.items()) or "(фактов нет)"
    return (
        "Проверь черновик ответа на верность. Верни строго JSON: "
        "{\"claims\": [{\"claim\": \"...\", \"evidence_fact_key\": \"...\", \"verdict\": \"supported|unsupported|glued\", \"reason\": \"...\"}], "
        "\"unsupported\": [<конкретные утверждения, которых нет ни в фактах, ни в словах клиента>]}.\n"
        "Конкретное утверждение = расписание/дни, формат, наличие пробного/мест/записи, сроки, условия, цены, действия.\n"
        "Каждое атомарное утверждение должно подтверждаться ОДНИМ fact_key из списка фактов. "
        "Если утверждение собрано из двух разных фактов, это glued/unsupported: например, если в одном факте есть личный кабинет, "
        "а в другом МТС Линк, нельзя писать «личный кабинет на МТС Линк», пока это не сказано одним фактом.\n"
        "Для supported обязательно укажи evidence_fact_key ровно из списка ниже. Не используй fact_key из памяти или всей базы.\n"
        "Не считай нарушением общую вежливость и предложение помочь.\n"
        f"Факты:\n{facts_block}\n"
        f"Слова клиента:\n{client_words}\n"
        f"Черновик:\n{draft}\n"
        "Только JSON."
    )


def check_claim_faithfulness(
    draft: str,
    *,
    facts: Mapping[str, str],
    client_words: str,
    faithfulness_fn: Callable[[str], object] | None,
) -> FaithfulnessResult:
    if faithfulness_fn is None:
        return FaithfulnessResult(unsupported=(), available=True)
    prompt = build_faithfulness_prompt(draft, facts=facts, client_words=client_words)
    try:
        raw = faithfulness_fn(prompt)
    except Exception:
        return FaithfulnessResult(unsupported=(), available=False)
    data: object = raw
    if isinstance(raw, str):
        try:
            data = _extract_json_object(raw)
        except Exception:
            return FaithfulnessResult(unsupported=(), available=False)
    claims: list[FaithfulnessClaim] = []
    unsupported: list[str] = []
    if isinstance(data, MappingABC) and "claims" in data:
        raw_claims = data.get("claims")
        if not isinstance(raw_claims, SequenceABC) or isinstance(raw_claims, (str, bytes, bytearray)):
            return FaithfulnessResult(unsupported=(), claims=(), available=False)
        for item in raw_claims:
            if not isinstance(item, MappingABC):
                return FaithfulnessResult(unsupported=(), claims=(), available=False)
            claim = str(item.get("claim") or "").strip()
            evidence_key = str(item.get("evidence_fact_key") or "").strip()
            verdict = str(item.get("verdict") or "").strip().casefold()
            reason = str(item.get("reason") or "").strip()
            if not claim:
                continue
            parsed = FaithfulnessClaim(
                claim=claim,
                evidence_fact_key=evidence_key,
                verdict=verdict,
                reason=reason,
            )
            claims.append(parsed)
            if verdict in {"unsupported", "glued", "not_supported", "false"}:
                unsupported.append(claim)
                continue
            if verdict != "supported":
                unsupported.append(claim)
                continue
            fact_text = facts.get(evidence_key)
            if not evidence_key or fact_text is None:
                unsupported.append(claim)
                continue
            if not claim_anchors_supported_by_fact(claim, fact_text):
                unsupported.append(claim)
        legacy_items = data.get("unsupported") or []
        if isinstance(legacy_items, SequenceABC) and not isinstance(legacy_items, (str, bytes, bytearray)):
            unsupported.extend(str(item).strip() for item in legacy_items if str(item).strip())
        elif legacy_items:
            return FaithfulnessResult(unsupported=(), claims=(), available=False)
        return FaithfulnessResult(
            unsupported=tuple(dict.fromkeys(item for item in unsupported if item)),
            claims=tuple(claims),
            available=True,
        )
    if isinstance(data, MappingABC) and "unsupported" in data:
        items = data.get("unsupported") or []
    elif isinstance(data, SequenceABC) and not isinstance(data, (str, bytes, bytearray)):
        items = data
    else:
        return FaithfulnessResult(unsupported=(), available=False)
    return FaithfulnessResult(
        unsupported=tuple(str(item).strip() for item in items if str(item).strip()),
        available=True,
    )


_RISKY_ENTITY_ALIASES: Mapping[str, tuple[str, ...]] = {
    "platform:mts_link": ("мтс линк", "мтс-линк", "mts link", "mts-link"),
    "platform:webinar": ("webinar", "webinar.ru"),
    "platform:zoom": ("zoom", "зум"),
    "platform:tallanto": ("tallanto", "талланто"),
    "platform:getcourse": ("getcourse", "геткурс"),
    "product:lvsh": ("лвш", "летняя выездная школа"),
    "product:formula_fizteha": ("формула физтеха",),
    "product:intensive": ("интенсив", "интенсивы"),
    "product:city_camp": ("городская летняя школа", "городской летний лагерь"),
    "address:sretenka": ("сретенка", "сретенке", "сретенский"),
    "address:patsaeva": ("пацаева",),
    "address:institutskiy": ("институтский пер", "институтский переулок"),
    "address:krasnoselskaya": ("верхняя красносельская",),
}
_ADDRESS_GENERIC_RE = re.compile(r"\b(?:ул\.|улиц[аеуы]|д\.|дом|каб\.|кабинет|метро)\s+[а-яa-z0-9-]+", re.I)
_ROLE_PERSON_RE = re.compile(
    r"\b(?:преподаватель|учитель|куратор|менеджер|администратор)\s+([А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+){1,2})"
)
_DATE_ANCHOR_RE = re.compile(
    r"(?<!\d)(\d{1,2})[. ](0?\d{1,2}|январ\w*|феврал\w*|март\w*|апрел\w*|ма[йя]|июн\w*|июл\w*|август\w*|сентябр\w*|октябр\w*|ноябр\w*|декабр\w*)(?:[. ](20\d{2}))?",
    re.I,
)
_CONDITION_ANCHOR_ALIASES: Mapping[str, tuple[str, ...]] = {
    "condition:weekdays": ("по будням", "будни", "будний"),
    "condition:weekends": ("по выходным", "выходные", "суббот", "воскрес"),
    "condition:evening": ("вечером", "вечернее"),
    "condition:morning": ("утром", "утреннее"),
    "condition:free": ("бесплат",),
    "condition:trial": ("пробное", "пробный", "фрагмент занятия"),
    "condition:refund": ("возврат", "вернуть деньги", "вернём деньги"),
    "condition:bank": ("банк", "т-банк", "рассроч"),
    "format:online": ("онлайн", "дистанционно"),
    "format:offline": ("очно", "очная", "очный"),
}
_SUBJECT_ANCHOR_ALIASES: Mapping[str, tuple[str, ...]] = {
    "subject:recording": ("запис", "пересмотр"),
    "subject:cabinet": ("личный кабинет", "личном кабинете", "личного кабинета"),
    "subject:matkap": ("маткап", "материнск", "сфр"),
    "subject:discount": ("скидк",),
    "subject:second_subject": ("второй предмет", "2-й предмет", "вторым предмет"),
    "subject:documents": ("документ", "заявлен", "договор"),
}
_MONTH_ANCHOR_BY_PREFIX = {
    "январ": "01",
    "феврал": "02",
    "март": "03",
    "апрел": "04",
    "ма": "05",
    "июн": "06",
    "июл": "07",
    "август": "08",
    "сентябр": "09",
    "октябр": "10",
    "ноябр": "11",
    "декабр": "12",
}


def claim_anchors_supported_by_fact(claim: str, fact_text: str) -> bool:
    claim_anchors = concrete_anchors(claim)
    if not claim_anchors:
        return True
    fact_anchors = concrete_anchors(fact_text)
    return claim_anchors <= fact_anchors


def concrete_anchors(text: str) -> set[str]:
    source = str(text or "")
    low = source.casefold().replace("ё", "е")
    anchors: set[str] = {f"number:{number}" for number in _numbers(source) if number not in {"2026", "2027"}}
    for match in _DATE_ANCHOR_RE.finditer(source):
        normalized = _normalize_date_anchor(match)
        if normalized:
            anchors.add(f"date:{normalized}")
    anchors.update(_entity_anchors(source))
    for key, aliases in _CONDITION_ANCHOR_ALIASES.items():
        if any(alias in low for alias in aliases):
            anchors.add(key)
    for key, aliases in _SUBJECT_ANCHOR_ALIASES.items():
        if any(alias in low for alias in aliases):
            anchors.add(key)
    return anchors


def new_concrete_anchors(candidate: str, *, original: str, facts: Mapping[str, str]) -> set[str]:
    allowed = concrete_anchors(original)
    for fact_text in facts.values():
        allowed.update(concrete_anchors(fact_text))
    return concrete_anchors(candidate) - allowed


def unsupported_named_entities(
    text: str,
    *,
    facts: Mapping[str, str],
    active_brand: str,
    client_message: str = "",
) -> list[str]:
    text_anchors = _entity_anchors(text)
    if not text_anchors:
        return []
    fact_anchors: set[str] = set()
    for fact_text in facts.values():
        fact_anchors.update(_entity_anchors(fact_text))
    allowed = set(fact_anchors)
    allowed.update(_active_brand_entity_anchors(active_brand))
    client_anchors = _entity_anchors(client_message)
    if client_anchors and _is_safe_entity_echo(text):
        allowed.update(client_anchors)
    return sorted(text_anchors - allowed)


def _entity_anchors(text: str) -> set[str]:
    source = str(text or "")
    low = source.casefold().replace("ё", "е")
    anchors: set[str] = set()
    for key, aliases in _RISKY_ENTITY_ALIASES.items():
        if any(alias in low for alias in aliases):
            anchors.add(key)
    if _ADDRESS_GENERIC_RE.search(source):
        anchors.add("address:generic")
    for match in _ROLE_PERSON_RE.finditer(source):
        name = _norm_text(match.group(1))
        if name:
            anchors.add(f"person:{name}")
    return anchors


def _active_brand_entity_anchors(active_brand: str) -> set[str]:
    brand = _normalize_brand(active_brand)
    if brand == "foton":
        return {"brand:foton"}
    if brand == "unpk":
        return {"brand:unpk"}
    return set()


def _is_safe_entity_echo(text: str) -> bool:
    low = str(text or "").casefold()
    return bool(re.search(r"менеджер|провер|уточн|подтверд", low, re.I))


def _normalize_date_anchor(match: re.Match[str]) -> str:
    try:
        day = int(match.group(1))
    except Exception:
        return ""
    if day < 1 or day > 31:
        return ""
    raw_month = match.group(2).casefold().replace("ё", "е")
    month = ""
    if raw_month.isdigit():
        value = int(raw_month)
        if 1 <= value <= 12:
            month = f"{value:02d}"
    else:
        for prefix, number in _MONTH_ANCHOR_BY_PREFIX.items():
            if raw_month.startswith(prefix):
                month = number
                break
    if not month:
        return ""
    year = match.group(3) or ""
    return f"{day:02d}.{month}" + (f".{year}" if year else "")


def form_check(draft: str, *, previous_bot_texts: Sequence[str] = ()) -> tuple[FormFinding, ...]:
    out: list[FormFinding] = []
    low = _norm_text(draft)
    if any(low.startswith(item) or item in low[:60] for item in _STOCK_OPENERS):
        out.append(FormFinding("stock_opener", "канцелярский штамп-зачин"))
    for previous in previous_bot_texts:
        prev = _norm_text(previous)
        if len(prev) > 25 and len(low) > 25 and _similarity(prev, low) > 0.85:
            out.append(FormFinding("near_repeat", "почти дословный повтор предыдущего ответа"))
            break
    if not re.search(r"[?]|подобрать|подскаж|помоч|следующий шаг|записать|уточн", low):
        out.append(FormFinding("no_next_step", "нет мягкого следующего шага"))
    if any(item in low for item in _CLERICAL):
        out.append(FormFinding("clerical", "канцелярит"))
    return tuple(out)


def build_warmth_prompt(
    draft: str,
    *,
    client_state: str,
    form_issues: Sequence[str],
    facts: Mapping[str, str],
) -> str:
    facts_block = "\n".join(f"- {key}: {value}" for key, value in facts.items()) or "(нет фактов)"
    return (
        "Перепиши ответ живее и теплее, сохранив весь смысл и факты. Меняй только форму.\n"
        "Фактические предложения исходного ответа копируй дословно: цены, даты, платформы, адреса, условия, формат, сроки, документы. "
        "Разрешено менять только зачин, связки, порядок коротких фраз и мягкий финальный следующий шаг.\n"
        f"Ситуация клиента: {client_state or 'обычная'} (подстрой регистр; не называй эмоцию вслух).\n"
        f"Что поправить по форме: {', '.join(form_issues) or 'тон/прямота'}.\n"
        "Жёстко: не вводи новых чисел/дат/имён/условий вне фактов; не упоминай другой бренд; "
        "не раскрывай ИИ; не обещай возврат/результат.\n"
        "Не склеивай разные факты в новое утверждение: если в фактах отдельно есть личный кабинет и отдельно МТС Линк, "
        "нельзя писать, что личный кабинет находится на МТС Линк, если это прямо не сказано. "
        "Не добавляй платформу, предмет, формат, срок или документ, которых нет в исходном ответе или одном подтверждённом факте.\n"
        "Сначала прямой ответ, потом 1-2 пояснения, один мягкий следующий шаг. Без штампов и канцелярита.\n"
        f"Факты, источник конкретики:\n{facts_block}\n"
        f"Ответ:\n{draft}\n"
        "Верни только переписанный текст."
    )


def warmth_rewrite(
    draft: str,
    *,
    client_state: str,
    form_issues: Sequence[str],
    facts: Mapping[str, str],
    warmth_fn: Callable[[str], str] | None,
) -> str | None:
    if warmth_fn is None:
        return None
    prompt = build_warmth_prompt(draft, client_state=client_state, form_issues=form_issues, facts=facts)
    try:
        candidate = str(warmth_fn(prompt) or "").strip()
    except Exception:
        return None
    return candidate or None


def run_pipeline(
    *,
    conversation: Sequence[Mapping[str, str]],
    active_brand: str,
    fact_store: FactStore,
    understand_fn: Callable[[str], object] | None,
    draft_fn: Callable[[str], str] | None,
    context: Mapping[str, Any] | None = None,
    tone_guide: str = "",
    style_examples: Sequence[str] = (),
    repair_fn: Callable[[str], str] | None = None,
    faithfulness_fn: Callable[[str], object] | None = None,
    warmth_fn: Callable[[str], str] | None = None,
    toggles: Toggles | None = None,
) -> DialogueContractPipelineResult:
    toggles = toggles or Toggles()
    client_words = str(conversation[-1].get("text") or "") if conversation else ""
    contract = understand(
        conversation=conversation,
        active_brand=active_brand,
        fact_key_catalog=fact_store.catalog,
        understand_fn=understand_fn,
        context=context,
    )
    if contract.is_p0:
        if _asks_refund_policy(contract) and not _current_refund_dispute_signal(
            client_words=client_words,
            contract=contract,
        ):
            contract = replace(contract, is_p0=False, p0_reason="", answerability="answer_self")
        else:
            return DialogueContractPipelineResult(
                draft_text=_dry_p0_text(conversation=conversation),
                route="manager_only",
                manager_only=True,
                contract=contract,
                fallback_reason="p0",
            )

    retrieval = retrieve_facts(
        needed_fact_keys=contract.all_needed_fact_keys(),
        active_brand=active_brand,
        fact_store=fact_store,
    )
    retrieval = _augment_with_soft_guidance(retrieval, contract=contract, active_brand=active_brand, fact_store=fact_store)
    retrieval = _augment_with_format_guidance(retrieval, contract=contract, active_brand=active_brand, fact_store=fact_store)
    retrieval = _augment_with_known_absence(retrieval, contract=contract, active_brand=active_brand, fact_store=fact_store)
    retrieval = _augment_with_presale_refund_policy(
        retrieval,
        contract=contract,
        active_brand=active_brand,
        fact_store=fact_store,
    )
    if _asks_refund_policy(contract) and not _presale_refund_policy_text(retrieval.facts):
        return DialogueContractPipelineResult(
            draft_text=_safe_fallback_text(contract, facts=retrieval.facts),
            route="draft_for_manager",
            manager_only=False,
            contract=contract,
            facts=retrieval.facts,
            missing=retrieval.missing,
            fallback_reason="refund_policy_manager_only",
        )
    slot_question = _single_missing_slot_question(contract, retrieval)
    if slot_question:
        return DialogueContractPipelineResult(
            draft_text=slot_question,
            route="bot_answer_self",
            manager_only=False,
            contract=contract,
            facts=retrieval.facts,
            missing=retrieval.missing,
            fallback_reason="single_missing_slot_question",
        )
    exact_answer_available = _has_exact_retrieved_answer_part(contract, retrieval)
    soft_weekend = _soft_weekend_guidance_text(retrieval.facts)
    if soft_weekend and _asks_weekend_or_slot(contract):
        fallback = _safe_fallback_text(contract, facts=retrieval.facts)
        return DialogueContractPipelineResult(
            draft_text=_avoid_repeating_text(fallback, conversation=conversation, contract=contract, facts=retrieval.facts),
            route="bot_answer_self",
            manager_only=False,
            contract=contract,
            facts=retrieval.facts,
            missing=retrieval.missing,
            fallback_reason="soft_weekend_guidance",
        )
    schedule_answer = _class_schedule_publication_answer(contract, retrieval.facts, conversation=conversation)
    if schedule_answer:
        return DialogueContractPipelineResult(
            draft_text=schedule_answer,
            route="bot_answer_self",
            manager_only=False,
            contract=contract,
            facts=retrieval.facts,
            missing=retrieval.missing,
            fallback_reason="schedule_publication_answer",
        )
    direct_answer = _direct_exact_fact_answer(contract, retrieval)
    if direct_answer:
        return DialogueContractPipelineResult(
            draft_text=_avoid_repeating_text(direct_answer, conversation=conversation, contract=contract, facts=retrieval.facts),
            route="bot_answer_self",
            manager_only=False,
            contract=contract,
            facts=retrieval.facts,
            missing=retrieval.missing,
            fallback_reason="direct_exact_fact_answer",
        )
    force_draft_for_manager = (
        contract.answerability != "answer_self"
        and not exact_answer_available
        and not _has_retrieved_self_answer_part(contract, retrieval)
        and not (_asks_refund_policy(contract) and _presale_refund_policy_text(retrieval.facts))
    )
    if force_draft_for_manager and (
        _asks_refund_policy(contract)
        or not retrieval.facts
        or (_soft_weekend_guidance_text(retrieval.facts) and not _has_self_answerable_subquestion(contract))
    ):
        fallback = _safe_fallback_text(contract, facts=retrieval.facts)
        return DialogueContractPipelineResult(
            draft_text=_avoid_repeating_text(fallback, conversation=conversation, contract=contract, facts=retrieval.facts),
            route="draft_for_manager",
            manager_only=False,
            contract=contract,
            facts=retrieval.facts,
            missing=retrieval.missing,
            fallback_reason="contract_manager_only",
        )
    if draft_fn is None:
        fallback = _safe_fallback_text(contract, facts=retrieval.facts)
        return DialogueContractPipelineResult(
            draft_text=_avoid_repeating_text(fallback, conversation=conversation, contract=contract, facts=retrieval.facts),
            route="draft_for_manager",
            manager_only=False,
            contract=contract,
            facts=retrieval.facts,
            missing=retrieval.missing,
            fallback_reason="no_draft_fn",
        )
    prompt = build_draft_prompt(
        conversation=conversation,
        contract=contract,
        facts=retrieval.facts,
        missing=retrieval.missing,
        tone_guide=tone_guide,
        style_examples=style_examples,
        toggles=toggles,
    )
    try:
        draft = str(draft_fn(prompt) or "").strip()
    except Exception:
        draft = ""
    if not draft:
        fallback = _safe_fallback_text(contract, facts=retrieval.facts)
        return DialogueContractPipelineResult(
            draft_text=_avoid_repeating_text(fallback, conversation=conversation, contract=contract, facts=retrieval.facts),
            route="draft_for_manager",
            manager_only=False,
            contract=contract,
            facts=retrieval.facts,
            missing=retrieval.missing,
            fallback_reason="draft_error",
        )

    draft = _specialize_grade_range_answer(draft, contract=contract, facts=retrieval.facts)
    repaired = False
    findings, unsupported, semantic_available = _hard_check(
        draft,
        facts=retrieval.facts,
        contract=contract,
        client_words=client_words,
        faithfulness_fn=faithfulness_fn,
        toggles=toggles,
        context=context,
    )
    if not semantic_available:
        fallback = _safe_fallback_text(contract, facts=retrieval.facts)
        return DialogueContractPipelineResult(
            draft_text=_avoid_repeating_text(fallback, conversation=conversation, contract=contract, facts=retrieval.facts),
            route="draft_for_manager",
            manager_only=False,
            contract=contract,
            facts=retrieval.facts,
            missing=retrieval.missing,
            fallback_reason="semantic_check_unavailable",
        )

    attempts = 0
    while (findings or unsupported) and repair_fn is not None and attempts < MAX_REPAIR_ATTEMPTS:
        attempts += 1
        instr = "; ".join(
            [finding.detail for finding in findings]
            + [f"неподтверждённое утверждение: {item}" for item in unsupported]
        )
        try:
            candidate = str(repair_fn(_repair_prompt(draft, instr, retrieval.facts)) or "").strip()
        except Exception:
            break
        if not candidate:
            break
        draft = candidate
        repaired = True
        findings, unsupported, semantic_available = _hard_check(
            draft,
            facts=retrieval.facts,
            contract=contract,
            client_words=client_words,
            faithfulness_fn=faithfulness_fn,
            toggles=toggles,
            context=context,
        )
        if not semantic_available:
            fallback = _safe_fallback_text(contract, facts=retrieval.facts)
            return DialogueContractPipelineResult(
                draft_text=_avoid_repeating_text(fallback, conversation=conversation, contract=contract, facts=retrieval.facts),
                route="draft_for_manager",
                manager_only=False,
                contract=contract,
                facts=retrieval.facts,
                missing=retrieval.missing,
                repaired=repaired,
                fallback_reason="semantic_check_unavailable",
            )

    if findings or unsupported:
        verified_fallback = _hard_failure_exact_fact_fallback(contract, retrieval)
        if verified_fallback and _can_autonomously_replace_failed_draft(findings):
            fallback_findings, fallback_unsupported, fallback_semantic_available = _hard_check(
                verified_fallback,
                facts=retrieval.facts,
                contract=contract,
                client_words=client_words,
                faithfulness_fn=faithfulness_fn,
                toggles=toggles,
                context=context,
            )
            if fallback_semantic_available and not fallback_findings and not fallback_unsupported:
                return DialogueContractPipelineResult(
                    draft_text=_avoid_repeating_text(
                        verified_fallback,
                        conversation=conversation,
                        contract=contract,
                        facts=retrieval.facts,
                    ),
                    route="bot_answer_self",
                    manager_only=False,
                    contract=contract,
                    facts=retrieval.facts,
                    missing=retrieval.missing,
                    findings=tuple(findings),
                    unsupported_claims=tuple(unsupported),
                    repaired=repaired,
                    fallback_reason="verified_fact_fallback_after_hard_check",
                )
        fallback = _safe_fallback_text(contract, facts=retrieval.facts)
        return DialogueContractPipelineResult(
            draft_text=_avoid_repeating_text(fallback, conversation=conversation, contract=contract, facts=retrieval.facts),
            route="draft_for_manager",
            manager_only=False,
            contract=contract,
            facts=retrieval.facts,
            missing=retrieval.missing,
            findings=tuple(findings),
            unsupported_claims=tuple(unsupported),
            repaired=repaired,
            fallback_reason="hard_verification_failed",
        )

    form_findings: tuple[FormFinding, ...] = ()
    warmed = False
    warmth_attempted = False
    warmth_mode = _normalize_warmth_mode(toggles.warmth_mode)
    warmth_rejected_reason = ""
    warmth_rejected_findings: tuple[VerificationFinding, ...] = ()
    warmth_rejected_unsupported: tuple[str, ...] = ()
    warmth_semantic_available = True
    if toggles.form_warmth:
        previous_bot_texts = [item.get("text", "") for item in conversation if item.get("role") == "bot"]
        form_findings = form_check(draft, previous_bot_texts=previous_bot_texts)
        should_attempt_warmth = (
            warmth_fn is not None
            and not force_draft_for_manager
            and (warmth_mode == "all_eligible" or bool(form_findings))
        )
        if should_attempt_warmth:
            warmth_attempted = True
            warm_candidate = warmth_rewrite(
                draft,
                client_state=contract.client_state,
                form_issues=[finding.code for finding in form_findings],
                facts=retrieval.facts,
                warmth_fn=warmth_fn,
            )
            if not warm_candidate:
                warmth_rejected_reason = "empty_candidate"
            else:
                warm_findings, warm_unsupported, warm_semantic_available = _hard_check(
                    warm_candidate,
                    facts=retrieval.facts,
                    contract=contract,
                    client_words=client_words,
                    faithfulness_fn=faithfulness_fn,
                    toggles=toggles,
                    context=context,
                )
                added_warm_anchors = new_concrete_anchors(warm_candidate, original=draft, facts=retrieval.facts)
                if warm_semantic_available and not warm_findings and (not warm_unsupported or not added_warm_anchors):
                    draft = _specialize_grade_range_answer(warm_candidate, contract=contract, facts=retrieval.facts)
                    warmed = True
                else:
                    warmth_rejected_findings = tuple(warm_findings)
                    warmth_rejected_unsupported = tuple(warm_unsupported)
                    if added_warm_anchors:
                        warmth_rejected_reason = "new_concrete_anchor"
                    elif not warm_semantic_available:
                        warmth_rejected_reason = "semantic_check_unavailable"
                    elif warm_findings:
                        warmth_rejected_reason = "hard_check_failed"
                    elif warm_unsupported:
                        warmth_rejected_reason = "unsupported_claims"
                    else:
                        warmth_rejected_reason = "unknown_rejection"

    return DialogueContractPipelineResult(
        draft_text=_avoid_repeating_text(draft, conversation=conversation, contract=contract, facts=retrieval.facts),
        route="draft_for_manager" if force_draft_for_manager else "bot_answer_self",
        manager_only=False,
        contract=contract,
        facts=retrieval.facts,
        missing=retrieval.missing,
        form_findings=form_findings,
        warmed=warmed,
        warmth_attempted=warmth_attempted,
        warmth_mode=warmth_mode,
        warmth_rejected_reason=warmth_rejected_reason,
        warmth_rejected_findings=warmth_rejected_findings,
        warmth_rejected_unsupported=warmth_rejected_unsupported,
        warmth_semantic_available=warmth_semantic_available,
        repaired=repaired,
    )


def verify_output(
    draft_text: str,
    *,
    facts: Mapping[str, str],
    active_brand: str,
    contract: AnswerContract | None = None,
    denied_topics: Sequence[str] = (),
    forbidden_substitutions: Sequence[str] = (),
    client_message: str = "",
    context: Mapping[str, Any] | None = None,
) -> list[VerificationFinding]:
    text = str(draft_text or "")
    low = text.casefold()
    findings: list[VerificationFinding] = []
    brand = _normalize_brand(active_brand)
    for token in _BRAND_TOKENS.get(brand, ()):
        if _brand_token_present(low, token):
            findings.append(VerificationFinding("brand_leak", f"чужой бренд/токен: {token}"))
            break
    backed_numbers = _numbers(" ".join(str(value) for value in facts.values()))
    client_numbers = _numbers(client_message)
    introduced = _numbers(text) - backed_numbers
    introduced = {num for num in introduced if not _is_allowed_ungrounded_number(num, client_numbers=client_numbers)}
    if introduced:
        findings.append(VerificationFinding("fact_grounding", f"числа вне подтверждённых фактов: {sorted(introduced)}"))
    unsupported_entities = unsupported_named_entities(
        text,
        facts=facts,
        active_brand=active_brand,
        client_message=client_message,
    )
    if unsupported_entities:
        findings.append(VerificationFinding("unsupported_entity", f"сущность вне фактов хода: {unsupported_entities}"))
    if contract is not None:
        findings.extend(_wrong_intent_fact_findings(text, contract=contract, facts=facts))
    for topic in tuple(denied_topics) + tuple(forbidden_substitutions):
        normalized = str(topic or "").strip().casefold()
        if normalized and normalized in low:
            findings.append(VerificationFinding("forbidden_scope", f"ответ затрагивает запрещённую тему: {topic}"))
            break
    if has_meta_leak(text) or _sanitize_blocks(text) or any(marker in low for marker in _META_MARKERS):
        findings.append(VerificationFinding("meta_leak", "служебная пометка или сырой JSON/fact_id/source_id"))
    if _AI_SELF_DISCLOSURE_RE.search(text) and not _client_asked_identity(client_message):
        findings.append(VerificationFinding("ai_disclosure", "самораскрытие без прямого вопроса клиента"))
    if _P0_PROMISE_RE.search(text):
        findings.append(VerificationFinding("p0_promise", "обещание возврата/результата/поступления"))
    safety = classify_answer_safety(client_message=client_message, context=context, route="bot_answer_self")
    if safety.p0_required and not p0_pre_gate(client_message, context=context):
        findings.append(VerificationFinding("p0_semantic_risk", "семантический P0 требует менеджера"))
    return findings


def _hard_check(
    draft: str,
    *,
    facts: Mapping[str, str],
    contract: AnswerContract,
    client_words: str,
    faithfulness_fn: Callable[[str], object] | None,
    toggles: Toggles,
    context: Mapping[str, Any] | None,
) -> tuple[tuple[VerificationFinding, ...], tuple[str, ...], bool]:
    verification_text = _handoff_factual_claim_text(draft)
    pure_handoff = _is_pure_handoff_text(draft) and verification_text is None
    text_to_check = verification_text or draft
    findings = list(
        verify_output(
            text_to_check,
            facts=facts,
            active_brand=contract.active_brand,
            contract=contract,
            denied_topics=contract.denied_topics,
            forbidden_substitutions=contract.forbidden_substitutions,
            client_message=client_words,
            context=context,
        )
    )
    findings.extend(_existence_yes_no_findings(text_to_check, contract=contract, facts=facts))
    findings.extend(_payment_method_findings(text_to_check, contract=contract, facts=facts))
    unsupported: tuple[str, ...] = ()
    semantic_available = True
    if toggles.semantic_faithfulness and not pure_handoff:
        result = check_claim_faithfulness(
            text_to_check,
            facts=facts,
            client_words=client_words,
            faithfulness_fn=faithfulness_fn,
        )
        unsupported = _unsupported_claims_without_current_fact_support(
            result.unsupported,
            facts=facts,
            contract=contract,
        )
        semantic_available = result.available
    return tuple(findings), unsupported, semantic_available


def _unsupported_claims_without_current_fact_support(
    unsupported: Sequence[str],
    *,
    facts: Mapping[str, str],
    contract: AnswerContract,
) -> tuple[str, ...]:
    kept: list[str] = []
    for claim in unsupported:
        text = str(claim or "").strip()
        if not text:
            continue
        if _claim_supported_by_current_subquestion_fact(text, facts=facts, contract=contract):
            continue
        kept.append(text)
    return tuple(dict.fromkeys(kept))


def _claim_supported_by_current_subquestion_fact(
    claim: str,
    *,
    facts: Mapping[str, str],
    contract: AnswerContract,
) -> bool:
    if not concrete_anchors(claim):
        return False
    for fact_key, fact_text in facts.items():
        if not claim_anchors_supported_by_fact(claim, fact_text):
            continue
        if _fact_matches_current_subquestion(str(fact_key), str(fact_text or ""), contract=contract, claim=claim):
            return True
    return False


def _fact_matches_current_subquestion(
    fact_key: str,
    fact_text: str,
    *,
    contract: AnswerContract,
    claim: str,
) -> bool:
    subquestions = contract.subquestions or (
        Subquestion(
            text=contract.current_question,
            needed_fact_keys=contract.needed_fact_keys,
            question_type=contract.question_type,
            existence_target=contract.existence_target,
        ),
    )
    fact_topics = _semantic_topic_anchors(f"{fact_key} {fact_text}")
    claim_topics = _semantic_topic_anchors(claim)
    for subquestion in subquestions:
        key_matches_subquestion = any(
            _fact_key_matches_required_key(fact_key, required)
            for required in subquestion.needed_fact_keys
            if required
        )
        subq_text = " ".join(
            part
            for part in (subquestion.text, subquestion.existence_target, contract.current_question)
            if part
        )
        subq_topics = _semantic_topic_anchors(subq_text)
        specific_subq_topics = _specific_semantic_topics(subq_topics)
        specific_claim_topics = _specific_semantic_topics(claim_topics)
        if specific_subq_topics and not (specific_subq_topics & fact_topics):
            continue
        if specific_claim_topics and not (specific_claim_topics & fact_topics):
            continue
        shared_subq_fact = subq_topics & fact_topics
        shared_subq_claim = subq_topics & claim_topics
        if key_matches_subquestion and (not subq_topics or shared_subq_fact or shared_subq_claim):
            return True
        if shared_subq_fact and (not claim_topics or claim_topics & fact_topics) and (not claim_topics or shared_subq_claim):
            return True
    return False


def _specific_semantic_topics(topics: set[str]) -> set[str]:
    generic = {"topic:discount", "topic:price", "period:semester", "period:year"}
    return {topic for topic in topics if topic not in generic}


def _fact_key_matches_required_key(fact_key: str, required_key: str) -> bool:
    if str(fact_key or "").strip() == str(required_key or "").strip():
        return True
    return key_matches(required_key, fact_key)


def _semantic_topic_anchors(text: str) -> set[str]:
    source = str(text or "").casefold().replace("ё", "е")
    anchors: set[str] = set()
    if re.search(r"скидк|discount", source, re.I):
        anchors.add("topic:discount")
    if re.search(
        r"многодет|двое\s+дет|двумя\s+детьми|двух\s+дет|два\s+реб[её]н|2\s*(?:реб[её]н|дет)|семейн",
        source,
        re.I,
    ):
        anchors.add("topic:discount_family")
    if re.search(r"втор\w+\s+предмет|2-?й\s+предмет|second[_\s-]?subject", source, re.I):
        anchors.add("topic:discount_second_subject")
    if re.search(r"друг|refer|привед", source, re.I):
        anchors.add("topic:discount_referral")
    if re.search(r"цен|стоим|сколько\s+стоит|price|tuition|руб|₽", source, re.I):
        anchors.add("topic:price")
    if re.search(r"семестр|semester", source, re.I):
        anchors.add("period:semester")
    if re.search(r"\bгод\b|year", source, re.I):
        anchors.add("period:year")
    if re.search(r"онлайн|online", source, re.I):
        anchors.add("format:online")
    if re.search(r"очно|очная|очный|offline|ochno", source, re.I):
        anchors.add("format:offline")
    if re.search(r"рассроч|installment|банк|т-банк|tbank|t-bank", source, re.I):
        anchors.add("payment:installment")
    if re.search(r"долями|dolyami", source, re.I):
        anchors.add("payment:dolyami")
    if re.search(r"перевод|по\s+сч[её]ту|квитанц|реквизит|invoice", source, re.I):
        anchors.add("payment:invoice")
    if re.search(r"запис|пересмотр|recording", source, re.I):
        anchors.add("topic:recording")
    if re.search(r"расписан|дни\s+занят|по\s+дням|schedule", source, re.I):
        anchors.add("topic:schedule")
    if re.search(r"адрес|где\s+вы|находит|метро|location|address", source, re.I):
        anchors.add("topic:address")
    for match in re.finditer(r"(?<!\d)([1-9]|10|11)\s*(?:класс|кл\b|class)", source, re.I):
        anchors.add(f"class:{match.group(1)}")
    for match in re.finditer(r"(?:grade|class)[_.\s-]?([1-9]|10|11)", source, re.I):
        anchors.add(f"class:{match.group(1)}")
    return anchors


def _handoff_factual_claim_text(text: str) -> str | None:
    source = " ".join(str(text or "").split())
    if not source or not _is_handoff_text(source):
        return None
    parts = [
        part.strip(" \t\n\r-—:;,.")
        for part in re.split(r"[.;]\s+|\s+[—-]\s+", source)
        if part.strip(" \t\n\r-—:;,.")
    ]
    claim_parts = [
        part
        for part in parts
        if _FACTUAL_CLAIM_RE.search(part) and not _is_handoff_text(part)
    ]
    if claim_parts:
        return ". ".join(claim_parts)
    if _FACTUAL_CLAIM_RE.search(source) and not _is_pure_handoff_text(source):
        return source
    return None


def _is_pure_handoff_text(text: str) -> bool:
    low = str(text or "").casefold()
    return (
        _is_handoff_text(low)
        and not re.search(r"\b(?:не\s+знаю|нет\s+(?:информации|данных|ответа)|не\s+могу\s+ответить)\b", low, re.I)
        and not _FACTUAL_CLAIM_RE.search(low)
    )


def _dry_p0_text(*, conversation: Sequence[Mapping[str, str]] | None = None) -> str:
    bot_turns = 0
    if conversation:
        bot_turns = sum(1 for item in conversation if str(item.get("role") or "") == "bot")
    return _DRY_P0_TEXTS[bot_turns % len(_DRY_P0_TEXTS)]


_REFUND_POLICY_TEXTS: tuple[str, ...] = (
    "Порядок возврата или отмены до начала занятий подтвердит менеджер по договору. Не буду подменять это общими правилами курса — передам вопрос именно про возврат.",
    "По возврату и отмене лучше не отвечать общими правилами курса. Передам менеджеру именно этот вопрос, он сверит условия по договору.",
    "Возврат и отмена зависят от условий договора и выбранного курса. Передам менеджеру именно эту тему, без обещаний по сумме или решению.",
)


def _refund_policy_handoff_text(*, conversation: Sequence[Mapping[str, str]] | None = None) -> str:
    bot_turns = 0
    if conversation:
        bot_turns = sum(1 for item in conversation if str(item.get("role") or "") == "bot")
    return _REFUND_POLICY_TEXTS[bot_turns % len(_REFUND_POLICY_TEXTS)]


def _safe_fallback_text(contract: AnswerContract, *, facts: Mapping[str, str] | None = None) -> str:
    known_absence = _known_absence_text(contract, facts or {})
    if known_absence:
        return known_absence
    presale_refund = _presale_refund_policy_text(facts or {})
    if presale_refund and _asks_refund_policy(contract):
        return presale_refund
    soft_weekend = _soft_weekend_guidance_text(facts or {})
    if soft_weekend and _asks_weekend_or_slot(contract):
        return (
            "По общему ориентиру бывают разные варианты слотов, в том числе по выходным. "
            "Но точное расписание конкретной группы без проверки не подтверждаю — менеджер сверит ваш класс, предмет и площадку."
        )
    schedule_publication = _class_schedule_publication_answer(contract, facts or {})
    if schedule_publication:
        return schedule_publication
    if _asks_refund_policy(contract):
        return _refund_policy_handoff_text()
    detail = _client_safe_question_detail(contract.current_question)
    secondary = _secondary_fact_text(contract, facts or {})
    if secondary:
        detail_part = f": {detail}" if detail else ""
        return (
            f"По спрошенному пункту точного подтверждения нет — менеджер уточнит точную деталь{detail_part}. "
            f"Из подтверждённого, как отдельная справка: {secondary} "
            "Если хотите, передам менеджеру именно ваш способ или условие."
        )
    if detail:
        return f"Сейчас точно ответить не могу. Передам менеджеру уточнить точную деталь: {detail}. Он свяжется с вами."
    return "Сейчас точно ответить не могу. Передам вопрос менеджеру — он уточнит и свяжется с вами."


def _client_safe_question_detail(value: str, *, max_chars: int = 120) -> str:
    text = " ".join(str(value or "").split())
    if not text:
        return ""
    text = re.sub(
        r"^\s*клиент\s+(?:спрашивает|уточняет|интересуется|хочет\s+понять|просит\s+уточнить)\s*(?:,|:|—|-)?\s*",
        "",
        text,
        flags=re.I,
    ).strip(" \t\n\r:;,.—-")
    if not text or text.casefold().startswith("клиент "):
        return ""
    if len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "…"
    return text


def _secondary_fact_text(contract: AnswerContract, facts: Mapping[str, str]) -> str:
    if not facts:
        return ""
    payment_targets = _payment_method_target_anchors(contract)
    if payment_targets:
        for key, text in facts.items():
            if _is_camp_or_lvsh_fact(key, str(text or "")) and not _contract_mentions_camp_or_lvsh(contract):
                continue
            fact_anchors = _payment_method_anchors_from_text(str(text or ""))
            if fact_anchors and not payment_targets.issubset(fact_anchors):
                return _short_fact_sentence(str(text or ""))
    return ""


def _short_fact_sentence(text: str, *, max_chars: int = 170) -> str:
    cleaned = " ".join(str(text or "").split())
    first = re.split(r"(?<=[.!?])\s+", cleaned, maxsplit=1)[0].strip()
    value = first or cleaned
    if len(value) > max_chars:
        value = value[: max_chars - 1].rstrip() + "…"
    return value


def _avoid_repeating_text(
    text: str,
    *,
    conversation: Sequence[Mapping[str, str]] | None,
    contract: AnswerContract,
    facts: Mapping[str, str] | None = None,
) -> str:
    source = str(text or "").strip()
    if not source or not _is_handoff_text(source):
        return source
    prior_bot_texts = [str(item.get("text") or "") for item in (conversation or ()) if str(item.get("role") or "") == "bot"]
    if not any(_near_repeat(source, prior) for prior in prior_bot_texts[-4:]):
        return source
    if _asks_refund_policy(contract) and _presale_refund_policy_text(facts or {}):
        fact = _presale_refund_policy_text(facts or {})
        return (
            f"По возврату ориентир тот же: {_short_fact_sentence(fact)} "
            "Точные пункты договора менеджер подтвердит по выбранному курсу."
        )
    return (
        "Не буду повторять общий ответ: точную деталь по этому вопросу подтвердит менеджер. "
        "Передам ему именно этот пункт."
    )


def _is_handoff_text(text: str) -> bool:
    low = str(text or "").casefold()
    return bool(re.search(r"менеджер|передам|уточнит|подтвердит|сверит", low, re.I))


def _near_repeat(left: str, right: str) -> bool:
    left_norm = _repeat_norm(left)
    right_norm = _repeat_norm(right)
    if not left_norm or not right_norm:
        return False
    if left_norm == right_norm:
        return True
    if _is_handoff_text(left) and _is_handoff_text(right):
        if _payment_method_anchors_from_text(left) & _payment_method_anchors_from_text(right):
            return True
    left_tokens = set(left_norm.split())
    right_tokens = set(right_norm.split())
    if len(left_tokens | right_tokens) < 5:
        return False
    if len(left_tokens & right_tokens) / max(1, min(len(left_tokens), len(right_tokens))) >= 0.78:
        return True
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens) >= 0.82


def _repeat_norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-zа-яё0-9]+", " ", str(text or "").casefold().replace("ё", "е"))).strip()


def _specialize_grade_range_answer(draft: str, *, contract: AnswerContract, facts: Mapping[str, str]) -> str:
    grade = _client_grade_from_contract(contract)
    if not grade:
        return draft
    value = int(grade)
    fact_text = " ".join(str(item or "") for item in facts.values())
    supported_ranges: list[tuple[int, int]] = []
    for match in re.finditer(r"\b(\d{1,2})\s*[–-]\s*(\d{1,2})\s+класс", fact_text, re.I):
        low, high = int(match.group(1)), int(match.group(2))
        if low <= value <= high:
            supported_ranges.append((low, high))
    if not supported_ranges:
        return draft
    result = str(draft or "")
    for low, high in supported_ranges:
        result = re.sub(
            rf"\b{low}\s*[–-]\s*{high}\s+классов\b",
            f"{value} класса",
            result,
            flags=re.I,
        )
        result = re.sub(
            rf"\b{low}\s*[–-]\s*{high}\s+класс\b",
            f"{value} класс",
            result,
            flags=re.I,
        )
    return result


def _client_grade_from_contract(contract: AnswerContract) -> str:
    text = " ".join(
        part
        for part in (
            contract.current_question,
            contract.client_state,
            " ".join(item.text for item in contract.subquestions),
            " ".join(str(slot.value) for slot in contract.known_slots.values() if slot.source),
        )
        if part
    )
    match = re.search(r"(?<!\d)([1-9]|10|11)\s*(?:класс|кл\b)", text, re.I)
    if match:
        return match.group(1)
    return ""


def _augment_with_soft_guidance(
    retrieval: RetrievalResult,
    *,
    contract: AnswerContract,
    active_brand: str,
    fact_store: FactStore,
) -> RetrievalResult:
    if not _asks_weekend_or_slot(contract):
        return retrieval
    brand = _normalize_brand(active_brand)
    store = fact_store.store.get(brand, {})
    extra: dict[str, str] = {}
    for key, text in store.items():
        key_low = str(key or "").casefold()
        text_low = str(text or "").casefold()
        if "objection" not in key_low and "возраж" not in text_low:
            continue
        if "выход" not in text_low and "слот" not in text_low:
            continue
        extra[key] = text
        if len(extra) >= 2:
            break
    if not extra:
        return retrieval
    facts = dict(retrieval.facts)
    for key, text in extra.items():
        facts.setdefault(key, text)
    matched = dict(retrieval.matched_keys)
    matched["soft_guidance.weekend_slots"] = tuple(extra.keys())
    return RetrievalResult(facts=facts, missing=retrieval.missing, matched_keys=matched)


def _augment_with_format_guidance(
    retrieval: RetrievalResult,
    *,
    contract: AnswerContract,
    active_brand: str,
    fact_store: FactStore,
) -> RetrievalResult:
    if not _asks_training_format_choice(contract):
        return retrieval
    brand = _normalize_brand(active_brand)
    store = fact_store.store.get(brand, {})
    extra: dict[str, str] = {}
    for key, text in store.items():
        combined = f"{key} {text}".casefold().replace("ё", "е")
        if "online_courses_format" in combined or "онлайн-курсы" in combined:
            extra[key] = text
            break
    if not extra:
        return retrieval
    facts = dict(retrieval.facts)
    for key, text in extra.items():
        facts.setdefault(key, text)
    matched = dict(retrieval.matched_keys)
    matched["format.online"] = tuple(extra.keys())
    return RetrievalResult(facts=facts, missing=retrieval.missing, matched_keys=matched)


def _augment_with_known_absence(
    retrieval: RetrievalResult,
    *,
    contract: AnswerContract,
    active_brand: str,
    fact_store: FactStore,
) -> RetrievalResult:
    if not _is_existence_yes_no_contract(contract):
        return retrieval
    target_anchors = _existence_target_anchors(contract)
    if not target_anchors:
        return retrieval
    brand = _normalize_brand(active_brand)
    store = fact_store.store.get(brand, {})
    extra: dict[str, str] = {}
    for key, text in store.items():
        if key in retrieval.facts:
            continue
        if _is_negative_existence_fact_for_target(text, target_anchors=target_anchors):
            extra[key] = text
            break
    if not extra:
        return retrieval
    facts = dict(retrieval.facts)
    facts.update(extra)
    matched = dict(retrieval.matched_keys)
    matched["known_absence.existence_yes_no"] = tuple(extra.keys())
    return RetrievalResult(facts=facts, missing=retrieval.missing, matched_keys=matched)


def _augment_with_presale_refund_policy(
    retrieval: RetrievalResult,
    *,
    contract: AnswerContract,
    active_brand: str,
    fact_store: FactStore,
) -> RetrievalResult:
    if not _asks_refund_policy(contract):
        return retrieval
    if _presale_refund_policy_text(retrieval.facts):
        return retrieval
    brand = _normalize_brand(active_brand)
    store = fact_store.store.get(brand, {})
    extra: dict[str, str] = {}
    for key, text in store.items():
        combined = f"{key} {text}".casefold().replace("ё", "е")
        if "refund_presale_policy" in combined or (
            "остаток неистраченных средств" in combined and "возврат" in combined
        ):
            extra[key] = text
            break
    if not extra:
        return retrieval
    facts = dict(retrieval.facts)
    facts.update(extra)
    matched = dict(retrieval.matched_keys)
    matched["refund_policy.current"] = tuple(extra.keys())
    return RetrievalResult(facts=facts, missing=tuple(item for item in retrieval.missing if item != "refund_policy.current"), matched_keys=matched)


def _asks_weekend_or_slot(contract: AnswerContract) -> bool:
    text = " ".join(
        [
            contract.current_question,
            contract.client_state,
            " ".join(contract.continued_topics),
            " ".join(item.text for item in contract.subquestions),
        ]
    ).casefold()
    return bool(re.search(r"выходн|суббот|воскрес|слот", text, re.I))


def _soft_weekend_guidance_text(facts: Mapping[str, str]) -> str:
    for key, text in facts.items():
        combined = f"{key} {text}".casefold()
        if ("objection" in combined or "возраж" in combined) and ("выход" in combined or "слот" in combined):
            return str(text or "")
    return ""


def _wrong_intent_fact_findings(
    draft: str,
    *,
    contract: AnswerContract,
    facts: Mapping[str, str],
) -> list[VerificationFinding]:
    findings: list[VerificationFinding] = []
    if _asks_class_schedule_days(contract) and _draft_uses_contact_hours_as_schedule(draft, facts):
        findings.append(
            VerificationFinding(
                "wrong_intent_fact",
                "Контактные часы нельзя выдавать как дни занятий группы.",
            )
        )
    if not _asks_address(contract) and _draft_uses_address_fact(draft, facts):
        findings.append(
            VerificationFinding(
                "wrong_intent_fact",
                "Адресный факт нельзя выдавать как ответ на неадресный вопрос.",
            )
        )
    if not _contract_mentions_camp_or_lvsh(contract) and _draft_uses_camp_or_lvsh_fact(draft, facts):
        findings.append(
            VerificationFinding(
                "wrong_intent_fact",
                "Лагерный/ЛВШ факт нельзя выдавать как справку вне лагерного контекста.",
            )
        )
    return findings


def _class_schedule_publication_answer(
    contract: AnswerContract,
    facts: Mapping[str, str],
    *,
    conversation: Sequence[Mapping[str, str]] | None = None,
) -> str:
    if not _asks_class_schedule_days(contract):
        return ""
    for key, text in facts.items():
        combined = f"{key} {text}".casefold().replace("ё", "е")
        if not re.search(r"расписани", combined, re.I):
            continue
        if not re.search(r"появ|опубли|июн|середин[ае]\s+сентябр", combined, re.I):
            continue
        if re.search(r"контакт|contacts|10[:.]?00|18[:.]?00|пн\s*[–-]\s*вс", combined, re.I):
            continue
        fact = _short_fact_sentence(str(text or ""), max_chars=220)
        if not fact:
            continue
        prefix = _format_context_prefix(contract, facts)
        answer = f"{prefix}{fact} Точные дни конкретной группы сейчас не подтверждаю."
        prior_bot_texts = [str(item.get("text") or "") for item in (conversation or ()) if str(item.get("role") or "") == "bot"]
        if any(_near_repeat(answer, prior) for prior in prior_bot_texts[-4:]):
            return (
                f"{prefix}По дням точного ответа пока нет: расписание опубликуют в июне. "
                "Без подтверждения не буду называть будни или выходные как факт."
            )
        return answer
    return ""


def _format_context_prefix(contract: AnswerContract, facts: Mapping[str, str]) -> str:
    if not _asks_training_format_choice(contract):
        return ""
    facts_text = " ".join(str(value or "") for value in facts.values()).casefold().replace("ё", "е")
    parts: list[str] = []
    if re.search(r"онлайн-?курс|онлайн\s+формат|online", facts_text, re.I):
        parts.append("Есть онлайн-формат.")
    if re.search(r"очн\w+\s+курс|очные\s+курсы|очно", facts_text, re.I):
        parts.append("Есть очные курсы.")
    return (" ".join(dict.fromkeys(parts)) + " ") if parts else ""


def _asks_training_format_choice(contract: AnswerContract) -> bool:
    text = _contract_intent_text(contract)
    return bool(re.search(r"онлайн\s+или\s+очно|очно\s+или\s+онлайн|онлайн.+очно|очно.+онлайн|формат", text, re.I))


def _asks_class_schedule_days(contract: AnswerContract) -> bool:
    text = _contract_intent_text(contract)
    if re.search(r"контакт|на\s+связи|звонить|телефон|офис\s+работ", text, re.I):
        return False
    return bool(
        re.search(r"по\s+каким\s+дням|дни\s+занят|когда\s+занят|расписани", text, re.I)
        and re.search(r"занят|групп|курс|класс|предмет|математ|физик|информат|очно|онлайн", text, re.I)
    )


def _contract_intent_text(contract: AnswerContract) -> str:
    return " ".join(
        part
        for part in (
            contract.current_question,
            contract.client_state,
            contract.existence_target,
            " ".join(contract.continued_topics),
            " ".join(contract.switched_topics),
            " ".join(item.text for item in contract.subquestions),
            " ".join(item.existence_target for item in contract.subquestions),
        )
        if part
    ).casefold().replace("ё", "е")


def _draft_uses_contact_hours_as_schedule(draft: str, facts: Mapping[str, str]) -> bool:
    text = str(draft or "").casefold().replace("ё", "е")
    contact_values = [
        str(value or "").casefold().replace("ё", "е")
        for key, value in facts.items()
        if re.search(r"contact|contacts|schedule|режим|график", str(key or ""), re.I)
        and re.search(r"10[:.]?00|18[:.]?00|пн\s*[–-]\s*вс|понедельник|ежедневн|на\s+связи", str(value or "").casefold(), re.I)
    ]
    if not contact_values:
        return False
    if not re.search(r"10[:.]?00|18[:.]?00|пн\s*[–-]\s*вс|понедельник|ежедневн", text, re.I):
        return False
    return bool(re.search(r"расписани|занят|по\s+дням|дни", text, re.I))


def _draft_uses_address_fact(draft: str, facts: Mapping[str, str]) -> bool:
    text = str(draft or "").casefold().replace("ё", "е")
    if not text:
        return False
    for key, value in facts.items():
        key_low = str(key or "").casefold()
        if not re.search(r"address|addresses|metro|location", key_low, re.I):
            continue
        tail = _fact_tail(str(value or "")).casefold().replace("ё", "е")
        if tail and len(tail) >= 4 and tail in text:
            return True
    return False


def _draft_uses_camp_or_lvsh_fact(draft: str, facts: Mapping[str, str]) -> bool:
    text = str(draft or "").casefold().replace("ё", "е")
    if not re.search(r"лвш|менделеев|лагер", text, re.I):
        return False
    return any(_is_camp_or_lvsh_fact(key, str(value or "")) for key, value in facts.items())


def _is_camp_or_lvsh_fact(key: str, text: str) -> bool:
    combined = f"{key} {text}".casefold().replace("ё", "е")
    return bool(re.search(r"лвш|lvsh|менделеев|лагер|camp", combined, re.I))


def _contract_mentions_camp_or_lvsh(contract: AnswerContract) -> bool:
    return bool(re.search(r"лвш|менделеев|лагер|camp|летн", _contract_intent_text(contract), re.I))


def _has_self_answerable_subquestion(contract: AnswerContract) -> bool:
    return any(item.answerable == "self" for item in contract.subquestions)


def _has_retrieved_self_answer_part(contract: AnswerContract, retrieval: RetrievalResult) -> bool:
    for subquestion in contract.subquestions:
        if subquestion.answerable != "self":
            continue
        keys = tuple(key for key in subquestion.needed_fact_keys if key)
        if not keys:
            continue
        if all(key not in retrieval.missing and retrieval.matched_keys.get(key) for key in keys):
            return True
    return False


def _has_exact_retrieved_answer_part(contract: AnswerContract, retrieval: RetrievalResult) -> bool:
    """True only for a fact matched to the current subquestion and its scope.

    This deliberately does not mean "any fact exists": neighboring payment, refund
    or schedule facts must not promote a cautious manager route to an autonomous
    client answer.
    """

    subquestions = contract.subquestions or (
        Subquestion(
            text=contract.current_question,
            answerable="manager",
            needed_fact_keys=contract.needed_fact_keys,
            question_type=contract.question_type,
            existence_target=contract.existence_target,
        ),
    )
    for subquestion in subquestions:
        keys = tuple(key for key in subquestion.needed_fact_keys if key)
        if not keys:
            continue
        if any(key in retrieval.missing or not retrieval.matched_keys.get(key) for key in keys):
            continue
        if _retrieved_keys_match_question_scope(contract, subquestion, retrieval, keys):
            return True
    return False


def _direct_exact_fact_answer(contract: AnswerContract, retrieval: RetrievalResult) -> str:
    if not _has_exact_retrieved_answer_part(contract, retrieval):
        return ""
    if _asks_address(contract):
        address = _first_address_from_facts(retrieval.facts)
        if not address:
            return ""
        city = address.get("city") or "Москве"
        location = address.get("address") or ""
        metro = address.get("metro") or ""
        if not location:
            return ""
        parts = [f"В {city}: {location}"]
        if metro:
            parts.append(f"метро {metro}")
        return "; ".join(parts) + ". Если хотите, менеджер поможет выбрать удобную площадку."
    payment = _direct_payment_answer_from_facts(contract, retrieval.facts)
    if payment:
        return payment
    return ""


def _hard_failure_exact_fact_fallback(contract: AnswerContract, retrieval: RetrievalResult) -> str:
    if not _has_exact_retrieved_answer_part(contract, retrieval):
        return ""
    price = _direct_price_answer_from_facts(contract, retrieval.facts)
    if price:
        return price
    format_answer = _direct_format_answer_from_facts(contract, retrieval.facts)
    if format_answer:
        return format_answer
    recording = _direct_recording_answer_from_facts(contract, retrieval.facts)
    if recording:
        return recording
    return _direct_exact_fact_answer(contract, retrieval)


def _can_autonomously_replace_failed_draft(findings: Sequence[VerificationFinding]) -> bool:
    """Only factual drift may be repaired into an autonomous exact-fact answer.

    Brand leaks, meta leaks, P0 promises and wrong-scope facts stay fail-safe and
    go to manager review.
    """

    return all(finding.code == "fact_grounding" for finding in findings)


def _direct_price_answer_from_facts(contract: AnswerContract, facts: Mapping[str, str]) -> str:
    if not _asks_price(contract):
        return ""
    items: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    facts_text = " ".join(str(value or "") for value in facts.values()).casefold().replace("ё", "е")
    for key, text in facts.items():
        combined = f"{key} {text}"
        if "₽" not in combined:
            continue
        if not re.search(r"цен|стоим|price|₽", combined, re.I):
            continue
        low = combined.casefold().replace("ё", "е")
        label = ""
        if re.search(r"semester|семестр", low, re.I):
            label = "семестр"
        elif re.search(r"(?:^|[._\s])year(?:$|[._\s])|\bгод\b", low, re.I):
            label = "год"
        amount_match = re.search(r"\d[\d\s]{2,}\s*₽", str(text or ""))
        if not label or not amount_match:
            continue
        amount = " ".join(amount_match.group(0).replace("₽", " ₽").split())
        marker = (label, amount)
        if marker in seen:
            continue
        seen.add(marker)
        items.append(marker)
    if not items:
        return ""
    order = {"семестр": 0, "год": 1}
    items.sort(key=lambda item: order.get(item[0], 99))
    price_part = ", ".join(f"{label} — {amount}" for label, amount in items[:2])
    scope_parts: list[str] = []
    if re.search(r"онлайн", facts_text, re.I):
        scope_parts.append("онлайн")
    if re.search(r"5\s*[-–]\s*11\s+класс", facts_text, re.I):
        scope_parts.append("5-11 классы")
    if re.search(r"2026\s*/\s*27", facts_text, re.I):
        scope_parts.append("2026/27 учебный год")
    scope = f" ({', '.join(scope_parts)})" if scope_parts else ""
    return f"По подтверждённым ценам{scope}: {price_part}."


def _direct_format_answer_from_facts(contract: AnswerContract, facts: Mapping[str, str]) -> str:
    if not _asks_training_format_choice(contract):
        return ""
    online_fact = ""
    offline_fact = ""
    for key, text in facts.items():
        combined = f"{key} {text}".casefold().replace("ё", "е")
        if not online_fact and ("online_courses_format" in combined or "онлайн-курсы" in combined):
            online_fact = _short_fact_sentence(str(text or ""), max_chars=220)
        if not offline_fact and re.search(r"очные\s+курсы|очно", combined, re.I):
            offline_fact = _short_fact_sentence(str(text or ""), max_chars=180)
    parts: list[str] = []
    if online_fact:
        parts.append(f"По онлайн-формату подтверждено: {online_fact}")
    if offline_fact and not re.search(r"контакт|10[:.]?00|18[:.]?00|пн\s*[–-]\s*вс", offline_fact.casefold(), re.I):
        parts.append(f"По очному формату: {offline_fact}")
    if not parts:
        return ""
    return " ".join(parts) + " Конкретную группу по предмету и классу менеджер подтвердит."


def _direct_recording_answer_from_facts(contract: AnswerContract, facts: Mapping[str, str]) -> str:
    text = _contract_intent_text(contract)
    if not re.search(r"запис|пересмотр|мтс|mts|link|линк", text, re.I):
        return ""
    recording_fact = ""
    platform_fact = ""
    for key, value in facts.items():
        combined = f"{key} {value}".casefold().replace("ё", "е")
        if not recording_fact and re.search(r"record|запис|пересмотр", combined, re.I):
            recording_fact = _short_fact_sentence(str(value or ""), max_chars=180)
        if not platform_fact and (re.search(r"мтс|mts|link|линк|webinar", combined, re.I) or str(key or "").endswith(".name")):
            platform_fact = _short_fact_sentence(str(value or ""), max_chars=140)
    if recording_fact and platform_fact:
        return f"Да: {recording_fact} {platform_fact}"
    if recording_fact:
        return f"Да: {recording_fact}"
    return ""


def _asks_address(contract: AnswerContract) -> bool:
    text = " ".join(
        [
            contract.current_question,
            contract.client_state,
            " ".join(item.text for item in contract.subquestions),
        ]
    ).casefold().replace("ё", "е")
    return bool(re.search(r"адрес|где\s+вы|где\s+находит|куда\s+ехать|куда\s+ездить", text, re.I))


def _asks_price(contract: AnswerContract) -> bool:
    text = " ".join(
        [
            contract.current_question,
            contract.client_state,
            " ".join(item.text for item in contract.subquestions),
            " ".join(contract.needed_fact_keys),
        ]
    ).casefold().replace("ё", "е")
    return bool(re.search(r"цен|стоим|price|сколько\s+стоит", text, re.I))


def _first_address_from_facts(facts: Mapping[str, str]) -> dict[str, str]:
    groups: dict[str, dict[str, str]] = {}
    for key, text in facts.items():
        match = re.search(r"addresses\.(\d+)\.(address|city|metro)", str(key or ""), re.I)
        if not match:
            continue
        group, field = match.group(1), match.group(2).casefold()
        value = _fact_tail(text)
        if value:
            groups.setdefault(group, {})[field] = value
    for group in sorted(groups, key=lambda item: int(item) if item.isdigit() else 999):
        if groups[group].get("address"):
            return groups[group]
    return {}


def _direct_payment_answer_from_facts(contract: AnswerContract, facts: Mapping[str, str]) -> str:
    targets = _payment_method_target_anchors(contract)
    if "monthly_no_bank" in targets and _has_monthly_no_bank_support(facts):
        return (
            "Да: отдельной банковской рассрочки нет, а помесячная оплата доступна. "
            "Условия по выбранной программе менеджер подтвердит."
        )
    if "direct_invoice" not in targets:
        return ""
    for text in facts.values():
        if "direct_invoice" not in _payment_method_anchors_from_text(str(text or "")):
            continue
        fact = _short_fact_sentence(str(text or ""), max_chars=220)
        if not fact:
            continue
        return f"По подтверждённым способам оплаты: {fact} Детали по выбранной программе менеджер подтвердит."
    return ""


def _fact_tail(text: str) -> str:
    value = str(text or "").strip()
    if "—" in value:
        value = value.rsplit("—", 1)[-1].strip()
    elif ":" in value:
        value = value.rsplit(":", 1)[-1].strip()
    return value.strip(" .")


def _retrieved_keys_match_question_scope(
    contract: AnswerContract,
    subquestion: Subquestion,
    retrieval: RetrievalResult,
    keys: Sequence[str],
) -> bool:
    matched_text = _matched_fact_text_for_required_keys(retrieval, keys)
    if not matched_text:
        return False
    if _asks_refund_policy(contract):
        return bool(_presale_refund_policy_text(_matched_fact_mapping_for_required_keys(retrieval, keys)))
    payment_targets = _payment_method_target_anchors(contract)
    if "monthly_no_bank" in payment_targets:
        if _has_monthly_no_bank_support(retrieval.facts):
            return True
        payment_targets = set(payment_targets) - {"monthly_no_bank"}
    if payment_targets:
        return any(_fact_supports_payment_target(text, target_anchors=payment_targets) for text in matched_text.values())
    question_text = _subquestion_scope_text(contract, subquestion)
    question_low = question_text.casefold().replace("ё", "е")
    if re.search(r"помесячн\w*.*сумм|сумм\w*\s+в\s+месяц|сколько\s+.*(?:в|за)\s+месяц|месячн\w*\s+сумм", question_low, re.I):
        return any(
            re.search(r"сумм\w*\s+в\s+месяц|ежемесячн\w*\s+сумм|помесячн\w*\s+сумм|руб\w*\s+в\s+месяц|₽\s*/\s*мес", value.casefold(), re.I)
            for value in matched_text.values()
        )
    if re.search(r"выходн|суббот|воскрес|будн|по\s+каким\s+дням|дни\s+занят", question_low, re.I):
        # Publication/contact-hour facts are useful context, but they are not an
        # exact answer to "which days/weekends?" unless the same fact names that
        # schedule scope directly.
        return any(
            re.search(r"выходн|суббот|воскрес|будн|слот", f"{key} {value}".casefold(), re.I)
            and "возраж" not in f"{key} {value}".casefold()
            and "objection" not in f"{key} {value}".casefold()
            for key, value in matched_text.items()
        )
    return True


def _matched_fact_text_for_required_keys(retrieval: RetrievalResult, keys: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for required in keys:
        for key in retrieval.matched_keys.get(required) or ():
            if key in retrieval.facts:
                result[key] = str(retrieval.facts[key])
    return result


def _matched_fact_mapping_for_required_keys(retrieval: RetrievalResult, keys: Sequence[str]) -> dict[str, str]:
    return _matched_fact_text_for_required_keys(retrieval, keys)


def _subquestion_scope_text(contract: AnswerContract, subquestion: Subquestion) -> str:
    return " ".join(
        part
        for part in (
            contract.current_question,
            contract.existence_target,
            subquestion.text,
            subquestion.existence_target,
        )
        if part
    )


def _asks_refund_policy(contract: AnswerContract) -> bool:
    text = " ".join(
        [
            contract.current_question,
            contract.client_state,
            " ".join(contract.continued_topics),
            " ".join(contract.switched_topics),
            " ".join(item.text for item in contract.subquestions),
        ]
    ).casefold()
    if re.search(r"налог|вычет|фнс|ндфл|маткап|материнск", text, re.I):
        return False
    return bool(re.search(r"refund|возврат|верн[её]т|вернут|деньг|отмен|передума", text, re.I))


def _current_turn_asks_refund_policy(contract: AnswerContract, *, client_words: str) -> bool:
    text = " ".join(
        part
        for part in (
            client_words,
            contract.current_question,
            " ".join(item.text for item in contract.subquestions),
            contract.existence_target,
            " ".join(item.existence_target for item in contract.subquestions),
        )
        if part
    ).casefold().replace("ё", "е")
    if re.search(r"налог|вычет|фнс|ндфл|маткап|материнск", text, re.I):
        return False
    return bool(re.search(r"refund|возврат|верн[её]т|вернут|деньг|отмен|передума", text, re.I))


def _presale_refund_policy_text(facts: Mapping[str, str]) -> str:
    for key, text in facts.items():
        combined = f"{key} {text}".casefold().replace("ё", "е")
        if "refund_presale_policy" in combined or "остаток неистраченных средств" in combined:
            return _client_presale_refund_text(str(text or ""))
    return ""


def _client_presale_refund_text(text: str) -> str:
    low = str(text or "").casefold().replace("ё", "е")
    if "остаток неистраченных средств" in low:
        return (
            "Да, при досрочном отказе возвращается остаток неистраченных средств. "
            "Конкретный порядок оформления менеджер подтвердит по выбранному курсу и договору."
        )
    return _short_fact_sentence(text, max_chars=220)


def _current_refund_dispute_signal(*, client_words: str, contract: AnswerContract) -> bool:
    current_text = " ".join(
        part
        for part in (
            client_words,
            contract.current_question,
            " ".join(item.text for item in contract.subquestions),
            contract.client_state,
        )
        if part
    )
    normalized = current_text.casefold().replace("ё", "е")
    if not normalized.strip():
        return False
    p0_codes = set(codes_from_text(normalized))
    if p0_codes.intersection({"refund", "payment_dispute", "complaint", "legal"}):
        return True
    safety = classify_answer_safety(client_message=normalized)
    if safety.p0_required:
        return True
    return bool(
        re.search(
            r"верните\s+(?:мне\s+)?деньг|отдайте\s+(?:мне\s+)?(?:деньг|оплат)|"
            r"хочу\s+вернуть\s+(?:деньг|оплат)|требую\s+верн|"
            r"заняти[йя]\s+нет|доступа\s+нет|оплатил[аи]?\b|недовол|обман|развод|"
            r"уже\s+(?:месяц|недел\w*|дн\w*)\s+жд|жду\s+(?:месяц|недел\w*|дн\w*)|"
            r"никто\s+.*не\s+отвеч|нормально\s+не\s+отвеч|"
            r"чарджб[еэ]к|оспор(?:ю|ить|ил)|наруш\w*\s+(?:моих\s+|наших\s+)?прав",
            normalized,
            re.I,
        )
    )


def _single_missing_slot_question(contract: AnswerContract, retrieval: RetrievalResult) -> str:
    if contract.is_p0 or _asks_refund_policy(contract):
        return ""
    if retrieval.facts or len(retrieval.missing) != 1:
        return ""
    text = " ".join([contract.current_question, " ".join(item.text for item in contract.subquestions), retrieval.missing[0]])
    low = text.casefold().replace("ё", "е")
    if re.search(r"цен|стоим|оплат|рассроч|долями|банк|возврат|договор|жалоб|суд|юрист", low, re.I):
        return ""
    if re.search(r"\bкласс|grade|student_grade", low, re.I):
        return "Подскажите, пожалуйста, класс ученика — тогда сориентирую точнее."
    if re.search(r"предмет|subject", low, re.I):
        return "Подскажите, пожалуйста, предмет — тогда сориентирую точнее."
    if re.search(r"формат|очно|онлайн|format", low, re.I):
        return "Подскажите, пожалуйста, какой формат удобнее: очно или онлайн?"
    return ""


def _is_existence_yes_no_contract(contract: AnswerContract) -> bool:
    if contract.question_type == "existence_yes_no":
        return True
    return any(item.question_type == "existence_yes_no" for item in contract.subquestions)


def _contract_existence_text(contract: AnswerContract) -> str:
    parts = [contract.existence_target, contract.current_question]
    for item in contract.subquestions:
        if item.question_type == "existence_yes_no":
            parts.extend([item.existence_target, item.text])
    return " ".join(part for part in parts if part)


def _existence_target_anchors(contract: AnswerContract) -> set[str]:
    text = _contract_existence_text(contract).casefold().replace("ё", "е")
    anchors: set[str] = set()
    if re.search(r"банк|банковск|т-банк|t-банк", text, re.I):
        anchors.add("bank")
    if re.search(r"рассроч|частями|долями", text, re.I):
        anchors.add("installment")
    if re.search(r"пробн|фрагмент", text, re.I):
        anchors.add("trial")
    if re.search(r"запис|пересмотр", text, re.I):
        anchors.add("recording")
    return anchors


def _fact_has_existence_anchors(text: str, *, target_anchors: set[str]) -> bool:
    low = str(text or "").casefold().replace("ё", "е")
    if "bank" in target_anchors and not re.search(r"банк|банковск|т-банк|t-банк", low, re.I):
        return False
    if "installment" in target_anchors and not re.search(r"рассроч|частями|долями", low, re.I):
        return False
    if "trial" in target_anchors and not re.search(r"пробн|фрагмент", low, re.I):
        return False
    if "recording" in target_anchors and not re.search(r"запис|пересмотр", low, re.I):
        return False
    return True


def _is_negative_existence_fact_for_target(text: str, *, target_anchors: set[str]) -> bool:
    low = str(text or "").casefold().replace("ё", "е")
    if not _fact_has_existence_anchors(low, target_anchors=target_anchors):
        return False
    return bool(re.search(r"\bнет\b|не\s+доступ|не\s+предусмотр|отсутств", low, re.I))


def _is_positive_existence_fact_for_target(text: str, *, target_anchors: set[str]) -> bool:
    low = str(text or "").casefold().replace("ё", "е")
    if _is_negative_existence_fact_for_target(low, target_anchors=target_anchors):
        return False
    if not _fact_has_existence_anchors(low, target_anchors=target_anchors):
        return False
    return bool(re.search(r"\bесть\b|доступ|можно|оформ", low, re.I))


def _known_absence_text(contract: AnswerContract, facts: Mapping[str, str]) -> str:
    target_anchors = _existence_target_anchors(contract)
    if not target_anchors:
        return ""
    for text in facts.values():
        if _is_negative_existence_fact_for_target(str(text or ""), target_anchors=target_anchors):
            return str(text or "")
    return ""


def _existence_yes_no_findings(
    draft: str,
    *,
    contract: AnswerContract,
    facts: Mapping[str, str],
) -> list[VerificationFinding]:
    if not _is_existence_yes_no_contract(contract):
        return []
    target_anchors = _existence_target_anchors(contract)
    if not target_anchors:
        return []
    fact_values = [str(text or "") for text in facts.values()]
    has_negative = any(_is_negative_existence_fact_for_target(text, target_anchors=target_anchors) for text in fact_values)
    has_positive = any(_is_positive_existence_fact_for_target(text, target_anchors=target_anchors) for text in fact_values)
    first_sentence = re.split(r"[.!?\n]", str(draft or "").strip(), maxsplit=1)[0].casefold().replace("ё", "е")
    findings: list[VerificationFinding] = []
    affirmative = bool(
        re.search(r"^\s*(да\b|можно\b|доступн|есть\s+вариант|получится\b|оформляется\b)", first_sentence, re.I)
    )
    negative = bool(re.search(r"^\s*(нет\b|не\s+доступ|не\s+предусмотр|отсутств)", first_sentence, re.I))
    if affirmative and not has_positive:
        findings.append(
            VerificationFinding(
                "unsupported_existence_affirmation",
                "Вопрос про наличие X получил утвердительный ответ без явного положительного факта про X.",
            )
        )
    if negative and not has_negative:
        findings.append(
            VerificationFinding(
                "unsupported_existence_negative",
                "Отрицательный ответ «нет» разрешён только при явном отрицательном факте про X.",
            )
        )
    return findings


def _payment_method_target_anchors(contract: AnswerContract) -> set[str]:
    text = " ".join(
        [
            contract.current_question,
            contract.existence_target,
            " ".join(item.text for item in contract.subquestions),
            " ".join(item.existence_target for item in contract.subquestions),
        ]
    ).casefold().replace("ё", "е")
    anchors: set[str] = set()
    if re.search(r"долями", text, re.I):
        anchors.add("dolyami")
    if re.search(r"банк|банковск|т-банк|t-банк", text, re.I) and re.search(r"рассроч|кредит|частями", text, re.I):
        anchors.add("bank_installment")
    if re.search(r"(прям\w*\s+перевод|перевод\w*\s+на\s+счет|перевод\w*\s+на\s+сч[её]т|по\s+счету|по\s+сч[её]ту|ежемесячн\w*\s+счет|ежемесячн\w*\s+сч[её]т|напрямую\s+(?:вам|центру)|без\s+банка|вам\s+платить)", text, re.I):
        anchors.add("direct_invoice")
    if re.search(r"помесячн", text, re.I) and re.search(r"без\s+банка|банк\s+не\s+участв|не\s+через\s+банк", text, re.I):
        anchors.add("monthly_no_bank")
    return anchors


def _payment_method_anchors_from_text(text: str) -> set[str]:
    low = str(text or "").casefold().replace("ё", "е")
    anchors: set[str] = set()
    if re.search(r"долями", low, re.I):
        anchors.add("dolyami")
    if re.search(r"т-банк|t-банк|банковск\w*\s+рассроч|рассроч\w*\s+через\s+банк|рассроч", low, re.I):
        anchors.add("bank_installment")
    if re.search(r"(прям\w*\s+перевод|перевод\w*\s+на\s+счет|перевод\w*\s+на\s+сч[её]т|по\s+счету|по\s+сч[её]ту|счет\s+кажд\w*\s+месяц|сч[её]т\s+кажд\w*\s+месяц|реквизит|квитанц|qr-?код|qr\s)", low, re.I):
        anchors.add("direct_invoice")
    if re.search(r"помесячн", low, re.I):
        anchors.add("monthly")
    if re.search(r"банковск\w*\s+рассроч\w*\s+нет|отдельн\w*\s+банковск\w*\s+рассроч\w*\s+нет|без\s+банка|банк\s+не\s+участв", low, re.I):
        anchors.add("no_bank")
    return anchors


def _has_monthly_no_bank_support(facts: Mapping[str, str]) -> bool:
    anchors: set[str] = set()
    for text in facts.values():
        anchors.update(_payment_method_anchors_from_text(str(text or "")))
    return "monthly" in anchors and "no_bank" in anchors


def _fact_supports_payment_target(text: str, *, target_anchors: set[str]) -> bool:
    if "monthly_no_bank" in target_anchors:
        return False
    fact_anchors = _payment_method_anchors_from_text(text)
    return bool(target_anchors) and target_anchors.issubset(fact_anchors)


def _payment_method_findings(
    draft: str,
    *,
    contract: AnswerContract,
    facts: Mapping[str, str],
) -> list[VerificationFinding]:
    target_anchors = _payment_method_target_anchors(contract)
    if not target_anchors:
        return []
    fact_values = [str(text or "") for text in facts.values()]
    has_target_fact = any(_fact_supports_payment_target(text, target_anchors=target_anchors) for text in fact_values)
    draft_anchors = _payment_method_anchors_from_text(draft)
    first_sentence = re.split(r"[.!?\n]", str(draft or "").strip(), maxsplit=1)[0].casefold().replace("ё", "е")
    affirmative = bool(
        re.search(r"^\s*(да\b|можно\b|доступн|есть\s+вариант|получится\b|оформляется\b)", first_sentence, re.I)
    )
    findings: list[VerificationFinding] = []
    neighbor_anchors = draft_anchors - target_anchors
    if neighbor_anchors and not has_target_fact:
        findings.append(
            VerificationFinding(
                "neighbor_payment_method_as_answer",
                "Ответ подменяет конкретно спрошенный способ оплаты соседним способом, даже если соседний факт реален.",
            )
        )
    if (affirmative or bool(draft_anchors & target_anchors)) and not has_target_fact:
        findings.append(
            VerificationFinding(
                "unsupported_payment_method_affirmation",
                "Утверждать конкретный способ оплаты можно только при факте именно про этот способ.",
            )
        )
    return findings


def _repair_prompt(draft: str, instruction: str, facts: Mapping[str, str]) -> str:
    facts_block = "\n".join(f"- {key}: {value}" for key, value in facts.items()) or "(нет фактов)"
    return (
        "Исправь ровно это, смысл и маршрут не меняй, новых фактов вне списка не вводи.\n"
        f"Замечания: {instruction}\n"
        f"Факты:\n{facts_block}\n"
        f"Черновик:\n{draft}\n"
        "Верни только исправленный текст."
    )


def _parse_subquestions(data: Mapping[str, Any], catalog: Sequence[str]) -> tuple[Subquestion, ...]:
    subquestions: list[Subquestion] = []
    raw = data.get("subquestions")
    if isinstance(raw, SequenceABC) and not isinstance(raw, (str, bytes, bytearray)):
        for item in raw:
            if not isinstance(item, MappingABC):
                continue
            keys = tuple(_valid_contract_key(key, catalog) for key in _seq(item.get("needed_fact_keys")))
            keys = tuple(key for key in dict.fromkeys(keys) if key)
            answerable = str(item.get("answerable") or "manager").strip()
            if answerable not in {"self", "manager"}:
                answerable = "manager"
            subquestions.append(
                Subquestion(
                    text=str(item.get("text") or "").strip()[:300],
                    answerable=answerable,
                    needed_fact_keys=keys,
                    next_step=str(item.get("next_step") or "").strip()[:180],
                    question_type=_normalize_question_type(
                        item.get("question_type"),
                        fallback_text=str(item.get("text") or ""),
                    ),
                    existence_target=str(item.get("existence_target") or "").strip()[:180],
                )
            )
    return tuple(subquestions)


def _normalize_question_type(value: object, *, fallback_text: str = "") -> str:
    raw = str(value or "").strip().casefold()
    if raw == "existence_yes_no":
        return "existence_yes_no"
    text = str(fallback_text or "").casefold().replace("ё", "е")
    if re.search(r"\b(?:есть\s+ли|бывает\s+ли|доступн\w*\s+ли|можно\s+ли|предусмотрен\w*\s+ли)\b", text, re.I):
        return "existence_yes_no"
    return ""


def _clean_slots(value: object) -> Mapping[str, Slot]:
    if not isinstance(value, MappingABC):
        return {}
    result: dict[str, Slot] = {}
    for key, item in value.items():
        clean_key = str(key or "").strip()[:80]
        if not clean_key:
            continue
        if isinstance(item, MappingABC):
            slot_value = str(item.get("value") or "").strip()[:180]
            source = str(item.get("source") or "").strip()[:120]
        else:
            slot_value = str(item or "").strip()[:180]
            source = ""
        if slot_value:
            result[clean_key] = Slot(value=slot_value, source=source)
    return result


def _valid_contract_key(value: str, catalog: Sequence[str]) -> str:
    key = str(value or "").strip()
    if not key or _MONEY_OR_VALUE_RE.search(key):
        return ""
    if key in catalog:
        return key
    if "." in key or "_" in key:
        return key[:180]
    return key[:120]


def _matched_fact_keys(required: str, store: Mapping[str, str]) -> list[str]:
    if required in store:
        return [required]
    matched = [key for key in store if key_matches(required, key)]
    if matched:
        return matched
    required_norm = _normalize_lookup(required)
    return [key for key in store if required_norm and required_norm in _normalize_lookup(key)]


def _prioritize_catalog(catalog: Sequence[str], *, context: Mapping[str, Any] | None) -> tuple[str, ...]:
    unique = tuple(dict.fromkeys(str(item or "").strip() for item in catalog if str(item or "").strip()))
    if not isinstance(context, MappingABC):
        return unique[:MAX_CATALOG_KEYS]
    required: list[str] = []
    for source in (
        context.get("required_fact_keys"),
        (context.get("conversation_intent_plan") or {}).get("required_fact_keys")
        if isinstance(context.get("conversation_intent_plan"), MappingABC)
        else (),
        (context.get("facts_context") or {}).get("required_fact_keys")
        if isinstance(context.get("facts_context"), MappingABC)
        else (),
    ):
        required.extend(_seq(source))
    scored: list[tuple[int, str]] = []
    for key in unique:
        score = 0
        if any(key == req or key_matches(req, key) for req in required):
            score += 100
        if _key_mentions_current_text(key, context):
            score += 10
        scored.append((score, key))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return tuple(key for _, key in scored[:MAX_CATALOG_KEYS])


def _key_mentions_current_text(key: str, context: Mapping[str, Any]) -> bool:
    text = " ".join(
        str(item or "")
        for item in (
            context.get("current_message", ""),
            " ".join(context.get("recent_messages") or ())
            if isinstance(context.get("recent_messages"), SequenceABC)
            else "",
        )
    ).casefold()
    if not text:
        return False
    key_norm = str(key or "").replace("_", " ").replace(".", " ").casefold()
    tokens = [token for token in re.findall(r"[a-zа-яё]{4,}", key_norm) if len(token) >= 4]
    return any(token in text for token in tokens[:8])


def _snapshot_path_from_context(context: Mapping[str, Any] | None) -> Path:
    if isinstance(context, MappingABC):
        for key in ("snapshot_path", "knowledge_snapshot_path", "kb_snapshot_path"):
            value = context.get(key)
            if value:
                return Path(str(value))
    return DEFAULT_KB_SNAPSHOT_PATH


@lru_cache(maxsize=8)
def _load_snapshot(path: str | Path) -> Mapping[str, Any]:
    snapshot_path = Path(path)
    if not snapshot_path.is_absolute():
        snapshot_path = Path.cwd() / snapshot_path
    try:
        return json.loads(snapshot_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _snapshot_facts(snapshot: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    facts = snapshot.get("facts") if isinstance(snapshot, MappingABC) else None
    if not isinstance(facts, SequenceABC) or isinstance(facts, (str, bytes, bytearray)):
        return ()
    return tuple(item for item in facts if isinstance(item, MappingABC))


def _client_safe_fact(fact: Mapping[str, Any]) -> bool:
    return (
        fact.get("allowed_for_client_answer") is True
        and fact.get("forbidden_for_client") is not True
        and fact.get("internal_only") is not True
        and str(fact.get("client_safe_text") or fact.get("fact_text") or "").strip()
    )


def _join_fact_text(previous: str, new: str) -> str:
    if new in previous:
        return previous
    return f"{previous}; {new}"[:1200]


def _fact_value_text(value: object) -> str:
    if isinstance(value, MappingABC):
        for key in ("client_safe_text", "fact_text", "manager_display_text", "text", "value"):
            if value.get(key):
                return str(value.get(key)).strip()
        return json.dumps(value, ensure_ascii=False)[:900]
    return str(value or "").strip()[:900]


def _seq(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value.strip(),) if value.strip() else ()
    if isinstance(value, SequenceABC) and not isinstance(value, (bytes, bytearray)):
        return tuple(str(item or "").strip() for item in value if str(item or "").strip())
    return ()


def _extract_json_object(text: str) -> Mapping[str, Any]:
    raw = str(text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        payload = json.loads(raw)
    except Exception:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            return {}
        payload = json.loads(raw[start : end + 1])
    return payload if isinstance(payload, MappingABC) else {}


def _numbers(text: str) -> set[str]:
    normalized = re.sub(r"(?<=\d)[\s\u00a0](?=\d)", "", str(text or ""))
    return set(_NUMBER_RE.findall(normalized))


def _is_allowed_ungrounded_number(value: str, *, client_numbers: set[str]) -> bool:
    if value in client_numbers:
        return True
    try:
        number = int(value)
    except Exception:
        return False
    # Years and short grades may be contextual; percentages, parts, prices and months
    # must be grounded by facts, so we do not blanket-allow all small numbers.
    return number in {2026, 2027}


def _sanitize_blocks(text: str) -> bool:
    sanitized = sanitize_answer(text, mode="bot")
    blocking_flags = {
        "raw_json_leak",
        "internal_metadata_leak",
        "bot_placeholder_leak",
        "unsafe_placeholder_leak",
        "personal_placeholder_leak",
    }
    return bool(set(sanitized.flags) & blocking_flags)


def _client_asked_identity(text: str) -> bool:
    return bool(re.search(r"\b(?:ты|вы)\s+(?:бот|ии|нейросет|gpt)|с\s+кем\s+я\s+общ", str(text or ""), re.I))


def _brand_token_present(low_text: str, token: str) -> bool:
    token_low = str(token or "").casefold()
    if not token_low:
        return False
    if re.fullmatch(r"[a-zа-яё0-9]+", token_low):
        return bool(re.search(rf"(?<![a-zа-яё0-9]){re.escape(token_low)}(?![a-zа-яё0-9])", low_text))
    return token_low in low_text


def _normalize_brand(value: str) -> str:
    text = str(value or "").strip().casefold()
    if text in {"foton", "фотон"}:
        return "foton"
    if text in {"unpk", "унпк", "унпк мфти", "mipt", "мфти"}:
        return "unpk"
    return text or "unknown"


def _normalize_lookup(value: str) -> str:
    return re.sub(r"[^a-zа-яё0-9]+", "", str(value or "").casefold())


def _clamp_float(value: object) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0


def _truthy(value: object) -> bool:
    return str(value or "").strip().casefold() in {"1", "true", "yes", "on", "y", "да"}


def _norm_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().casefold())


def _similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    # Tiny local similarity implementation avoids importing difflib in the hot path
    # through global module side effects.
    import difflib

    return difflib.SequenceMatcher(None, left, right).ratio()
