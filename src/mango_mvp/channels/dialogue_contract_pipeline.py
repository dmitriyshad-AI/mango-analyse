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
from mango_mvp.channels.dialogue_debug_trace import trace_event, trace_span
from mango_mvp.channels.fact_retrieval import key_matches
from mango_mvp.channels.humanity_guards import has_meta_leak
from mango_mvp.channels.p0_recall_spec import codes_from_text, hard_codes_from_text, soft_codes_from_text
from mango_mvp.insights.sanitizers import sanitize_answer


DIALOGUE_CONTRACT_PIPELINE_ENV = "TELEGRAM_DIALOGUE_CONTRACT_PIPELINE"
ESTIMATE_MODE_ENV = "TELEGRAM_A_ESTIMATE_MODE"
FREE_NUMBER_GATE_ENV = "TELEGRAM_A_FREE_NUMBER_GATE"
TRAVEL_COMPOSE_ENV = "TELEGRAM_A_TRAVEL_COMPOSE"
QUALITY_PARTIAL_YIELD_ENV = "TELEGRAM_Q_PARTIAL_YIELD"
QUALITY_THREAD_MEMORY_ENV = "TELEGRAM_Q_THREAD_MEMORY"
QUALITY_COMPOSITE_ENV = "TELEGRAM_Q_COMPOSITE"
QUALITY_NEXT_STEP_ENV = "TELEGRAM_Q_NEXT_STEP"
QUALITY_CLARIFY_SCOPE_ENV = "TELEGRAM_Q_CLARIFY_SCOPE"
QUALITY_USEFUL_HANDOFF_ENV = "TELEGRAM_Q_USEFUL_HANDOFF"
DIALOGUE_CONTRACT_SCHEMA_VERSION = "dialogue_contract_v2_2026_05_26"
DEFAULT_KB_SNAPSHOT_PATH = Path(
    "product_data/knowledge_base/kb_release_20260603_v6_5_summer_format_cleanup/kb_release_v3_snapshot.json"
)
MAX_CATALOG_KEYS = 240
MAX_REPAIR_ATTEMPTS = 2
PLANNER_INTENT_VALUES: tuple[str, ...] = (
    "teacher",
    "recording",
    "address",
    "document",
    "matkap",
    "tax",
    "olympiad",
    "platform_access",
    "installment",
    "payment_method",
    "discount",
    "pricing",
    "format",
    "trial",
    "camp_lvsh",
    "enrollment_process",
    "schedule",
    "refund_policy",
    "general_consultation",
)

_MONEY_OR_VALUE_RE = re.compile(
    r"(?:₽|руб(?:\.|лей|ля|ль)?|%)|\b\d[\d\s\u00a0]{2,}\s*(?:р\.|руб|₽)\b",
    re.I,
)
_NUMBER_RE = re.compile(r"\d+")
_ESTIMATE_DOMAINS = ("travel_time", "route_logistics", "general_advice")
_ESTIMATE_NUMBER_TOKEN_RE = re.compile(
    r"\d[\d\s\u00a0]*(?:[.,]\d+)?\s*"
    r"(?:₽|%|руб(?:\.|лей|ля|ль)?|р\.|месяц(?:ев|а)?|дн(?:ей|я)?|"
    r"раз(?:а)?|балл(?:ов|а)?|минут(?:ы|у)?|час(?:а|ов)?|км|километр(?:а|ов)?)?",
    re.I,
)
_PRODUCT_QUESTION_RE = re.compile(
    r"цена|стоит|стоимост|сколько\s+стоит|скидк|рассрочк|долями|тариф|расписан|"
    r"во\s+сколько|какие\s+дни|время\s+занят|дат[аеуы]|смен[аеуы]|лагер|формат|"
    r"сколько\s+длится|длительност|документ|справк|возврат|вернут|оплат|мест[ао]|"
    r"записать|запис[ьи]|₽|%",
    re.I,
)
_PRODUCT_NUMBER_CTX_RE = re.compile(
    r"₽|руб|р\.|%|скидк|рассрочк|долями|\b\d{1,2}:\d{2}\b|семестр|за\s+год|"
    r"стоит|цена|тариф|сколько\s+длится|длительност|урок|занят|январ|феврал|март|"
    r"апрел|ма[яй]|июн|июл|август|сентяб|октяб|ноябр|декабр|смен[аеуы]",
    re.I,
)
_TRAVEL_ESTIMATE_TEXT_RE = re.compile(
    r"дорог|ехать|доехать|добират|добраться|как\s+проехать|маршрут|пешком|электрич|метро|автобус|"
    r"такси|станци|остановк|лобн|долгопрудн|пацаев|сретенк|красносельск",
    re.I,
)
_TRAVEL_ESTIMATE_PRODUCT_BLOCK_RE = re.compile(
    r"цена|стоимост|сколько\s+стоит|стоит\s+курс|скидк|рассрочк|долями|тариф|расписан|"
    r"какие\s+дни|во\s+сколько|сколько\s+длится|длительност|дат[аеуы]|смен[аеуы]|лагер|"
    r"формат|документ|справк|возврат|вернут|оплат|мест[ао]\b|запис",
    re.I,
)
_FREE_NUMBER_PRODUCT_CTX_RE = re.compile(
    r"₽|руб|р\.|%|процент|скидк|рассрочк|долями|цена|стоит|стоимост|тариф|семестр|за\s+год|оплат|"
    r"\b\d{1,2}:\d{2}\b|расписан|по\s+(?:понедельник|вторник|сред|четверг|пятниц|суббот|воскресень)|"
    r"\b(?:пн|вт|ср|чт|пт|сб|вс)\b|\bв\s+(?:1[0-9]|2[0-3])\b|"
    r"\d{1,2}[-–]\d{1,2}\.\d{1,2}|\b\d{1,2}[./]\d{1,2}(?:[./]\d{2,4})?\b|"
    r"январ|феврал|март|апрел|ма[яй]|июн|июл|август|сентяб|октяб|ноябр|декабр|"
    r"смен[аеуы]|заезд|лагер|лвш|\bлш\b|интенсив|мест[ао]\b|балл|групп|сфр|фнс|справк|"
    r"ак\.?\s*ч|занятий|недел|длится|длительност|академ|раз(?:а)?\s+в\s+недел|час[аов]*\s+в\s+недел",
    re.I,
)
_FREE_NUMBER_TOKEN_RE = re.compile(
    r"\b20\d{2}/\d{2}\b|"
    r"\b\d{1,2}:\d{2}\b|"
    r"\b\d+(?:[.,]\d+)?\s*[-–]\s*\d+(?:[.,]\d+)?(?:\s*(?:км|километр(?:а|ов)?|минут(?:ы|у)?|час(?:а|ов)?|год(?:а)?|лет|недел(?:и|ь)?|заняти(?:й|я)|балл(?:ов|а)?|процент(?:ов|а)?|%))?|"
    r"\b\d{1,2}[./]\d{1,2}(?:[./]\d{2,4})?\b|"
    r"\b\d[\d\s\u00a0]*(?:[.,]\d+)?\s*(?:к|тыс\.?|тысяч|₽|руб(?:\.|лей|ля|ль)?|р\.|процент(?:ов|а)?|%|минут(?:ы|у)?|час(?:а|ов)?|км|километр(?:а|ов)?|год(?:а)?|лет|недел(?:и|ь)?|заняти(?:й|я)|балл(?:ов|а)?|ак\.?\s*ч(?:\.|аса|асов)?|раз(?:а)?)?\b",
    re.I,
)
_FREE_NUMBER_UNCERTAINTY_MARKERS = (
    "ориентировоч",
    "примерно",
    "навскидк",
    "скорее всего",
    "не уверен",
    "точно подскажет менеджер",
    "точную информацию уточнит менеджер",
    "около",
    "порядка",
    "в районе",
    "приблизительно",
    "где-то",
    "обычно",
    "как правило",
    "в среднем",
    "чаще всего",
)
_STRUCTURAL_NUMBER_OK = {str(number) for number in range(1, 12)}
_YEAR_NUMBER_OK = {"2024", "2025", "2026", "2027", "2024/25", "2025/26", "2026/27"}
_INDIVIDUAL_CHILD_RE = re.compile(
    r"мой\s+(?:реб[её]нок|сын|дочь|дочк\w*)|моя\s+(?:дочь|дочк\w*)|"
    r"у\s+моего|у\s+моей|потянет\s+ли|справится\s+ли|"
    r"отста[её]т|не\s+тянет|что\s+с\s+ним|что\s+с\s+ней|подойд[её]т\s+ли\s+(?:моему|нам)|"
    r"уровень\s+моего",
    re.I,
)
_INDIVIDUAL_CHILD_CONFIDENT_RE = re.compile(
    r"^\s*да\b|точно\s+(?:справится|потянет|подойд[её]т)|\b(?:справится|потянет|подойд[её]т)\b",
    re.I,
)
_UNCERTAINTY_MARKERS = (
    "ориентировочно",
    "примерно",
    "навскидку",
    "точно подскажет менеджер",
    "не возьмусь утверждать точно",
    "точную информацию уточнит менеджер",
)
_ESTIMATE_PRESSURE_RE = re.compile(
    r"срочно\s+записыва\w+|мест\s+почти\s+нет|надо\s+успеть|иначе\s+не\s+попад[её]те|"
    r"лучше\s+не\s+тянуть",
    re.I,
)
_ESTIMATE_GUARANTEE_RE = re.compile(
    r"гарантир|100\s*%|обязательно\s+(?:поступ|сдад|сдаст|получ)|точно\s+(?:поступ|сдад|сдаст|получ)|"
    r"исправим\s+на\s+(?:5|пят)|подтянем\s+на\s+(?:5|пят)|точно\s+станет",
    re.I,
)
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
_GENERIC_HANDOFF_TEXTS: tuple[str, ...] = (
    "Чтобы не ошибиться, передам вопрос менеджеру — он сверит детали и вернётся с ответом.",
    "Не хочу гадать по неподтверждённому пункту: менеджер проверит его и вернётся с ответом.",
    "Здесь лучше сверить условия: передам вопрос менеджеру, он ответит по точным данным.",
    "Передам этот пункт менеджеру, чтобы он проверил его по актуальным данным и ответил вам.",
)
_DETAIL_HANDOFF_TEXTS: tuple[str, ...] = (
    "Чтобы не ошибиться, менеджер уточнит именно про {detail} и вернётся с ответом.",
    "Не хочу гадать по неподтверждённому пункту: менеджер проверит именно {detail} и ответит вам.",
    "По пункту «{detail}» нужна точная сверка — передам его менеджеру.",
    "Передам менеджеру именно вопрос про {detail}, чтобы он проверил актуальные условия.",
)
_HANDOFF_EXHAUSTED_TEXTS: tuple[str, ...] = (
    "Вижу, это важно — отдельно отмечу менеджеру, чтобы он ответил именно по этому пункту.",
    "Зафиксирую этот пункт отдельно для менеджера, чтобы он вернулся не общим ответом, а по сути вопроса.",
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
    planner_intent: str = ""
    planner_subvariant: str = ""
    planner_slots: Mapping[str, str] = field(default_factory=dict)
    planner_confidence: float = 0.0
    answer_mode: str = "confirmed_only"
    estimate_domain: str = "none"
    estimate_confidence: float = 0.0
    selling: Mapping[str, Any] = field(
        default_factory=lambda: {
            "objection": "none",
            "exit_signal": False,
            "anxiety": False,
            "unmet_need": "",
            "readiness": "exploring",
        }
    )
    forbidden_substitutions: tuple[str, ...] = ()
    client_state: str = ""
    answerability: str = "manager_only"
    question_type: str = ""
    existence_target: str = ""
    is_p0: bool = False
    p0_reason: str = ""
    p0_source: str = ""
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
            "planner_intent": self.planner_intent,
            "planner_subvariant": self.planner_subvariant,
            "planner_slots": dict(self.planner_slots),
            "planner_confidence": self.planner_confidence,
            "answer_mode": self.answer_mode,
            "estimate_domain": self.estimate_domain,
            "estimate_confidence": self.estimate_confidence,
            "selling": dict(self.selling),
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
            "p0_source": self.p0_source,
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
    is_manager_deferral: bool = False
    reason_class: str = ""
    reason_evidence: Mapping[str, Any] = field(default_factory=dict)
    semantic_match_attempted: bool = False
    semantic_match_replaced: bool = False
    semantic_match_reason: str = ""
    warmed: bool = False
    warmth_attempted: bool = False
    warmth_mode: str = ""
    warmth_rejected_reason: str = ""
    warmth_rejected_findings: tuple[VerificationFinding, ...] = ()
    warmth_rejected_unsupported: tuple[str, ...] = ()
    warmth_semantic_available: bool = True
    repaired: bool = False
    recovery_candidate: str = ""
    is_estimate: bool = False
    estimate_applied: bool = False
    estimate_domain: str = "none"
    estimate_answer_mode: str = "confirmed_only"
    partial_yield_applied: bool = False
    partial_yield_fact_keys: tuple[str, ...] = ()
    partial_yield_missing: tuple[str, ...] = ()
    composite_applied: bool = False
    composite_fact_keys: tuple[str, ...] = ()
    composite_missing: tuple[str, ...] = ()
    next_step_applied: bool = False
    next_step_text: str = ""

    def __post_init__(self) -> None:
        route = str(self.route or "").strip()
        fallback_reason = str(self.fallback_reason or "").strip()
        is_manager_deferral = route != "bot_answer_self"
        reason_class = str(self.reason_class or "").strip()
        if is_manager_deferral and not reason_class:
            reason_class = _pipeline_reason_class(
                fallback_reason=fallback_reason,
                contract=self.contract,
                findings=self.findings,
                unsupported_claims=self.unsupported_claims,
            )
        if not is_manager_deferral:
            reason_class = ""
        evidence = dict(self.reason_evidence or {})
        if is_manager_deferral:
            if fallback_reason:
                evidence.setdefault("fallback_reason", fallback_reason)
            if reason_class == "p0_deferral":
                evidence.setdefault("p0_source", self.contract.p0_source or "model")
                if self.contract.p0_reason:
                    evidence.setdefault("p0_reason", self.contract.p0_reason)
        else:
            evidence = {}
        object.__setattr__(self, "fallback_reason", fallback_reason)
        object.__setattr__(self, "is_manager_deferral", is_manager_deferral)
        object.__setattr__(self, "reason_class", reason_class)
        object.__setattr__(self, "reason_evidence", evidence)


def _pipeline_reason_class(
    *,
    fallback_reason: str,
    contract: AnswerContract,
    findings: Sequence[VerificationFinding] = (),
    unsupported_claims: Sequence[str] = (),
) -> str:
    reason = str(fallback_reason or "").strip().casefold()
    finding_codes = {str(item.code or "").strip() for item in findings if str(item.code or "").strip()}
    for visible_code in ("estimate_individual_child_advice", "estimate_general_advice_risk"):
        if visible_code in finding_codes:
            return visible_code
    if contract.is_p0 or reason.startswith("p0") or reason == "prior_hard_p0_refund_claim":
        return "p0_deferral"
    if "refund" in reason:
        return "refund"
    if "high_risk" in reason:
        return "high_risk"
    if reason == "low_confidence":
        return "low_confidence"
    if reason == "payment":
        return "payment"
    if reason == "terminal":
        return "terminal"
    if reason in {"no_fact_or_unverified", "empty_facts_no_fabrication", "estimate_guard_failed"}:
        return "no_fact_or_unverified"
    if reason in {"semantic_check_unavailable", "draft_error", "no_draft_fn"}:
        return "provider_runtime"
    if reason in {"hard_verification_failed", "authoritative_output_gate_blocked"} or findings or unsupported_claims:
        return "output_safety"
    if reason in {"contract_manager_only", "policy_permission"}:
        return "policy_permission"
    if not reason:
        return "policy_permission"
    return reason


_MANAGER_DEFERRAL_TEXT_RE = re.compile(
    r"(?:передам\s+(?:вопрос|его|её|обращение|менеджеру|ответственному)|"
    r"(?:менеджер|сотрудник|ответственный)[^.?!\n]{0,140}"
    r"(?:уточнит|сверит|проверит|подтвердит|свяжется|верн[её]тся))",
    re.I,
)


def _manager_deferral_text(text: str) -> bool:
    return bool(_MANAGER_DEFERRAL_TEXT_RE.search(str(text or "")))


def _force_draft_for_manager_reason_class(contract: AnswerContract, retrieval: RetrievalResult) -> str:
    intent_text = _contract_intent_text(contract).casefold().replace("ё", "е")
    if _asks_refund_policy(contract):
        return "refund"
    if _payment_method_target_anchors(contract) or re.search(r"оплат|рассроч|долями|счет|сч[её]т|payment", intent_text, re.I):
        return "payment"
    if re.search(r"terminal|prompt|инъекц|служебн|системн|ignore previous|инструкц", intent_text, re.I):
        return "terminal"
    confidence_values = tuple(
        value
        for value in (contract.confidence, contract.planner_confidence)
        if isinstance(value, (int, float)) and value > 0
    )
    if confidence_values and min(confidence_values) < 0.70:
        return "low_confidence"
    if contract.all_needed_fact_keys() and not retrieval.facts:
        return "no_fact_or_unverified"
    return "policy_permission"


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


def estimate_mode_enabled(context: Mapping[str, Any] | None = None) -> bool:
    if isinstance(context, MappingABC) and context.get(ESTIMATE_MODE_ENV) is not None:
        return _truthy(context.get(ESTIMATE_MODE_ENV))
    return _truthy(os.getenv(ESTIMATE_MODE_ENV))


def free_number_gate_enabled(context: Mapping[str, Any] | None = None) -> bool:
    if isinstance(context, MappingABC) and context.get(FREE_NUMBER_GATE_ENV) is not None:
        return _truthy(context.get(FREE_NUMBER_GATE_ENV))
    return _truthy(os.getenv(FREE_NUMBER_GATE_ENV))


def travel_compose_enabled(context: Mapping[str, Any] | None = None) -> bool:
    if isinstance(context, MappingABC) and context.get(TRAVEL_COMPOSE_ENV) is not None:
        return _truthy(context.get(TRAVEL_COMPOSE_ENV))
    return _truthy(os.getenv(TRAVEL_COMPOSE_ENV))


def quality_partial_yield_enabled(context: Mapping[str, Any] | None = None) -> bool:
    if isinstance(context, MappingABC) and context.get(QUALITY_PARTIAL_YIELD_ENV) is not None:
        return _truthy(context.get(QUALITY_PARTIAL_YIELD_ENV))
    return _truthy(os.getenv(QUALITY_PARTIAL_YIELD_ENV))


def quality_thread_memory_enabled(context: Mapping[str, Any] | None = None) -> bool:
    if isinstance(context, MappingABC) and context.get(QUALITY_THREAD_MEMORY_ENV) is not None:
        return _truthy(context.get(QUALITY_THREAD_MEMORY_ENV))
    return _truthy(os.getenv(QUALITY_THREAD_MEMORY_ENV))


def quality_composite_enabled(context: Mapping[str, Any] | None = None) -> bool:
    if isinstance(context, MappingABC) and context.get(QUALITY_COMPOSITE_ENV) is not None:
        return _truthy(context.get(QUALITY_COMPOSITE_ENV))
    return _truthy(os.getenv(QUALITY_COMPOSITE_ENV))


def quality_next_step_enabled(context: Mapping[str, Any] | None = None) -> bool:
    if isinstance(context, MappingABC) and context.get(QUALITY_NEXT_STEP_ENV) is not None:
        return _truthy(context.get(QUALITY_NEXT_STEP_ENV))
    return _truthy(os.getenv(QUALITY_NEXT_STEP_ENV))


def quality_clarify_scope_enabled(context: Mapping[str, Any] | None = None) -> bool:
    if isinstance(context, MappingABC) and context.get(QUALITY_CLARIFY_SCOPE_ENV) is not None:
        return _truthy(context.get(QUALITY_CLARIFY_SCOPE_ENV))
    return _truthy(os.getenv(QUALITY_CLARIFY_SCOPE_ENV))


def quality_useful_handoff_enabled(context: Mapping[str, Any] | None = None) -> bool:
    if isinstance(context, MappingABC) and context.get(QUALITY_USEFUL_HANDOFF_ENV) is not None:
        return _truthy(context.get(QUALITY_USEFUL_HANDOFF_ENV))
    return _truthy(os.getenv(QUALITY_USEFUL_HANDOFF_ENV))


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
    planner_values = ", ".join(PLANNER_INTENT_VALUES)
    known_slots: Mapping[str, Any] = {}
    topic_focus: Mapping[str, Any] = {}
    if isinstance(context, MappingABC):
        memory = context.get("dialogue_memory_view") if isinstance(context.get("dialogue_memory_view"), MappingABC) else {}
        known_slots = memory.get("known_slots") if isinstance(memory.get("known_slots"), MappingABC) else {}
        topic_focus = memory.get("topic_focus") if isinstance(memory.get("topic_focus"), MappingABC) else {}
    return (
        "Ты разбираешь диалог с родителем о курсах учебного центра.\n"
        f"Активный бренд: {_normalize_brand(active_brand)}. Клиентский ответ потом будет только по этому бренду.\n"
        "Верни строго JSON без пояснений:\n"
        "{ current_question, client_state, continued_topics[], denied_topics[], switched_topics[], forbidden_substitutions[],\n"
        "  known_slots: { имя: {value, source} },\n"
        "  planner_intent, planner_subvariant, planner_slots: {slot:value}, planner_confidence:0..1,\n"
        "  answer_mode:'confirmed_only'|'estimate_allowed', "
        "estimate_domain:'travel_time'|'route_logistics'|'general_advice'|'none', estimate_confidence:0..1,\n"
        "  selling: {objection:'price'|'none', exit_signal:bool, anxiety:bool, unmet_need:str, readiness:'exploring'|'comparing'|'ready'},\n"
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
        "- Если реплика — уточнение/эллипсис (короткий вопрос про класс/формат/цену/срок без названия предмета или продукта), "
        "ВОССТАНОВИ тему из истории, known_slots и topic_focus: в current_question и needed_fact_keys укажи полную тему "
        "(предмет+формат+класс+продукт), а не только новую деталь.\n"
        "- product_family из topic_focus важен: если тема была 'camp' (лагерь/смена), уточнение остаётся про смену, "
        "НЕ подменяй обычным курсом или олимпиадой.\n"
        "- Если клиент ЯВНО назвал другой предмет/продукт, заполни switched_topics и НЕ склеивай новую тему со старой.\n"
        "- known_slots указывай ТОЛЬКО с источником: 'client_turn_N' или 'fact:<key>'. Без источника слот не указывай.\n"
        "- client_state — ситуация/тон клиента для выбора регистра; не нужно потом произносить эмоцию вслух.\n"
        "- needed_fact_keys: только ключи или смысловые ключи из каталога; значения, суммы, даты и проценты не пиши.\n"
        "- Если нужен спорный возврат, жалоба, юридическая угроза или спорная оплата: is_p0=true, answerability=manager_only.\n"
        "- Жалобу/недовольство распознавай по смыслу, даже без слова «жалоба»: "
        "«ребёнок ничего не понял», «зря заплатили», «толку нет», «не нравится как ведут» — "
        "это is_p0=true, p0_reason='complaint', answerability=manager_only. "
        "Если в той же реплике есть вопрос о курсе, приоритет у жалобы, не собирай данные ребёнка.\n"
        "- Если прямого факта нет, но в каталоге есть ключ, ПО СМЫСЛУ покрывающий вопрос — поставь его в "
        "needed_fact_keys и answerable='self'. Если вопрос неоднозначен — задай ОДИН уточняющий подвопрос, не "
        "уходи к менеджеру. answerability=manager_only ТОЛЬКО при P0 или когда в каталоге реально нет покрывающего "
        "ключа. current_question заполняй всегда.\n"
        f"- planner_intent — главное намерение для выбора правила; выбери одно из: {planner_values}. "
        "Если не уверен, ставь general_consultation и planner_confidence ниже 0.70.\n"
        "- planner_subvariant — короткая разновидность внутри намерения, например online/offline/weekend/start_date/"
        "license/how_to_login/second_subject/live_seats; если не нужно — пустая строка.\n"
        "- planner_slots — только явно понятые или восстановленные из памяти слоты: grade, subject, format, product, "
        "product_family, payment_method. Не добавляй active_brand: бренд задаётся каналом, а не текстом клиента.\n"
        "- На эллипсисе используй topic_focus и known_slots для planner_intent/planner_slots так же, как для current_question: "
        "«а очно?» после цены информатики остаётся pricing/format по той же теме, не general_consultation.\n"
        "- answer_mode='estimate_allowed' ТОЛЬКО для низкорисковой оценки: дорога/логистика/география "
        "(как добраться, сколько ехать, расстояние) или общий педагогический совет в общем виде. "
        "Для дороги ставь estimate_domain='travel_time' или 'route_logistics'; для общего совета — 'general_advice'. "
        "Для всего продуктового — цены, скидки, расписание, даты, смены, лагерь, формат-условия, длительность урока, "
        "документы, возврат, оплата, места, запись — ставь answer_mode='confirmed_only', estimate_domain='none'. "
        "Диагноз конкретного ребёнка, обещание результата/поступления или вопрос «потянет ли мой ребёнок» — "
        "confirmed_only/none. Если сомневаешься — confirmed_only/none.\n"
        "- selling — только для мягких коммерческих сигналов, НЕ для P0. objection='price', если клиент прямо или по смыслу "
        "сомневается в цене/бюджете: «дорого», «серьёзная сумма для семьи», «не потянем», «есть дешевле?». "
        "exit_signal=true, если клиент уходит подумать/сравнить/обсудить: «подумаю», «посоветуюсь с мужем/семьёй», "
        "«посмотрю другие варианты». anxiety=true, если клиент боится ошибиться, недоверяет или прямо спрашивает, "
        "нормальный ли центр; НЕ путай с юридической угрозой или претензией. unmet_need — короткий внутренний ярлык "
        "невысказанной потребности без дословной цитаты клиента, например 'нужна мягкая поддержка по физике'; не ставь туда "
        "ПДн и не обещай оценку. readiness='ready', если клиент явно готов записываться/платить/просит следующий шаг; "
        "например «куда платить», «как записаться», «готовы оформить». "
        "readiness='comparing', если сравнивает варианты; иначе 'exploring'. Для нейтрального «сколько стоит/расскажите» "
        "ставь objection='none', exit_signal=false, anxiety=false, unmet_need='', readiness='exploring'. "
        "Реальный возврат, жалоба или спор оплаты остаются is_p0=true и selling не должен менять маршрут.\n"
        f"Уже известные данные: {json.dumps(dict(known_slots), ensure_ascii=False)}\n"
        f"Фокус темы из памяти: {json.dumps(dict(topic_focus), ensure_ascii=False)}\n"
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
            p0_source="floor" if p0_reason_pregate else "",
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

    model_p0 = bool(data.get("is_p0"))
    is_p0 = model_p0 or bool(p0_reason_pregate)
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
        planner_intent=_clean_planner_intent(data.get("planner_intent")),
        planner_subvariant=str(data.get("planner_subvariant") or "").strip()[:80],
        planner_slots=_clean_planner_slots(data.get("planner_slots")),
        planner_confidence=_clamp_float(data.get("planner_confidence")),
        answer_mode=_clean_answer_mode(data.get("answer_mode")),
        estimate_domain=_clean_estimate_domain(data.get("estimate_domain")),
        estimate_confidence=_clamp_float(data.get("estimate_confidence")),
        selling=_clean_selling(data.get("selling")),
        forbidden_substitutions=tuple(_seq(data.get("forbidden_substitutions"))),
        client_state=str(data.get("client_state") or "").strip()[:180],
        answerability=answerability,
        question_type=question_type,
        existence_target=existence_target,
        is_p0=is_p0,
        p0_reason=str(data.get("p0_reason") or p0_reason_pregate or "").strip()[:200],
        p0_source="floor" if p0_reason_pregate else ("model" if model_p0 else ""),
        confidence=_clamp_float(data.get("confidence")),
    )


def _clean_planner_intent(value: object) -> str:
    intent = str(value or "").strip().casefold()
    return intent if intent in PLANNER_INTENT_VALUES else ""


def _clean_answer_mode(value: object) -> str:
    return "estimate_allowed" if str(value or "").strip().casefold() == "estimate_allowed" else "confirmed_only"


def _clean_estimate_domain(value: object) -> str:
    domain = str(value or "").strip().casefold()
    return domain if domain in (*_ESTIMATE_DOMAINS, "none") else "none"


def _clean_selling(value: object) -> Mapping[str, Any]:
    default = {
        "objection": "none",
        "exit_signal": False,
        "anxiety": False,
        "unmet_need": "",
        "readiness": "exploring",
    }
    if not isinstance(value, MappingABC):
        return default
    objection = str(value.get("objection") or "none").strip().casefold()
    if objection != "price":
        objection = "none"
    readiness = str(value.get("readiness") or "exploring").strip().casefold()
    if readiness not in {"exploring", "comparing", "ready"}:
        readiness = "exploring"
    unmet_need = " ".join(str(value.get("unmet_need") or "").split())[:120]
    return {
        "objection": objection,
        "exit_signal": bool(value.get("exit_signal")),
        "anxiety": bool(value.get("anxiety")),
        "unmet_need": unmet_need,
        "readiness": readiness,
    }


def _clean_planner_slots(raw: object) -> Mapping[str, str]:
    if not isinstance(raw, MappingABC):
        return {}
    allowed = {"grade", "subject", "format", "product", "product_family", "payment_method"}
    result: dict[str, str] = {}
    for key, value in raw.items():
        name = str(key or "").strip().casefold()
        if name not in allowed:
            continue
        text = str(value or "").strip()
        if text:
            result[name] = text[:120]
    return result


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


_MEMORY_TOPIC_MARKERS_RE = re.compile(
    r"информат|физик|математ|хими|биолог|русск|англ|обществ|истори|литерат|географ|"
    r"лвш|лагер|смен|олимпиад|физтех|выездн|camp|lvsh|olympiad|phystech",
    re.I,
)


def _augment_contract_with_memory_topic(
    contract: AnswerContract,
    *,
    context: Mapping[str, Any] | None,
    fact_key_catalog: Sequence[str],
) -> AnswerContract:
    if contract.is_p0 or contract.switched_topics or not isinstance(context, MappingABC):
        return contract
    memory = _thread_memory_view_for_contract(contract, context=context)
    if not isinstance(memory, MappingABC):
        return contract
    thread_memory = quality_thread_memory_enabled(context)
    focus = _memory_focus_for_contract(memory, include_known_slots=thread_memory)
    if not focus:
        return contract
    subject = str(focus.get("subject") or "").strip()
    if not subject or _contract_has_topic(contract):
        return contract
    topic_keys = _keys_for_topic(focus, fact_key_catalog=fact_key_catalog, contract=contract)
    if not topic_keys:
        return contract
    current_question = _compose_topic_question(contract.current_question, focus)
    trace_event(
        context,
        "thread_memory_topic",
        {
            "applied": True,
            "flag_enabled": thread_memory,
            "current_question": current_question,
            "needed_fact_keys": list(topic_keys),
            "focus": dict(focus),
        },
    )
    return replace_contract_topic(
        contract,
        current_question=current_question,
        needed_fact_keys=topic_keys,
        memory_focus=focus if thread_memory else None,
    )


def _memory_focus_for_contract(memory: Mapping[str, Any], *, include_known_slots: bool) -> Mapping[str, str]:
    focus: dict[str, str] = {}
    raw_focus = memory.get("topic_focus")
    if isinstance(raw_focus, MappingABC):
        for key in ("subject", "grade", "format", "product", "product_family"):
            value = str(raw_focus.get(key) or "").strip()
            if value:
                focus[key] = value[:120]
    if not include_known_slots:
        return focus
    known_slots = memory.get("known_slots")
    if not isinstance(known_slots, MappingABC):
        return focus
    slot_sources = memory.get("slot_sources") if isinstance(memory.get("slot_sources"), MappingABC) else {}
    for key in ("subject", "grade", "format", "product", "product_family"):
        if focus.get(key):
            continue
        raw = known_slots.get(key)
        if raw is None and key == "grade":
            raw = known_slots.get("class")
        value = _memory_slot_value(raw)
        if not value:
            continue
        source = _memory_slot_source(raw, slot_sources.get(key) if isinstance(slot_sources, MappingABC) else "")
        if source and not _memory_slot_source_allowed(source):
            continue
        focus[key] = value[:120]
    return focus


def _thread_memory_view_for_contract(
    contract: AnswerContract,
    *,
    context: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    if not isinstance(context, MappingABC):
        return None
    memory = context.get("dialogue_memory_view")
    if not isinstance(memory, MappingABC):
        return memory if isinstance(memory, MappingABC) else None
    if not quality_thread_memory_enabled(context):
        return memory
    return _suppress_stale_thread_memory(memory, contract=contract)


def _suppress_stale_thread_memory(memory: Mapping[str, Any], *, contract: AnswerContract) -> Mapping[str, Any]:
    focus = memory.get("topic_focus") if isinstance(memory.get("topic_focus"), MappingABC) else {}
    known_slots = memory.get("known_slots") if isinstance(memory.get("known_slots"), MappingABC) else {}
    if not focus and not known_slots:
        return memory
    text = _contract_intent_text(contract)
    fields_to_drop: set[str] = set()
    clear_narrative = False

    current_family = _explicit_product_family_from_text(text)
    previous_family = str(focus.get("product_family") or _memory_slot_value(known_slots.get("product_family")) or "").strip()
    if current_family and previous_family and current_family != previous_family:
        fields_to_drop.update({"product", "product_family", "format"})
        clear_narrative = True

    current_subject = _explicit_subject_from_text(text)
    previous_subject = str(focus.get("subject") or _memory_slot_value(known_slots.get("subject")) or "").strip()
    if current_subject and previous_subject and current_subject != _canonical_subject(previous_subject):
        fields_to_drop.add("subject")
        clear_narrative = True

    current_camp_scope = _camp_scope_from_text(text)
    previous_camp_scope = _camp_scope_from_memory(focus, known_slots)
    if current_camp_scope and previous_camp_scope and current_camp_scope != previous_camp_scope:
        fields_to_drop.add("product")
        clear_narrative = True

    if _explicit_service_topic_from_text(text):
        fields_to_drop.update({"subject", "grade", "format", "product", "product_family"})
        clear_narrative = True

    if not fields_to_drop and not clear_narrative:
        return memory

    sanitized: dict[str, Any] = dict(memory)
    replacement_focus: dict[str, str] = {}
    if current_family:
        replacement_focus["product_family"] = current_family
    if current_camp_scope:
        replacement_focus["product_family"] = "camp"
        replacement_focus["product"] = "lvsh_mendeleevo" if current_camp_scope == "residential_lvsh" else "city_camp"
    if focus:
        new_focus = {key: value for key, value in dict(focus).items() if key not in fields_to_drop}
        new_focus.update(replacement_focus)
        if new_focus:
            sanitized["topic_focus"] = new_focus
        else:
            sanitized.pop("topic_focus", None)
    elif replacement_focus:
        sanitized["topic_focus"] = replacement_focus
    if known_slots:
        new_known = {key: value for key, value in dict(known_slots).items() if key not in fields_to_drop}
        if new_known:
            sanitized["known_slots"] = new_known
        else:
            sanitized.pop("known_slots", None)
    slot_sources = sanitized.get("slot_sources")
    if isinstance(slot_sources, MappingABC):
        new_sources = {key: value for key, value in dict(slot_sources).items() if key not in fields_to_drop}
        if new_sources:
            sanitized["slot_sources"] = new_sources
        else:
            sanitized.pop("slot_sources", None)
    if clear_narrative:
        for key in ("open_question", "last_bot_commitments", "conversation_summary_short"):
            sanitized.pop(key, None)
    return sanitized


def _explicit_product_family_from_text(text: str) -> str:
    low = str(text or "").casefold().replace("ё", "е")
    if re.search(r"не\s+лагер|не\s+лвш|вместо\s+лагер|обычн\w*\s+курс|регулярн\w*\s+курс", low, re.I):
        return "regular_course"
    if re.search(r"лвш|лагер|летн\w*\s+школ|смен|менделеев|выездн|каникул", low, re.I):
        return "camp"
    if re.search(r"\bкурс\b|онлайн-курс|очные\s+курсы|регулярн", low, re.I) and not re.search(r"лагер|лвш|смен", low, re.I):
        return "regular_course"
    return ""


def _explicit_service_topic_from_text(text: str) -> str:
    low = str(text or "").casefold().replace("ё", "е")
    if re.search(r"налог|вычет|3-ндфл|фнс|кнд", low, re.I):
        return "tax"
    if re.search(r"маткап|материнск", low, re.I):
        return "matkap"
    if re.search(r"справк|договор|сертификат|чек|квитанц|документ", low, re.I):
        return "document"
    return ""


def _canonical_subject(value: str) -> str:
    raw = str(value or "").casefold().replace("ё", "е")
    if "информ" in raw:
        return "информатика"
    if "физ" in raw:
        return "физика"
    if "мат" in raw:
        return "математика"
    if "хим" in raw:
        return "химия"
    if "био" in raw:
        return "биология"
    if "рус" in raw:
        return "русский"
    if "анг" in raw:
        return "английский"
    return raw.strip()


def _explicit_subject_from_text(text: str) -> str:
    low = str(text or "").casefold().replace("ё", "е")
    for subject in ("информатика", "физика", "математика", "химия", "биология", "русский", "английский"):
        if _focus_aliases("subject", subject) and _key_has_any_topic_alias(low, _focus_aliases("subject", subject)):
            return subject
    return ""


def _camp_scope_from_memory(focus: Mapping[str, Any], known_slots: Mapping[str, Any]) -> str:
    product = str(focus.get("product") or _memory_slot_value(known_slots.get("product")) or "").strip()
    family = str(focus.get("product_family") or _memory_slot_value(known_slots.get("product_family")) or "").strip()
    return _camp_scope_from_text(f"{product} {family}")


def _memory_slot_value(raw: object) -> str:
    if isinstance(raw, MappingABC):
        return str(raw.get("value") or "").strip()
    return str(raw or "").strip()


def _memory_slot_source(raw: object, fallback: object = "") -> str:
    if isinstance(raw, MappingABC):
        return str(raw.get("source") or fallback or "").strip()
    return str(fallback or "").strip()


def _memory_slot_source_allowed(source: str) -> bool:
    normalized = str(source or "").strip().casefold()
    if not normalized:
        return True
    return (
        normalized in {"dialogue_memory", "memory_llm", "provided_context", "client_confirmed"}
        or normalized.startswith("client_turn")
        or normalized.startswith("fact:")
    )


def _contract_has_topic(contract: AnswerContract) -> bool:
    return bool(_MEMORY_TOPIC_MARKERS_RE.search(_contract_intent_text(contract)))


def _compose_topic_question(question: str, focus: Mapping[str, Any]) -> str:
    base = str(question or "").strip() or "уточнение по текущей теме"
    current_format = _format_from_text(base)
    parts: list[str] = []
    subject = str(focus.get("subject") or "").strip()
    grade = _grade_from_text(base) or str(focus.get("grade") or "").strip()
    format_value = current_format or str(focus.get("format") or "").strip()
    product = str(focus.get("product") or "").strip()
    product_family = str(focus.get("product_family") or "").strip()
    if subject:
        parts.append(f"предмет {subject}")
    if grade:
        parts.append(f"{grade} класс")
    if format_value:
        parts.append(f"формат {format_value}")
    if product:
        parts.append(f"продукт {product}")
    if product_family:
        family_text = "лагерь/смена" if product_family == "camp" else "регулярный курс" if product_family == "regular_course" else product_family
        parts.append(f"тип продукта {family_text}")
    if not parts:
        return base
    return f"{base}. Тема: {', '.join(parts)}."


def replace_contract_topic(
    contract: AnswerContract,
    *,
    current_question: str,
    needed_fact_keys: Sequence[str],
    memory_focus: Mapping[str, Any] | None = None,
) -> AnswerContract:
    keys = tuple(dict.fromkeys(str(item or "").strip() for item in needed_fact_keys if str(item or "").strip()))
    if not keys:
        return contract
    subquestions = contract.subquestions or (
        Subquestion(
            text=contract.current_question,
            answerable="self" if contract.answerability == "answer_self" else "manager",
            needed_fact_keys=(),
            question_type=contract.question_type,
            existence_target=contract.existence_target,
        ),
    )
    updated: list[Subquestion] = []
    for item in subquestions:
        if item.answerable == "self" or contract.answerability == "answer_self":
            updated.append(
                replace(
                    item,
                    text=current_question,
                    answerable="self" if contract.answerability == "answer_self" else item.answerable,
                    needed_fact_keys=tuple(dict.fromkeys((*item.needed_fact_keys, *keys))),
                )
            )
        else:
            updated.append(item)
    updates: dict[str, Any] = {
        "current_question": current_question,
        "subquestions": tuple(updated),
    }
    if isinstance(memory_focus, MappingABC) and memory_focus:
        planner_slots = dict(contract.planner_slots)
        known_slots = dict(contract.known_slots)
        for key in ("subject", "grade", "format", "product", "product_family"):
            value = _memory_focus_value_for_contract(key, memory_focus, contract=contract, current_question=current_question)
            if not value:
                continue
            planner_slots.setdefault(key, value[:120])
            known_slots.setdefault(key, Slot(value=value[:120], source="dialogue_memory"))
        updates["planner_slots"] = planner_slots
        updates["known_slots"] = known_slots
    return replace(contract, **updates)


def _memory_focus_value_for_contract(
    key: str,
    memory_focus: Mapping[str, Any],
    *,
    contract: AnswerContract,
    current_question: str,
) -> str:
    if key == "format":
        explicit_format = _format_from_text(contract.current_question) or _format_from_text(current_question)
        if explicit_format:
            return explicit_format
    if key == "grade":
        explicit_grade = _grade_from_text(contract.current_question) or _grade_from_text(current_question)
        if explicit_grade:
            return explicit_grade
    return str(memory_focus.get(key) or "").strip()


def _keys_for_topic(
    focus: Mapping[str, Any],
    *,
    fact_key_catalog: Sequence[str],
    contract: AnswerContract,
) -> tuple[str, ...]:
    subject_aliases = _focus_aliases("subject", focus.get("subject"))
    if not subject_aliases:
        return ()
    grade_aliases = _focus_aliases("grade", focus.get("grade"))
    format_aliases = _focus_aliases("format", _format_from_text(_contract_intent_text(contract)) or focus.get("format"))
    product_aliases = _focus_aliases("product", focus.get("product"))
    family = str(focus.get("product_family") or "").strip().casefold()
    family_aliases = _focus_aliases("product_family", family)
    intent_aliases = _contract_query_aliases(contract)

    scored: list[tuple[int, str]] = []
    for key in tuple(dict.fromkeys(str(item or "").strip() for item in fact_key_catalog if str(item or "").strip())):
        has_subject = _key_has_any_topic_alias(key, subject_aliases)
        has_family = _key_has_any_topic_alias(key, family_aliases)
        if not has_subject and not (family == "camp" and has_family):
            continue
        if family == "camp" and family_aliases and not has_family:
            continue
        if family == "regular_course" and _key_has_any_topic_alias(key, _focus_aliases("product_family", "camp")):
            continue
        score = 40 if has_subject else 28
        if grade_aliases and _key_has_any_topic_alias(key, grade_aliases):
            score += 18
        if format_aliases and _key_has_any_topic_alias(key, format_aliases):
            score += 16
        if product_aliases and _key_has_any_topic_alias(key, product_aliases):
            score += 14
        if family_aliases and has_family:
            score += 12
        if intent_aliases and _key_has_any_topic_alias(key, intent_aliases):
            score += 10
        scored.append((score, key))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return tuple(key for _, key in scored[:8])


def _focus_aliases(field: str, value: object) -> tuple[str, ...]:
    raw = str(value or "").strip().casefold().replace("ё", "е")
    if not raw:
        return ()
    if field == "subject":
        if "информ" in raw:
            return ("информат", "informatics", "computer_science", "computer")
        if "физ" in raw:
            return ("физик", "physics")
        if "мат" in raw:
            return ("математ", "math")
        if "хим" in raw:
            return ("хим", "chem")
        if "био" in raw:
            return ("биолог", "bio")
        if "рус" in raw:
            return ("русск", "russian")
        if "анг" in raw:
            return ("англ", "english")
        return tuple(part for part in re.split(r"[\s,;/]+", raw) if part)
    if field == "grade":
        match = re.search(r"\b([1-9]|1[01])\b", raw)
        if not match:
            return ()
        grade = match.group(1)
        return (f"grade{grade}", f"class{grade}", f"{grade}klass", f"klass{grade}", f"{grade}класс")
    if field == "format":
        if "онлайн" in raw or "online" in raw:
            return ("online", "онлайн")
        if "очно" in raw or "офлайн" in raw or "offline" in raw or "ochno" in raw:
            return ("offline", "ochno", "очно", "офлайн")
        return ()
    if field == "product_family":
        if raw == "camp" or "лагер" in raw or "смен" in raw or "лвш" in raw:
            return ("camp", "lvsh", "лвш", "лагер", "смен", "mendeleevo", "менделеев")
        if raw == "regular_course":
            return ("regular", "regular_course", "course", "курс")
        return ()
    if field == "product":
        aliases = [part for part in re.split(r"[\s,;/]+", raw) if len(part) >= 3]
        if "лвш" in raw:
            aliases.extend(["lvsh", "camp", "лагер", "смен"])
        return tuple(dict.fromkeys(aliases))
    return ()


def _contract_query_aliases(contract: AnswerContract) -> tuple[str, ...]:
    text = _contract_intent_text(contract)
    aliases: list[str] = []
    if re.search(r"цен|стоим|сколько|оплат", text, re.I):
        aliases.extend(("price", "prices", "cost", "tuition", "стоим", "цен"))
    if re.search(r"онлайн|online", text, re.I):
        aliases.extend(("online", "онлайн"))
    if re.search(r"очно|офлайн|offline|ochno", text, re.I):
        aliases.extend(("offline", "ochno", "очно", "офлайн"))
    if re.search(r"распис|дни|когда|выходн|будн", text, re.I):
        aliases.extend(("schedule", "days", "weekly", "распис", "дни", "weekend"))
    if re.search(r"запис|материал|кабинет", text, re.I):
        aliases.extend(("recording", "materials", "cabinet", "запис"))
    return tuple(dict.fromkeys(aliases))


def _format_from_text(text: str) -> str:
    low = str(text or "").casefold().replace("ё", "е")
    if re.search(r"онлайн|online", low, re.I):
        return "онлайн"
    if re.search(r"очно|офлайн|offline|ochno", low, re.I):
        return "очно"
    return ""


def _grade_from_text(text: str) -> str:
    match = re.search(r"\b([1-9]|1[01])\s*(?:класс|кл\.?|grade)?\b", str(text or "").casefold().replace("ё", "е"))
    return match.group(1) if match else ""


def _key_has_any_topic_alias(key: str, aliases: Sequence[str]) -> bool:
    if not aliases:
        return False
    raw = str(key or "").casefold().replace("ё", "е")
    norm = _normalize_lookup(raw)
    for alias in aliases:
        alias_raw = str(alias or "").casefold().replace("ё", "е")
        alias_norm = _normalize_lookup(alias_raw)
        if alias_raw and alias_raw in raw:
            return True
        if alias_norm and alias_norm in norm:
            return True
    return False


def p0_pre_gate(text: str, *, context: Mapping[str, Any] | None = None) -> str | None:
    codes = hard_codes_from_text(text)
    if codes:
        result = ",".join(codes)
        trace_event(context, "p0_pre_gate", {"source": "regex", "codes": list(codes), "result": result})
        return result
    soft_codes = soft_codes_from_text(text)
    if soft_codes:
        trace_event(context, "p0_pre_gate", {"source": "regex_soft", "codes": list(soft_codes), "result": ""})
    decision = classify_answer_safety(client_message=text, context=context)
    if decision.p0_required:
        result = ",".join(decision.risk_codes or (decision.primary_risk or "p0",))
        trace_event(
            context,
            "p0_pre_gate",
            {"source": "classifier", "codes": list(decision.risk_codes or ()), "primary_risk": decision.primary_risk, "result": result},
        )
        return result
    trace_event(context, "p0_pre_gate", {"source": "classifier", "codes": [], "result": ""})
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


def _resolve_answer_mode(
    *,
    contract: AnswerContract,
    question_text: str,
    has_p0: bool,
    has_kb_fact: bool,
    estimate_enabled: bool = False,
    free_number_gate: bool = False,
) -> tuple[str, str]:
    if has_p0:
        return "confirmed_only", "none"
    if free_number_gate:
        if has_kb_fact:
            return "confirmed_only", "none"
        if contract.answer_mode == "estimate_allowed" and contract.estimate_domain in _ESTIMATE_DOMAINS:
            return "estimate_allowed", contract.estimate_domain
        if contract.answerability == "answer_self":
            return "estimate_allowed", "general_advice"
        return "confirmed_only", "none"
    if _is_product_question(
        question_text,
        planner_intent=contract.planner_intent,
        needed_fact_keys=contract.all_needed_fact_keys(),
    ):
        return "confirmed_only", "none"
    if has_kb_fact:
        return "confirmed_only", "none"
    if contract.answer_mode == "estimate_allowed" and contract.estimate_domain in _ESTIMATE_DOMAINS:
        return "estimate_allowed", contract.estimate_domain
    return "confirmed_only", "none"


def _is_product_question(
    text: str,
    *,
    planner_intent: str = "",
    needed_fact_keys: Sequence[str] = (),
) -> bool:
    combined = " ".join(str(item or "") for item in (text, planner_intent, *needed_fact_keys))
    if _PRODUCT_QUESTION_RE.search(combined):
        return True
    if _INDIVIDUAL_CHILD_RE.search(str(text or "")):
        return True
    normalized_intent = str(planner_intent or "").casefold()
    if any(
        marker in normalized_intent
        for marker in (
            "price",
            "pricing",
            "discount",
            "schedule",
            "enroll",
            "format",
            "camp",
            "refund",
            "payment",
            "docs",
            "document",
            "trial",
            "installment",
        )
    ):
        return True
    return bool(tuple(item for item in needed_fact_keys if str(item or "").strip()))


def _estimate_policy_context(
    *,
    contract: AnswerContract,
    retrieval: RetrievalResult,
    enabled: bool,
    free_number_gate: bool = False,
    question_text: str,
) -> Mapping[str, Any]:
    resolved_mode, resolved_domain = _resolve_answer_mode(
        contract=contract,
        question_text=question_text,
        has_p0=contract.is_p0,
        has_kb_fact=bool(retrieval.facts),
        estimate_enabled=enabled,
        free_number_gate=free_number_gate,
    )
    return {
        "enabled": enabled,
        "free_number_gate": free_number_gate,
        "planner_answer_mode": contract.answer_mode,
        "planner_estimate_domain": contract.estimate_domain,
        "planner_estimate_confidence": contract.estimate_confidence,
        "answer_mode": resolved_mode,
        "estimate_domain": resolved_domain,
        "has_kb_fact": bool(retrieval.facts),
        "individual_child_question": bool(_INDIVIDUAL_CHILD_RE.search(str(question_text or ""))),
        "product_question": _is_product_question(
            question_text,
            planner_intent=contract.planner_intent,
            needed_fact_keys=contract.all_needed_fact_keys(),
        ),
    }


def _quality_partial_yield_travel_domain(
    *,
    contract: AnswerContract,
    client_words: str,
    retrieval: RetrievalResult,
    context: Mapping[str, Any] | None,
    free_number_gate: bool,
) -> str:
    enabled = quality_partial_yield_enabled(context) or travel_compose_enabled(context)
    number_gate = free_number_gate or travel_compose_enabled(context)
    if (
        not enabled
        or not number_gate
        or retrieval.facts
        or contract.is_p0
    ):
        return ""
    combined = " ".join(
        part
        for part in (
            client_words,
            contract.current_question,
            " ".join(item.text for item in contract.subquestions if item.text),
        )
        if part
    )
    if not _TRAVEL_ESTIMATE_TEXT_RE.search(combined):
        return ""
    product_guard_text = client_words or combined
    if _TRAVEL_ESTIMATE_PRODUCT_BLOCK_RE.search(product_guard_text):
        return ""
    normalized = combined.casefold().replace("ё", "е")
    if re.search(r"как\s+добраться|маршрут|проехать|электрич|метро|автобус|такси|станци|остановк", normalized, re.I):
        return "route_logistics"
    return "travel_time"


def build_draft_prompt(
    *,
    conversation: Sequence[Mapping[str, str]],
    contract: AnswerContract,
    facts: Mapping[str, str],
    missing: Sequence[str],
    tone_guide: str = "",
    style_examples: Sequence[str] = (),
    toggles: Toggles | None = None,
    dialogue_memory_view: Mapping[str, Any] | None = None,
) -> str:
    toggles = toggles or Toggles()
    hist = "\n".join(f"{item.get('role', '?')}: {item.get('text', '')}" for item in conversation)
    facts_block = "\n".join(f"- {key}: {value}" for key, value in facts.items()) or "(нет подтверждённых фактов под этот вопрос)"
    memory_block = _format_memory_block(dialogue_memory_view)
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
        "Если в фактах есть ответ на вопрос ПО СМЫСЛУ — отвечай из него, даже если формулировка факта не совпадает с вопросом дословно. "
        "Считай совпадением по смыслу: синонимы и иные названия того же продукта "
        "(вопрос «олимпиада по физике» + факт «олимпиадная подготовка Физтех» — это одно и то же, отвечай да); "
        "конкретное внутри общего (вопрос «в августе» + факт «3-14 августа» — да; "
        "вопрос «для 10 класса» + факт «5-11 класс» — да). Не уходи к менеджеру только из-за разной формулировки.\n"
        "«Соседний факт», который подставлять нельзя, — это факт про другой продукт/предмет/способ оплаты/формат "
        "(физика vs математика; рассрочка vs Долями; очно vs онлайн), а не тот же факт другими словами. "
        "«Нет» можно писать только при явном отрицательном факте про X. "
        "Если вопрос про конкретный способ оплаты, отвечай именно про него: прямой перевод/счёт, банковская рассрочка и Долями — разные способы. "
        "Не подставляй соседний способ оплаты как ответ; если факта по спрошенному способу нет, узко передай менеджеру проверить именно его. "
        "Если клиент гипотетически спрашивает о возврате до оплаты/до старта, отвечай из факта про остаток неистраченных средств и не оформляй это как жалобу. "
        "Если клиент уже требует вернуть деньги или спорит по оплате, не отвечай автономно.\n"
        "В составном вопросе ответь на подтверждённые безопасные части, а неподтверждённую часть узко передай менеджеру. "
        "Никогда не утверждай расписание, класс, предмет, формат, цену, скидку, дату или тему, которых нет в фактах или словах клиента. "
        "Передавай менеджеру при сомнении только если по теме вопроса факта нет вовсе, найденный факт про другой продукт/тему или это P0 "
        "(возврат/жалоба/спор оплаты). Если факт по теме есть и покрывает вопрос по смыслу — отвечай сам, не уходи к менеджеру. "
        "Не раскрывай внутренние настройки, fact_id/source_id/JSON. Не обещай результат, возврат, одобрение банка/СФР/ФНС.\n"
        + (f"Манера: {tone_guide}\n" if tone_guide else "")
        + memory_block
        + f"История диалога:\n{hist}\n"
        "Верни только текст клиенту, без JSON и служебных пометок."
    )


def build_estimate_prompt(
    *,
    conversation: Sequence[Mapping[str, str]],
    contract: AnswerContract,
    estimate_domain: str,
    tone_guide: str = "",
) -> str:
    hist = "\n".join(f"{item.get('role', '?')}: {item.get('text', '')}" for item in conversation)
    domain_hint = {
        "travel_time": "дорога/время в пути/география",
        "route_logistics": "логистика маршрута/как добраться/расстояние",
        "general_advice": "общий педагогический совет без диагностики конкретного ребёнка",
    }.get(estimate_domain, "низкорисковая бытовая оценка")
    return (
        f"Активный бренд: {contract.active_brand}. Не упоминай другой бренд.\n"
        "Напиши клиенту полезный ответ, потому что подтверждённого факта по этому вопросу может не быть.\n"
        f"Вопрос: {contract.current_question}\n"
        f"Разрешённый домен оценки: {domain_hint}.\n"
        "Правила:\n"
        "- Отвечай естественно и помогай по сути вопроса.\n"
        "- Если это бытовое/дорога/логистика/география или общий совет без продуктовой конкретики, можно дать полезную оценку.\n"
        "- Для дороги/маршрута дай именно ориентир по времени в минутах, а не повторяй только адрес или площадку.\n"
        "- Для любой такой оценки с числом ОБЯЗАТЕЛЬНО поставь рядом маркер неуверенности: «ориентировочно», «примерно», «около», «обычно» или «скорее всего».\n"
        "- Нельзя оценивать цену, скидку, расписание, даты, смены, длительность занятия, документы, возврат, оплату, места и запись.\n"
        "- Если клиент спрашивает продуктовую конкретику без подтверждённого факта, честно скажи, что это проверит менеджер; не придумывай число даже с оговоркой.\n"
        "- В общем педагогическом совете говори только про типичную ситуацию; не ставь диагноз конкретному ребёнку и не обещай результат.\n"
        "- Не добавляй ₽, проценты, даты занятий, расписание или условия курса.\n"
        "- Если точность зависит от маршрута/расписания транспорта, так и скажи мягко.\n"
        + (f"Манера: {tone_guide}\n" if tone_guide else "")
        + f"История диалога:\n{hist}\n"
        "Верни только текст клиенту, без JSON и служебных пометок."
    )


def _format_memory_block(view: Mapping[str, Any] | None) -> str:
    if not view:
        return ""
    open_question = view.get("open_question") or {}
    open_question_text = str(open_question.get("text") or "") if isinstance(open_question, MappingABC) else str(open_question or "")
    known_slots = view.get("known_slots") or {}
    do_not_ask_again = view.get("do_not_ask_again") or ()
    commitments = view.get("last_bot_commitments") or ()
    topic_focus = view.get("topic_focus") or {}
    summary = str(view.get("conversation_summary_short") or "")
    lines = ["Рабочая память переписки (используй, но P0/бренд/факт-гарды важнее памяти):"]
    if summary:
        lines.append(f"- кратко: {summary}")
    if topic_focus:
        lines.append(f"- фокус темы: {json.dumps(topic_focus, ensure_ascii=False)}")
    if open_question_text:
        lines.append(f"- открытый вопрос клиента (закрой первым, если безопасно): {open_question_text}")
    if known_slots:
        lines.append(f"- уже известно (НЕ переспрашивай): {json.dumps(known_slots, ensure_ascii=False)}")
    if do_not_ask_again:
        lines.append(f"- не спрашивай заново: {', '.join(str(item) for item in do_not_ask_again)}")
    if commitments:
        lines.append(f"- бот уже обещал (не меняй без факта): {'; '.join(str(item) for item in commitments)}")
    return "\n".join(lines) + "\n\n"


def _format_established_topic_block(topic: Mapping[str, Any] | None) -> str:
    if not topic:
        return ""
    compact = {str(key): str(value) for key, value in topic.items() if str(value or "").strip()}
    if not compact:
        return ""
    return (
        f"Установленная тема диалога: {json.dumps(compact, ensure_ascii=False)}.\n"
        "Если клиент уточняет класс или формат уже установленной темы (тот же предмет/продукт), "
        "не ставь wrong_scope только из-за смены класса/формата; проверяй утверждение по факту той же темы. "
        "Это НЕ разрешает подменять продукт, предмет или семью продукта: лагерь/смена, обычный курс и олимпиада "
        "остаются разными scope, а противоречие факту остаётся contradicted.\n"
    )


def _established_topic_from_context(
    context: Mapping[str, Any] | None,
    *,
    contract: AnswerContract | None = None,
) -> Mapping[str, Any] | None:
    if not isinstance(context, MappingABC):
        return None
    memory = (
        _thread_memory_view_for_contract(contract, context=context)
        if contract is not None
        else context.get("dialogue_memory_view")
    )
    if not isinstance(memory, MappingABC):
        return None
    topic: dict[str, Any] = {}
    focus = memory.get("topic_focus")
    if isinstance(focus, MappingABC):
        for key in ("subject", "grade", "format", "product", "product_family"):
            value = focus.get(key)
            if str(value or "").strip():
                topic[key] = value
    known_slots = memory.get("known_slots")
    if isinstance(known_slots, MappingABC):
        for key in ("subject", "grade", "format", "product"):
            value = known_slots.get(key)
            if key not in topic and str(value or "").strip():
                topic[key] = value
    return topic or None


def build_faithfulness_prompt(
    draft: str,
    *,
    facts: Mapping[str, str],
    client_words: str,
    established_topic: Mapping[str, Any] | None = None,
) -> str:
    facts_block = "\n".join(f"- {key}: {value}" for key, value in facts.items()) or "(фактов нет)"
    established_topic_block = _format_established_topic_block(established_topic)
    return (
        "Проверь черновик ответа на верность. Верни строго JSON: "
        "{\"claims\": [{\"claim\": \"...\", \"evidence_fact_key\": \"...\", "
        "\"verdict\": \"supported|unsupported|glued|wrong_scope|contradicted\", \"reason\": \"...\"}], "
        "\"unsupported\": [<конкретные утверждения, которых нет ни в фактах, ни в словах клиента>]}.\n"
        "Конкретное утверждение = расписание/дни, формат (онлайн/очно), тема и направление "
        "(обычный курс / лагерь / смена / олимпиада / интенсив), класс, наличие пробного/мест/записи, "
        "сроки, условия, цены, действия.\n"
        "Каждое атомарное утверждение должно подтверждаться ОДНИМ fact_key из списка фактов.\n"
        "ТЕМА И ФОРМАТ — строго: утверждение о формате/теме/направлении/классе подтверждено ТОЛЬКО "
        "фактом про ТОТ ЖЕ продукт и тему, что в вопросе клиента. Лагерь/смена ≠ обычный курс ≠ "
        "олимпиадная подготовка: если клиент спрашивает про летнюю смену или лагерь, а факт/ответ про "
        "обычный курс или олимпиаду — verdict = wrong_scope, даже если предмет/класс совпали.\n"
        "ВЫБОР ФОРМАТА: если клиент спросил «онлайн или очно» (или не указал формат), а черновик "
        "утверждает конкретный формат, для которого в фактах нет однозначного подтверждения именно "
        "по спрошенному продукту/классу — verdict = unsupported (нельзя выбирать формат за клиента).\n"
        "РАСПИСАНИЕ/ДНИ/ВРЕМЯ: дни недели, «в будни», «по вторникам», «вечером», частота — unsupported, "
        "если нет факта-расписания именно для этого продукта/класса.\n"
        "ОТРИЦАНИЕ И СПЕЦИФИКА: утверждение об отсутствии/полноте («других форматов нет», «только это», "
        "«это всё, что есть») или о специфике курса («фокус на ОГЭ/ЕГЭ», «экзаменационный курс», "
        "«подготовка к олимпиаде») — unsupported, если нет прямого подтверждающего факта; отсутствие "
        "других вариантов нельзя выводить из того, что их нет в списке.\n"
        "ПРОТИВОРЕЧИЕ: если утверждение противоречит факту (черновик: онлайн, а факт: очно; черновик: "
        "9 класс, а факт: 10) — verdict = contradicted.\n"
        "Если утверждение собрано из двух разных фактов — glued.\n"
        "Для supported обязательно укажи evidence_fact_key ровно из списка ниже.\n"
        + established_topic_block
        + "Не считай нарушением общую вежливость и предложение помочь.\n"
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
    established_topic: Mapping[str, Any] | None = None,
) -> FaithfulnessResult:
    if faithfulness_fn is None:
        return FaithfulnessResult(unsupported=(), available=True)
    prompt = build_faithfulness_prompt(draft, facts=facts, client_words=client_words, established_topic=established_topic)
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
            if verdict in {"unsupported", "glued", "not_supported", "false", "wrong_scope", "contradicted"}:
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


def build_semantic_match_prompt(*, question: str, facts: Mapping[str, str], draft: str) -> str:
    facts_block = "\n".join(f"- {value}" for value in facts.values()) or "(фактов нет)"
    return (
        f"Клиент спросил: {question}\n"
        f"У нас есть подтверждённые факты:\n{facts_block}\n"
        f"Черновик ответа бота: {draft}\n"
        "Вопрос: отвечают ли эти факты на вопрос клиента ПО СМЫСЛУ, и про ТОТ ЖЕ продукт/тему?\n"
        "Правила: «олимпиадная подготовка Физтех» = ответ на «олимпиада по физике» (covers=true). "
        "«в августе» покрывается фактом «3-14 августа» (covers=true). "
        "Но летняя СМЕНА/ЛАГЕРЬ ≠ обычный регулярный курс: если спросили про смену, а факт про "
        "регулярный курс — same_product=false. Другой предмет/способ оплаты/формат — same_product=false.\n"
        "Верни строго JSON: {\"covers\": true|false, \"same_product\": true|false}."
    )


def _semantic_match(
    semantic_match_fn: Callable[[str], object],
    *,
    contract: AnswerContract,
    retrieval: RetrievalResult,
    client_words: str,
    draft: str,
) -> Mapping[str, Any]:
    prompt = build_semantic_match_prompt(
        question=_semantic_match_question_text(contract, client_words=client_words),
        facts=retrieval.facts,
        draft=draft,
    )
    try:
        raw = semantic_match_fn(prompt)
    except Exception:
        return {}
    data: object = raw
    if isinstance(raw, str):
        try:
            data = _extract_json_object(raw)
        except Exception:
            return {}
    if not isinstance(data, MappingABC):
        return {}
    return {
        "covers": data.get("covers"),
        "same_product": data.get("same_product"),
        "reason": str(data.get("reason") or ""),
    }


def _semantic_match_question_text(contract: AnswerContract, *, client_words: str) -> str:
    return " ".join(
        part
        for part in (
            client_words,
            contract.current_question,
            contract.existence_target,
            " ".join(item.text for item in contract.subquestions),
            " ".join(item.existence_target for item in contract.subquestions),
        )
        if part
    )


def _quality_handoff_estimate_domain(
    *,
    contract: AnswerContract,
    client_words: str,
    context: Mapping[str, Any] | None,
) -> str:
    enabled = quality_partial_yield_enabled(context) or travel_compose_enabled(context)
    number_gate = free_number_gate_enabled(context) or travel_compose_enabled(context)
    if not enabled or not number_gate or contract.is_p0:
        return ""
    combined = " ".join(
        part
        for part in (
            client_words,
            contract.current_question,
            " ".join(item.text for item in contract.subquestions if item.text),
            contract.existence_target,
        )
        if part
    )
    if not combined.strip():
        return ""
    product_guard_text = client_words or combined
    if _TRAVEL_ESTIMATE_PRODUCT_BLOCK_RE.search(product_guard_text):
        return ""
    if _TRAVEL_ESTIMATE_TEXT_RE.search(combined):
        normalized = combined.casefold().replace("ё", "е")
        if re.search(r"как\s+добраться|маршрут|проехать|электрич|метро|автобус|такси|станци|остановк", normalized, re.I):
            return "route_logistics"
        return "travel_time"
    if (
        contract.answer_mode == "estimate_allowed"
        and contract.estimate_domain in _ESTIMATE_DOMAINS
        and not _INDIVIDUAL_CHILD_RE.search(combined)
    ):
        return contract.estimate_domain
    return ""


def _semantic_recover_or_handoff(
    *,
    contract: AnswerContract,
    retrieval: RetrievalResult,
    draft: str,
    semantic_match_fn: Callable[[str], object] | None,
    faithfulness_fn: Callable[[str], object] | None,
    client_words: str,
    conversation: Sequence[Mapping[str, str]],
    context: Mapping[str, Any] | None,
    toggles: Toggles,
    previous_bot_texts: Sequence[str] = (),
) -> DialogueContractPipelineResult | None:
    if (
        semantic_match_fn is None
        or not retrieval.facts
        or contract.answerability != "answer_self"
        or contract.is_p0
    ):
        return None
    verdict = _semantic_match(
        semantic_match_fn,
        contract=contract,
        retrieval=retrieval,
        client_words=client_words,
        draft=draft,
    )
    covers = _truthy(verdict.get("covers"))
    same_product = _truthy(verdict.get("same_product"))
    if not (covers and same_product):
        trace_event(
            context,
            "semantic_recover",
            {"replaced": False, "covers": covers, "same_product": same_product},
        )
        return None
    replacement = _verified_empty_handoff_replacement(
        draft,
        contract=contract,
        retrieval=retrieval,
        client_words=client_words,
        faithfulness_fn=faithfulness_fn,
        toggles=toggles,
        context=context,
        previous_bot_texts=previous_bot_texts,
        allow_key_coverage=True,
    )
    if not replacement:
        trace_event(
            context,
            "semantic_recover",
            {"replaced": False, "covers": covers, "same_product": same_product, "composer_empty": True},
        )
        return None
    trace_event(context, "semantic_recover", {"replaced": True, "covers": covers, "same_product": same_product})
    return DialogueContractPipelineResult(
        draft_text=_avoid_repeating_text(
            replacement,
            conversation=conversation,
            contract=contract,
            facts=retrieval.facts,
        ),
        route="bot_answer_self",
        manager_only=False,
        contract=contract,
        facts=retrieval.facts,
        missing=retrieval.missing,
        fallback_reason="semantic_recover",
        semantic_match_attempted=True,
        semantic_match_replaced=True,
        semantic_match_reason=str(verdict.get("reason") or "").strip(),
        repaired=True,
        recovery_candidate=replacement,
    )


def _cite_only_recover_result_before_handoff(
    *,
    contract: AnswerContract,
    retrieval: RetrievalResult,
    draft: str,
    client_words: str,
    conversation: Sequence[Mapping[str, str]],
    faithfulness_fn: Callable[[str], object] | None,
    toggles: Toggles,
    context: Mapping[str, Any] | None,
    previous_bot_texts: Sequence[str] = (),
    tone_guide: str = "",
    fallback_reason: str = "cite_only_recover",
    allow_key_coverage: bool = False,
    original_findings: Sequence[VerificationFinding] = (),
    original_unsupported: Sequence[str] = (),
    draft_fn: Callable[[str], str] | None = None,
) -> DialogueContractPipelineResult | None:
    partial = _partial_yield_result_before_handoff(
        contract=contract,
        retrieval=retrieval,
        client_words=client_words,
        conversation=conversation,
        faithfulness_fn=faithfulness_fn,
        toggles=toggles,
        context=context,
        previous_bot_texts=previous_bot_texts,
        source_reason=fallback_reason,
    )
    if partial:
        return partial
    estimate = _quality_estimate_result_before_handoff(
        contract=contract,
        retrieval=retrieval,
        client_words=client_words,
        conversation=conversation,
        faithfulness_fn=faithfulness_fn,
        draft_fn=draft_fn,
        toggles=toggles,
        context=context,
        previous_bot_texts=previous_bot_texts,
        tone_guide=tone_guide,
        source_reason=fallback_reason,
    )
    if estimate:
        return estimate
    recovered = _cite_only_recover_before_handoff(
        contract=contract,
        retrieval=retrieval,
        draft=draft,
        client_words=client_words,
        faithfulness_fn=faithfulness_fn,
        toggles=toggles,
        context=context,
        previous_bot_texts=previous_bot_texts,
        allow_key_coverage=allow_key_coverage,
        original_findings=original_findings,
        original_unsupported=original_unsupported,
    )
    if not recovered:
        return None
    trace_event(context, "cite_only_recover", {"replaced": True, "fallback_reason": fallback_reason})
    result = DialogueContractPipelineResult(
        draft_text=_avoid_repeating_text(
            recovered,
            conversation=conversation,
            contract=contract,
            facts=retrieval.facts,
        ),
        route="bot_answer_self",
        manager_only=False,
        contract=contract,
        facts=retrieval.facts,
        missing=retrieval.missing,
        fallback_reason=fallback_reason,
        repaired=True,
        recovery_candidate=recovered,
    )
    return _quality_next_step_result(
        result,
        conversation=conversation,
        client_words=client_words,
        faithfulness_fn=faithfulness_fn,
        toggles=toggles,
        context=context,
        previous_bot_texts=previous_bot_texts,
    )


def _quality_estimate_result_before_handoff(
    *,
    contract: AnswerContract,
    retrieval: RetrievalResult,
    client_words: str,
    conversation: Sequence[Mapping[str, str]],
    faithfulness_fn: Callable[[str], object] | None,
    draft_fn: Callable[[str], str] | None,
    toggles: Toggles,
    context: Mapping[str, Any] | None,
    previous_bot_texts: Sequence[str],
    tone_guide: str,
    source_reason: str,
) -> DialogueContractPipelineResult | None:
    if draft_fn is None:
        return None
    if _cite_only_recover_blocked(contract, client_words=client_words, context=context):
        trace_event(context, "estimate_recover", {"applied": False, "reason": "blocked_risk", "source_reason": source_reason})
        return None
    domain = _quality_handoff_estimate_domain(contract=contract, client_words=client_words, context=context)
    if not domain:
        trace_event(context, "estimate_recover", {"applied": False, "reason": "not_estimate_domain", "source_reason": source_reason})
        return None
    estimate_contract = replace(
        contract,
        current_question=client_words or contract.current_question,
        subquestions=(),
        planner_intent="general_consultation",
        planner_subvariant="",
        answer_mode="estimate_allowed",
        estimate_domain=domain,
        answerability="answer_self",
        question_type="",
        existence_target="",
    )
    prompt = build_estimate_prompt(
        conversation=conversation,
        contract=estimate_contract,
        estimate_domain=domain,
        tone_guide=tone_guide,
    )
    try:
        candidate = str(draft_fn(prompt) or "").strip()
    except Exception:
        candidate = ""
    gate_context = _estimate_number_gate_context(context)
    candidate = _ensure_estimate_uncertainty_marker(candidate, context=gate_context)
    if not candidate:
        trace_event(context, "estimate_recover", {"applied": False, "reason": "empty_candidate", "source_reason": source_reason, "domain": domain})
        return None
    findings, unsupported, semantic_available = _hard_check(
        candidate,
        facts=retrieval.facts,
        contract=estimate_contract,
        client_words=client_words,
        faithfulness_fn=faithfulness_fn,
        toggles=toggles,
        context=gate_context,
        previous_bot_texts=previous_bot_texts,
    )
    if findings or unsupported or not semantic_available:
        trace_event(
            context,
            "estimate_recover",
            {
                "applied": False,
                "reason": "hard_check_failed" if semantic_available else "semantic_unavailable",
                "source_reason": source_reason,
                "domain": domain,
                "findings": [finding.code for finding in findings],
                "unsupported": list(unsupported),
            },
        )
        return None
    final = _avoid_repeating_text(candidate, conversation=conversation, contract=estimate_contract, facts=retrieval.facts)
    trace_event(context, "estimate_recover", {"applied": True, "source_reason": source_reason, "domain": domain})
    result = DialogueContractPipelineResult(
        draft_text=final,
        route="bot_answer_self",
        manager_only=False,
        contract=estimate_contract,
        facts=retrieval.facts,
        missing=retrieval.missing,
        fallback_reason=f"estimate_{source_reason}",
        repaired=True,
        recovery_candidate=final,
        is_estimate=True,
        estimate_applied=True,
        estimate_domain=domain,
        estimate_answer_mode="estimate_allowed",
    )
    return _quality_next_step_result(
        result,
        conversation=conversation,
        client_words=client_words,
        faithfulness_fn=faithfulness_fn,
        toggles=toggles,
        context=gate_context,
        previous_bot_texts=previous_bot_texts,
    )


def _estimate_number_gate_context(context: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if free_number_gate_enabled(context) or not travel_compose_enabled(context):
        return context
    merged = dict(context or {})
    merged[FREE_NUMBER_GATE_ENV] = True
    return merged


_NEXT_STEP_ALREADY_RE = re.compile(
    r"\?|если\s+(?:хотите|подходит|удобно)|напишите|подскажите|дальше|следующ|"
    r"менеджер[^.?!\n]{0,90}(?:поможет|подбер|сверит|проверит|подскажет|свяжется|оформ)",
    re.I,
)
_NEXT_STEP_PII_RE = re.compile(r"\b(?:телефон|номер|почт|email|e-mail|фио|фамили|паспорт|снилс)\b", re.I)
_NEXT_STEP_CONCRETE_RE = re.compile(
    r"(?:₽|руб|%|\b\d|\b(?:январ|феврал|март|апрел|ма[йя]|июн|июл|август|сентябр|октябр|ноябр|декабр)\w*)",
    re.I,
)
_NEXT_STEP_PRESSURE_RE = re.compile(
    r"сроч|мест\s+(?:почти\s+)?нет|надо\s+успеть|иначе|не\s+тяните|лучше\s+не\s+тянуть|гарантир|точно\s+получ",
    re.I,
)


def _quality_next_step_result(
    result: DialogueContractPipelineResult,
    *,
    conversation: Sequence[Mapping[str, str]],
    client_words: str,
    faithfulness_fn: Callable[[str], object] | None,
    toggles: Toggles,
    context: Mapping[str, Any] | None,
    previous_bot_texts: Sequence[str],
    draft_fn: Callable[[str], str] | None = None,
    tone_guide: str = "",
) -> DialogueContractPipelineResult:
    del draft_fn, tone_guide
    if not quality_next_step_enabled(context):
        return result
    if result.route != "bot_answer_self" or result.manager_only or result.contract.is_p0:
        trace_event(context, "quality_next_step", {"applied": False, "reason": "non_autonomous_or_p0"})
        return result
    if _cite_only_recover_blocked(result.contract, client_words=client_words, context=context):
        trace_event(context, "quality_next_step", {"applied": False, "reason": "blocked_risk"})
        return result
    if _has_next_step(result.draft_text):
        trace_event(context, "quality_next_step", {"applied": False, "reason": "already_has_next_step"})
        return result
    if toggles.semantic_faithfulness and faithfulness_fn is None:
        trace_event(context, "quality_next_step", {"applied": False, "reason": "faithfulness_fn_missing"})
        return result
    step = _quality_next_step_text(result.contract, client_words=client_words)
    if not step:
        trace_event(context, "quality_next_step", {"applied": False, "reason": "empty_step"})
        return result
    candidate = f"{result.draft_text.rstrip(' .')} {step}".strip()
    facts = _facts_with_derived_answer(result.facts, candidate)
    findings, unsupported, semantic_available = _hard_check(
        candidate,
        facts=facts,
        contract=result.contract,
        client_words=client_words,
        faithfulness_fn=faithfulness_fn,
        toggles=toggles,
        context=context,
        previous_bot_texts=previous_bot_texts,
    )
    if findings or unsupported or not semantic_available:
        trace_event(
            context,
            "quality_next_step",
            {
                "applied": False,
                "reason": "hard_check_failed" if semantic_available else "semantic_unavailable",
                "findings": [finding.code for finding in findings],
                "unsupported": list(unsupported),
            },
        )
        return result
    final = _avoid_repeating_text(candidate, conversation=conversation, contract=result.contract, facts=result.facts)
    trace_event(context, "quality_next_step", {"applied": True, "step": step})
    form_findings = tuple(finding for finding in result.form_findings if finding.code != "no_next_step")
    return replace(
        result,
        draft_text=final,
        form_findings=form_findings,
        next_step_applied=True,
        next_step_text=step,
    )


def _has_next_step(text: str) -> bool:
    value = str(text or "")
    return bool(_NEXT_STEP_ALREADY_RE.search(value))


def _quality_next_step_text(contract: AnswerContract, *, client_words: str) -> str:
    explicit = _explicit_contract_next_step(contract)
    if explicit:
        return explicit
    text = " ".join(part for part in (client_words, contract.current_question, contract.planner_intent) if part)
    low = text.casefold().replace("ё", "е")
    if any(marker in low for marker in ("цен", "стоим", "сколько", "оплат", "рассроч", "долями", "скид")):
        if not _known_slot_value(contract, "grade") or not _known_slot_value(contract, "format"):
            return "Напишите класс ребёнка и удобный формат — подберём подходящий вариант."
        return "Если подходит, менеджер поможет подобрать удобный вариант оплаты и группу."
    if any(marker in low for marker in ("распис", "день", "время", "старт", "когда", "группа")):
        if not _known_slot_value(contract, "grade") or not _known_slot_value(contract, "subject"):
            return "Напишите класс и предмет — менеджер сверит ближайшую подходящую группу."
        return "Если хотите, менеджер сверит ближайшую подходящую группу."
    if any(marker in low for marker in ("запис", "пробн", "оформ", "поступить")):
        return "Если хотите продолжить, менеджер подскажет ближайший шаг по записи."
    if not _known_slot_value(contract, "grade") or not _known_slot_value(contract, "subject"):
        return "Напишите класс ребёнка и предмет — подберём подходящий вариант."
    return "Если хотите, менеджер подскажет ближайший подходящий шаг."


def _explicit_contract_next_step(contract: AnswerContract) -> str:
    for item in contract.subquestions:
        step = _clean_next_step_text(item.next_step)
        if step:
            return step
    return ""


def _clean_next_step_text(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    if _NEXT_STEP_PII_RE.search(text) or _NEXT_STEP_CONCRETE_RE.search(text) or _NEXT_STEP_PRESSURE_RE.search(text):
        return ""
    if len(text) > 160:
        return ""
    return text.rstrip(".") + "."


def _known_slot_value(contract: AnswerContract, key: str) -> str:
    slot = contract.known_slots.get(key)
    if slot and slot.value:
        return slot.value
    return str(contract.planner_slots.get(key) or "").strip()


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
    semantic_match_fn: Callable[[str], object] | None = None,
    warmth_fn: Callable[[str], str] | None = None,
    toggles: Toggles | None = None,
) -> DialogueContractPipelineResult:
    toggles = toggles or Toggles()
    client_words = str(conversation[-1].get("text") or "") if conversation else ""
    previous_bot_texts = [str(item.get("text") or "") for item in conversation if str(item.get("role") or "") == "bot"]
    had_hard_p0_claim = _dialogue_had_hard_p0_claim(context)
    with trace_span(context, "understand", {"client_message": client_words, "active_brand": active_brand}) as trace:
        contract = understand(
            conversation=conversation,
            active_brand=active_brand,
            fact_key_catalog=fact_store.catalog,
            understand_fn=understand_fn,
            context=context,
        )
        contract = _augment_contract_with_memory_topic(
            contract,
            context=context,
            fact_key_catalog=fact_store.catalog,
        )
        trace.update(
            {
                "answerability": contract.answerability,
                "is_p0": contract.is_p0,
                "p0_reason": contract.p0_reason,
                "needed_fact_keys": list(contract.all_needed_fact_keys()),
                "subquestions": [item.to_json_dict() for item in contract.subquestions],
                "planner_intent": contract.planner_intent,
                "planner_subvariant": contract.planner_subvariant,
                "planner_confidence": contract.planner_confidence,
                "selling": dict(contract.selling),
            }
        )
    if contract.is_p0:
        if _asks_refund_policy(contract) and not _current_refund_dispute_signal(
            client_words=client_words,
            contract=contract,
        ) and not had_hard_p0_claim:
            contract = replace(contract, is_p0=False, p0_reason="", p0_source="", answerability="answer_self")
        else:
            text = _p0_handoff_text(contract, conversation=conversation)
            text = _avoid_repeating_text(text, conversation=conversation, contract=contract, facts={})
            trace_event(context, "build_draft", {"route": "manager_only", "fallback_reason": "p0", "draft": text})
            return DialogueContractPipelineResult(
                draft_text=text,
                route="manager_only",
                manager_only=True,
                contract=contract,
                fallback_reason="p0",
            )

    with trace_span(context, "retrieve_facts", {"needed_fact_keys": list(contract.all_needed_fact_keys())}) as trace:
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
            context=context,
        )
        retrieval = _scope_camp_retrieval_for_contract(retrieval, contract=contract, context=context)
        trace.update(
            {
                "fact_keys": list(retrieval.facts.keys()),
                "missing": list(retrieval.missing),
                "matched_keys": {key: list(value) for key, value in retrieval.matched_keys.items()},
            }
        )
    estimate_enabled = estimate_mode_enabled(context)
    free_number_enabled = free_number_gate_enabled(context) or travel_compose_enabled(context)
    estimate_policy = dict(_estimate_policy_context(
        contract=contract,
        retrieval=retrieval,
        enabled=estimate_enabled or free_number_enabled,
        free_number_gate=free_number_enabled,
        question_text=client_words or contract.current_question,
    ))
    partial_travel_domain = _quality_partial_yield_travel_domain(
        contract=contract,
        client_words=client_words,
        retrieval=retrieval,
        context=context,
        free_number_gate=free_number_enabled,
    )
    if partial_travel_domain:
        estimate_policy = {
            **estimate_policy,
            "enabled": True,
            "answer_mode": "estimate_allowed",
            "estimate_domain": partial_travel_domain,
            "partial_yield_travel": True,
        }
    contract = replace(
        contract,
        answer_mode=str(estimate_policy["answer_mode"]),
        estimate_domain=str(estimate_policy["estimate_domain"]),
    )
    if estimate_policy["enabled"] and estimate_policy["individual_child_question"] and not retrieval.facts:
        contract = replace(contract, answerability="manager_only")
    trace_event(context, "estimate_answer_mode", estimate_policy)
    if _asks_refund_policy(contract) and had_hard_p0_claim:
        guarded_contract = replace(contract, is_p0=True, p0_reason=contract.p0_reason or "prior_hard_p0_refund_claim")
        fallback = _safe_fallback_text(guarded_contract, facts=retrieval.facts, context=context)
        return DialogueContractPipelineResult(
            draft_text=_avoid_repeating_text(fallback, conversation=conversation, contract=guarded_contract, facts=retrieval.facts),
            route="manager_only",
            manager_only=True,
            contract=guarded_contract,
            facts=retrieval.facts,
            missing=retrieval.missing,
            fallback_reason="prior_hard_p0_refund_claim",
        )
    if _asks_refund_policy(contract) and not _presale_refund_policy_text(retrieval.facts):
        fallback = _safe_fallback_text(contract, facts=retrieval.facts, context=context)
        return DialogueContractPipelineResult(
            draft_text=_avoid_repeating_text(fallback, conversation=conversation, contract=contract, facts=retrieval.facts),
            route="draft_for_manager",
            manager_only=False,
            contract=contract,
            facts=retrieval.facts,
            missing=retrieval.missing,
            fallback_reason="refund_policy_manager_only",
        )
    early_estimate_domain = _quality_handoff_estimate_domain(
        contract=contract,
        client_words=client_words,
        context=context,
    )
    if early_estimate_domain and not _cite_only_recover_blocked(contract, client_words=client_words, context=context):
        recovered_estimate = _quality_estimate_result_before_handoff(
            contract=contract,
            retrieval=retrieval,
            client_words=client_words,
            conversation=conversation,
            faithfulness_fn=faithfulness_fn,
            draft_fn=draft_fn,
            toggles=toggles,
            context=context,
            previous_bot_texts=previous_bot_texts,
            tone_guide=tone_guide,
            source_reason="pre_handoff_estimate",
        )
        if recovered_estimate:
            return recovered_estimate
        fallback = _safe_fallback_text(contract, facts={}, context=context)
        return DialogueContractPipelineResult(
            draft_text=_avoid_repeating_text(fallback, conversation=conversation, contract=contract, facts=()),
            route="draft_for_manager",
            manager_only=False,
            contract=contract,
            facts=retrieval.facts,
            missing=retrieval.missing,
            fallback_reason="estimate_guard_failed",
        )
    if (
        estimate_policy["enabled"]
        and contract.answer_mode == "estimate_allowed"
        and contract.estimate_domain in _ESTIMATE_DOMAINS
        and draft_fn is not None
    ):
        estimate_prompt = build_estimate_prompt(
            conversation=conversation,
            contract=contract,
            estimate_domain=contract.estimate_domain,
            tone_guide=tone_guide,
        )
        try:
            estimate_draft = str(draft_fn(estimate_prompt) or "").strip()
        except Exception:
            estimate_draft = ""
        estimate_draft = _ensure_estimate_uncertainty_marker(estimate_draft, context=context)
        trace_event(
            context,
            "estimate_compose",
            {"attempted": True, "domain": contract.estimate_domain, "draft": estimate_draft},
        )
        if estimate_draft:
            findings, unsupported, semantic_available = _hard_check(
                estimate_draft,
                facts=retrieval.facts,
                contract=contract,
                client_words=client_words,
                faithfulness_fn=faithfulness_fn,
                toggles=toggles,
                context=context,
                previous_bot_texts=previous_bot_texts,
            )
            if semantic_available and not findings and not unsupported:
                final_estimate = _avoid_repeating_text(
                    estimate_draft,
                    conversation=conversation,
                    contract=contract,
                    facts=retrieval.facts,
                )
                trace_event(context, "estimate_gate", {"passed": True, "domain": contract.estimate_domain, "draft": final_estimate})
                result = DialogueContractPipelineResult(
                    draft_text=final_estimate,
                    route="bot_answer_self",
                    manager_only=False,
                    contract=contract,
                    facts=retrieval.facts,
                    missing=retrieval.missing,
                    fallback_reason="",
                    is_estimate=True,
                    estimate_applied=True,
                    estimate_domain=contract.estimate_domain,
                    estimate_answer_mode=contract.answer_mode,
                )
                return _quality_next_step_result(
                    result,
                    conversation=conversation,
                    client_words=client_words,
                    faithfulness_fn=faithfulness_fn,
                    toggles=toggles,
                    context=context,
                    previous_bot_texts=previous_bot_texts,
                )
            trace_event(
                context,
                "estimate_gate",
                {
                    "passed": False,
                    "domain": contract.estimate_domain,
                    "findings": [{"code": item.code, "detail": item.detail} for item in findings],
                    "unsupported": list(unsupported),
                    "semantic_available": semantic_available,
                },
            )
        fallback = _safe_fallback_text(contract, facts=retrieval.facts, context=context)
        partial = _partial_yield_result_before_handoff(
            contract=contract,
            retrieval=retrieval,
            client_words=client_words,
            conversation=conversation,
            faithfulness_fn=faithfulness_fn,
            toggles=toggles,
            context=context,
            previous_bot_texts=previous_bot_texts,
            source_reason="estimate_guard_failed",
        )
        if partial:
            return partial
        return DialogueContractPipelineResult(
            draft_text=_avoid_repeating_text(fallback, conversation=conversation, contract=contract, facts=retrieval.facts),
            route="draft_for_manager",
            manager_only=False,
            contract=contract,
            facts=retrieval.facts,
            missing=retrieval.missing,
            fallback_reason="estimate_guard_failed",
            is_estimate=False,
            estimate_domain=contract.estimate_domain,
            estimate_answer_mode=contract.answer_mode,
        )
    composite = _quality_composite_result_before_draft(
        contract=contract,
        retrieval=retrieval,
        client_words=client_words,
        conversation=conversation,
        faithfulness_fn=faithfulness_fn,
        toggles=toggles,
        context=context,
        previous_bot_texts=previous_bot_texts,
    )
    if composite:
        return _quality_next_step_result(
            composite,
            conversation=conversation,
            client_words=client_words,
            faithfulness_fn=faithfulness_fn,
            toggles=toggles,
            context=context,
            previous_bot_texts=previous_bot_texts,
            draft_fn=draft_fn,
            tone_guide=tone_guide,
        )
    clarify_question = _scope_clarification_question(contract, retrieval, client_words=client_words, context=context)
    if clarify_question:
        return DialogueContractPipelineResult(
            draft_text=clarify_question,
            route="bot_answer_self",
            manager_only=False,
            contract=contract,
            facts=retrieval.facts,
            missing=retrieval.missing,
            fallback_reason="scope_clarification_question",
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
        fallback = _safe_fallback_text(contract, facts=retrieval.facts, context=context)
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
        result = DialogueContractPipelineResult(
            draft_text=schedule_answer,
            route="bot_answer_self",
            manager_only=False,
            contract=contract,
            facts=retrieval.facts,
            missing=retrieval.missing,
            fallback_reason="schedule_publication_answer",
        )
        return _quality_next_step_result(
            result,
            conversation=conversation,
            client_words=client_words,
            faithfulness_fn=faithfulness_fn,
            toggles=toggles,
            context=context,
            previous_bot_texts=previous_bot_texts,
            draft_fn=None,
            tone_guide=tone_guide,
        )
    direct_answer = _direct_exact_fact_answer(contract, retrieval)
    if direct_answer:
        direct_draft = _avoid_repeating_text(direct_answer, conversation=conversation, contract=contract, facts=retrieval.facts)
        result = DialogueContractPipelineResult(
            draft_text=direct_draft,
            route="bot_answer_self",
            manager_only=False,
            contract=contract,
            facts=retrieval.facts,
            missing=retrieval.missing,
            fallback_reason="direct_exact_fact_answer",
            recovery_candidate=_stashed_recovery_candidate(
                direct_draft,
                contract=contract,
                retrieval=retrieval,
                client_words=client_words,
                context=context,
            ),
        )
        return _quality_next_step_result(
            result,
            conversation=conversation,
            client_words=client_words,
            faithfulness_fn=faithfulness_fn,
            toggles=toggles,
            context=context,
            previous_bot_texts=previous_bot_texts,
            draft_fn=draft_fn,
            tone_guide=tone_guide,
        )
    force_draft_for_manager = (
        contract.answerability != "answer_self"
        and not exact_answer_available
        and not _has_retrieved_self_answer_part(contract, retrieval)
        and not (_asks_refund_policy(contract) and _presale_refund_policy_text(retrieval.facts) and not had_hard_p0_claim)
    )
    needs_facts = bool(contract.all_needed_fact_keys())
    empty_factual_answer_self = (
        contract.answerability == "answer_self"
        and needs_facts
        and not retrieval.facts
        and not exact_answer_available
        and not _has_retrieved_self_answer_part(contract, retrieval)
    )
    if empty_factual_answer_self:
        fallback = _safe_fallback_text(contract, facts=retrieval.facts, context=context)
        final_text = _avoid_repeating_text(fallback, conversation=conversation, contract=contract, facts=retrieval.facts)
        route = "draft_for_manager" if _manager_deferral_text(final_text) else "bot_answer_self"
        trace_event(
            context,
            "build_draft",
            {
                "route": route,
                "fallback_reason": "empty_facts_no_fabrication",
                "draft": final_text,
            },
        )
        return DialogueContractPipelineResult(
            draft_text=final_text,
            route=route,
            manager_only=False,
            contract=contract,
            facts=retrieval.facts,
            missing=retrieval.missing,
            fallback_reason="empty_facts_no_fabrication",
            reason_class="no_fact_or_unverified" if route != "bot_answer_self" else "",
            reason_evidence={"source": "empty_facts_text_discriminator"} if route != "bot_answer_self" else {},
        )
    force_manager_reason_class = _force_draft_for_manager_reason_class(contract, retrieval) if force_draft_for_manager else ""
    if force_draft_for_manager and (
        _asks_refund_policy(contract)
        or not retrieval.facts
        or (_soft_weekend_guidance_text(retrieval.facts) and not _has_self_answerable_subquestion(contract))
    ):
        fallback = _safe_fallback_text(contract, facts=retrieval.facts, context=context)
        recovered = _cite_only_recover_result_before_handoff(
            contract=contract,
            retrieval=retrieval,
            draft=fallback,
            client_words=client_words,
            conversation=conversation,
            faithfulness_fn=faithfulness_fn,
            toggles=toggles,
            context=context,
            previous_bot_texts=previous_bot_texts,
        )
        if recovered:
            return recovered
        return DialogueContractPipelineResult(
            draft_text=_avoid_repeating_text(fallback, conversation=conversation, contract=contract, facts=retrieval.facts),
            route="draft_for_manager",
            manager_only=False,
            contract=contract,
            facts=retrieval.facts,
            missing=retrieval.missing,
            fallback_reason=force_manager_reason_class,
            reason_class=force_manager_reason_class,
            reason_evidence={"source": "force_draft_for_manager"},
        )
    if draft_fn is None:
        fallback = _safe_fallback_text(contract, facts=retrieval.facts, context=context)
        recovered = _cite_only_recover_result_before_handoff(
            contract=contract,
            retrieval=retrieval,
            draft=fallback,
            client_words=client_words,
            conversation=conversation,
            faithfulness_fn=faithfulness_fn,
            toggles=toggles,
            context=context,
            previous_bot_texts=previous_bot_texts,
        )
        if recovered:
            return recovered
        return DialogueContractPipelineResult(
            draft_text=_avoid_repeating_text(fallback, conversation=conversation, contract=contract, facts=retrieval.facts),
            route="draft_for_manager",
            manager_only=False,
            contract=contract,
            facts=retrieval.facts,
            missing=retrieval.missing,
            fallback_reason="no_draft_fn",
        )
    with trace_span(
        context,
        "build_draft",
        {"answerability": contract.answerability, "fact_keys": list(retrieval.facts.keys()), "missing": list(retrieval.missing)},
    ) as trace:
        prompt_memory_view = _thread_memory_view_for_contract(contract, context=context)
        prompt = build_draft_prompt(
            conversation=conversation,
            contract=contract,
            facts=retrieval.facts,
            missing=retrieval.missing,
            tone_guide=tone_guide,
            style_examples=style_examples,
            toggles=toggles,
            dialogue_memory_view=prompt_memory_view,
        )
        trace["prompt_chars"] = len(prompt)
        try:
            draft = str(draft_fn(prompt) or "").strip()
        except Exception:
            draft = ""
        trace["draft"] = draft
    if not draft:
        fallback = _safe_fallback_text(contract, facts=retrieval.facts, context=context)
        recovered = _cite_only_recover_result_before_handoff(
            contract=contract,
            retrieval=retrieval,
            draft=fallback,
            client_words=client_words,
            conversation=conversation,
            faithfulness_fn=faithfulness_fn,
            toggles=toggles,
            context=context,
            previous_bot_texts=previous_bot_texts,
        )
        if recovered:
            return recovered
        return DialogueContractPipelineResult(
            draft_text=_avoid_repeating_text(fallback, conversation=conversation, contract=contract, facts=retrieval.facts),
            route="draft_for_manager",
            manager_only=False,
            contract=contract,
            facts=retrieval.facts,
            missing=retrieval.missing,
            fallback_reason="draft_error",
        )

    repaired = False
    semantic_match_attempted = False
    semantic_match_replaced = False
    semantic_match_reason = ""
    semantic_match_blocked_replacement = False
    if (
        _is_pure_handoff_text(draft)
        and contract.answerability == "answer_self"
        and not contract.is_p0
        and _key_coverage_ok(contract, retrieval)
    ):
        replacement = _verified_empty_handoff_replacement(
            draft,
            contract=contract,
            retrieval=retrieval,
            client_words=client_words,
            faithfulness_fn=faithfulness_fn,
            toggles=toggles,
            context=context,
            previous_bot_texts=previous_bot_texts,
            allow_key_coverage=True,
        )
        if replacement:
            trace_event(context, "key_coverage_gate", {"replaced": True})
            draft = replacement
            repaired = True

    if (
        semantic_match_fn is not None
        and _looks_like_handoff(draft)
        and contract.answerability == "answer_self"
        and not contract.is_p0
        and retrieval.facts
    ):
        semantic_match_attempted = True
        semantic_verdict = _semantic_match(
            semantic_match_fn,
            contract=contract,
            retrieval=retrieval,
            client_words=client_words,
            draft=draft,
        )
        covers = _truthy(semantic_verdict.get("covers"))
        same_product = _truthy(semantic_verdict.get("same_product"))
        semantic_match_reason = str(semantic_verdict.get("reason") or "").strip()
        if covers and same_product:
            replacement = _verified_empty_handoff_replacement(
                draft,
                contract=contract,
                retrieval=retrieval,
                client_words=client_words,
                faithfulness_fn=faithfulness_fn,
                toggles=toggles,
                context=context,
                previous_bot_texts=previous_bot_texts,
                allow_key_coverage=True,
            )
            if replacement:
                trace_event(
                    context,
                    "semantic_match_gate",
                    {"replaced": True, "covers": covers, "same_product": same_product},
                )
                draft = replacement
                repaired = True
                semantic_match_replaced = True
        else:
            semantic_match_blocked_replacement = True

    draft = _specialize_grade_range_answer(draft, contract=contract, facts=retrieval.facts)
    findings, unsupported, semantic_available = _hard_check(
        draft,
        facts=retrieval.facts,
        contract=contract,
        client_words=client_words,
        faithfulness_fn=faithfulness_fn,
        toggles=toggles,
        context=context,
        previous_bot_texts=previous_bot_texts,
    )
    if not semantic_available:
        fallback = _safe_fallback_text(contract, facts=retrieval.facts, context=context)
        recovered = _cite_only_recover_result_before_handoff(
            contract=contract,
            retrieval=retrieval,
            draft=fallback,
            faithfulness_fn=None,
            client_words=client_words,
            conversation=conversation,
            context=context,
            toggles=toggles,
            previous_bot_texts=previous_bot_texts,
            original_findings=findings,
            original_unsupported=unsupported,
            allow_key_coverage=True,
            draft_fn=draft_fn,
            tone_guide=tone_guide,
        )
        if recovered:
            return recovered
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
            previous_bot_texts=previous_bot_texts,
        )
        if not semantic_available:
            fallback = _safe_fallback_text(contract, facts=retrieval.facts, context=context)
            recovered = _cite_only_recover_result_before_handoff(
                contract=contract,
                retrieval=retrieval,
                draft=fallback,
                faithfulness_fn=None,
                client_words=client_words,
                conversation=conversation,
                context=context,
                toggles=toggles,
                previous_bot_texts=previous_bot_texts,
                original_findings=findings,
                original_unsupported=unsupported,
                allow_key_coverage=True,
                draft_fn=draft_fn,
                tone_guide=tone_guide,
            )
            if recovered:
                return recovered
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
                previous_bot_texts=previous_bot_texts,
            )
            if fallback_semantic_available and not fallback_findings and not fallback_unsupported:
                verified_draft = _avoid_repeating_text(
                    verified_fallback,
                    conversation=conversation,
                    contract=contract,
                    facts=retrieval.facts,
                )
                result = DialogueContractPipelineResult(
                    draft_text=verified_draft,
                    route="bot_answer_self",
                    manager_only=False,
                    contract=contract,
                    facts=retrieval.facts,
                    missing=retrieval.missing,
                    findings=tuple(findings),
                    unsupported_claims=tuple(unsupported),
                    repaired=repaired,
                    fallback_reason="verified_fact_fallback_after_hard_check",
                    recovery_candidate=_stashed_recovery_candidate(
                        verified_draft,
                        contract=contract,
                        retrieval=retrieval,
                        client_words=client_words,
                        context=context,
                    ),
                )
                return _quality_next_step_result(
                    result,
                    conversation=conversation,
                    client_words=client_words,
                    faithfulness_fn=faithfulness_fn,
                    toggles=toggles,
                    context=context,
                    previous_bot_texts=previous_bot_texts,
                )
        fallback = _safe_fallback_text(contract, facts=retrieval.facts, context=context)
        recovered = _cite_only_recover_result_before_handoff(
            contract=contract,
            retrieval=retrieval,
            draft=fallback,
            faithfulness_fn=faithfulness_fn,
            client_words=client_words,
            conversation=conversation,
            context=context,
            toggles=toggles,
            previous_bot_texts=previous_bot_texts,
            original_findings=findings,
            original_unsupported=unsupported,
            allow_key_coverage=True,
            draft_fn=draft_fn,
            tone_guide=tone_guide,
        )
        if recovered:
            return recovered
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

    composition = "" if semantic_match_blocked_replacement else _composition_answer(contract, retrieval, current_draft=draft)
    if composition and composition != draft:
        composition_facts = _facts_with_derived_answer(retrieval.facts, composition)
        comp_findings, comp_unsupported, comp_semantic_available = _hard_check(
            composition,
            facts=composition_facts,
            contract=contract,
            client_words=client_words,
            faithfulness_fn=faithfulness_fn,
            toggles=toggles,
            context=context,
            previous_bot_texts=previous_bot_texts,
        )
        if comp_semantic_available and not comp_findings and not comp_unsupported:
            draft = composition
            repaired = True

    coverage_findings = (
        ()
        if semantic_match_blocked_replacement
        else _coverage_findings(
            draft,
            contract=contract,
            retrieval=retrieval,
            force_draft_for_manager=force_draft_for_manager,
            context=context,
        )
    )
    coverage_attempts = 0
    while coverage_findings and repair_fn is not None and coverage_attempts < MAX_REPAIR_ATTEMPTS:
        coverage_attempts += 1
        try:
            candidate = str(
                repair_fn(_coverage_repair_prompt(draft, coverage_findings, retrieval.facts))
                or ""
            ).strip()
        except Exception:
            break
        if not candidate:
            break
        candidate = _specialize_grade_range_answer(candidate, contract=contract, facts=retrieval.facts)
        candidate_findings, candidate_unsupported, candidate_semantic_available = _hard_check(
            candidate,
            facts=retrieval.facts,
            contract=contract,
            client_words=client_words,
            faithfulness_fn=faithfulness_fn,
            toggles=toggles,
            context=context,
            previous_bot_texts=previous_bot_texts,
        )
        if not candidate_semantic_available:
            fallback = _safe_fallback_text(contract, facts=retrieval.facts, context=context)
            recovered = _cite_only_recover_result_before_handoff(
                contract=contract,
                retrieval=retrieval,
                draft=fallback,
                faithfulness_fn=None,
                client_words=client_words,
                conversation=conversation,
                context=context,
                toggles=toggles,
                previous_bot_texts=previous_bot_texts,
                original_findings=candidate_findings,
                original_unsupported=candidate_unsupported,
                allow_key_coverage=True,
                draft_fn=draft_fn,
                tone_guide=tone_guide,
            )
            if recovered:
                return recovered
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
        if candidate_findings or candidate_unsupported:
            break
        candidate_coverage = _coverage_findings(
            candidate,
            contract=contract,
            retrieval=retrieval,
            force_draft_for_manager=force_draft_for_manager,
            context=context,
        )
        draft = candidate
        repaired = True
        coverage_findings = candidate_coverage

    if coverage_findings:
        cite_only = _composition_answer(contract, retrieval, current_draft=draft) or _coverage_cite_only_answer(contract, retrieval)
        if cite_only:
            cite_facts = _facts_with_derived_answer(retrieval.facts, cite_only)
            cite_findings, cite_unsupported, cite_semantic_available = _hard_check(
                cite_only,
                facts=cite_facts,
                contract=contract,
                client_words=client_words,
                faithfulness_fn=faithfulness_fn,
                toggles=toggles,
                context=context,
                previous_bot_texts=previous_bot_texts,
            )
            cite_coverage = _coverage_findings(
                cite_only,
                contract=contract,
                retrieval=retrieval,
                force_draft_for_manager=force_draft_for_manager,
                context=context,
            )
            if cite_semantic_available and not cite_findings and not cite_unsupported and not cite_coverage:
                draft = cite_only
                repaired = True

    replacement = "" if semantic_match_blocked_replacement else _verified_empty_handoff_replacement(
        draft,
        contract=contract,
        retrieval=retrieval,
        client_words=client_words,
        faithfulness_fn=faithfulness_fn,
        toggles=toggles,
        context=context,
        previous_bot_texts=previous_bot_texts,
    )
    if replacement:
        draft = replacement
        repaired = True

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
                    previous_bot_texts=previous_bot_texts,
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

    final_draft = _avoid_repeating_text(draft, conversation=conversation, contract=contract, facts=retrieval.facts)
    result = DialogueContractPipelineResult(
        draft_text=final_draft,
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
        semantic_match_attempted=semantic_match_attempted,
        semantic_match_replaced=semantic_match_replaced,
        semantic_match_reason=semantic_match_reason,
        repaired=repaired,
        recovery_candidate=_stashed_recovery_candidate(
            final_draft,
            contract=contract,
            retrieval=retrieval,
            client_words=client_words,
            context=context,
        ),
        fallback_reason=force_manager_reason_class if force_draft_for_manager else "",
        reason_class=force_manager_reason_class if force_draft_for_manager else "",
        reason_evidence={"source": "force_draft_for_manager_final"} if force_draft_for_manager else {},
    )
    return _quality_next_step_result(
        result,
        conversation=conversation,
        client_words=client_words,
        faithfulness_fn=faithfulness_fn,
        toggles=toggles,
        context=context,
        previous_bot_texts=previous_bot_texts,
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
    previous_bot_texts: Sequence[str] = (),
    answer_mode: str | None = None,
    estimate_domain: str | None = None,
    is_estimate: bool | None = None,
) -> list[VerificationFinding]:
    text = str(draft_text or "")
    low = text.casefold()
    findings: list[VerificationFinding] = []
    brand = _normalize_brand(active_brand)
    for token in _BRAND_TOKENS.get(brand, ()):
        if _brand_token_present(low, token):
            findings.append(VerificationFinding("brand_leak", f"чужой бренд/токен: {token}"))
            break
    gate_answer_mode = _gate_answer_mode(contract=contract, context=context, explicit=answer_mode)
    gate_estimate_domain = _gate_estimate_domain(contract=contract, context=context, explicit=estimate_domain)
    gate_is_estimate = _gate_is_estimate(contract=contract, context=context, explicit=is_estimate)
    gate_free_number = free_number_gate_enabled(context)
    if gate_free_number:
        findings.extend(
            _free_number_gate_findings(
                text,
                facts=facts,
                client_message=client_message,
                context=context,
                estimate_domain=gate_estimate_domain,
            )
        )
    else:
        findings.extend(
            _answer_mode_number_findings(
                text,
                facts=facts,
                client_message=client_message,
                contract=contract,
                answer_mode=gate_answer_mode,
                estimate_domain=gate_estimate_domain,
            )
        )
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
        if _preemptive_format_choice_finding(low, contract=contract):
            findings.append(
                VerificationFinding(
                    "preemptive_format",
                    "клиент спросил выбор формата, а ответ навязывает один формат без альтернативы",
                )
            )
    unconfirmed_schedule = _unconfirmed_schedule_finding(low, facts=facts, client_message=client_message)
    if unconfirmed_schedule is not None:
        findings.append(unconfirmed_schedule)
    self_contradiction = _self_contradiction_finding(text, low, previous_bot_texts=previous_bot_texts)
    if self_contradiction is not None:
        findings.append(self_contradiction)
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
    if (
        not gate_free_number
        and gate_answer_mode == "estimate_allowed"
        and gate_is_estimate
        and not _has_uncertainty_marker(text)
    ):
        findings.append(VerificationFinding("estimate_without_uncertainty_marker", "оценка без явного маркера неуверенности"))
    if gate_answer_mode == "estimate_allowed" and gate_estimate_domain == "general_advice":
        findings.extend(_general_advice_estimate_findings(text, client_message=client_message))
    if not any(finding.code == "estimate_individual_child_advice" for finding in findings):
        findings.extend(_individual_child_diagnosis_findings(text, client_message=client_message))
    safety = classify_answer_safety(client_message=client_message, context=context, route="bot_answer_self")
    if safety.p0_required and not p0_pre_gate(client_message, context=context):
        findings.append(VerificationFinding("p0_semantic_risk", "семантический P0 требует менеджера"))
    return findings


def _answer_mode_number_findings(
    text: str,
    *,
    facts: Mapping[str, str],
    client_message: str,
    contract: AnswerContract | None,
    answer_mode: str,
    estimate_domain: str = "none",
) -> list[VerificationFinding]:
    backed_numbers = _numbers(" ".join(str(value) for value in facts.values()))
    client_numbers = _numbers(client_message)
    introduced: set[str] = set()
    product_introduced: set[str] = set()
    token_map = _number_token_map(text)
    for num in _numbers(text) - backed_numbers:
        tokens = token_map.get(num, ())
        is_product = any(
            _is_product_number_context(text, token)
            and not _is_route_estimate_number_context(text, token, estimate_domain=estimate_domain)
            for token in tokens
        )
        if _is_allowed_ungrounded_number(num, client_numbers=client_numbers) and (
            not is_product or any(_is_client_grade_number_context(text, token) for token in tokens)
        ):
            continue
        if answer_mode == "estimate_allowed" and is_product:
            product_introduced.add(num)
            continue
        if answer_mode != "estimate_allowed":
            introduced.add(num)
    findings: list[VerificationFinding] = []
    if product_introduced:
        findings.append(
            VerificationFinding(
                "unsupported_product_claim",
                f"продуктовые числа вне подтверждённых фактов: {sorted(product_introduced)}",
            )
        )
    if introduced:
        findings.append(VerificationFinding("fact_grounding", f"числа вне подтверждённых фактов: {sorted(introduced)}"))
    return findings


def _free_number_gate_findings(
    text: str,
    *,
    facts: Mapping[str, str],
    client_message: str,
    context: Mapping[str, Any] | None,
    estimate_domain: str = "none",
) -> list[VerificationFinding]:
    fact_surfaces = _free_number_surfaces(" ".join(str(value) for value in facts.values()))
    client_surfaces = _free_number_surfaces(_client_number_context_text(client_message, context=context))
    product_tokens: list[str] = []
    general_without_marker: list[str] = []
    for token, start, end in _free_number_token_matches(text):
        surfaces = _free_number_surfaces(token)
        if not surfaces:
            continue
        if fact_surfaces.intersection(surfaces):
            continue
        if _is_route_estimate_number_context(text, token, estimate_domain=estimate_domain):
            pass
        elif _is_free_product_number_context(text, token, start=start, end=end):
            product_tokens.append(token)
            continue
        if _is_free_structural_number(token, surfaces, text=text, start=start, end=end) or client_surfaces.intersection(surfaces):
            continue
        if not _has_free_uncertainty_marker_near(text, token, start=start, end=end):
            general_without_marker.append(token)
    findings: list[VerificationFinding] = []
    if product_tokens:
        findings.append(
            VerificationFinding(
                "unsupported_product_number",
                f"продуктовые числа вне подтверждённых фактов: {sorted(dict.fromkeys(product_tokens))}",
            )
        )
    if general_without_marker:
        findings.append(
            VerificationFinding(
                "general_number_without_marker",
                f"общее негрунтованное число без маркера неуверенности рядом: {sorted(dict.fromkeys(general_without_marker))}",
            )
        )
    return findings


def _client_number_context_text(client_message: str, *, context: Mapping[str, Any] | None) -> str:
    parts = [str(client_message or "")]
    if not isinstance(context, MappingABC):
        return " ".join(part for part in parts if part)
    for key in ("conversation", "messages", "dialogue", "turns"):
        value = context.get(key)
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            continue
        for item in value:
            if not isinstance(item, MappingABC):
                continue
            role = str(item.get("role") or item.get("speaker") or "").casefold()
            if role and role not in {"client", "user", "customer"}:
                continue
            text = item.get("text") or item.get("message") or item.get("content")
            if text:
                parts.append(str(text))
    return " ".join(part for part in parts if part)


def _free_number_tokens(text: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(token for token, _start, _end in _free_number_token_matches(text)))


def _free_number_token_matches(text: str) -> tuple[tuple[str, int, int], ...]:
    tokens: list[tuple[str, int, int]] = []
    for match in _FREE_NUMBER_TOKEN_RE.finditer(str(text or "")):
        token = match.group(0).strip()
        if token:
            tokens.append((token, match.start(), match.end()))
    return tuple(tokens)


def _free_number_surfaces(text: str) -> set[str]:
    surfaces: set[str] = set()
    for token in _free_number_tokens(text):
        surfaces.update(_normalize_free_number_token(token))
    surfaces.update(_free_number_word_surfaces(text))
    return {surface for surface in surfaces if surface}


def _normalize_free_number_token(token: str) -> set[str]:
    raw = str(token or "").casefold().replace("ё", "е").replace("\u00a0", " ").strip()
    if not raw:
        return set()
    raw = raw.replace("–", "-").replace("—", "-")
    raw = re.sub(r"\s+", " ", raw)
    surfaces: set[str] = set()
    time_match = re.fullmatch(r"\d{1,2}:\d{2}", raw)
    if time_match:
        hour, minute = raw.split(":", 1)
        return {f"{int(hour)}:{minute}", f"{int(hour):02d}:{minute}"}
    academic_year = re.fullmatch(r"(20\d{2})/(\d{2})", raw)
    if academic_year:
        year, short = academic_year.groups()
        return {f"{year}/{short}"}
    date_match = re.fullmatch(r"(\d{1,2})[./](\d{1,2})(?:[./](\d{2,4}))?", raw)
    if date_match:
        day, month, year = date_match.groups()
        day_i = int(day)
        month_i = int(month)
        surfaces.add(f"{day_i}.{month_i}")
        surfaces.add(f"{day_i:02d}.{month_i:02d}")
        if year:
            surfaces.add(f"{day_i}.{month_i}.{year}")
            surfaces.add(f"{day_i:02d}.{month_i:02d}.{year}")
        return surfaces
    percent = bool(re.search(r"%|процент", raw, re.I))
    thousand = bool(re.search(r"(?<=\d)\s*к\b|\b(?:тыс\.?|тысяч)\b", raw, re.I))
    unitless = re.sub(
        r"(?:₽|руб(?:\.|лей|ля|ль)?|р\.|процент(?:ов|а)?|%|минут(?:ы|у)?|час(?:а|ов)?|"
        r"км|километр(?:а|ов)?|год(?:а)?|лет|недел(?:и|ь)?|заняти(?:й|я)|балл(?:ов|а)?|"
        r"ак\.?\s*ч(?:\.|аса|асов)?|раз(?:а)?|тыс\.?|тысяч|(?<=\d)\s*к\b)",
        "",
        raw,
        flags=re.I,
    ).strip()
    unitless = re.sub(r"\s+", "", unitless)
    range_match = re.fullmatch(r"(\d+(?:[.,]\d+)?)\s*-\s*(\d+(?:[.,]\d+)?)", unitless)
    if range_match:
        left = _normalize_decimal_surface(range_match.group(1))
        right = _normalize_decimal_surface(range_match.group(2))
        surfaces.update({left, right, f"{left}-{right}"})
        if percent:
            surfaces.update({f"{left}%", f"{right}%", f"{left}-{right}%"})
        return surfaces
    number_match = re.search(r"\d+(?:[.,]\d+)?", unitless)
    if not number_match:
        return set()
    value = _normalize_decimal_surface(number_match.group(0))
    if thousand:
        value = _multiply_thousand_surface(value)
    surfaces.add(value)
    if percent:
        surfaces.add(f"{value}%")
    return surfaces


def _normalize_decimal_surface(value: str) -> str:
    normalized = str(value or "").replace(",", ".").strip()
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return normalized


def _multiply_thousand_surface(value: str) -> str:
    try:
        number = float(value)
    except Exception:
        return value
    multiplied = number * 1000
    if multiplied.is_integer():
        return str(int(multiplied))
    return str(multiplied).rstrip("0").rstrip(".")


def _is_free_product_number_context(
    text: str,
    token: str,
    *,
    start: int | None = None,
    end: int | None = None,
) -> bool:
    raw = str(text or "")
    item = str(token or "").strip()
    if start is not None and end is not None:
        if _is_client_grade_number_context_at(raw, start=start, end=end):
            return False
        window = _free_number_context_window(raw, start=start, end=end, radius=35)
        if _is_decimal_year_range(item) and not _FREE_NUMBER_PRODUCT_CTX_RE.search(window.replace(item, " ")):
            return False
        return bool(_FREE_NUMBER_PRODUCT_CTX_RE.search(window))
    if _is_client_grade_number_context(raw, item):
        return False
    if not item:
        return bool(_FREE_NUMBER_PRODUCT_CTX_RE.search(raw))
    index = raw.find(item)
    if index < 0:
        return bool(_FREE_NUMBER_PRODUCT_CTX_RE.search(raw))
    window = raw[max(0, index - 35) : index + len(item) + 35]
    if _is_decimal_year_range(item) and not _FREE_NUMBER_PRODUCT_CTX_RE.search(window.replace(item, " ")):
        return False
    return bool(_FREE_NUMBER_PRODUCT_CTX_RE.search(window))


def _is_free_structural_number(
    token: str,
    surfaces: set[str],
    *,
    text: str,
    start: int | None = None,
    end: int | None = None,
) -> bool:
    if surfaces.intersection(_YEAR_NUMBER_OK):
        return True
    if start is not None and end is not None:
        return bool(surfaces.intersection(_STRUCTURAL_NUMBER_OK) and _is_client_grade_number_context_at(text, start=start, end=end))
    return bool(surfaces.intersection(_STRUCTURAL_NUMBER_OK) and _is_client_grade_number_context(text, token))


def _has_free_uncertainty_marker_near(
    text: str,
    token: str,
    *,
    start: int | None = None,
    end: int | None = None,
) -> bool:
    raw = str(text or "")
    item = str(token or "").strip()
    if not item:
        segment = raw
    elif start is not None and end is not None:
        segment = raw[max(0, start - 60) : end + 60]
    else:
        index = raw.find(item)
        segment = raw if index < 0 else raw[max(0, index - 60) : index + len(item) + 60]
    low = segment.casefold().replace("ё", "е")
    return any(marker in low for marker in _FREE_NUMBER_UNCERTAINTY_MARKERS)


def _has_free_uncertainty_marker(text: str) -> bool:
    low = str(text or "").casefold().replace("ё", "е")
    return any(marker in low for marker in _FREE_NUMBER_UNCERTAINTY_MARKERS)


def _ensure_estimate_uncertainty_marker(text: str, *, context: Mapping[str, Any] | None) -> str:
    value = str(text or "").strip()
    if not value or not (free_number_gate_enabled(context) or travel_compose_enabled(context)) or _has_free_uncertainty_marker(value):
        return value
    if not _estimate_text_needs_uncertainty_marker(value):
        return value
    return f"Ориентировочно: {value}"


def _estimate_text_needs_uncertainty_marker(text: str) -> bool:
    value = str(text or "").strip()
    if not value:
        return False
    low = value.casefold().replace("ё", "е")
    if re.fullmatch(r"(?:пожалуйста|рада?\s+был[аио]?\s+помочь|обращайтесь|спасибо)[!. ]*", low, re.I):
        return False
    for token, start, end in _free_number_token_matches(value):
        if _is_free_product_number_context(value, token, start=start, end=end):
            continue
        window = _free_number_context_window(value, start=start, end=end, radius=60).casefold().replace("ё", "е")
        if re.search(r"минут|час|км|километр|дорог|ехать|доехать|пешком|электрич|метро|маршрут|обычно|примерно|около", window, re.I):
            return True
    return False


def _is_client_grade_number_context_at(text: str, *, start: int, end: int) -> bool:
    raw = str(text or "")
    window = raw[max(0, start - 16) : end + 24].casefold().replace("ё", "е")
    return bool(re.search(r"\bкласс(?:а|е|ов|ы)?\b|\bкл\.?\b", window, re.I))


def _free_number_context_window(text: str, *, start: int, end: int, radius: int) -> str:
    raw = str(text or "")
    left = max(0, start - radius)
    right = min(len(raw), end + radius)
    for separator in ".?!;\n":
        pos = raw.rfind(separator, left, start)
        if pos >= 0:
            left = max(left, pos + 1)
        pos = raw.find(separator, end, right)
        if pos >= 0:
            right = min(right, pos)
    return raw[left:right]


def _is_decimal_year_range(token: str) -> bool:
    item = str(token or "").casefold().replace("ё", "е")
    return bool(re.search(r"\d+[.,]\d+\s*[-–]\s*\d+(?:[.,]\d+)?", item) and re.search(r"год|лет", item))


def _free_number_word_surfaces(text: str) -> set[str]:
    normalized = str(text or "").casefold().replace("ё", "е")
    surfaces: set[str] = set()
    if re.search(r"\b(?:два|двое|двух|второ[йеюя])\b", normalized):
        surfaces.add("2")
    if re.search(r"\b(?:три|трое|трех|трёх|трети[йеюя])\b", normalized):
        surfaces.add("3")
    return surfaces


def _number_token_map(text: str) -> Mapping[str, tuple[str, ...]]:
    result: dict[str, list[str]] = {}
    for match in _ESTIMATE_NUMBER_TOKEN_RE.finditer(str(text or "")):
        token = match.group(0).strip()
        if not token:
            continue
        for number in _numbers(token):
            result.setdefault(number, []).append(token)
    return {key: tuple(value) for key, value in result.items()}


def _is_product_number_context(text: str, token: str) -> bool:
    raw = str(text or "")
    item = str(token or "").strip()
    if not item:
        return bool(_PRODUCT_NUMBER_CTX_RE.search(raw))
    index = raw.find(item)
    if index < 0:
        return bool(_PRODUCT_NUMBER_CTX_RE.search(raw))
    window = raw[max(0, index - 25) : index + len(item) + 25]
    return bool(_PRODUCT_NUMBER_CTX_RE.search(window))


def _is_route_estimate_number_context(text: str, token: str, *, estimate_domain: str) -> bool:
    if estimate_domain not in {"travel_time", "route_logistics"}:
        return False
    raw = str(text or "")
    item = str(token or "").strip()
    if not item:
        return False
    index = raw.find(item)
    if index < 0:
        return False
    window = raw[max(0, index - 45) : index + len(item) + 45].casefold().replace("ё", "е")
    if not re.search(r"минут|час|км|километр", item.casefold(), re.I):
        return False
    return bool(re.search(r"ехать|дорог|пешком|электрич|метро|автобус|маршрут|такси|станци", window, re.I))


def _is_client_grade_number_context(text: str, token: str) -> bool:
    raw = str(text or "")
    item = str(token or "").strip()
    if not item:
        return False
    index = raw.find(item)
    if index < 0:
        return False
    window = raw[max(0, index - 16) : index + len(item) + 24].casefold().replace("ё", "е")
    return bool(re.search(r"\bкласс(?:а|е|ов|ы)?\b|\bкл\.?\b", window, re.I))


def _has_uncertainty_marker(text: str) -> bool:
    low = str(text or "").casefold()
    return any(marker in low for marker in _UNCERTAINTY_MARKERS)


def _general_advice_estimate_findings(text: str, *, client_message: str) -> list[VerificationFinding]:
    combined = " ".join([str(client_message or ""), str(text or "")])
    findings: list[VerificationFinding] = []
    if _INDIVIDUAL_CHILD_RE.search(combined):
        findings.append(VerificationFinding("estimate_individual_child_advice", "оценка похожа на диагноз конкретного ребёнка"))
    if _ESTIMATE_PRESSURE_RE.search(text) or _ESTIMATE_GUARANTEE_RE.search(text):
        findings.append(VerificationFinding("estimate_general_advice_risk", "совет содержит давление или обещание результата"))
    return findings


def _individual_child_diagnosis_findings(text: str, *, client_message: str) -> list[VerificationFinding]:
    if not _INDIVIDUAL_CHILD_RE.search(str(client_message or "")):
        return []
    if not _INDIVIDUAL_CHILD_CONFIDENT_RE.search(str(text or "")):
        return []
    return [VerificationFinding("estimate_individual_child_advice", "ответ уверенно оценивает конкретного ребёнка")]


def _estimate_gate_payload_from_context(context: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not isinstance(context, MappingABC):
        return {}
    direct = context.get("estimate_mode")
    if isinstance(direct, MappingABC):
        return direct
    pipeline = context.get("dialogue_contract_pipeline")
    if isinstance(pipeline, MappingABC):
        estimate = pipeline.get("estimate")
        if isinstance(estimate, MappingABC):
            return estimate
    return {}


def _gate_answer_mode(
    *,
    contract: AnswerContract | None,
    context: Mapping[str, Any] | None,
    explicit: str | None,
) -> str:
    if explicit is not None:
        return _clean_answer_mode(explicit)
    payload = _estimate_gate_payload_from_context(context)
    if payload.get("answer_mode") is not None:
        return _clean_answer_mode(payload.get("answer_mode"))
    if contract is not None:
        return _clean_answer_mode(contract.answer_mode)
    return "confirmed_only"


def _gate_estimate_domain(
    *,
    contract: AnswerContract | None,
    context: Mapping[str, Any] | None,
    explicit: str | None,
) -> str:
    if explicit is not None:
        return _clean_estimate_domain(explicit)
    payload = _estimate_gate_payload_from_context(context)
    if payload.get("estimate_domain") is not None:
        return _clean_estimate_domain(payload.get("estimate_domain"))
    if contract is not None:
        return _clean_estimate_domain(contract.estimate_domain)
    return "none"


def _gate_is_estimate(
    *,
    contract: AnswerContract | None,
    context: Mapping[str, Any] | None,
    explicit: bool | None,
) -> bool:
    if explicit is not None:
        return bool(explicit)
    payload = _estimate_gate_payload_from_context(context)
    if payload.get("is_estimate") is not None:
        return _truthy(payload.get("is_estimate"))
    return bool(contract is not None and contract.answer_mode == "estimate_allowed")


def _preemptive_format_choice_finding(answer_low: str, *, contract: AnswerContract) -> bool:
    if not _asks_training_format_choice(contract) or _contract_mentions_camp_or_lvsh(contract):
        return False
    normalized = str(answer_low or "").casefold().replace("ё", "е")
    asserts_single = bool(
        re.search(r"\bэто\s+онлайн\b|\bтолько\s+онлайн\b|\bэто\s+очно\b|\bтолько\s+очно\b", normalized, re.I)
    )
    mentions_both = "онлайн" in normalized and "очно" in normalized
    return asserts_single and not mentions_both


_SCHEDULE_SPECIFICITY_ALIASES: Mapping[str, tuple[str, ...]] = {
    "weekday": ("по будням", "в будни", "будни", "будний", "будням"),
    "weekend": ("по выходным", "выходные", "выходным", "суббот", "воскрес"),
    "monday": ("по понедельникам", "понедельник", "понедельникам"),
    "tuesday": ("по вторникам", "вторник", "вторникам"),
    "wednesday": ("по средам", "среда", "средам"),
    "thursday": ("по четвергам", "четверг", "четвергам"),
    "friday": ("по пятницам", "пятница", "пятницам"),
    "evening": ("вечерам", "вечером", "вечерн"),
    "morning": ("утрам", "по утрам", "утром", "утренн"),
}


def _unconfirmed_schedule_finding(
    answer_low: str,
    *,
    facts: Mapping[str, str],
    client_message: str,
) -> VerificationFinding | None:
    answer_anchors = _schedule_specificity_anchors(answer_low)
    if not answer_anchors:
        return None
    if _schedule_specificity_is_declined(answer_low):
        return None
    fact_text = " ".join(str(value or "") for value in facts.values()).casefold().replace("ё", "е")
    client_text = str(client_message or "").casefold().replace("ё", "е")
    backed = _schedule_specificity_anchors(fact_text) | _schedule_specificity_anchors(client_text)
    unconfirmed = tuple(sorted(answer_anchors - backed))
    if not unconfirmed:
        return None
    return VerificationFinding(
        "unconfirmed_schedule",
        f"ответ называет дни/время без факта-расписания: {list(unconfirmed)}",
    )


def _schedule_specificity_anchors(text: str) -> set[str]:
    normalized = str(text or "").casefold().replace("ё", "е")
    return {
        anchor
        for anchor, aliases in _SCHEDULE_SPECIFICITY_ALIASES.items()
        if any(_schedule_alias_present(normalized, alias) for alias in aliases)
    }


def _schedule_alias_present(normalized_text: str, alias: str) -> bool:
    normalized_alias = str(alias or "").casefold().replace("ё", "е")
    if not normalized_alias:
        return False
    return bool(re.search(rf"(?<![а-яa-z]){re.escape(normalized_alias)}", normalized_text, re.I))


def _schedule_specificity_is_declined(text: str) -> bool:
    normalized = str(text or "").casefold().replace("ё", "е")
    return bool(
        re.search(
            r"не\s+буду\s+называть|не\s+называю|не\s+подтверждаю|без\s+подтверждени[яй]|точн\w*\s+дн\w*\s+.*\bнет\b",
            normalized,
            re.I,
        )
    )


def _self_contradiction_finding(
    text: str,
    answer_low: str,
    *,
    previous_bot_texts: Sequence[str],
) -> VerificationFinding | None:
    cur_pcts = set(re.findall(r"(\d{1,2})\s*%", text))
    if not cur_pcts or "скидк" not in answer_low:
        return None
    cur_scopes = _discount_scope_anchors(answer_low)
    for previous in previous_bot_texts:
        prev_text = str(previous or "")
        prev_low = prev_text.casefold().replace("ё", "е")
        if "скидк" not in prev_low:
            continue
        prev_pcts = set(re.findall(r"(\d{1,2})\s*%", prev_text))
        if not prev_pcts or not prev_pcts.isdisjoint(cur_pcts):
            continue
        prev_scopes = _discount_scope_anchors(prev_low)
        if cur_scopes and prev_scopes and cur_scopes.isdisjoint(prev_scopes):
            continue
        return VerificationFinding(
            "self_contradiction",
            f"процент скидки противоречит ранее названному ботом: было {sorted(prev_pcts)}, стало {sorted(cur_pcts)}",
        )
    return None


_DISCOUNT_SCOPE_ALIASES: Mapping[str, tuple[str, ...]] = {
    "second_subject": ("второй предмет", "2-й предмет", "вторым предмет", "второго предмет"),
    "third_subject": ("третий предмет", "3-й предмет", "третьим предмет", "третьего предмет", "последующ"),
    "multichild": ("многодет", "двое детей", "2 детей", "несколько детей"),
    "sibling": ("брат", "сестр", "ребенок", "ребёнок", "детей"),
}


def _discount_scope_anchors(text: str) -> set[str]:
    normalized = str(text or "").casefold().replace("ё", "е")
    return {
        anchor
        for anchor, aliases in _DISCOUNT_SCOPE_ALIASES.items()
        if any(alias in normalized for alias in aliases)
    }


def _hard_check(
    draft: str,
    *,
    facts: Mapping[str, str],
    contract: AnswerContract,
    client_words: str,
    faithfulness_fn: Callable[[str], object] | None,
    toggles: Toggles,
    context: Mapping[str, Any] | None,
    previous_bot_texts: Sequence[str] = (),
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
            previous_bot_texts=previous_bot_texts,
        )
    )
    findings.extend(_existence_yes_no_findings(text_to_check, contract=contract, facts=facts))
    findings.extend(_payment_method_findings(text_to_check, contract=contract, facts=facts))
    unsupported: tuple[str, ...] = ()
    semantic_available = True
    if toggles.semantic_faithfulness:
        result = check_claim_faithfulness(
            text_to_check,
            facts=facts,
            client_words=client_words,
            faithfulness_fn=faithfulness_fn,
            established_topic=_established_topic_from_context(context, contract=contract),
        )
        semantic_available = result.available
        if not pure_handoff:
            unsupported = _unsupported_claims_without_current_fact_support(
                result.unsupported,
                facts=facts,
                contract=contract,
            )
    trace_event(
        context,
        "_hard_check",
        {
            "draft": draft,
            "pure_handoff": pure_handoff,
            "verification_text": text_to_check,
            "findings": [{"code": finding.code, "detail": finding.detail} for finding in findings],
            "unsupported": list(unsupported),
            "semantic_available": semantic_available,
        },
    )
    return tuple(findings), unsupported, semantic_available


@dataclass(frozen=True)
class _CoverageFinding:
    subquestion: str
    required_key: str
    fact_key: str
    fact_text: str


def _coverage_findings(
    draft: str,
    *,
    contract: AnswerContract,
    retrieval: RetrievalResult,
    force_draft_for_manager: bool,
    context: Mapping[str, Any] | None = None,
) -> tuple[_CoverageFinding, ...]:
    if force_draft_for_manager or contract.is_p0 or contract.answerability != "answer_self":
        return ()
    if _is_handoff_text(draft) and not _handoff_factual_claim_text(draft):
        return ()
    findings: list[_CoverageFinding] = []
    subquestions = contract.subquestions or (
        Subquestion(
            text=contract.current_question,
            answerable="self" if contract.answerability == "answer_self" else "manager",
            needed_fact_keys=contract.needed_fact_keys,
            question_type=contract.question_type,
            existence_target=contract.existence_target,
        ),
    )
    for subquestion in subquestions:
        if subquestion.answerable != "self":
            continue
        keys = tuple(key for key in subquestion.needed_fact_keys if key)
        if not keys:
            continue
        if not _retrieved_keys_match_question_scope(contract, subquestion, retrieval, keys):
            continue
        for required_key in keys:
            matched = list(_matched_scope_fact_keys_for_required_key(contract, subquestion, retrieval, required_key))
            if not matched:
                continue
            if any(_answer_cites_fact(draft, retrieval.facts[key]) for key in matched):
                continue
            first_key = matched[0]
            findings.append(
                _CoverageFinding(
                    subquestion=subquestion.text or contract.current_question,
                    required_key=required_key,
                    fact_key=first_key,
                    fact_text=str(retrieval.facts[first_key]),
                )
            )
    trace_event(
        context,
        "coverage_check",
        {
            "findings": [
                {"required_key": item.required_key, "fact_key": item.fact_key}
                for item in findings
            ]
        },
    )
    return tuple(findings)


def _answer_cites_fact(answer: str, fact_text: str) -> bool:
    answer_text = str(answer or "")
    fact = str(fact_text or "")
    if not answer_text.strip() or not fact.strip():
        return False
    fact_value_anchors = _coverage_value_anchors(fact)
    if fact_value_anchors:
        return bool(fact_value_anchors & _coverage_value_anchors(answer_text))
    answer_anchors = concrete_anchors(answer_text)
    fact_anchors = concrete_anchors(fact)
    if fact_anchors:
        return bool(answer_anchors & fact_anchors)
    if _semantic_topic_anchors(answer_text) & _semantic_topic_anchors(fact):
        return True
    answer_low = answer_text.casefold().replace("ё", "е")
    return any(token in answer_low for token in _coverage_terms(fact))


def _coverage_value_anchors(text: str) -> set[str]:
    source = str(text or "")
    low = source.casefold().replace("ё", "е")
    anchors: set[str] = set()
    for match in re.finditer(r"\d[\d\s\u00a0]{2,}\s*(?:₽|руб(?:\.|лей|ля|ль)?|р\.)", source, re.I):
        digits = re.sub(r"\D", "", match.group(0))
        if digits:
            anchors.add(f"money:{digits}")
    for match in re.finditer(r"\b(\d{1,3})\s*%", source, re.I):
        anchors.add(f"percent:{match.group(1)}")
    for match in _DATE_ANCHOR_RE.finditer(source):
        normalized = _normalize_date_anchor(match)
        if normalized:
            anchors.add(f"date:{normalized}")
    if re.search(r"январ|феврал|март|апрел|ма[йя]|июн|июл|август|сентябр|октябр|ноябр|декабр", low, re.I):
        for number in _numbers(source):
            anchors.add(f"date_number:{number}")
    return anchors


def _coverage_terms(text: str) -> tuple[str, ...]:
    low = str(text or "").casefold().replace("ё", "е")
    tokens = re.findall(r"[а-яa-z][а-яa-z0-9-]{4,}", low, re.I)
    stop = {
        "фотон",
        "унпк",
        "клиент",
        "клиента",
        "можно",
        "действует",
        "подтвердит",
        "менеджер",
        "учебный",
        "учебного",
        "курса",
        "курсы",
    }
    return tuple(dict.fromkeys(token for token in tokens if token not in stop))[:8]


def _coverage_repair_prompt(
    draft: str,
    findings: Sequence[_CoverageFinding],
    facts: Mapping[str, str],
) -> str:
    required = "\n".join(
        f"- {item.fact_key}: {_short_fact_sentence(item.fact_text, max_chars=220)}"
        for item in findings
    )
    facts_block = "\n".join(f"- {key}: {value}" for key, value in facts.items()) or "(нет фактов)"
    return (
        "Исправь ответ: он обязан прямо использовать подтверждённые факты ниже. "
        "Не добавляй новых чисел, дат, адресов или условий.\n"
        f"Факты, которые обязательно надо назвать:\n{required}\n"
        f"Все факты хода:\n{facts_block}\n"
        f"Черновик:\n{draft}\n"
        "Верни только клиентский ответ."
    )


def _coverage_cite_only_answer(contract: AnswerContract, retrieval: RetrievalResult) -> str:
    return _coverage_cite_only_answer_from_findings(
        _coverage_findings(
            "",
            contract=contract,
            retrieval=retrieval,
            force_draft_for_manager=False,
        )
    )


def _key_coverage_cite_only_answer(contract: AnswerContract, retrieval: RetrievalResult) -> str:
    return _coverage_cite_only_answer_from_findings(_key_coverage_findings(contract, retrieval))


def _coverage_cite_only_answer_from_findings(findings: Sequence[_CoverageFinding]) -> str:
    if not findings:
        return ""
    snippets: list[str] = []
    seen: set[str] = set()
    for item in findings:
        snippet = _short_fact_sentence(item.fact_text, max_chars=220)
        if not snippet or snippet in seen:
            continue
        seen.add(snippet)
        snippets.append(snippet)
    if not snippets:
        return ""
    if len(snippets) == 1:
        return f"По подтверждённым данным: {snippets[0]}"
    return "По подтверждённым данным: " + " ".join(snippets[:3])


def _key_coverage_findings(contract: AnswerContract, retrieval: RetrievalResult) -> tuple[_CoverageFinding, ...]:
    if contract.is_p0 or contract.answerability != "answer_self":
        return ()
    findings: list[_CoverageFinding] = []
    subquestions = contract.subquestions or (
        Subquestion(
            text=contract.current_question,
            answerable="self" if contract.answerability == "answer_self" else "manager",
            needed_fact_keys=contract.needed_fact_keys,
            question_type=contract.question_type,
            existence_target=contract.existence_target,
        ),
    )
    for subquestion in subquestions:
        if subquestion.answerable != "self":
            continue
        for required_key in tuple(key for key in subquestion.needed_fact_keys if key):
            for fact_key in _matched_scope_fact_keys_for_required_key(contract, subquestion, retrieval, required_key):
                findings.append(
                    _CoverageFinding(
                        subquestion=subquestion.text or contract.current_question,
                        required_key=required_key,
                        fact_key=fact_key,
                        fact_text=str(retrieval.facts[fact_key]),
                    )
                )
    return tuple(findings)


def _key_coverage_ok(contract: AnswerContract, retrieval: RetrievalResult) -> bool:
    if contract.is_p0 or contract.answerability != "answer_self":
        return False
    for subquestion in _contract_subquestions(contract):
        if subquestion.answerable != "self":
            continue
        for required_key in tuple(key for key in subquestion.needed_fact_keys if key):
            if _matched_scope_fact_keys_for_required_key(contract, subquestion, retrieval, required_key):
                return True
    return False


def _quality_composite_result_before_draft(
    *,
    contract: AnswerContract,
    retrieval: RetrievalResult,
    client_words: str,
    conversation: Sequence[Mapping[str, str]],
    faithfulness_fn: Callable[[str], object] | None,
    toggles: Toggles,
    context: Mapping[str, Any] | None,
    previous_bot_texts: Sequence[str],
) -> DialogueContractPipelineResult | None:
    if not quality_composite_enabled(context):
        return None
    if _cite_only_recover_blocked(contract, client_words=client_words, context=context) or _composite_has_hard_p0_part(
        contract,
        client_words=client_words,
        context=context,
    ):
        trace_event(context, "composite_answer", {"applied": False, "reason": "p0_or_high_risk"})
        return None
    subquestions = _contract_subquestions(contract)
    if len(subquestions) < 2 or not any(item.answerable == "self" for item in subquestions):
        return None
    if toggles.semantic_faithfulness and faithfulness_fn is None:
        trace_event(context, "composite_answer", {"applied": False, "reason": "faithfulness_fn_missing"})
        return None
    findings, missing_details = _partial_yield_findings_and_missing(contract, retrieval)
    if not findings:
        trace_event(context, "composite_answer", {"applied": False, "reason": "no_grounded_parts"})
        return None
    candidate = _composite_candidate_from_parts(findings, missing_details)
    if not candidate:
        return None
    candidate_facts = _facts_with_derived_answer(retrieval.facts, candidate)
    check_findings, unsupported, semantic_available = _partial_yield_full_check(
        candidate,
        facts=candidate_facts,
        contract=contract,
        client_words=client_words,
        faithfulness_fn=faithfulness_fn,
        toggles=toggles,
        context=context,
        previous_bot_texts=previous_bot_texts,
    )
    if check_findings or unsupported or not semantic_available:
        trace_event(
            context,
            "composite_answer",
            {
                "applied": False,
                "reason": "hard_check_failed" if semantic_available else "semantic_unavailable",
                "findings": [finding.code for finding in check_findings],
                "unsupported": list(unsupported),
            },
        )
        return None
    final = _avoid_repeating_text(candidate, conversation=conversation, contract=contract, facts=retrieval.facts)
    fact_keys = tuple(dict.fromkeys(item.fact_key for item in findings if item.fact_key))
    trace_event(
        context,
        "composite_answer",
        {
            "applied": True,
            "fact_keys": list(fact_keys),
            "missing": list(missing_details),
        },
    )
    return DialogueContractPipelineResult(
        draft_text=final,
        route="bot_answer_self",
        manager_only=False,
        contract=contract,
        facts=retrieval.facts,
        missing=retrieval.missing,
        fallback_reason="composite_partial_yield" if missing_details else "composite_grounded_answer",
        repaired=True,
        recovery_candidate=final,
        composite_applied=True,
        composite_fact_keys=fact_keys,
        composite_missing=tuple(missing_details),
    )


def _contract_subquestions(contract: AnswerContract) -> tuple[Subquestion, ...]:
    return contract.subquestions or (
        Subquestion(
            text=contract.current_question,
            answerable="self" if contract.answerability == "answer_self" else "manager",
            needed_fact_keys=contract.needed_fact_keys,
            question_type=contract.question_type,
            existence_target=contract.existence_target,
        ),
    )


def _composite_has_hard_p0_part(
    contract: AnswerContract,
    *,
    client_words: str,
    context: Mapping[str, Any] | None,
) -> bool:
    for text in (client_words, contract.current_question, *(item.text for item in contract.subquestions)):
        if str(text or "").strip() and p0_pre_gate(str(text), context=context):
            return True
    return False


def _composite_candidate_from_parts(
    findings: Sequence[_CoverageFinding],
    missing_details: Sequence[str],
) -> str:
    grounded = _coverage_cite_only_answer_from_findings(findings)
    if not grounded:
        return ""
    parts = [grounded.rstrip(" .")]
    if missing_details:
        missing_text = _partial_yield_missing_text(missing_details)
        parts.append(f"{missing_text} менеджер сверит точный ответ; я передам ему этот вопрос")
    return ". ".join(part for part in parts if part).rstrip(".") + "."


def _partial_yield_result_before_handoff(
    *,
    contract: AnswerContract,
    retrieval: RetrievalResult,
    client_words: str,
    conversation: Sequence[Mapping[str, str]],
    faithfulness_fn: Callable[[str], object] | None,
    toggles: Toggles,
    context: Mapping[str, Any] | None,
    previous_bot_texts: Sequence[str],
    source_reason: str,
) -> DialogueContractPipelineResult | None:
    if not quality_partial_yield_enabled(context):
        return None
    if toggles.semantic_faithfulness and faithfulness_fn is None:
        trace_event(
            context,
            "partial_yield",
            {"applied": False, "reason": "faithfulness_fn_missing", "source_reason": source_reason},
        )
        return None
    if _cite_only_recover_blocked(contract, client_words=client_words, context=context):
        trace_event(
            context,
            "partial_yield",
            {"applied": False, "reason": "blocked_risk", "source_reason": source_reason},
        )
        return None
    candidate, fact_keys, missing_details = _partial_yield_candidate(contract, retrieval)
    if not candidate:
        trace_event(
            context,
            "partial_yield",
            {"applied": False, "reason": "empty_candidate", "source_reason": source_reason},
        )
        return None
    candidate_facts = _facts_with_derived_answer(retrieval.facts, candidate)
    findings, unsupported, semantic_available = _partial_yield_full_check(
        candidate,
        facts=candidate_facts,
        contract=contract,
        client_words=client_words,
        faithfulness_fn=faithfulness_fn,
        toggles=toggles,
        context=context,
        previous_bot_texts=previous_bot_texts,
    )
    if findings or unsupported or not semantic_available:
        trace_event(
            context,
            "partial_yield",
            {
                "applied": False,
                "reason": "hard_check_failed" if semantic_available else "semantic_unavailable",
                "source_reason": source_reason,
                "findings": [finding.code for finding in findings],
                "unsupported": list(unsupported),
            },
        )
        return None
    final = _avoid_repeating_text(candidate, conversation=conversation, contract=contract, facts=retrieval.facts)
    trace_event(
        context,
        "partial_yield",
        {
            "applied": True,
            "source_reason": source_reason,
            "fact_keys": list(fact_keys),
            "missing": list(missing_details),
        },
    )
    return DialogueContractPipelineResult(
        draft_text=final,
        route="bot_answer_self",
        manager_only=False,
        contract=contract,
        facts=retrieval.facts,
        missing=retrieval.missing,
        fallback_reason=f"partial_yield_{source_reason}",
        repaired=True,
        recovery_candidate=final,
        partial_yield_applied=True,
        partial_yield_fact_keys=tuple(fact_keys),
        partial_yield_missing=tuple(missing_details),
    )


def _partial_yield_full_check(
    draft: str,
    *,
    facts: Mapping[str, str],
    contract: AnswerContract,
    client_words: str,
    faithfulness_fn: Callable[[str], object] | None,
    toggles: Toggles,
    context: Mapping[str, Any] | None,
    previous_bot_texts: Sequence[str],
) -> tuple[tuple[VerificationFinding, ...], tuple[str, ...], bool]:
    findings = list(
        verify_output(
            draft,
            facts=facts,
            active_brand=contract.active_brand,
            contract=contract,
            denied_topics=contract.denied_topics,
            forbidden_substitutions=contract.forbidden_substitutions,
            client_message=client_words,
            context=context,
            previous_bot_texts=previous_bot_texts,
        )
    )
    findings.extend(_existence_yes_no_findings(draft, contract=contract, facts=facts))
    findings.extend(_payment_method_findings(draft, contract=contract, facts=facts))
    unsupported: tuple[str, ...] = ()
    semantic_available = True
    if toggles.semantic_faithfulness:
        result = check_claim_faithfulness(
            draft,
            facts=facts,
            client_words=client_words,
            faithfulness_fn=faithfulness_fn,
            established_topic=_established_topic_from_context(context, contract=contract),
        )
        semantic_available = result.available
        unsupported = _unsupported_claims_without_current_fact_support(
            result.unsupported,
            facts=facts,
            contract=contract,
        )
    return tuple(findings), unsupported, semantic_available


def _partial_yield_candidate(
    contract: AnswerContract,
    retrieval: RetrievalResult,
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    if contract.is_p0 or not retrieval.facts:
        return "", (), ()
    findings, missing_details = _partial_yield_findings_and_missing(contract, retrieval)
    if not findings or not missing_details:
        return "", (), ()
    grounded = _coverage_cite_only_answer_from_findings(findings)
    if not grounded:
        return "", (), ()
    missing_text = _partial_yield_missing_text(missing_details)
    text = f"{grounded.rstrip(' .')}. {missing_text} менеджер сверит точный ответ; я передам ему этот вопрос."
    fact_keys = tuple(dict.fromkeys(item.fact_key for item in findings if item.fact_key))
    return text, fact_keys, tuple(missing_details)


def _partial_yield_findings_and_missing(
    contract: AnswerContract,
    retrieval: RetrievalResult,
) -> tuple[tuple[_CoverageFinding, ...], tuple[str, ...]]:
    if _has_foreign_brand_matched_self_fact(contract, retrieval):
        return (), ()
    subquestions = contract.subquestions or (
        Subquestion(
            text=contract.current_question,
            answerable="self" if contract.answerability == "answer_self" else "manager",
            needed_fact_keys=contract.needed_fact_keys,
            question_type=contract.question_type,
            existence_target=contract.existence_target,
        ),
    )
    findings: list[_CoverageFinding] = []
    missing: list[str] = []
    for subquestion in subquestions:
        keys = tuple(key for key in subquestion.needed_fact_keys if key)
        if not keys:
            if subquestion.answerable != "self":
                missing.append(_client_safe_question_detail(subquestion.text or contract.current_question))
            continue
        if subquestion.answerable != "self":
            missing.append(_client_safe_question_detail(subquestion.text or contract.current_question))
            continue
        if not _retrieved_keys_match_question_scope(contract, subquestion, retrieval, keys):
            missing.append(_client_safe_question_detail(subquestion.text or contract.current_question))
            continue
        has_fact = False
        for required_key in keys:
            matched = list(_matched_scope_fact_keys_for_required_key(contract, subquestion, retrieval, required_key))
            if not matched:
                missing.append(_client_safe_question_detail(subquestion.text or required_key))
                continue
            has_fact = True
            first_key = matched[0]
            findings.append(
                _CoverageFinding(
                    subquestion=subquestion.text or contract.current_question,
                    required_key=required_key,
                    fact_key=first_key,
                    fact_text=str(retrieval.facts[first_key]),
                )
            )
        if not has_fact:
            missing.append(_client_safe_question_detail(subquestion.text or contract.current_question))
    if not missing and retrieval.missing:
        missing.extend(_client_safe_question_detail(item) for item in retrieval.missing)
    clean_missing = tuple(dict.fromkeys(item for item in missing if item))
    return tuple(findings), clean_missing


def _partial_yield_missing_text(missing_details: Sequence[str]) -> str:
    details = tuple(dict.fromkeys(_client_safe_question_detail(item) for item in missing_details if item))
    if not details:
        return "По остальной части вопроса"
    if len(details) == 1:
        return f"По части «{details[0]}»"
    return "По остальным частям вопроса"


def _composition_answer(contract: AnswerContract, retrieval: RetrievalResult, *, current_draft: str = "") -> str:
    for builder in (
        _compose_n_subjects_discount,
        _compose_nearest_camp_shift,
        _compose_price_plus_format,
        _compose_installment_summary,
    ):
        answer = builder(contract, retrieval, current_draft=current_draft)
        if answer:
            return answer
    return ""


def _verified_empty_handoff_replacement(
    draft: str,
    *,
    contract: AnswerContract,
    retrieval: RetrievalResult,
    client_words: str,
    faithfulness_fn: Callable[[str], object] | None,
    toggles: Toggles,
    context: Mapping[str, Any] | None,
    previous_bot_texts: Sequence[str] = (),
    allow_key_coverage: bool = False,
) -> str:
    if not _should_replace_empty_handoff(
        draft,
        contract=contract,
        retrieval=retrieval,
        allow_key_coverage=allow_key_coverage,
    ):
        return ""
    return _cite_only_recover_before_handoff(
        contract=contract,
        retrieval=retrieval,
        draft=draft,
        client_words=client_words,
        faithfulness_fn=faithfulness_fn,
        toggles=toggles,
        context=context,
        previous_bot_texts=previous_bot_texts,
        allow_key_coverage=allow_key_coverage,
    )


def _cite_only_recover_before_handoff(
    *,
    contract: AnswerContract,
    retrieval: RetrievalResult,
    draft: str,
    client_words: str,
    faithfulness_fn: Callable[[str], object] | None,
    toggles: Toggles,
    context: Mapping[str, Any] | None,
    previous_bot_texts: Sequence[str] = (),
    allow_key_coverage: bool = False,
    original_findings: Sequence[VerificationFinding] = (),
    original_unsupported: Sequence[str] = (),
) -> str:
    if _cite_only_recover_blocked(contract, client_words=client_words, context=context):
        trace_event(context, "cite_only_recover", {"replaced": False, "reason": "blocked_risk"})
        return ""
    if not _original_failure_allows_cite_only_recover(original_findings, original_unsupported):
        trace_event(context, "cite_only_recover", {"replaced": False, "reason": "unsafe_original_failure"})
        return ""
    has_scope = _key_coverage_ok(contract, retrieval) if allow_key_coverage else _has_exact_retrieved_answer_part(contract, retrieval)
    if not has_scope:
        trace_event(context, "cite_only_recover", {"replaced": False, "reason": "no_exact_scope"})
        return ""
    if allow_key_coverage and _asks_class_schedule_days(contract):
        matched = _matched_fact_text_for_required_keys(retrieval, contract.all_needed_fact_keys())
        if matched and all(_is_contact_hours_fact(key, value) for key, value in matched.items()):
            trace_event(context, "cite_only_recover", {"replaced": False, "reason": "contact_hours_not_class_schedule"})
            return ""
    replacement = (
        _composition_answer(contract, retrieval, current_draft=draft)
        or _hard_failure_exact_fact_fallback(contract, retrieval)
        or (
            _key_coverage_cite_only_answer(contract, retrieval)
            if allow_key_coverage
            else _exact_scope_cite_only_answer(contract, retrieval)
        )
    )
    if not replacement:
        trace_event(context, "cite_only_recover", {"replaced": False, "reason": "empty_candidate"})
        return ""
    replacement_facts = _facts_with_derived_answer(retrieval.facts, replacement)
    findings, unsupported, semantic_available = _hard_check(
        replacement,
        facts=replacement_facts,
        contract=contract,
        client_words=client_words,
        faithfulness_fn=faithfulness_fn,
        toggles=toggles,
        context=context,
        previous_bot_texts=previous_bot_texts,
    )
    if findings or unsupported:
        trace_event(
            context,
            "cite_only_recover",
            {
                "replaced": False,
                "reason": "hard_check_failed",
                "findings": [finding.code for finding in findings],
                "unsupported": list(unsupported),
            },
        )
        return ""
    if semantic_available or not new_concrete_anchors(replacement, original="", facts=retrieval.facts):
        return replacement
    trace_event(context, "cite_only_recover", {"replaced": False, "reason": "semantic_unavailable_new_anchor"})
    return ""


def _stashed_recovery_candidate(
    draft: str,
    *,
    contract: AnswerContract,
    retrieval: RetrievalResult,
    client_words: str,
    context: Mapping[str, Any] | None,
) -> str:
    text = str(draft or "").strip()
    if (
        not text
        or not retrieval.facts
        or contract.answerability != "answer_self"
        or _looks_like_handoff(text)
        or _cite_only_recover_blocked(contract, client_words=client_words, context=context)
    ):
        return ""
    return text


_CITE_ONLY_RECOVERABLE_FINDING_CODES = {
    "fact_grounding",
    "unsupported_named_entity",
    "wrong_intent_fact",
}


def _original_failure_allows_cite_only_recover(
    findings: Sequence[VerificationFinding],
    unsupported: Sequence[str],
) -> bool:
    if any(finding.code not in _CITE_ONLY_RECOVERABLE_FINDING_CODES for finding in findings):
        return False
    return all(_unsupported_item_is_missing_answer(item) for item in unsupported)


def _unsupported_item_is_missing_answer(item: str) -> bool:
    text = str(item or "").casefold().replace("ё", "е")
    return bool(re.search(r"нет\s+ответ|не\s+ответ|не\s+использ|handoff|передам|менеджер|уточн", text, re.I))


def _cite_only_recover_blocked(
    contract: AnswerContract,
    *,
    client_words: str,
    context: Mapping[str, Any] | None,
) -> bool:
    if contract.is_p0 or _asks_refund_policy(contract):
        return True
    safety = classify_answer_safety(
        client_message=" ".join(part for part in (client_words, contract.current_question, contract.client_state) if part),
        context=context,
        route="manager_only",
    )
    return bool(safety.zero_collect_required or safety.primary_risk in {"complaint", "refund", "payment_dispute", "legal"})


def _exact_scope_cite_only_answer(contract: AnswerContract, retrieval: RetrievalResult) -> str:
    return _coverage_cite_only_answer_from_findings(_exact_scope_coverage_findings(contract, retrieval))


def _exact_scope_coverage_findings(contract: AnswerContract, retrieval: RetrievalResult) -> tuple[_CoverageFinding, ...]:
    findings: list[_CoverageFinding] = []
    subquestions = contract.subquestions or (
        Subquestion(
            text=contract.current_question,
            answerable=contract.answerability,
            needed_fact_keys=contract.needed_fact_keys,
            question_type=contract.question_type,
            existence_target=contract.existence_target,
        ),
    )
    for subquestion in subquestions:
        keys = tuple(key for key in subquestion.needed_fact_keys if key)
        if not keys or any(key in retrieval.missing or not retrieval.matched_keys.get(key) for key in keys):
            continue
        if not _retrieved_keys_match_question_scope(contract, subquestion, retrieval, keys):
            continue
        for required_key in keys:
            for fact_key in _matched_scope_fact_keys_for_required_key(contract, subquestion, retrieval, required_key):
                findings.append(
                    _CoverageFinding(
                        subquestion=subquestion.text or contract.current_question,
                        required_key=required_key,
                        fact_key=fact_key,
                        fact_text=str(retrieval.facts[fact_key]),
                    )
                )
                break
    return tuple(findings)


def _should_replace_empty_handoff(
    draft: str,
    *,
    contract: AnswerContract,
    retrieval: RetrievalResult,
    allow_key_coverage: bool = False,
) -> bool:
    if contract.is_p0 or contract.answerability != "answer_self":
        return False
    has_coverage = (
        _key_coverage_ok(contract, retrieval)
        if allow_key_coverage
        else _has_exact_retrieved_answer_part(contract, retrieval)
    )
    if not has_coverage:
        return False
    if _draft_cites_any_retrieved_self_fact(draft, contract=contract, retrieval=retrieval):
        return False
    handoff_like = _is_handoff_text(draft) or (allow_key_coverage and _looks_like_handoff(draft))
    return handoff_like and _handoff_factual_claim_text(draft) is None


def _draft_cites_any_retrieved_self_fact(
    draft: str,
    *,
    contract: AnswerContract,
    retrieval: RetrievalResult,
) -> bool:
    subquestions = contract.subquestions or (
        Subquestion(
            text=contract.current_question,
            answerable="self" if contract.answerability == "answer_self" else "manager",
            needed_fact_keys=contract.needed_fact_keys,
            question_type=contract.question_type,
            existence_target=contract.existence_target,
        ),
    )
    for subquestion in subquestions:
        if subquestion.answerable != "self":
            continue
        keys = tuple(key for key in subquestion.needed_fact_keys if key)
        if not keys or not _retrieved_keys_match_question_scope(contract, subquestion, retrieval, keys):
            continue
        for required_key in keys:
            for fact_key in _matched_scope_fact_keys_for_required_key(contract, subquestion, retrieval, required_key):
                if _answer_cites_fact(draft, retrieval.facts[fact_key]):
                    return True
    return False


def _facts_with_derived_answer(facts: Mapping[str, str], answer: str) -> Mapping[str, str]:
    merged = dict(facts)
    if answer:
        merged["__derived.phase1_composition"] = answer
    return merged


def _compose_n_subjects_discount(contract: AnswerContract, retrieval: RetrievalResult, *, current_draft: str = "") -> str:
    text = _contract_intent_text(contract)
    subject_count = _requested_subject_count(text)
    if subject_count < 2:
        return ""
    base = _price_for_composition(contract, retrieval.facts)
    pct = _second_subject_discount_pct(contract, retrieval.facts)
    if base is None or pct is None:
        return ""
    discounted = [round(base * (100 - pct) / 100) for _ in range(subject_count - 1)]
    total = base + sum(discounted)
    total_text = _format_rub(total)
    if total_text in str(current_draft or ""):
        return ""
    parts = [f"первый предмет — {_format_rub(base)}"]
    for index, amount in enumerate(discounted, start=2):
        parts.append(f"{index}-й предмет со скидкой {pct}% — {_format_rub(amount)}")
    return (
        f"Если брать {subject_count} предмета, по подтверждённым фактам: "
        f"{', '.join(parts)}. Итого — {total_text}. "
        "Скидки не суммируются; менеджер подтвердит группу и оформление."
    )


def _compose_nearest_camp_shift(contract: AnswerContract, retrieval: RetrievalResult, *, current_draft: str = "") -> str:
    if not _contract_mentions_camp_or_lvsh(contract):
        return ""
    text = _contract_intent_text(contract)
    if not re.search(r"ближайш|даты|когда|смен", text, re.I):
        return ""
    date_fact = ""
    price_fact = ""
    included_fact = ""
    for key, value in retrieval.facts.items():
        combined = f"{key} {value}".casefold().replace("ё", "е")
        if not _is_camp_or_lvsh_fact(key, str(value or "")):
            continue
        sentence = _short_fact_sentence(str(value or ""), max_chars=220)
        if not date_fact and re.search(r"\d{1,2}\s*[–-]\s*\d{1,2}|январ|феврал|март|апрел|ма[йя]|июн|июл|август", combined, re.I):
            date_fact = sentence
        elif not price_fact and re.search(r"₽|руб|цен|стоим", combined, re.I):
            price_fact = sentence
        elif not included_fact and re.search(r"входит|включ", combined, re.I):
            included_fact = sentence
    if not date_fact:
        return ""
    parts = [date_fact]
    if price_fact:
        parts.append(price_fact)
    if included_fact:
        parts.append(included_fact)
    return " ".join(parts) + " По наличию мест менеджер сверит актуальную группу."


def _compose_price_plus_format(contract: AnswerContract, retrieval: RetrievalResult, *, current_draft: str = "") -> str:
    if not (_asks_price(contract) or _asks_training_format_choice(contract)):
        return ""
    if _contract_mentions_camp_or_lvsh(contract):
        camp_facts = _camp_or_lvsh_facts(retrieval.facts, contract=contract)
        if not camp_facts:
            return ""
        price = _direct_price_answer_from_facts(contract, camp_facts)
        format_answer = _direct_camp_format_answer_from_facts(contract, camp_facts)
        if price and format_answer:
            return f"{price} {format_answer}"
        return price or format_answer
    scoped_facts = _scope_matched_facts_for_contract(contract, retrieval)
    price = _direct_price_answer_from_facts(contract, scoped_facts)
    if not price:
        return ""
    if _answer_cites_fact(current_draft, " ".join(scoped_facts.values())):
        return ""
    format_answer = _direct_format_answer_from_facts(contract, scoped_facts)
    if format_answer:
        return f"{price} {format_answer}"
    return price


def _compose_installment_summary(contract: AnswerContract, retrieval: RetrievalResult, *, current_draft: str = "") -> str:
    targets = _payment_method_target_anchors(contract)
    text = _contract_intent_text(contract)
    if not targets and not re.search(r"рассроч|частями|оплат", text, re.I):
        return ""
    if _is_existence_yes_no_contract(contract) and _answer_cites_fact(current_draft, " ".join(retrieval.facts.values())):
        return ""
    payment = _direct_payment_answer_from_facts(contract, retrieval.facts)
    if payment:
        return payment
    installment_facts: list[str] = []
    for key, value in retrieval.facts.items():
        combined = f"{key} {value}".casefold().replace("ё", "е")
        if re.search(r"рассроч|частями|долями|т-банк|t-банк", combined, re.I):
            installment_facts.append(_short_fact_sentence(str(value or ""), max_chars=220))
    if not installment_facts:
        return ""
    return "По подтверждённым вариантам оплаты: " + " ".join(dict.fromkeys(installment_facts[:2]))


def _requested_subject_count(text: str) -> int:
    low = str(text or "").casefold().replace("ё", "е")
    if not re.search(r"предмет", low, re.I):
        return 0
    number_words = {"два": 2, "две": 2, "три": 3, "четыре": 4}
    for word, value in number_words.items():
        if re.search(rf"\b{word}\b", low, re.I):
            return value
    ordinal_stems = {"втор": 2, "трет": 3, "четверт": 4}
    for stem, value in ordinal_stems.items():
        if re.search(rf"\b{stem}\w*\s+предмет", low, re.I):
            return value
    ordinal_match = re.search(r"\b([2-4])\s*[-–]?\s*(?:й|ии|ий|ой|го|му|м)?\s+предмет", low, re.I)
    if ordinal_match:
        return int(ordinal_match.group(1))
    match = re.search(r"\b([2-4])\s*(?:предмет|курс)", low, re.I)
    if match:
        return int(match.group(1))
    if re.search(r"втор\w+\s+предмет|2-?й\s+предмет", low, re.I):
        return 2
    return 0


def _price_for_composition(contract: AnswerContract, facts: Mapping[str, str]) -> int | None:
    preferred_period = "year" if re.search(r"\bгод\b|year", _contract_intent_text(contract), re.I) else ""
    preferred_format = "online" if re.search(r"онлайн|online", _contract_intent_text(contract), re.I) else ""
    if not preferred_format and re.search(r"очно|очная|очный|offline", _contract_intent_text(contract), re.I):
        preferred_format = "offline"
    candidates: list[tuple[int, int]] = []
    for key, value in facts.items():
        combined = f"{key} {value}".casefold().replace("ё", "е")
        if "₽" not in combined and "руб" not in combined:
            continue
        if "discount" in combined or "скидк" in combined:
            continue
        score = 0
        if preferred_period and (preferred_period in combined or "год" in combined):
            score += 3
        if preferred_format == "online" and re.search(r"онлайн|online", combined, re.I):
            score += 2
        if preferred_format == "offline" and re.search(r"очно|очная|очный|offline", combined, re.I):
            score += 2
        amount = _first_money_amount(value)
        if amount:
            candidates.append((score, amount))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _second_subject_discount_pct(contract: AnswerContract, facts: Mapping[str, str]) -> int | None:
    preferred_format = "online" if re.search(r"онлайн|online", _contract_intent_text(contract), re.I) else ""
    if not preferred_format and re.search(r"очно|очная|очный|offline", _contract_intent_text(contract), re.I):
        preferred_format = "offline"
    candidates: list[tuple[int, int]] = []
    for key, value in facts.items():
        combined = f"{key} {value}".casefold().replace("ё", "е")
        if not re.search(
            r"втор\w+\s+предмет|последующ\w+\s+предмет|2-?й\s+предмет|second[_\s-]?subject",
            combined,
            re.I,
        ):
            continue
        match = re.search(r"\b(\d{1,2})\s*%", combined)
        if not match:
            continue
        score = 0
        if preferred_format == "online" and re.search(r"онлайн|online", combined, re.I):
            score += 2
        if preferred_format == "offline" and re.search(r"очно|очная|очный|offline", combined, re.I):
            score += 2
        candidates.append((score, int(match.group(1))))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _first_money_amount(text: str) -> int | None:
    match = re.search(r"\d[\d\s\u00a0]{2,}\s*(?:₽|руб(?:\.|лей|ля|ль)?|р\.)", str(text or ""), re.I)
    if not match:
        return None
    digits = re.sub(r"\D", "", match.group(0))
    return int(digits) if digits else None


def _format_rub(value: int) -> str:
    return f"{int(value):,}".replace(",", " ") + " ₽"


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


def _p0_handoff_text(
    contract: AnswerContract,
    *,
    conversation: Sequence[Mapping[str, str]] | None = None,
) -> str:
    reason = f"{contract.p0_reason} {contract.client_state}".casefold().replace("ё", "е")
    if "complaint" in reason or "жалоб" in reason:
        return _complaint_handoff_text(conversation=conversation)
    return _dry_p0_text(conversation=conversation)


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


_COMPLAINT_HANDOFF_TEXTS: tuple[str, ...] = (
    "Понимаю, что ситуация неприятная, и хочу, чтобы её разобрали внимательно. "
    "Передам менеджеру — он свяжется с вами и поможет.",
    "Спасибо, что написали. Такую ситуацию правильнее разобрать с менеджером — "
    "передам ему, он свяжется и во всём разберётся.",
    "Понимаю вас. Чтобы решить вопрос по существу, передам менеджеру — "
    "он свяжется с вами напрямую.",
)


def _complaint_handoff_text(*, conversation: Sequence[Mapping[str, str]] | None = None) -> str:
    bot_turns = 0
    if conversation:
        bot_turns = sum(1 for item in conversation if str(item.get("role") or "") == "bot")
    return _COMPLAINT_HANDOFF_TEXTS[bot_turns % len(_COMPLAINT_HANDOFF_TEXTS)]


def _safe_fallback_text(
    contract: AnswerContract,
    *,
    facts: Mapping[str, str] | None = None,
    context: Mapping[str, Any] | None = None,
) -> str:
    def traced(text: str, reason: str) -> str:
        trace_event(
            context,
            "_safe_fallback_text",
            {
                "reason": reason,
                "current_question": contract.current_question,
                "answerability": contract.answerability,
                "fact_keys": list((facts or {}).keys()),
                "text": text,
            },
        )
        return text

    safety = classify_answer_safety(
        client_message=contract.current_question or "",
        context=context,
        route="manager_only",
    )
    if safety.zero_collect_required:
        if safety.primary_risk == "complaint":
            return traced(_complaint_handoff_text(), "complaint_zero_collect")
        if safety.primary_risk == "refund":
            return traced(_refund_policy_handoff_text(), "refund_zero_collect")
        return traced(
            "Сейчас точно ответить не могу. Передам вопрос менеджеру — он свяжется с вами.",
            "p0_zero_collect",
        )

    known_absence = _known_absence_text(contract, facts or {})
    if known_absence:
        return traced(known_absence, "known_absence")
    presale_refund = _presale_refund_policy_text(facts or {})
    if presale_refund and _asks_refund_policy(contract) and not _dialogue_had_hard_p0_claim(context):
        return traced(presale_refund, "presale_refund")
    soft_weekend = _soft_weekend_guidance_text(facts or {})
    if soft_weekend and _asks_weekend_or_slot(contract):
        return traced(
            "По общему ориентиру бывают разные варианты слотов, в том числе по выходным. "
            "Но точное расписание конкретной группы без проверки не подтверждаю — менеджер сверит ваш класс, предмет и площадку.",
            "soft_weekend",
        )
    schedule_publication = _class_schedule_publication_answer(contract, facts or {})
    if schedule_publication:
        return traced(schedule_publication, "schedule_publication")
    if _asks_refund_policy(contract):
        return traced(_refund_policy_handoff_text(), "refund_policy_handoff")
    detail = _client_safe_question_detail(contract.current_question)
    useful = _useful_handoff_text(contract, facts or {}, context=context)
    if useful:
        return traced(useful, "useful_handoff")
    secondary = _partial_orientation_text(contract, facts or {})
    if secondary:
        detail_part = f": {detail}" if detail else ""
        return traced(
            f"Из подтверждённого: {secondary} "
            f"По спрошенной детали менеджер сверит точный ответ{detail_part} и вернётся к вам.",
            "secondary_fact",
        )
    if detail:
        return traced(_detail_handoff_text(detail), "question_detail")
    return traced(_generic_handoff_text(), "generic")


def _useful_handoff_text(
    contract: AnswerContract,
    facts: Mapping[str, str],
    *,
    context: Mapping[str, Any] | None,
) -> str:
    if not quality_useful_handoff_enabled(context):
        return ""
    if contract.is_p0 or _asks_refund_policy(contract):
        return ""
    orientation = _partial_orientation_text(contract, facts)
    if not orientation:
        return ""
    open_point = _handoff_open_point_label(contract)
    return (
        f"Из подтверждённого: {orientation} "
        f"{open_point} менеджер сверит по актуальным данным и вернётся к вам."
    )


def _handoff_open_point_label(contract: AnswerContract) -> str:
    text = _contract_intent_text(contract).casefold().replace("ё", "е")
    if re.search(r"цен|стоим|стоит|сколько\s+стоит|оплат|руб|₽|price", text, re.I):
        return "По цене или условиям именно нужного варианта"
    if re.search(r"распис|дни|когда|старт|выходн|будн", text, re.I):
        return "По расписанию или старту конкретной группы"
    if re.search(r"формат|онлайн|очно", text, re.I):
        return "По формату именно для вашего варианта"
    if re.search(r"адрес|площадк|где\s+вы|куда\s+ехать", text, re.I):
        return "По площадке для вашего варианта"
    return "По открытому пункту"


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
    label = _question_detail_topic_label(text)
    if label:
        return label
    if _looks_like_raw_question_detail(text):
        return ""
    if len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "…"
    return text


def _question_detail_topic_label(value: str) -> str:
    text = str(value or "").casefold().replace("ё", "е")
    if not text:
        return ""
    if re.search(r"сын|дочк|дочь|реб[её]н|школьник|ученик|справит|потянет|пробел|уровен|индивидуальн|подойд[её]т\s+ли", text, re.I):
        return "индивидуальную ситуацию ребёнка"
    if re.search(r"прям\w*\s+перевод|помесячн\w*[^.?!]{0,40}(?:счет|счёт)|(?:счет|счёт)[^.?!]{0,40}перевод", text, re.I):
        return "оплату прямым переводом на счёт"
    if re.search(r"цен|стоим|сколько\s+стоит|оплат|счет|счёт|руб|₽|тариф|рассроч|долями", text, re.I):
        return "цену или условия оплаты"
    if re.search(r"распис|дни|когда|старт|выходн|будн|время|во\s+сколько", text, re.I):
        return "расписание или старт конкретной группы"
    if re.search(r"формат|онлайн|очно|дистанц", text, re.I):
        return "формат занятий"
    if re.search(r"адрес|площадк|где\s+вы|куда\s+ехать|дорог|доехать|добират|маршрут|метро|электрич", text, re.I):
        return "дорогу или площадку"
    if re.search(r"маткап|материнск|сфр|налог|вычет|фнс|документ|справк|договор", text, re.I):
        return "документы или порядок оформления"
    if re.search(r"пробн|фрагмент", text, re.I):
        return "пробный формат или фрагмент занятия"
    if re.search(r"лагер|лвш|смен|мест[ао]\b", text, re.I):
        return "смену или условия лагеря"
    if re.search(r"запис|оформ|поступить|заявк", text, re.I):
        return "порядок записи"
    return ""


def _looks_like_raw_question_detail(value: str) -> bool:
    text = " ".join(str(value or "").split())
    low = text.casefold().replace("ё", "е")
    if len(text) > 70:
        return True
    return bool(re.search(r"\b(?:можно|сможет|есть|будет|подойдет|подойд[её]т|получится|стоит|сколько|когда|как|где|почему|нужно)\s+ли\b|\?$", low, re.I))


def _secondary_fact_text(contract: AnswerContract, facts: Mapping[str, str]) -> str:
    if not facts:
        return ""
    payment_targets = _payment_method_target_anchors(contract)
    if payment_targets:
        for key, text in facts.items():
            if _is_camp_or_lvsh_fact(key, str(text or "")) and not _contract_mentions_camp_or_lvsh(contract):
                continue
            if not _fact_scope_matches_question(
                contract,
                Subquestion(text=contract.current_question, answerable="self", needed_fact_keys=()),
                str(key),
                str(text or ""),
            ):
                continue
            fact_anchors = _payment_method_anchors_from_text(str(text or ""))
            if fact_anchors and payment_targets.issubset(fact_anchors):
                return _short_fact_sentence(str(text or ""))
    return ""


def _partial_orientation_text(contract: AnswerContract, facts: Mapping[str, str]) -> str:
    secondary = _secondary_fact_text(contract, facts)
    if secondary:
        return secondary
    if not facts:
        return ""
    if _payment_method_target_anchors(contract):
        return ""
    if contract.is_p0 or _asks_refund_policy(contract):
        return ""
    active_brand = str(contract.active_brand or "").casefold()
    for key, text in facts.items():
        value = str(text or "").strip()
        if not value:
            continue
        if not _fact_is_safe_partial_orientation(contract, str(key), value, active_brand=active_brand):
            continue
        return _short_fact_sentence(value)
    return ""


def _fact_is_safe_partial_orientation(
    contract: AnswerContract,
    key: str,
    text: str,
    *,
    active_brand: str,
) -> bool:
    combined = f"{key} {text}".casefold().replace("ё", "е")
    if active_brand:
        for token in _BRAND_TOKENS.get(active_brand, ()):
            if token and token in combined:
                return False
    asks_camp = _contract_mentions_camp_or_lvsh(contract)
    is_camp_fact = _is_camp_or_lvsh_fact(key, text)
    if asks_camp and not is_camp_fact:
        return False
    if is_camp_fact and not asks_camp:
        return False
    if not _fact_scope_matches_question(
        contract,
        Subquestion(text=contract.current_question, answerable="self", needed_fact_keys=()),
        key,
        text,
    ):
        return False
    if _asks_class_schedule_days(contract) and _is_contact_hours_fact(key, text):
        return False
    if _is_address_fact(key, text) and not _asks_address(contract):
        return False
    return True


def _is_contact_hours_fact(key: str, text: str) -> bool:
    combined = f"{key} {text}".casefold().replace("ё", "е")
    return bool(
        re.search(r"contact|contacts|режим|график|на\s+связи|10[:.]?00|18[:.]?00|пн\s*[–-]\s*вс|ежедневн", combined, re.I)
    )


def _is_address_fact(key: str, text: str) -> bool:
    combined = f"{key} {text}".casefold().replace("ё", "е")
    return bool(re.search(r"address|addresses|metro|location|адрес|метро|сретенк|скорняжн|москва|чистые\s+пруды", combined, re.I))


def _generic_handoff_text() -> str:
    return _GENERIC_HANDOFF_TEXTS[0]


def _detail_handoff_text(detail: str) -> str:
    clean = _client_safe_question_detail(detail) or "эту деталь"
    return _DETAIL_HANDOFF_TEXTS[0].format(detail=clean)


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
    if contract.is_p0:
        return _select_unused_handoff_variant(_DRY_P0_TEXTS, prior_bot_texts, fallback=source)
    if _is_complaint_handoff_text(source):
        return _select_unused_handoff_variant(_COMPLAINT_HANDOFF_TEXTS, prior_bot_texts, fallback=source)
    if _asks_refund_policy(contract) and _presale_refund_policy_text(facts or {}):
        fact = _presale_refund_policy_text(facts or {})
        return (
            f"По возврату ориентир тот же: {_short_fact_sentence(fact)} "
            "Точные пункты договора менеджер подтвердит по выбранному курсу."
        )
    if _is_refund_handoff_text(source) or _asks_refund_policy(contract):
        return _select_unused_handoff_variant(_REFUND_POLICY_TEXTS, prior_bot_texts, fallback=source)
    detail = _client_safe_question_detail(contract.current_question) or "эту деталь"
    rendered = tuple(item.format(detail=detail) for item in (*_DETAIL_HANDOFF_TEXTS, *_GENERIC_HANDOFF_TEXTS))
    exhausted = _select_unused_handoff_variant(
        _HANDOFF_EXHAUSTED_TEXTS,
        prior_bot_texts,
        fallback=_HANDOFF_EXHAUSTED_TEXTS[-1],
    )
    return _select_unused_handoff_variant(rendered, prior_bot_texts, fallback=exhausted)


def _select_unused_handoff_variant(
    variants: Sequence[str],
    prior_bot_texts: Sequence[str],
    *,
    fallback: str,
) -> str:
    for candidate in variants:
        text = str(candidate or "").strip()
        if text and not any(_near_repeat(text, prior) for prior in prior_bot_texts):
            return text
    return fallback


def _is_refund_handoff_text(text: str) -> bool:
    low = str(text or "").casefold().replace("ё", "е")
    return "возврат" in low or "отмен" in low


def _is_complaint_handoff_text(text: str) -> bool:
    low = str(text or "").casefold().replace("ё", "е")
    return "ситуац" in low and ("неприят" in low or "разбер" in low)


def _is_handoff_text(text: str) -> bool:
    low = str(text or "").casefold()
    return bool(re.search(r"менеджер|передам|уточнит|подтвердит|сверит", low, re.I))


def _looks_like_handoff(text: str) -> bool:
    low = str(text or "").casefold().replace("ё", "е")
    if _is_handoff_text(low):
        return True
    return bool(
        re.search(
            r"спасибо\s+за\s+сообщение|не\s+могу\s+точно\s+ответить|нет\s+точн(?:ой|ых)\s+(?:информации|данных)|"
            r"верн[её]тся\s+с\s+проверенн|свяжется\s+с\s+проверенн|уточн[ию]\s+и\s+верн",
            low,
            re.I,
        )
    )


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
    if _contract_mentions_camp_or_lvsh(contract):
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
    context: Mapping[str, Any] | None = None,
) -> RetrievalResult:
    if not _asks_refund_policy(contract):
        return retrieval
    if _dialogue_had_hard_p0_claim(context):
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


def _scope_camp_retrieval_for_contract(
    retrieval: RetrievalResult,
    *,
    contract: AnswerContract,
    context: Mapping[str, Any] | None,
) -> RetrievalResult:
    if not quality_thread_memory_enabled(context):
        return retrieval
    requested_scope = _camp_scope_from_contract(contract)
    if not requested_scope:
        return retrieval
    facts: dict[str, str] = {}
    removed: list[str] = []
    for key, value in retrieval.facts.items():
        fact_scope = _camp_scope_from_fact(str(key), str(value or ""))
        if fact_scope and fact_scope != requested_scope:
            removed.append(str(key))
            continue
        facts[str(key)] = str(value or "")
    if not removed:
        return retrieval
    matched: dict[str, tuple[str, ...]] = {}
    missing = list(retrieval.missing)
    for required, keys in retrieval.matched_keys.items():
        kept = tuple(key for key in keys if key in facts)
        if kept:
            matched[str(required)] = kept
        elif str(required) not in missing:
            missing.append(str(required))
    trace_event(
        context,
        "thread_memory_camp_scope_filter",
        {"requested_scope": requested_scope, "removed_fact_keys": removed},
    )
    return RetrievalResult(facts=facts, missing=tuple(dict.fromkeys(missing)), matched_keys=matched)


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


def _camp_or_lvsh_facts(facts: Mapping[str, str], *, contract: AnswerContract | None = None) -> dict[str, str]:
    requested_scope = _camp_scope_from_contract(contract) if contract is not None else ""
    selected: dict[str, str] = {}
    for key, value in facts.items():
        text = str(value or "")
        if not _is_camp_or_lvsh_fact(str(key), text):
            continue
        fact_scope = _camp_scope_from_fact(str(key), text)
        if requested_scope and fact_scope and fact_scope != requested_scope:
            continue
        selected[str(key)] = text
    return selected


def _is_camp_or_lvsh_fact(key: str, text: str) -> bool:
    combined = f"{key} {text}".casefold().replace("ё", "е")
    return bool(re.search(r"лвш|lvsh|менделеев|лагер|camp", combined, re.I))


def _contract_mentions_camp_or_lvsh(contract: AnswerContract) -> bool:
    return bool(re.search(r"лвш|менделеев|лагер|camp|летн", _contract_intent_text(contract), re.I))


def _camp_scope_from_contract(contract: AnswerContract | None) -> str:
    if contract is None:
        return ""
    return _camp_scope_from_text(_contract_intent_text(contract))


def _camp_scope_from_text(text: str) -> str:
    low = str(text or "").casefold().replace("ё", "е")
    residential = bool(
        re.search(r"лвш|lvsh|менделеев|выездн|трансфер|с\s+прожив", low, re.I)
        or (re.search(r"прожив", low, re.I) and not re.search(r"без\s+прожив|без\s+ночев", low, re.I))
    )
    city = bool(re.search(r"city_day_camp|city_camp|городск|дневн|без\s+прожив|без\s+ночев|лш\s+москв", low, re.I))
    if residential and not city:
        return "residential_lvsh"
    if city and not residential:
        return "city_day_camp"
    return ""


def _camp_scope_from_fact(key: str, text: str) -> str:
    return _camp_scope_from_text(f"{key} {text}")


def _has_self_answerable_subquestion(contract: AnswerContract) -> bool:
    return any(item.answerable == "self" for item in contract.subquestions)


def _has_retrieved_self_answer_part(contract: AnswerContract, retrieval: RetrievalResult) -> bool:
    for subquestion in contract.subquestions:
        if subquestion.answerable != "self":
            continue
        keys = tuple(key for key in subquestion.needed_fact_keys if key)
        if not keys:
            continue
        if all(
            key not in retrieval.missing and _matched_scope_fact_keys_for_required_key(contract, subquestion, retrieval, key)
            for key in keys
        ):
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
    facts = _scope_matched_facts_for_contract(contract, retrieval, include_manager=True)
    if not facts:
        return ""
    if _asks_address(contract):
        address = _first_address_from_facts(facts)
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
    payment = _direct_payment_answer_from_facts(contract, facts)
    if payment:
        return payment
    return ""


def _hard_failure_exact_fact_fallback(contract: AnswerContract, retrieval: RetrievalResult) -> str:
    if not _has_exact_retrieved_answer_part(contract, retrieval):
        return ""
    facts = _scope_matched_facts_for_contract(contract, retrieval, include_manager=True)
    if not facts:
        return ""
    price = _direct_price_answer_from_facts(contract, facts)
    if price:
        return price
    format_answer = _direct_format_answer_from_facts(contract, facts)
    if format_answer:
        return format_answer
    recording = _direct_recording_answer_from_facts(contract, facts)
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
    if _contract_mentions_camp_or_lvsh(contract):
        facts = _camp_or_lvsh_facts(facts, contract=contract)
        if not facts:
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
    if _contract_mentions_camp_or_lvsh(contract):
        return _direct_camp_format_answer_from_facts(contract, facts)
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


def _direct_camp_format_answer_from_facts(contract: AnswerContract, facts: Mapping[str, str]) -> str:
    if not _contract_mentions_camp_or_lvsh(contract):
        return ""
    text = _contract_intent_text(contract)
    if not re.search(r"формат|очно|онлайн|прожив|дневн|ночев", text, re.I):
        return ""
    for key, value in _camp_or_lvsh_facts(facts, contract=contract).items():
        combined = f"{key} {value}".casefold().replace("ё", "е")
        if not re.search(r"без\s+прожив|дневн|очная\s+городск|городск\w+\s+школ|городск\w+\s+лагер", combined, re.I):
            continue
        fact = _short_fact_sentence(str(value or ""), max_chars=220)
        if fact:
            return f"По лагерной смене подтверждено: {fact}"
    return ""


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
    return bool(re.search(r"адрес|площадк|где\s+вы|где\s+находит|куда\s+ехать|куда\s+ездить", text, re.I))


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
    matched_text = _matched_scope_fact_text_for_required_keys(contract, subquestion, retrieval, keys)
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
    has_camp_fact = any(_is_camp_or_lvsh_fact(key, value) for key, value in matched_text.items())
    if _contract_mentions_camp_or_lvsh(contract):
        return has_camp_fact
    if has_camp_fact:
        return False
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


def _matched_scope_fact_text_for_required_keys(
    contract: AnswerContract,
    subquestion: Subquestion,
    retrieval: RetrievalResult,
    keys: Sequence[str],
) -> dict[str, str]:
    result: dict[str, str] = {}
    for required in keys:
        for key in _matched_scope_fact_keys_for_required_key(contract, subquestion, retrieval, required):
            result[key] = str(retrieval.facts[key])
    return result


def _matched_scope_fact_keys_for_required_key(
    contract: AnswerContract,
    subquestion: Subquestion,
    retrieval: RetrievalResult,
    required_key: str,
) -> tuple[str, ...]:
    return tuple(
        key
        for key in retrieval.matched_keys.get(required_key, ())
        if key in retrieval.facts and _fact_scope_matches_question(contract, subquestion, key, str(retrieval.facts[key]))
    )


def _scope_matched_facts_for_contract(
    contract: AnswerContract,
    retrieval: RetrievalResult,
    *,
    include_manager: bool = False,
) -> dict[str, str]:
    scoped: dict[str, str] = {}
    for subquestion in _contract_subquestions(contract):
        if not include_manager and subquestion.answerable != "self":
            continue
        for required_key in tuple(key for key in subquestion.needed_fact_keys if key):
            for fact_key in _matched_scope_fact_keys_for_required_key(contract, subquestion, retrieval, required_key):
                scoped[fact_key] = str(retrieval.facts[fact_key])
    return scoped


def _has_foreign_brand_matched_self_fact(contract: AnswerContract, retrieval: RetrievalResult) -> bool:
    active_brand = _normalize_brand(contract.active_brand)
    tokens = tuple(token for token in _BRAND_TOKENS.get(active_brand, ()) if token)
    if not tokens:
        return False
    for subquestion in _contract_subquestions(contract):
        if subquestion.answerable != "self":
            continue
        for required_key in tuple(key for key in subquestion.needed_fact_keys if key):
            for fact_key in retrieval.matched_keys.get(required_key, ()):
                if fact_key not in retrieval.facts:
                    continue
                combined = f"{fact_key} {retrieval.facts[fact_key]}".casefold().replace("ё", "е")
                if any(token in combined for token in tokens):
                    return True
    return False


def _fact_scope_matches_question(
    contract: AnswerContract,
    subquestion: Subquestion,
    fact_key: str,
    fact_text: str,
) -> bool:
    combined = f"{fact_key} {fact_text}".casefold().replace("ё", "е")
    active_brand = _normalize_brand(contract.active_brand)
    for token in _BRAND_TOKENS.get(active_brand, ()):
        if token and token in combined:
            return False

    question_text = _subquestion_scope_text(contract, subquestion)
    requested_formats = _format_values_from_text(question_text)
    fact_formats = _format_values_from_text(combined)
    if requested_formats and fact_formats and requested_formats.isdisjoint(fact_formats):
        return False

    requested_grade = _grade_from_text(question_text)
    fact_grades = _grade_values_from_fact_scope(fact_key, fact_text)
    if requested_grade and fact_grades and requested_grade not in fact_grades:
        return False
    return True


def _format_values_from_text(text: str) -> set[str]:
    low = str(text or "").casefold().replace("ё", "е")
    values: set[str] = set()
    if re.search(r"онлайн|online|дистанционно", low, re.I):
        values.add("онлайн")
    if re.search(r"очно|очная|очный|офлайн|offline|ochno", low, re.I):
        values.add("очно")
    return values


def _grade_values_from_fact_scope(fact_key: str, fact_text: str) -> set[str]:
    key_source = str(fact_key or "").casefold().replace("ё", "е")
    source = f"{fact_key} {fact_text}".casefold().replace("ё", "е")
    grades: set[int] = set()
    for match in re.finditer(
        r"(?<!\d)([1-9]|10|11)\s*[-–_]\s*([1-9]|10|11)\s*(?:класс(?:а|е|ов|ы)?|кл\.?|grade|class|grades|classes)",
        source,
        re.I,
    ):
        low, high = int(match.group(1)), int(match.group(2))
        if low > high:
            low, high = high, low
        if 1 <= low <= 11 and 1 <= high <= 11:
            grades.update(range(low, high + 1))
    if re.search(r"price|prices|course|courses|grade|class|tuition|цена|стоим|курс", key_source, re.I):
        for match in re.finditer(r"(?<!\d)([1-9]|10|11)[._-]([1-9]|10|11)(?!\d)", key_source, re.I):
            low, high = int(match.group(1)), int(match.group(2))
            if low > high:
                low, high = high, low
            if 1 <= low <= 11 and 1 <= high <= 11:
                grades.update(range(low, high + 1))
    for match in re.finditer(r"(?<!\d)(?:grade|class|klass)[_.\s-]?([1-9]|10|11)(?!\d)", source, re.I):
        grades.add(int(match.group(1)))
    for match in re.finditer(r"(?<!\d)([1-9]|10|11)\s*(?:класс(?:а|е|ов|ы)?|кл\.?|class)(?!\d)", source, re.I):
        grades.add(int(match.group(1)))
    return {str(item) for item in sorted(grades)}


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


def _dialogue_had_hard_p0_claim(context: Mapping[str, Any] | None) -> bool:
    if not isinstance(context, Mapping):
        return False
    sources: list[Any] = []
    for key in ("dialogue_memory_view", "dialogue_memory"):
        value = context.get(key)
        if isinstance(value, Mapping):
            sources.append(value)
    p0_latch = context.get("p0_latch")
    if isinstance(p0_latch, Mapping):
        sources.append({"p0_latch": p0_latch})
    hard_codes = {"refund", "payment_dispute", "legal", "legal_threat", "complaint"}
    for source in sources:
        latch = source.get("p0_latch") if isinstance(source, Mapping) else None
        if isinstance(latch, Mapping):
            if bool(latch.get("had_hard_p0_claim")):
                return True
            codes = {str(item or "").strip() for item in (latch.get("codes") or ())}
            if codes.intersection(hard_codes):
                return True
        risk_flags = {str(item or "").strip() for item in (source.get("risk_flags") or ())} if isinstance(source, Mapping) else set()
        if risk_flags.intersection(hard_codes):
            return True
    return False


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


def _scope_clarification_question(
    contract: AnswerContract,
    retrieval: RetrievalResult,
    *,
    client_words: str,
    context: Mapping[str, Any] | None,
) -> str:
    if not quality_clarify_scope_enabled(context):
        return ""
    if _cite_only_recover_blocked(contract, client_words=client_words, context=context):
        return ""
    if _has_foreign_brand_matched_self_fact(contract, retrieval):
        return ""
    if not retrieval.facts or contract.answerability != "answer_self":
        return ""
    if not (_asks_price(contract) or _asks_class_schedule_days(contract)):
        return ""

    question_text = _contract_intent_text(contract)
    requested_formats = _format_values_from_text(question_text)
    available_formats = _format_values_from_facts(retrieval.facts)
    if not requested_formats and len(available_formats) > 1:
        return "Уточните, пожалуйста, какой формат нужен: онлайн или очно?"

    requested_grade = _grade_from_text(question_text)
    available_grades = _grade_values_from_retrieved_facts(retrieval.facts)
    if not requested_grade and len(available_grades) > 1:
        return "Уточните, пожалуйста, для какого класса нужен вариант?"
    return ""


def _format_values_from_facts(facts: Mapping[str, str]) -> set[str]:
    values: set[str] = set()
    for key, text in facts.items():
        values.update(_format_values_from_text(f"{key} {text}"))
    return values


def _grade_values_from_retrieved_facts(facts: Mapping[str, str]) -> set[str]:
    values: set[str] = set()
    for key, text in facts.items():
        values.update(_grade_values_from_fact_scope(str(key), str(text or "")))
    return values


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
