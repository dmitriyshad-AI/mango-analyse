"""Deterministic safety floor shared by the live direct path and legacy compatibility."""

from __future__ import annotations

import difflib
import json
import os
import re
from collections.abc import Mapping as MappingABC, Sequence as SequenceABC
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from mango_mvp.channels.answer_safety_classifier import classify_answer_safety
from mango_mvp.channels.dialogue_debug_trace import trace_event
from mango_mvp.channels.fact_venue_scope import (
    VENUE_SCOPE_VALUES,
    fact_venue,
    normalize_requested_scope,
    venue_scope_enabled,
    venue_scope_label,
)
from mango_mvp.channels.fact_scope_spec import blocked_neighbors_for, detect_fact_scopes, fact_scopes_allowed
from mango_mvp.channels.p0_recall_spec import hard_codes_from_text, is_benign_hypothetical_refund, soft_codes_from_text
from mango_mvp.insights.sanitizers import sanitize_answer


FREE_NUMBER_GATE_ENV = "TELEGRAM_A_FREE_NUMBER_GATE"


NUMBER_GATE_SCOPE_AWARE_ENV = "TELEGRAM_NUMBER_GATE_SCOPE_AWARE"


DIRECT_PATH_PILOT_CONFIG_ENV = "TELEGRAM_DIRECT_PATH_PILOT_CONFIG"


DIRECT_PATH_PILOT_CONFIG_VERSION = "pilot_gold_v1"


STEP4_NUMBER_GROUNDING_ENV = "TELEGRAM_STEP4_NUMBER_GROUNDING"


AUTONOMY_SCOPE_PRECISION_ENV = "TELEGRAM_AUTONOMY_SCOPE_PRECISION"


AUTHORITATIVE_GATE_SCOPE_RELEVANCE_FIX_ENV = "TELEGRAM_AUTHORITATIVE_GATE_SCOPE_RELEVANCE_FIX"


DIALOGUE_CONTRACT_SCHEMA_VERSION = "dialogue_contract_v2_2026_05_26"


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


_PRODUCT_NUMBER_CTX_RE = re.compile(
    r"₽|руб|р\.|%|скидк|рассрочк|долями|\b\d{1,2}:\d{2}\b|семестр|за\s+год|"
    r"мес|месяц|плат[её]ж|на\s+\d+\s+част|в\s+рассрочку\s+на|"
    r"стоит|цена|тариф|сколько\s+длится|длительност|урок|занят|январ|феврал|март|"
    r"апрел|ма[яй]|июн|июл|август|сентяб|октяб|ноябр|декабр|смен[аеуы]",
    re.I,
)


_FREE_NUMBER_PRODUCT_CTX_RE = re.compile(
    r"₽|руб|р\.|%|процент|скидк|рассрочк|долями|цена|стоит|стоимост|тариф|семестр|за\s+год|оплат|"
    r"мес|месяц|плат[её]ж|на\s+\d+\s+част|в\s+рассрочку\s+на|"
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
    r"\b\d+(?:[.,]\d+)?\s*[-–]\s*\d+(?:[.,]\d+)?(?:\s*(?:км|километр(?:а|ов)?|минут(?:ы|у)?|час(?:а|ов)?|год(?:а)?|лет|мес(?:\.|яц(?:ев|а)?)?|недел(?:и|ь)?|заняти(?:й|я)|балл(?:ов|а)?|процент(?:ов|а)?|%))?|"
    r"\b\d{1,2}[./]\d{1,2}(?:[./]\d{2,4})?\b|"
    r"\b\d[\d\s\u00a0]*(?:[.,]\d+)?\s*(?:к|тыс\.?|тысяч|₽|руб(?:\.|лей|ля|ль)?|р\.|процент(?:ов|а)?|%|минут(?:ы|у)?|час(?:а|ов)?|км|километр(?:а|ов)?|год(?:а)?|лет|мес(?:\.|яц(?:ев|а)?)?|недел(?:и|ь)?|заняти(?:й|я)|балл(?:ов|а)?|ак\.?\s*ч(?:\.|аса|асов)?|раз(?:а)?)?\b",
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


_PAYMENT_PLAN_COUNT_PREFIX = "payment_plan_count:"


_PAYMENT_PLAN_COUNT_RE = re.compile(
    r"(?<!\d)((?:\d{1,2}\s*(?:,|/|и|или|[-–])\s*)*\d{1,2})\s*"
    r"(?:мес\.?|месяц(?:ев|а)?|плат[её]ж(?:ей|а)?|част(?:ей|и|ями)?)",
    re.I,
)


_PAYMENT_PLAN_CONTEXT_RE = re.compile(
    r"мес\.?|месяц(?:ев|а)?|плат[её]ж(?:ей|а)?|част(?:ей|и|ями)?|рассрочк|долями",
    re.I,
)


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


_HANDOFF_FACTUAL_CLAIM_RE = re.compile(
    r"\b(?:обычно|как\s+правило|в\s+основном|чаще\s+всего)\b"
    r"|(?:\bвход(?:ит|ят)\b(?!\s+ли\b)|\bвключа(?:ет|ют|ется|ются)\b)"
    r"|\bдела(?:ем|ют)\s+упор\b|\bупор\s+(?:ид[её]т|дела(?:ем|ют))\b"
    r"|\bзаняти[яе]\s+(?:проход(?:ит|ят)|ид(?:е[тё]|ут)|дл(?:ит|ятся))\b"
    r"|\bкурс\s+(?:рассчитан|ид[её]т|подходит|включа(?:ет|ется))\b"
    r"|\bгрупп[аы]\s+(?:дел(?:ится|ятся)|есть|ид(?:е[тё]|ут))\b",
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


_GENERIC_HANDOFF_TEXTS: tuple[str, ...] = (
    "Чтобы не ошибиться, передам вопрос менеджеру — он сверит детали и вернётся с ответом.",
    "Не хочу гадать по неподтверждённому пункту: менеджер проверит его и вернётся с ответом.",
    "Здесь лучше сверить условия: передам вопрос менеджеру, он ответит по точным данным.",
    "Передам этот пункт менеджеру, чтобы он проверил его по актуальным данным и ответил вам.",
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
    runtime_error: str = ""

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
            "runtime_error": self.runtime_error,
        }


@dataclass(frozen=True)
class VerificationFinding:
    code: str
    detail: str


def free_number_gate_enabled(context: Mapping[str, Any] | None = None) -> bool:
    if isinstance(context, MappingABC):
        if context.get(STEP4_NUMBER_GROUNDING_ENV) is not None:
            if _truthy(context.get(STEP4_NUMBER_GROUNDING_ENV)):
                return True
            if context.get(FREE_NUMBER_GATE_ENV) is None:
                return False
        if context.get(FREE_NUMBER_GATE_ENV) is not None:
            return _truthy(context.get(FREE_NUMBER_GATE_ENV))
    return _truthy(os.getenv(FREE_NUMBER_GATE_ENV)) or _truthy(os.getenv(STEP4_NUMBER_GROUNDING_ENV))


def number_gate_scope_aware_enabled(context: Mapping[str, Any] | None = None) -> bool:
    if isinstance(context, MappingABC) and context.get(NUMBER_GATE_SCOPE_AWARE_ENV) is not None:
        return _truthy(context.get(NUMBER_GATE_SCOPE_AWARE_ENV))
    if NUMBER_GATE_SCOPE_AWARE_ENV in os.environ:
        return _truthy(os.getenv(NUMBER_GATE_SCOPE_AWARE_ENV))
    if isinstance(context, MappingABC):
        for key in (DIRECT_PATH_PILOT_CONFIG_ENV, "direct_path_pilot_config", "pilot_config"):
            if str(context.get(key) or "").strip() == DIRECT_PATH_PILOT_CONFIG_VERSION:
                return True
    return str(os.getenv(DIRECT_PATH_PILOT_CONFIG_ENV) or "").strip() == DIRECT_PATH_PILOT_CONFIG_VERSION


def autonomy_scope_precision_enabled(context: Mapping[str, Any] | None = None) -> bool:
    aliases = ("autonomy_scope_precision", "autonomy_scope_precision_enabled")
    if isinstance(context, MappingABC):
        for key in (AUTONOMY_SCOPE_PRECISION_ENV, *aliases):
            if key in context:
                return _truthy(context.get(key))
    if AUTONOMY_SCOPE_PRECISION_ENV in os.environ:
        return _truthy(os.getenv(AUTONOMY_SCOPE_PRECISION_ENV))
    from mango_mvp.channels.subscription_llm_parts.support import _pilot_profile_default_on_flag_enabled

    return _pilot_profile_default_on_flag_enabled(context, AUTONOMY_SCOPE_PRECISION_ENV, aliases=aliases)


def authoritative_gate_scope_relevance_fix_enabled(context: Mapping[str, Any] | None = None) -> bool:
    aliases = (
        "authoritative_gate_scope_relevance_fix",
        "authoritative_gate_scope_relevance_fix_enabled",
    )
    if isinstance(context, MappingABC):
        for key in (AUTHORITATIVE_GATE_SCOPE_RELEVANCE_FIX_ENV, *aliases):
            if key in context:
                return _truthy(context.get(key))
    if AUTHORITATIVE_GATE_SCOPE_RELEVANCE_FIX_ENV in os.environ:
        return _truthy(os.getenv(AUTHORITATIVE_GATE_SCOPE_RELEVANCE_FIX_ENV))
    return False


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
        runtime_error=str(data.get("runtime_error") or data.get("provider_runtime_error") or "").strip()[:120],
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


def _grade_from_text(text: str) -> str:
    match = re.search(r"\b([1-9]|1[01])\s*(?:класс|кл\.?|grade)?\b", str(text or "").casefold().replace("ё", "е"))
    return match.group(1) if match else ""


_ACTIVE_HARD_P0_LATCH_CODES = {"payment_dispute", "refund_claim", "legal", "legal_threat", "complaint", "p0"}


_P0_LATCH_REASON_PRIORITY = ("payment_dispute", "refund_claim", "refund", "legal", "legal_threat", "complaint", "p0")


def _p0_latch_sources(context: Mapping[str, Any] | None) -> list[Mapping[str, Any]]:
    if not isinstance(context, MappingABC):
        return []
    sources: list[Mapping[str, Any]] = []
    for key in ("dialogue_memory_view", "dialogue_memory"):
        value = context.get(key)
        if isinstance(value, MappingABC):
            sources.append(value)
    p0_latch = context.get("p0_latch")
    if isinstance(p0_latch, MappingABC):
        sources.append({"p0_latch": p0_latch})
    return sources


def _latch_is_active(latch: Mapping[str, Any]) -> bool:
    if "active" in latch:
        return bool(latch.get("active"))
    return bool(latch.get("codes") or latch.get("primary_risk"))


def _first_p0_latch_reason(codes: set[str], *, default: str = "p0") -> str:
    for code in _P0_LATCH_REASON_PRIORITY:
        if code in codes:
            return code
    return next(iter(codes), default)


def _has_presale_refund_evidence(context: Mapping[str, Any] | None, *, current_text: str = "") -> bool:
    if is_benign_hypothetical_refund(current_text):
        return True
    if not isinstance(context, MappingABC):
        return False
    plan = context.get("conversation_intent_plan")
    if isinstance(plan, MappingABC) and str(plan.get("refund_frame") or "") == "presale_policy":
        return True
    for source in _p0_latch_sources(context):
        if str(source.get("refund_frame") or "") == "presale_policy" or bool(source.get("semantic_non_p0")):
            return True
    recent = context.get("recent_messages")
    if isinstance(recent, SequenceABC) and not isinstance(recent, (str, bytes)):
        for item in recent:
            if is_benign_hypothetical_refund(str(item or "")):
                return True
    return False


def _active_hard_p0_latch_reason(context: Mapping[str, Any] | None, *, current_text: str = "") -> str:
    presale_evidence = _has_presale_refund_evidence(context, current_text=current_text)
    for source in _p0_latch_sources(context):
        latch = source.get("p0_latch")
        if isinstance(latch, MappingABC) and _latch_is_active(latch):
            codes = {str(item or "").strip() for item in (latch.get("codes") or ()) if str(item or "").strip()}
            primary = str(latch.get("primary_risk") or "").strip()
            suppress_refund_latch = _presale_refund_latch_can_release(
                codes,
                primary=primary,
                presale_evidence=presale_evidence,
                current_text=current_text,
            )
            if bool(latch.get("had_hard_p0_claim")):
                return primary or _first_p0_latch_reason(codes)
            hard = codes.intersection(_ACTIVE_HARD_P0_LATCH_CODES)
            if hard:
                return _first_p0_latch_reason(hard)
            if primary in _ACTIVE_HARD_P0_LATCH_CODES:
                return primary
            if ("refund" in codes or primary == "refund") and not presale_evidence:
                return "refund"
        risk_flags = {str(item or "").strip() for item in (source.get("risk_flags") or ()) if str(item or "").strip()}
        hard_flags = risk_flags.intersection(_ACTIVE_HARD_P0_LATCH_CODES)
        if hard_flags:
            return _first_p0_latch_reason(hard_flags)
        if "refund" in risk_flags and not presale_evidence:
            return "refund"
    return ""


def _presale_refund_latch_can_release(
    codes: set[str],
    *,
    primary: str,
    presale_evidence: bool,
    current_text: str = "",
) -> bool:
    if not presale_evidence or hard_codes_from_text(current_text):
        return False
    return codes.issubset({"refund"}) and primary in {"", "refund"}


def _has_only_benign_refund_latch(context: Mapping[str, Any] | None, *, current_text: str = "") -> bool:
    if not isinstance(context, MappingABC) or not _has_presale_refund_evidence(context, current_text=current_text):
        return False
    if _active_hard_p0_latch_reason(context, current_text=current_text):
        return False
    saw_refund_latch = False
    for source in _p0_latch_sources(context):
        latch = source.get("p0_latch")
        if isinstance(latch, MappingABC) and _latch_is_active(latch):
            if bool(latch.get("had_hard_p0_claim")):
                return False
            codes = {str(item or "").strip() for item in (latch.get("codes") or ()) if str(item or "").strip()}
            if codes.intersection(_ACTIVE_HARD_P0_LATCH_CODES):
                return False
            if "refund" in codes or str(latch.get("primary_risk") or "").strip() == "refund":
                saw_refund_latch = True
        risk_flags = {str(item or "").strip() for item in (source.get("risk_flags") or ()) if str(item or "").strip()}
        if risk_flags.intersection(_ACTIVE_HARD_P0_LATCH_CODES):
            return False
    return saw_refund_latch


def p0_pre_gate(text: str, *, context: Mapping[str, Any] | None = None) -> str | None:
    codes = hard_codes_from_text(text)
    if codes:
        result = ",".join(codes)
        trace_event(context, "p0_pre_gate", {"source": "regex", "codes": list(codes), "result": result})
        return result
    soft_codes = soft_codes_from_text(text)
    if soft_codes:
        trace_event(context, "p0_pre_gate", {"source": "regex_soft", "codes": list(soft_codes), "result": ""})
    latch_reason = _active_hard_p0_latch_reason(context, current_text=text)
    if latch_reason:
        trace_event(context, "p0_pre_gate", {"source": "active_p0_latch", "codes": [latch_reason], "result": latch_reason})
        return latch_reason
    decision = classify_answer_safety(client_message=text, context=context)
    if decision.p0_required:
        decision_codes = {str(item or "").strip() for item in (decision.risk_codes or ()) if str(item or "").strip()}
        if (
            (decision.primary_risk == "refund" or decision_codes == {"refund"})
            and _has_only_benign_refund_latch(context, current_text=text)
            and not hard_codes_from_text(text)
        ):
            trace_event(
                context,
                "p0_pre_gate",
                {"source": "classifier_suppressed_benign_refund_latch", "codes": list(decision.risk_codes or ()), "result": ""},
            )
            return None
        result = ",".join(decision.risk_codes or (decision.primary_risk or "p0",))
        trace_event(
            context,
            "p0_pre_gate",
            {"source": "classifier", "codes": list(decision.risk_codes or ()), "primary_risk": decision.primary_risk, "result": result},
        )
        return result
    trace_event(context, "p0_pre_gate", {"source": "classifier", "codes": [], "result": ""})
    return None


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
        findings.extend(_wrong_intent_fact_findings(text, contract=contract, facts=facts, context=context))
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
    scope_aware = number_gate_scope_aware_enabled(context)
    scoped_facts = _scope_aware_number_facts(facts) if scope_aware else ()
    product_tokens: list[str] = []
    wrong_scope_tokens: list[str] = []
    general_without_marker: list[str] = []
    for token, start, end in _free_number_token_matches(text):
        surfaces = _free_number_surfaces(token)
        if not surfaces:
            continue
        payment_plan_surfaces = _payment_plan_count_surfaces_for_token(text, token, start=start, end=end)
        match_surfaces = payment_plan_surfaces or surfaces
        if scope_aware:
            supported, wrong_scope_seen = _scope_aware_number_supported(
                match_surfaces,
                scoped_facts,
                text=text,
                token=token,
                start=start,
                end=end,
                client_message=client_message,
                context=context,
            )
            if supported:
                continue
            scope_product_context = _scope_aware_product_number_context(text, token, surfaces, start=start, end=end)
            if wrong_scope_seen and scope_product_context:
                wrong_scope_tokens.append(token)
                continue
        else:
            if payment_plan_surfaces:
                if fact_surfaces.intersection(payment_plan_surfaces):
                    continue
            elif fact_surfaces.intersection(surfaces):
                continue
        if _is_route_estimate_number_context(text, token, estimate_domain=estimate_domain):
            pass
        elif (
            _scope_aware_product_number_context(text, token, surfaces, start=start, end=end)
            if scope_aware
            else _is_free_product_number_context(text, token, start=start, end=end)
        ):
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
    if wrong_scope_tokens:
        findings.append(
            VerificationFinding(
                "wrong_scope",
                f"числа найдены только в фактах другого скоупа: {sorted(dict.fromkeys(wrong_scope_tokens))}",
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


@dataclass(frozen=True)
class _ScopeAwareNumberFact:
    key: str
    text: str
    surfaces: frozenset[str]
    scopes: frozenset[str]
    formats: frozenset[str]
    grades: frozenset[str]


def _scope_aware_number_facts(facts: Mapping[str, str]) -> tuple[_ScopeAwareNumberFact, ...]:
    result: list[_ScopeAwareNumberFact] = []
    for raw_key, raw_text in (facts or {}).items():
        key = str(raw_key or "").strip()
        text = str(raw_text or "").strip()
        if not key and not text:
            continue
        combined = f"{key} {text}"
        surfaces = frozenset(_free_number_surfaces(text))
        if not surfaces:
            continue
        result.append(
            _ScopeAwareNumberFact(
                key=key,
                text=text,
                surfaces=surfaces,
                scopes=frozenset(detect_fact_scopes(combined, fact_types=_number_scope_fact_types(key))),
                formats=frozenset(_format_values_from_text(combined)),
                grades=frozenset(_grade_values_from_fact_scope(key, text)),
            )
        )
    return tuple(result)


def _scope_aware_number_supported(
    surfaces: set[str],
    scoped_facts: Sequence[_ScopeAwareNumberFact],
    *,
    text: str,
    token: str,
    start: int,
    end: int,
    client_message: str,
    context: Mapping[str, Any] | None,
) -> tuple[bool, bool]:
    if not surfaces or not scoped_facts:
        return False, False
    query_text = _number_scope_query_text(
        text,
        token,
        start=start,
        end=end,
        client_message=client_message,
        context=context,
    )
    matched_wrong_scope = False
    for fact in scoped_facts:
        if not fact.surfaces.intersection(surfaces):
            continue
        if _scope_aware_number_fact_allowed(fact, query_text=query_text):
            return True, False
        matched_wrong_scope = True
    return False, matched_wrong_scope


def _scope_aware_number_fact_allowed(fact: _ScopeAwareNumberFact, *, query_text: str) -> bool:
    requested_formats = _format_values_from_text(query_text)
    if requested_formats and fact.formats and requested_formats.isdisjoint(fact.formats):
        return False
    requested_grade = _grade_from_text(query_text)
    if requested_grade and fact.grades and requested_grade not in fact.grades:
        return False
    requested_scopes = detect_fact_scopes(query_text)
    if not requested_scopes:
        return True
    if not fact.scopes:
        return True
    for scope in requested_scopes:
        if fact_scopes_allowed(set(fact.scopes), requested_scope=scope, blocked_neighbor_scopes=blocked_neighbors_for(scope)):
            return True
    return False


def _number_scope_query_text(
    text: str,
    token: str,
    *,
    start: int,
    end: int,
    client_message: str,
    context: Mapping[str, Any] | None,
) -> str:
    parts = [str(client_message or "")]
    parts.append(_free_number_context_window(str(text or ""), start=start, end=end, radius=80))
    if isinstance(context, MappingABC):
        for raw in (
            context.get("current_question"),
            context.get("question"),
            (context.get("conversation_intent_plan") or {}).get("primary_intent")
            if isinstance(context.get("conversation_intent_plan"), MappingABC)
            else "",
            (context.get("conversation_intent_plan") or {}).get("product_scope")
            if isinstance(context.get("conversation_intent_plan"), MappingABC)
            else "",
        ):
            if raw:
                parts.append(str(raw))
        memory = context.get("dialogue_memory_view") if isinstance(context.get("dialogue_memory_view"), MappingABC) else {}
        known_slots = memory.get("known_slots") if isinstance(memory.get("known_slots"), MappingABC) else {}
        for key in ("brand", "product", "format", "grade", "class", "subject"):
            value = known_slots.get(key) if isinstance(known_slots, MappingABC) else None
            if value:
                parts.append(f"{key}: {value}")
    return " ".join(part for part in parts if str(part or "").strip())


def _number_scope_fact_types(fact_key: str) -> tuple[str, ...]:
    key = str(fact_key or "").casefold().replace("ё", "е")
    result: list[str] = []
    if "installment" in key or "rassroch" in key or "рассроч" in key:
        result.append("installment")
    if "discount" in key or "скид" in key:
        result.append("discount")
    if "camp_lvsh" in key or "lvsh" in key or "лвш" in key:
        result.append("camp_lvsh")
    if "camp_city" in key or "city_day_camp" in key or "город" in key:
        result.append("camp_city")
    if "contact" in key or "office_hours" in key:
        result.append("contact")
    return tuple(dict.fromkeys(result))


def _scope_aware_product_number_context(
    text: str,
    token: str,
    surfaces: set[str],
    *,
    start: int,
    end: int,
) -> bool:
    if _is_free_product_number_context(text, token, start=start, end=end):
        return True
    if not surfaces or surfaces <= _STRUCTURAL_NUMBER_OK:
        return False
    window = _free_number_context_window(str(text or ""), start=start, end=end, radius=35)
    return bool(_FREE_NUMBER_PRODUCT_CTX_RE.search(window))


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
        surfaces.update(_payment_plan_count_surfaces_for_token(text, token))
    surfaces.update(_payment_plan_count_surfaces_from_text(text))
    surfaces.update(_free_number_word_surfaces(text))
    return {surface for surface in surfaces if surface}


def _payment_plan_count_surfaces_from_text(text: str) -> set[str]:
    surfaces: set[str] = set()
    for match in _PAYMENT_PLAN_COUNT_RE.finditer(str(text or "")):
        for raw_number in re.findall(r"\d{1,2}", match.group(1)):
            normalized = _normalize_decimal_surface(raw_number)
            if normalized:
                surfaces.add(f"{_PAYMENT_PLAN_COUNT_PREFIX}{normalized}")
    return surfaces


def _payment_plan_count_surfaces_for_token(
    text: str,
    token: str,
    *,
    start: int | None = None,
    end: int | None = None,
) -> set[str]:
    surfaces = _normalize_free_number_token(token)
    if not surfaces:
        return set()
    if start is not None and end is not None:
        window = _free_number_context_window(str(text or ""), start=start, end=end, radius=35)
    else:
        source = str(text or "")
        item = str(token or "").strip()
        index = source.find(item) if item else -1
        window = source[max(0, index - 35) : index + len(item) + 35] if index >= 0 else source
    if not _PAYMENT_PLAN_CONTEXT_RE.search(window):
        return set()
    return {f"{_PAYMENT_PLAN_COUNT_PREFIX}{surface}" for surface in surfaces if re.fullmatch(r"\d{1,2}", surface)}


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
        r"км|километр(?:а|ов)?|год(?:а)?|лет|мес(?:\.|яц(?:ев|а)?)?|недел(?:и|ь)?|заняти(?:й|я)|балл(?:ов|а)?|"
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


def _handoff_factual_claim_text(text: str) -> str | None:
    source = " ".join(str(text or "").split())
    if not source or not _is_handoff_text(source):
        return None
    parts = [
        part.strip(" \t\n\r-—:;,.")
        for part in re.split(r"[.;!?]\s+|\s+[—-]\s+", source)
        if part.strip(" \t\n\r-—:;,.")
    ]
    claim_parts = [
        part
        for part in parts
        if (_FACTUAL_CLAIM_RE.search(part) or _HANDOFF_FACTUAL_CLAIM_RE.search(part))
        and not _is_handoff_text(part)
    ]
    if claim_parts:
        return ". ".join(claim_parts)
    if (_FACTUAL_CLAIM_RE.search(source) or _HANDOFF_FACTUAL_CLAIM_RE.search(source)) and not _is_pure_handoff_text(source):
        return source
    return None


def _is_pure_handoff_text(text: str) -> bool:
    low = str(text or "").casefold()
    return (
        _is_handoff_text(low)
        and not re.search(r"\b(?:не\s+знаю|нет\s+(?:информации|данных|ответа)|не\s+могу\s+ответить)\b", low, re.I)
        and not _FACTUAL_CLAIM_RE.search(low)
    )


def _is_handoff_text(text: str) -> bool:
    low = str(text or "").casefold()
    return bool(re.search(r"менеджер|передам|уточнит|подтвердит|сверит", low, re.I))


def _wrong_intent_fact_findings(
    draft: str,
    *,
    contract: AnswerContract,
    facts: Mapping[str, str],
    context: Mapping[str, Any] | None = None,
) -> list[VerificationFinding]:
    findings: list[VerificationFinding] = []
    if _asks_class_schedule_days(contract) and _draft_uses_contact_hours_as_schedule(draft, facts):
        findings.append(
            VerificationFinding(
                "wrong_intent_fact",
                "Контактные часы нельзя выдавать как дни занятий группы.",
            )
        )
    scope_relevance_fix = authoritative_gate_scope_relevance_fix_enabled(context)
    address_allowed = _asks_address(contract) or (
        scope_relevance_fix and _scope_relevance_context_allows_address(context)
    ) or (
        scope_relevance_fix
        and _scope_relevance_context_allows_class_schedule(context)
        and _draft_uses_class_schedule_fact(draft, facts)
    )
    if not address_allowed and _draft_uses_address_fact(draft, facts):
        findings.append(
            VerificationFinding(
                "wrong_intent_fact",
                "Адресный факт нельзя выдавать как ответ на неадресный вопрос.",
            )
        )
    camp_allowed = _contract_mentions_camp_or_lvsh(contract) or (
        scope_relevance_fix and _scope_relevance_context_allows_camp_or_lvsh(context)
    )
    if not camp_allowed and _draft_uses_camp_or_lvsh_fact(draft, facts):
        findings.append(
            VerificationFinding(
                "wrong_intent_fact",
                "Лагерный/ЛВШ факт нельзя выдавать как справку вне лагерного контекста.",
            )
        )
    if venue_scope_enabled(context):
        findings.extend(_wrong_venue_scope_fact_findings(draft, facts=facts, context=context))
    return findings


_SCOPE_ADDRESS_MARKERS = {
    "address",
    "addresses",
    "location",
    "locations",
    "contact_address",
    "theme:015_address",
}


_SCOPE_CAMP_MARKERS = {
    "camp",
    "camp_lvsh",
    "city_camp",
    "city_day_camp",
    "lvsh",
    "lvsh_mendeleevo",
    "residential_lvsh",
    "theme:026_camp_general",
    "vyezd_camp",
    "лвш",
}


_SCOPE_CLASS_SCHEDULE_MARKERS = {
    "class_schedule",
    "theme:013_schedule",
}


def _scope_relevance_context_allows_address(context: Mapping[str, Any] | None) -> bool:
    return bool(_scope_relevance_context_tokens(context) & _SCOPE_ADDRESS_MARKERS)


def _scope_relevance_context_allows_camp_or_lvsh(context: Mapping[str, Any] | None) -> bool:
    tokens = _scope_relevance_context_tokens(context)
    if tokens & _SCOPE_CAMP_MARKERS:
        return True
    return any(
        token.startswith("lvsh_") or token.endswith("_camp") or token.startswith("theme:026_")
        for token in tokens
    )


def _scope_relevance_context_allows_class_schedule(context: Mapping[str, Any] | None) -> bool:
    return bool(_scope_relevance_context_tokens(context) & _SCOPE_CLASS_SCHEDULE_MARKERS)


def _scope_relevance_context_tokens(context: Mapping[str, Any] | None) -> set[str]:
    if not isinstance(context, Mapping):
        return set()
    tokens: set[str] = set()
    plan = context.get("conversation_intent_plan")
    if isinstance(plan, Mapping):
        _scope_relevance_add_fields(
            tokens,
            plan,
            fields=(
                "answer_topics",
                "topic_roles",
                "primary_intent",
                "product_family",
                "product_scope",
                "fact_scope",
                "topic_id",
                "training_format",
                "training_formats",
            ),
        )
        known_slots = plan.get("known_slots")
        if isinstance(known_slots, Mapping):
            _scope_relevance_add_fields(tokens, known_slots, fields=("product",))
    answer_contract = context.get("answer_contract")
    if isinstance(answer_contract, Mapping):
        _scope_relevance_add_fields(
            tokens,
            answer_contract,
            fields=("primary_intent", "answerable_safe_parts", "topic_id", "known_slots"),
        )
    direct = context.get("direct_path")
    if isinstance(direct, Mapping):
        model_intent = direct.get("model_intent")
        if isinstance(model_intent, Mapping):
            _scope_relevance_add_fields(tokens, model_intent, fields=("primary_intent", "scope", "sense"))
        _scope_relevance_add_coverage_tokens(tokens, direct.get("answer_coverage_plan"))
        reliable = direct.get("reliable_answerer")
        if isinstance(reliable, Mapping):
            _scope_relevance_add_fields(tokens, reliable, fields=("covered_facets_in_text",))
            _scope_relevance_add_coverage_tokens(tokens, reliable.get("answer_coverage_plan"))
        llm_retrieve = direct.get("llm_retrieve")
        if isinstance(llm_retrieve, Mapping):
            venue_meta = llm_retrieve.get("venue_scope")
            if isinstance(venue_meta, Mapping):
                _scope_relevance_add_fields(tokens, venue_meta, fields=("requested_scope", "requested_scope_raw"))
    direct_path_model_intent = context.get("direct_path_model_intent")
    if isinstance(direct_path_model_intent, Mapping):
        _scope_relevance_add_fields(tokens, direct_path_model_intent, fields=("primary_intent", "scope", "sense"))
    _scope_relevance_add_coverage_tokens(tokens, context.get("answer_coverage_plan"))
    return tokens


def _scope_relevance_add_coverage_tokens(tokens: set[str], coverage: Any) -> None:
    if not isinstance(coverage, Mapping):
        return
    _scope_relevance_add_fields(tokens, coverage, fields=("client_facets", "requested_scope"))
    covered = coverage.get("covered_facets")
    if isinstance(covered, SequenceABC) and not isinstance(covered, (str, bytes)):
        for item in covered:
            if isinstance(item, Mapping):
                _scope_relevance_add_fields(tokens, item, fields=("facet",))


def _scope_relevance_add_fields(tokens: set[str], mapping: Mapping[str, Any], *, fields: Sequence[str]) -> None:
    for field in fields:
        if field in mapping:
            _scope_relevance_add_value(tokens, mapping.get(field))


def _scope_relevance_add_value(tokens: set[str], value: Any) -> None:
    if isinstance(value, str):
        normalized = _scope_relevance_token(value)
        if normalized:
            tokens.add(normalized)
        return
    if isinstance(value, SequenceABC) and not isinstance(value, (str, bytes)):
        for item in value:
            _scope_relevance_add_value(tokens, item)
        return
    if isinstance(value, Mapping):
        for item in value.values():
            _scope_relevance_add_value(tokens, item)


def _scope_relevance_token(value: str) -> str:
    return " ".join(str(value or "").strip().casefold().replace("ё", "е").split()).replace(" ", "_")


def _wrong_venue_scope_fact_findings(
    draft: str,
    *,
    facts: Mapping[str, str],
    context: Mapping[str, Any] | None,
) -> list[VerificationFinding]:
    requested_scope = _venue_scope_requested_from_context(context)
    if requested_scope not in VENUE_SCOPE_VALUES:
        return []
    metadata = _venue_scope_fact_metadata(context)
    if not metadata:
        return []
    draft_norm = _venue_scope_normalized_text(draft)
    findings: list[VerificationFinding] = []
    for key, text in facts.items():
        meta = metadata.get(str(key))
        if not isinstance(meta, Mapping):
            continue
        venue = fact_venue(meta)
        if venue not in VENUE_SCOPE_VALUES or venue == requested_scope:
            continue
        fact_norm = _venue_scope_normalized_text(text)
        if fact_norm and fact_norm in draft_norm:
            findings.append(
                VerificationFinding(
                    "wrong_intent_fact",
                    f"Факт другой площадки: нужна {venue_scope_label(requested_scope)}, факт {key} помечен как {venue_scope_label(venue)}.",
                )
            )
    return findings


def _venue_scope_normalized_text(value: Any) -> str:
    return " ".join(str(value or "").casefold().replace("ё", "е").split())


def _venue_scope_requested_from_context(context: Mapping[str, Any] | None) -> str:
    if not isinstance(context, Mapping):
        return "unspecified"
    for container in (
        context.get("direct_path"),
        context.get("dialogue_contract_pipeline"),
        context.get("facts_context"),
    ):
        if not isinstance(container, Mapping):
            continue
        venue_meta = container.get("venue_scope") if isinstance(container.get("venue_scope"), Mapping) else {}
        requested = normalize_requested_scope(venue_meta.get("requested_scope") if isinstance(venue_meta, Mapping) else "")
        if requested in VENUE_SCOPE_VALUES:
            return requested
        requested = normalize_requested_scope(container.get("requested_scope"))
        if requested in VENUE_SCOPE_VALUES:
            return requested
        if isinstance(container.get("llm_retrieve"), Mapping):
            llm_venue_meta = container["llm_retrieve"].get("venue_scope")  # type: ignore[index]
            requested = normalize_requested_scope(
                llm_venue_meta.get("requested_scope") if isinstance(llm_venue_meta, Mapping) else ""
            )
            if requested in VENUE_SCOPE_VALUES:
                return requested
    return "unspecified"


def _venue_scope_fact_metadata(context: Mapping[str, Any] | None) -> Mapping[str, Mapping[str, Any]]:
    if not isinstance(context, Mapping):
        return {}
    candidates: list[Any] = [
        context.get("direct_path_fact_metadata"),
        context.get("wide_fact_metadata"),
    ]
    facts_context = context.get("facts_context")
    if isinstance(facts_context, Mapping):
        candidates.append(facts_context.get("fact_metadata"))
    direct = context.get("direct_path")
    if isinstance(direct, Mapping):
        candidates.append(direct.get("wide_fact_metadata"))
    result: dict[str, Mapping[str, Any]] = {}
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        for key, value in candidate.items():
            if str(key).strip() and isinstance(value, Mapping):
                result[str(key)] = value
    return result


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


def _draft_uses_class_schedule_fact(draft: str, facts: Mapping[str, str]) -> bool:
    text = str(draft or "").casefold().replace("ё", "е")
    if not text:
        return False
    for key, value in facts.items():
        combined = f"{key} {value}".casefold().replace("ё", "е")
        if not re.search(r"schedule|расписан|групп", combined, re.I):
            continue
        tail = _fact_tail(str(value or "")).casefold().replace("ё", "е")
        if tail and len(tail) >= 12 and tail in text:
            return True
        if re.search(r"\b(?:пн|вт|ср|чт|пт|сб|вс)\b|\b(?:понедельник|вторник|сред|четверг|пятниц|суббот|воскрес)", text, re.I):
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


def _asks_address(contract: AnswerContract) -> bool:
    text = " ".join(
        [
            contract.current_question,
            contract.client_state,
            " ".join(item.text for item in contract.subquestions),
        ]
    ).casefold().replace("ё", "е")
    pattern = r"адрес|площадк|где\s+вы|где\s+находит|куда\s+ехать|куда\s+ездить"
    if autonomy_scope_precision_enabled():
        pattern = (
            pattern
            + r"|как\s+(?:доехать|добраться|попасть|пройти|проехать)|"
            r"доехать|добраться|проезд|маршрут"
        )
    return bool(re.search(pattern, text, re.I))


def _fact_tail(text: str) -> str:
    value = str(text or "").strip()
    if "—" in value:
        value = value.rsplit("—", 1)[-1].strip()
    elif ":" in value:
        value = value.rsplit(":", 1)[-1].strip()
    return value.strip(" .")


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


def _clamp_float(value: object) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0


def _truthy(value: object) -> bool:
    return str(value or "").strip().casefold() in {"1", "true", "yes", "on", "y", "да"}


def _norm_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().casefold())


_WORD_CHARS = "0-9a-zа-я"


def _norm(text: object) -> str:
    s = str(text or "").casefold().replace("ё", "е")
    s = re.sub(rf"[^{_WORD_CHARS} ]", " ", s)
    return " ".join(s.split())


def _tokens(text: object) -> set[str]:
    return set(_norm(text).split())


def repeat_ratio(a: object, b: object) -> float:
    """Сходство двух реплик (0..1). 1.0 = практически одно и то же."""
    ta, tb = _tokens(a), _tokens(b)
    jaccard = len(ta & tb) / len(ta | tb) if ta and tb else 0.0
    a_norm, b_norm = _norm(a), _norm(b)
    sequence = 0.0
    if len(a_norm) >= 25 and len(b_norm) >= 25:
        sequence = difflib.SequenceMatcher(None, a_norm, b_norm).ratio()
    return max(jaccard, sequence)


def is_near_repeat(draft: object, prior_bot_texts: Sequence[object], *, threshold: float = 0.8) -> bool:
    """Черновик почти дублирует один из ПРЕДЫДУЩИХ ответов бота → повтор.
    Применять на ВСЕХ ветках (включая P0/handoff), кроме чисто служебного сухого
    P0-хендоффа на ПОВТОРНОЙ P0-реплике."""
    d = _norm(draft)
    if len(d.split()) < 4:
        return False
    return any(repeat_ratio(draft, p) >= threshold for p in prior_bot_texts)


_META_CLIENT_MARKERS: tuple[str, ...] = (
    "автономный ответ",
    "автономный ответ не требуется",
    "если менеджер решит",
    "безопасный вариант",
    "без служебных пометок",
    "не оформляю как жалобу",
    "не оформляю как заявление",
    "не буду оформлять это как",
    "передам ему контекст диалога",
    "не требует автономного",
    "manager_only",
    "client_safe",
    "fact:",
    "fact_id",
    "trace_id",
    "source_id",
)


_META_FACT_PHRASE_RE = re.compile(
    r"(?:"
    r"в\s+(?:факт\w+|баз\w+|данн\w+)\s+нет|"
    r"нет\s+в\s+(?:факт\w+|баз\w+|данн\w+)|"
    r"(?:не\s+указан\w*|не\s+уточнен\w*|отсутствует)[^.\n]{0,20}\b(?:факт\w+|баз\w+|данн\w+)"
    r")",
    re.I,
)


def _has_meta_fact_phrase(text: object) -> bool:
    return bool(_META_FACT_PHRASE_RE.search(str(text or "").casefold().replace("ё", "е")))


def has_meta_leak(text: object) -> bool:
    """Клиенту виден внутренний/служебный текст."""
    low = _norm(text)
    low_raw = str(text or "").casefold()
    return _has_meta_fact_phrase(text) or any((m in low) or (m in low_raw) for m in _META_CLIENT_MARKERS)


__all__ = (
    'AUTHORITATIVE_GATE_SCOPE_RELEVANCE_FIX_ENV',
    'AUTONOMY_SCOPE_PRECISION_ENV',
    'AnswerContract',
    'DIALOGUE_CONTRACT_SCHEMA_VERSION',
    'DIRECT_PATH_PILOT_CONFIG_ENV',
    'DIRECT_PATH_PILOT_CONFIG_VERSION',
    'FREE_NUMBER_GATE_ENV',
    'NUMBER_GATE_SCOPE_AWARE_ENV',
    'PLANNER_INTENT_VALUES',
    'STEP4_NUMBER_GROUNDING_ENV',
    'Slot',
    'Subquestion',
    'VerificationFinding',
    '_ACTIVE_HARD_P0_LATCH_CODES',
    '_ADDRESS_GENERIC_RE',
    '_AI_SELF_DISCLOSURE_RE',
    '_BRAND_TOKENS',
    '_CONDITION_ANCHOR_ALIASES',
    '_DATE_ANCHOR_RE',
    '_DISCOUNT_SCOPE_ALIASES',
    '_ESTIMATE_DOMAINS',
    '_ESTIMATE_GUARANTEE_RE',
    '_ESTIMATE_NUMBER_TOKEN_RE',
    '_ESTIMATE_PRESSURE_RE',
    '_FACTUAL_CLAIM_RE',
    '_FREE_NUMBER_PRODUCT_CTX_RE',
    '_FREE_NUMBER_TOKEN_RE',
    '_FREE_NUMBER_UNCERTAINTY_MARKERS',
    '_GENERIC_HANDOFF_TEXTS',
    '_HANDOFF_EXHAUSTED_TEXTS',
    '_HANDOFF_FACTUAL_CLAIM_RE',
    '_INDIVIDUAL_CHILD_CONFIDENT_RE',
    '_INDIVIDUAL_CHILD_RE',
    '_META_MARKERS',
    '_MONEY_OR_VALUE_RE',
    '_MONTH_ANCHOR_BY_PREFIX',
    '_NUMBER_RE',
    '_P0_LATCH_REASON_PRIORITY',
    '_P0_PROMISE_RE',
    '_PAYMENT_PLAN_CONTEXT_RE',
    '_PAYMENT_PLAN_COUNT_PREFIX',
    '_PAYMENT_PLAN_COUNT_RE',
    '_PRODUCT_NUMBER_CTX_RE',
    '_RISKY_ENTITY_ALIASES',
    '_ROLE_PERSON_RE',
    '_SCHEDULE_SPECIFICITY_ALIASES',
    '_SCOPE_ADDRESS_MARKERS',
    '_SCOPE_CAMP_MARKERS',
    '_SCOPE_CLASS_SCHEDULE_MARKERS',
    '_STRUCTURAL_NUMBER_OK',
    '_SUBJECT_ANCHOR_ALIASES',
    '_ScopeAwareNumberFact',
    '_UNCERTAINTY_MARKERS',
    '_YEAR_NUMBER_OK',
    '_active_brand_entity_anchors',
    '_active_hard_p0_latch_reason',
    '_answer_mode_number_findings',
    '_asks_address',
    '_asks_class_schedule_days',
    '_asks_training_format_choice',
    '_brand_token_present',
    '_clamp_float',
    '_clean_answer_mode',
    '_clean_estimate_domain',
    '_clean_planner_intent',
    '_clean_planner_slots',
    '_clean_selling',
    '_clean_slots',
    '_client_asked_identity',
    '_client_number_context_text',
    '_contract_intent_text',
    '_contract_mentions_camp_or_lvsh',
    '_discount_scope_anchors',
    '_draft_uses_address_fact',
    '_draft_uses_camp_or_lvsh_fact',
    '_draft_uses_class_schedule_fact',
    '_draft_uses_contact_hours_as_schedule',
    '_entity_anchors',
    '_estimate_gate_payload_from_context',
    '_extract_json_object',
    '_fact_tail',
    '_first_p0_latch_reason',
    '_format_values_from_text',
    '_free_number_context_window',
    '_free_number_gate_findings',
    '_free_number_surfaces',
    '_free_number_token_matches',
    '_free_number_tokens',
    '_free_number_word_surfaces',
    '_gate_answer_mode',
    '_gate_estimate_domain',
    '_gate_is_estimate',
    '_general_advice_estimate_findings',
    '_grade_from_text',
    '_grade_values_from_fact_scope',
    '_handoff_factual_claim_text',
    '_has_free_uncertainty_marker_near',
    '_has_only_benign_refund_latch',
    '_has_presale_refund_evidence',
    '_has_uncertainty_marker',
    '_individual_child_diagnosis_findings',
    '_is_allowed_ungrounded_number',
    '_is_camp_or_lvsh_fact',
    '_is_client_grade_number_context',
    '_is_client_grade_number_context_at',
    '_is_decimal_year_range',
    '_is_free_product_number_context',
    '_is_free_structural_number',
    '_is_handoff_text',
    '_is_product_number_context',
    '_is_pure_handoff_text',
    '_is_route_estimate_number_context',
    '_is_safe_entity_echo',
    '_latch_is_active',
    '_multiply_thousand_surface',
    '_norm_text',
    '_normalize_brand',
    '_normalize_date_anchor',
    '_normalize_decimal_surface',
    '_normalize_free_number_token',
    '_normalize_question_type',
    '_number_scope_fact_types',
    '_number_scope_query_text',
    '_number_token_map',
    '_numbers',
    '_p0_latch_sources',
    '_parse_subquestions',
    '_payment_plan_count_surfaces_for_token',
    '_payment_plan_count_surfaces_from_text',
    '_preemptive_format_choice_finding',
    '_presale_refund_latch_can_release',
    '_sanitize_blocks',
    '_schedule_alias_present',
    '_schedule_specificity_anchors',
    '_schedule_specificity_is_declined',
    '_scope_aware_number_fact_allowed',
    '_scope_aware_number_facts',
    '_scope_aware_number_supported',
    '_scope_aware_product_number_context',
    '_scope_relevance_add_coverage_tokens',
    '_scope_relevance_add_fields',
    '_scope_relevance_add_value',
    '_scope_relevance_context_allows_address',
    '_scope_relevance_context_allows_camp_or_lvsh',
    '_scope_relevance_context_allows_class_schedule',
    '_scope_relevance_context_tokens',
    '_scope_relevance_token',
    '_self_contradiction_finding',
    '_seq',
    '_truthy',
    '_unconfirmed_schedule_finding',
    '_valid_contract_key',
    '_venue_scope_fact_metadata',
    '_venue_scope_normalized_text',
    '_venue_scope_requested_from_context',
    '_wrong_intent_fact_findings',
    '_wrong_venue_scope_fact_findings',
    'authoritative_gate_scope_relevance_fix_enabled',
    'autonomy_scope_precision_enabled',
    'concrete_anchors',
    'free_number_gate_enabled',
    'number_gate_scope_aware_enabled',
    'p0_pre_gate',
    'parse_contract',
    'unsupported_named_entities',
    'verify_output',
    '_META_CLIENT_MARKERS',
    '_META_FACT_PHRASE_RE',
    '_WORD_CHARS',
    '_has_meta_fact_phrase',
    '_norm',
    '_tokens',
    'has_meta_leak',
    'is_near_repeat',
    'repeat_ratio',
)
