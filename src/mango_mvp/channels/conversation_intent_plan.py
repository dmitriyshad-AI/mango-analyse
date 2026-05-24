from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from mango_mvp.channels.answer_safety_classifier import codes_from_current_message
from mango_mvp.channels.new_lead_funnel import (
    extract_format,
    extract_grade,
    extract_product,
    extract_subjects,
    normalize_text,
)


CONVERSATION_INTENT_PLAN_SCHEMA_VERSION = "conversation_intent_plan_v1_2026_05_23"


@dataclass(frozen=True)
class ConversationIntentPlan:
    active_brand: str
    primary_intent: str
    topic_id: str
    direct_question: str = ""
    intent_confidence: float = 0.0
    topic_switch_decision: str = "continue"
    topic_switch_confidence: float = 0.0
    product_family: str = ""
    product_scope: str = ""
    known_slots: Mapping[str, str] = field(default_factory=dict)
    requested_slots: tuple[str, ...] = ()
    do_not_reask_slots: tuple[str, ...] = ()
    keyword_signals: tuple[str, ...] = ()
    risk_signals: tuple[str, ...] = ()
    required_fact_keys: tuple[str, ...] = ()
    answer_policy: str = "answer_directly_if_fact_verified"
    route_bias: str = "draft_for_manager"
    next_step_hint: str = ""
    fact_query_text: str = ""
    decision_notes: tuple[str, ...] = ()

    def to_prompt_view(self) -> Mapping[str, Any]:
        return {
            "schema_version": CONVERSATION_INTENT_PLAN_SCHEMA_VERSION,
            "active_brand": self.active_brand,
            "primary_intent": self.primary_intent,
            "topic_id": self.topic_id,
            "direct_question": self.direct_question,
            "intent_confidence": round(self.intent_confidence, 3),
            "topic_switch_decision": self.topic_switch_decision,
            "topic_switch_confidence": round(self.topic_switch_confidence, 3),
            "product_family": self.product_family,
            "product_scope": self.product_scope,
            "known_slots": dict(self.known_slots),
            "requested_slots": list(self.requested_slots),
            "do_not_reask_slots": list(self.do_not_reask_slots),
            "keyword_signals": list(self.keyword_signals),
            "risk_signals": list(self.risk_signals),
            "required_fact_keys": list(self.required_fact_keys),
            "answer_policy": self.answer_policy,
            "route_bias": self.route_bias,
            "next_step_hint": self.next_step_hint,
            "fact_query_text": self.fact_query_text,
            "decision_notes": list(self.decision_notes),
        }


def build_conversation_intent_plan(
    *,
    current_message: str,
    active_brand: str = "unknown",
    topic_id: str = "",
    known_slots: Mapping[str, Any] | None = None,
    dialogue_memory_view: Mapping[str, Any] | None = None,
    recent_messages: Sequence[str] = (),
) -> ConversationIntentPlan:
    """Build an internal conversation plan.

    Keywords are treated as signals. The final intent decision also uses known
    slots, previous product focus and whether the client is clearly switching
    topics.
    """

    brand = _normalize_brand(active_brand)
    text = str(current_message or "").strip()
    normalized = normalize_text(text)
    memory = dict(dialogue_memory_view or {})
    previous_focus = memory.get("topic_focus") if isinstance(memory.get("topic_focus"), Mapping) else {}
    memory_slots = memory.get("known_slots") if isinstance(memory.get("known_slots"), Mapping) else {}
    slots = _merge_slots(known_slots or {}, memory_slots, _extract_slots(text))
    open_question = memory.get("open_question") if isinstance(memory.get("open_question"), Mapping) else {}
    previous_question_kind = str(open_question.get("kind") or "")
    previous_product_family = str(previous_focus.get("product_family") or "")
    previous_product = str(previous_focus.get("product") or "")

    risk_signals = _risk_signals(normalized)
    keyword_signals = _keyword_signals(normalized)
    product_family, product_scope = _product_focus(
        normalized,
        slots=slots,
        previous_product_family=previous_product_family,
        previous_product=previous_product,
        recent_messages=recent_messages,
    )
    primary_intent = _primary_intent(
        normalized,
        keyword_signals=keyword_signals,
        risk_signals=risk_signals,
        previous_question_kind=previous_question_kind,
        previous_product_family=previous_product_family,
        product_family=product_family,
    )
    intent_topic = _topic_for_intent(primary_intent)
    resolved_topic = topic_id or intent_topic
    if primary_intent != "general_consultation":
        resolved_topic = intent_topic
    requested_slots = _requested_slots(normalized, primary_intent=primary_intent)
    do_not_reask = tuple(key for key, value in slots.items() if str(value or "").strip())
    required_fact_keys = _required_fact_keys(primary_intent, normalized)
    switch_decision, switch_confidence = _topic_switch_decision(
        normalized,
        primary_intent=primary_intent,
        product_family=product_family,
        previous_product_family=previous_product_family,
        previous_question_kind=previous_question_kind,
    )
    answer_policy, route_bias = _answer_policy(primary_intent, risk_signals=risk_signals)
    direct_question = _direct_question(text, previous_open=str(open_question.get("text") or ""))
    notes = _decision_notes(
        primary_intent=primary_intent,
        keyword_signals=keyword_signals,
        switch_decision=switch_decision,
        previous_product_family=previous_product_family,
        product_family=product_family,
    )
    fact_query = _fact_query_text(
        text,
        primary_intent=primary_intent,
        product_family=product_family,
        product_scope=product_scope,
        slots=slots,
        required_fact_keys=required_fact_keys,
    )
    return ConversationIntentPlan(
        active_brand=brand,
        primary_intent=primary_intent,
        topic_id=resolved_topic,
        direct_question=direct_question,
        intent_confidence=_intent_confidence(primary_intent, keyword_signals, risk_signals),
        topic_switch_decision=switch_decision,
        topic_switch_confidence=switch_confidence,
        product_family=product_family,
        product_scope=product_scope,
        known_slots=slots,
        requested_slots=requested_slots,
        do_not_reask_slots=do_not_reask,
        keyword_signals=keyword_signals,
        risk_signals=risk_signals,
        required_fact_keys=required_fact_keys,
        answer_policy=answer_policy,
        route_bias=route_bias,
        next_step_hint=_next_step_hint(primary_intent, slots=slots, risk_signals=risk_signals),
        fact_query_text=fact_query,
        decision_notes=notes,
    )


def _primary_intent(
    text: str,
    *,
    keyword_signals: Sequence[str],
    risk_signals: Sequence[str],
    previous_question_kind: str,
    previous_product_family: str,
    product_family: str,
) -> str:
    if "legal" in risk_signals:
        return "legal_threat"
    if "refund" in risk_signals:
        return "refund"
    if "complaint" in risk_signals:
        return "complaint"
    if "payment_dispute" in risk_signals:
        return "payment_dispute"
    if _asks_live_availability(text, previous_product_family=previous_product_family, product_family=product_family):
        return "live_availability"
    if _asks_price_fix(text):
        return "price_fix"
    if "schedule" in keyword_signals and re.search(r"\bво\s+сколько\b|\bраспис|\bвремя\b|\bдни\b|\bкогда\b", text):
        return "schedule"
    if "tax" in keyword_signals:
        return "tax"
    if "matkap" in keyword_signals:
        return "matkap"
    if "document" in keyword_signals:
        return "document"
    priority = (
        ("installment", "installment"),
        ("trial", "trial"),
        ("price", "pricing"),
        ("discount", "discount"),
        ("schedule", "schedule"),
        ("format", "format"),
        ("camp", "camp"),
        ("address", "address"),
        ("identity", "identity"),
        ("off_topic", "off_topic"),
    )
    for signal, intent in priority:
        if signal in keyword_signals:
            return intent
    if previous_question_kind in {"price", "price_fix"} and _is_followup(text):
        return "price_fix" if previous_question_kind == "price_fix" else "pricing"
    if previous_question_kind:
        return previous_question_kind
    return "general_consultation"


def _asks_live_availability(text: str, *, previous_product_family: str, product_family: str) -> bool:
    if any(marker in text for marker in ("не про мест", "не о мест", "не места", "я не про места")):
        return False
    asks_place = bool(re.search(r"\b(?:мест(?:о|а)?|налич\w*|брон\w*|заброни\w*)\b", text))
    asks_fix_place = bool(re.search(r"\bзакреп\w*\b", text) and re.search(r"\bмест(?:о|а)?\b", text))
    camp_context = product_family == "camp" or previous_product_family == "camp"
    return bool(camp_context and (asks_place or asks_fix_place))


def _asks_price_fix(text: str) -> bool:
    if any(marker in text for marker in ("мест", "брон", "заброни")):
        return False
    if any(marker in text for marker in ("зафикс", "закреп")) and any(
        marker in text for marker in ("цен", "услов", "текущ", "сейчас", "стоим")
    ):
        return True
    return any(marker in text for marker in ("по текущей цене", "по текущим условиям"))


def _keyword_signals(text: str) -> tuple[str, ...]:
    markers: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("price", ("цен", "стоим", "сколько", "прайс", "руб")),
        ("installment", ("рассроч", "долями", "частями", "помесяч", "банк", "одобр")),
        ("discount", ("скид", "акци", "промокод", "льгот")),
        ("trial", ("пробн", "фрагмент")),
        ("camp", ("лагер", "лвш", "смен", "менделеево", "прожив", "питан", "трансфер")),
        ("schedule", ("распис", "когда", "во сколько", "дни", "время")),
        ("format", ("онлайн", "очно", "офлайн", "дистанц", "формат")),
        ("address", ("адрес", "где", "площадк", "метро", "пацаева", "сретен", "красносель")),
        ("document", ("справк", "документ", "договор", "сертификат", "чек", "квитанц")),
        ("matkap", ("маткап", "материн")),
        ("tax", ("налог", "вычет", "фнс")),
        ("identity", ("вы бот", "ты бот", "кто вы", "с кем я общаюсь", "gpt")),
        ("off_topic", ("айфон", "iphone", "погода", "сочинение", "биткоин")),
    )
    return tuple(code for code, values in markers if any(marker in text for marker in values))


def _risk_signals(text: str) -> tuple[str, ...]:
    mapping = {
        "legal": "legal",
        "refund": "refund",
        "complaint": "complaint",
        "reputation_threat": "complaint",
        "payment_dispute": "payment_dispute",
    }
    return tuple(dict.fromkeys(mapping.get(code, code) for code in codes_from_current_message(text)))


def _has_refund_signal(text: str) -> bool:
    if re.search(r"\b(?:верн(?:ите|уть|ули|ем)|забрать|получить\s+назад)\b[^.!?]{0,45}\b(?:деньги|оплат[ауые]?|средств[а-я]*)\b", text):
        return True
    if re.search(r"\bденьги\s+назад\b", text):
        return True
    if re.search(r"\bрасторг\w*\b[^.!?]{0,45}\bдоговор\b|\bотказ\w*\b[^.!?]{0,45}\bобучен", text):
        return True
    # "возврат к теме/уроку/материалам" is not a refund request.
    if re.search(r"\bвозврат\w*\b", text) and not re.search(r"\bвозврат\w*\s+к\s+(?:тем|урок|материал|заняти)", text):
        return True
    return False


def _has_legal_signal(text: str) -> bool:
    return bool(
        re.search(r"\b(?:суд|иск|адвокат|юрист)\b", text)
        or re.search(r"\b(?:прокуратур|роспотреб|досудеб|претензионн)\w*\b", text)
        or re.search(r"\bпо\s+закону\b[^.!?]{0,60}\b(?:обязан|должн|наруш)", text)
    )


def _has_complaint_signal(text: str) -> bool:
    # Mild hesitation ("подумаю", "обсудить с менеджером") is not a complaint.
    if re.search(r"\b(?:подумаю|обсудить|обсудим|с менеджером обсудить)\b", text) and not re.search(
        r"\b(?:жалоб|претензи|недовол|возмущ|ужас|обман|плохо\s+уч|некомпетент)\w*\b",
        text,
    ):
        return False
    return bool(re.search(r"\b(?:жалоб|претензи|недовол|возмущ|ужас|обман|плохо\s+уч|некомпетент)\w*\b", text))


def _product_focus(
    text: str,
    *,
    slots: Mapping[str, str],
    previous_product_family: str,
    previous_product: str,
    recent_messages: Sequence[str],
) -> tuple[str, str]:
    if any(marker in text for marker in ("вместо лагер", "не лагер", "не лвш")) and any(
        marker in text for marker in ("курс", "онлайн", "очно", "физик", "математ", "информат")
    ):
        return "regular_course", str(slots.get("format") or "")
    if any(marker in text for marker in ("лвш", "лагер", "смен", "менделеево", "прожив", "питан", "трансфер")):
        if "город" in text:
            return "camp", "city_camp"
        return "camp", "lvsh_mendeleevo"
    if any(marker in text for marker in ("онлайн", "очно", "физик", "математ", "информат", "курс")):
        return "regular_course", str(slots.get("format") or "")
    if _is_followup(text) and previous_product_family:
        return previous_product_family, previous_product
    recent_text = normalize_text(" ".join(str(item or "") for item in recent_messages[-6:]))
    if previous_product_family:
        return previous_product_family, previous_product
    if "лвш" in recent_text or "лагер" in recent_text:
        return "camp", "lvsh_mendeleevo"
    return "", ""


def _topic_switch_decision(
    text: str,
    *,
    primary_intent: str,
    product_family: str,
    previous_product_family: str,
    previous_question_kind: str,
) -> tuple[str, float]:
    if not previous_product_family:
        return "new_or_unknown", 0.4
    explicit_switch = any(marker in text for marker in ("теперь", "другой", "вместо", "нет, все-таки", "не лагерь", "не курс"))
    if explicit_switch:
        return "confirmed_switch", 0.9
    if product_family and product_family != previous_product_family:
        if _is_followup(text):
            return "clarify_before_switch", 0.55
        return "confirmed_switch", 0.78
    if primary_intent == "live_availability" and previous_product_family == "camp":
        return "continue", 0.95
    if previous_question_kind and _is_followup(text):
        return "continue", 0.85
    return "continue", 0.7


def _topic_for_intent(intent: str) -> str:
    mapping = {
        "pricing": "theme:001_pricing",
        "price_fix": "theme:001_pricing",
        "installment": "theme:006_installment",
        "discount": "theme:005_discounts",
        "trial": "theme:023_trial_class",
        "camp": "theme:026_camp_general",
        "live_availability": "theme:026_camp_general",
        "schedule": "theme:013_schedule",
        "format": "theme:014_format",
        "address": "theme:015_address",
        "document": "theme:012_certificates",
        "matkap": "theme:007_matkap_payment",
        "tax": "theme:008_tax_deduction",
        "identity": "service:S5_general_consultation",
        "off_topic": "service:S3_out_of_scope",
        "refund": "theme:009_refund",
        "legal_threat": "theme:029_legal_question",
        "complaint": "theme:019b_negative_feedback",
        "payment_dispute": "theme:003_payment_status",
    }
    return mapping.get(intent, "service:S5_general_consultation")


def _required_fact_keys(intent: str, text: str) -> tuple[str, ...]:
    keys: list[str] = []
    if intent in {"pricing", "price_fix"}:
        keys.append("prices.current")
    if intent == "installment":
        keys.append("installment_terms.current")
    if intent == "discount":
        keys.append("discounts.current")
    if intent == "trial":
        keys.append("trial_class.current")
    if intent in {"camp", "live_availability"}:
        keys.append("programs.current")
        if intent == "live_availability":
            keys.append("availability.current")
    if intent == "schedule":
        keys.append("schedule.current")
    if intent == "format":
        keys.append("formats.current")
    if intent == "address":
        keys.append("locations.current")
    if intent == "document":
        keys.append("documents.current")
    if intent == "matkap":
        keys.append("matkap_documents.current")
    if intent == "tax":
        keys.append("tax_deduction_procedure.current")
    if any(marker in text for marker in ("прожив", "питан", "трансфер", "что входит")):
        keys.append("programs.current")
    return tuple(dict.fromkeys(keys))


def _answer_policy(intent: str, *, risk_signals: Sequence[str]) -> tuple[str, str]:
    if risk_signals or intent in {"refund", "legal_threat", "complaint", "payment_dispute"}:
        return "manager_only_p0", "manager_only"
    if intent == "live_availability":
        return "answer_safe_parts_then_manager_live_check", "draft_for_manager"
    if intent in {"pricing", "price_fix", "installment", "discount", "trial", "camp", "schedule", "format", "address", "document", "matkap", "tax"}:
        return "answer_directly_if_fact_verified", "bot_answer_self_for_pilot"
    return "help_then_one_question", "draft_for_manager"


def _requested_slots(text: str, *, primary_intent: str) -> tuple[str, ...]:
    slots: list[str] = []
    if any(marker in text for marker in ("класс", "кл ")):
        slots.append("grade")
    if any(marker in text for marker in ("предмет", "физик", "математ", "информат", "хим", "англ")):
        slots.append("subject")
    if any(marker in text for marker in ("онлайн", "очно", "формат")):
        slots.append("format")
    if primary_intent == "live_availability":
        slots.append("availability")
    return tuple(dict.fromkeys(slots))


def _next_step_hint(intent: str, *, slots: Mapping[str, str], risk_signals: Sequence[str]) -> str:
    if risk_signals:
        return "handoff_to_responsible_staff"
    if intent == "live_availability":
        return "manager_checks_availability_without_promising_seat"
    if intent in {"pricing", "price_fix"}:
        if slots.get("grade") and slots.get("subject") and slots.get("format"):
            return "answer_price_then_offer_application_or_manager_check"
        return "answer_known_price_or_ask_one_missing_selection_slot"
    if intent == "installment":
        return "answer_payment_options_then_offer_manager_setup"
    if intent == "trial":
        return "answer_trial_format_then_offer_manager_setup"
    if intent == "camp":
        return "answer_camp_fact_then_ask_class_or_shift_if_missing"
    return "answer_then_one_next_question"


def _fact_query_text(
    text: str,
    *,
    primary_intent: str,
    product_family: str,
    product_scope: str,
    slots: Mapping[str, str],
    required_fact_keys: Sequence[str],
) -> str:
    parts = [str(text or "").strip()]
    if primary_intent:
        parts.append(f"Намерение: {primary_intent}.")
    if product_family:
        parts.append(f"Продукт: {product_family} {product_scope}".strip() + ".")
    slot_parts = []
    for key in ("grade", "subject", "format", "product", "goal"):
        if slots.get(key):
            slot_parts.append(f"{key}={slots[key]}")
    if slot_parts:
        parts.append("Известные данные: " + ", ".join(slot_parts) + ".")
    if required_fact_keys:
        parts.append("Нужные факты: " + ", ".join(required_fact_keys) + ".")
    return " ".join(part for part in parts if part).strip()


def _decision_notes(
    *,
    primary_intent: str,
    keyword_signals: Sequence[str],
    switch_decision: str,
    previous_product_family: str,
    product_family: str,
) -> tuple[str, ...]:
    notes = ["keyword_as_signal_context_as_decision"]
    if keyword_signals:
        notes.append("keywords_detected:" + ",".join(keyword_signals[:6]))
    if switch_decision == "continue" and previous_product_family:
        notes.append(f"continue_previous_product:{previous_product_family}")
    if switch_decision == "clarify_before_switch":
        notes.append(f"weak_topic_switch:{previous_product_family}->{product_family}")
    if primary_intent == "live_availability":
        notes.append("seat_or_booking_words_do_not_mean_price_fix")
    return tuple(notes)


def _intent_confidence(intent: str, keyword_signals: Sequence[str], risk_signals: Sequence[str]) -> float:
    if risk_signals:
        return 0.95
    if intent in {"live_availability", "price_fix"}:
        return 0.9
    if keyword_signals:
        return 0.82
    return 0.55


def _direct_question(text: str, *, previous_open: str = "") -> str:
    clean = " ".join(str(text or "").split())[:260]
    if clean and ("?" in clean or _keyword_signals(normalize_text(clean))):
        return clean
    return str(previous_open or "")[:260]


def _merge_slots(*mappings: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for mapping in mappings:
        for key, value in mapping.items():
            normalized = _slot_key(key)
            if not normalized:
                continue
            cleaned = _slot_value(value)
            if cleaned:
                result[normalized] = cleaned
    return result


def _extract_slots(text: str) -> Mapping[str, str]:
    normalized = normalize_text(text)
    return {
        "grade": extract_grade(normalized),
        "subject": extract_subjects(normalized),
        "format": _normalize_format(extract_format(normalized)),
        "product": extract_product(normalized),
    }


def _slot_key(value: Any) -> str:
    key = str(value or "").strip()
    aliases = {
        "class": "grade",
        "student_grade": "grade",
        "course_subject": "subject",
        "interest_subject": "subject",
        "course_format": "format",
        "preferred_format": "format",
        "course_type": "product",
    }
    return aliases.get(key, key if key in {"grade", "subject", "format", "goal", "product", "city", "location"} else "")


def _slot_value(value: Any) -> str:
    if isinstance(value, Mapping):
        value = value.get("value")
    text = str(value or "").strip()
    if text.lower() in {"none", "false", "unknown"}:
        return ""
    return _normalize_format(text) if text.casefold() in {"online", "offline", "онлайн", "очно", "очный", "офлайн"} else text[:160]


def _normalize_format(value: str) -> str:
    text = str(value or "").strip().casefold()
    if text in {"online", "онлайн", "дистанционно", "дистанц"}:
        return "онлайн"
    if text in {"offline", "очно", "очный", "офлайн"}:
        return "очно"
    return str(value or "").strip()


def _is_followup(text: str) -> bool:
    stripped = str(text or "").strip()
    if len(stripped) <= 90 and re.match(r"^(а|и|да|нет|хорошо|понятно|тогда|это|как|что|сколько)\b", stripped):
        return True
    return any(marker in stripped for marker in ("это цена", "по этой", "как можно", "что от меня", "а если"))


def _normalize_brand(value: Any) -> str:
    text = str(value or "unknown").strip().casefold()
    if text in {"foton", "фотон"}:
        return "foton"
    if text in {"unpk", "унпк", "унпк мфти"}:
        return "unpk"
    return "unknown"
