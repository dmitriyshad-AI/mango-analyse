from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from mango_mvp.channels.p0_recall_spec import codes_from_text
from mango_mvp.channels.text_signals import has_any_marker, has_marker


NEW_LEAD_FUNNEL_SCHEMA_VERSION = "new_lead_funnel_v1_2026_05_23"
ANCHORED_BARE_GRADE_ENV = "TELEGRAM_ANCHORED_BARE_GRADE"

OFF_TOPIC_MARKERS = (
    "айфон",
    "iphone",
    "погода",
    "сочинение",
    "напиши код",
    "реши задачу не по обучению",
)

SUBJECT_MARKERS = (
    ("математ", "математика"),
    ("физик", "физика"),
    ("информат", "информатика"),
    ("программирован", "программирование"),
    ("русск", "русский язык"),
    ("англий", "английский язык"),
    ("хими", "химия"),
    ("биолог", "биология"),
)
_TRUE_ENV_VALUES = {"1", "true", "yes", "on", "y", "да"}
_BARE_GRADE_CANDIDATE_RE = re.compile(r"(?<![\dA-Za-zА-Яа-яЁё+])(?P<grade>[1-9]|1[01])(?![\dA-Za-zА-Яа-яЁё])")
_PHONE_RE = re.compile(r"(?:\+7|8)[\s\-()]*(?:\d[\s\-()]*){6,}")
_TIME_RE = re.compile(r"\b(?:[01]?\d|2[0-3])[:.]\d{2}\b")
_DATE_RE = re.compile(
    r"\b(?:[1-9]|1[01])\s*(?:январ|феврал|март|апрел|ма[йя]|июн|июл|август|сентябр|октябр|ноябр|декабр)\w*\b"
    r"|\b(?:[1-9]|1[01])[./](?:0?[1-9]|1[0-2])\b",
    re.I,
)
_MONEY_AFTER_RE = re.compile(r"^\s*(?:тыс|т\.?\s*р|руб|₽|000\b)", re.I)
_AGE_AFTER_RE = re.compile(r"^\s*(?:лет|года|годик|летн\w*)\b", re.I)
_AGE_BEFORE_RE = re.compile(r"(?:\bс|\bдля)\s*$", re.I)
_COUNT_AFTER_RE = re.compile(
    r"^\s*(?:дет(?:ей|и|ям|я)?|реб[её]н\w*|заняти\w*|урок\w*|раз(?:а|ов)?|человек|месяц\w*|недел\w*)\b",
    re.I,
)
_COUNT_BEFORE_RE = re.compile(r"(?:\bу\s+меня|\bнас|\bесть)\s*$", re.I)
_CHOICE_VERBS_RE = re.compile(r"\b(?:сравнив\w*|дума\w*|выбира\w*|реша\w*|смотр\w*|рассматрива\w*)\b", re.I)


@dataclass(frozen=True)
class LeadSlots:
    grade: str = ""
    subject: str = ""
    format: str = ""
    city: str = ""
    location: str = ""
    goal: str = ""
    product: str = ""
    camp_direction: str = ""
    shift: str = ""
    student_name: str = ""
    parent_name: str = ""
    phone_known: bool = False
    known_from: Mapping[str, str] = field(default_factory=dict)
    confidence_by_field: Mapping[str, float] = field(default_factory=dict)

    def to_json_dict(self) -> Mapping[str, Any]:
        return {
            "grade": self.grade,
            "subject": self.subject,
            "format": self.format,
            "city": self.city,
            "location": self.location,
            "goal": self.goal,
            "product": self.product,
            "camp_direction": self.camp_direction,
            "shift": self.shift,
            "student_name": self.student_name,
            "parent_name": self.parent_name,
            "phone_known": self.phone_known,
            "known_from": dict(self.known_from),
            "confidence_by_field": dict(self.confidence_by_field),
        }

    def filled_slots(self) -> Mapping[str, Any]:
        result = {
            key: value
            for key, value in self.to_json_dict().items()
            if key not in {"known_from", "confidence_by_field"} and value not in ("", False, None)
        }
        if self.phone_known:
            result["phone_known"] = True
        return result


@dataclass(frozen=True)
class LeadFunnelState:
    brand_id: str
    client_segment: str
    lead_stage: str
    topic_id: str = ""
    product_scope: str = "unknown"
    known_slots: LeadSlots = field(default_factory=LeadSlots)
    missing_slots: tuple[str, ...] = ()
    next_best_question: str = ""
    next_step_type: str = "wait_for_client"
    handoff_reason: str = ""
    safety_blockers: tuple[str, ...] = ()
    semantic_flags: tuple[str, ...] = ()
    confidence: float = 0.0

    def to_json_dict(self) -> Mapping[str, Any]:
        return {
            "schema_version": NEW_LEAD_FUNNEL_SCHEMA_VERSION,
            "brand_id": self.brand_id,
            "client_segment": self.client_segment,
            "lead_stage": self.lead_stage,
            "topic_id": self.topic_id,
            "product_scope": self.product_scope,
            "known_slots": self.known_slots.to_json_dict(),
            "filled_slots": dict(self.known_slots.filled_slots()),
            "missing_slots": list(self.missing_slots),
            "next_best_question": self.next_best_question,
            "next_step_type": self.next_step_type,
            "handoff_reason": self.handoff_reason,
            "safety_blockers": list(self.safety_blockers),
            "semantic_flags": list(self.semantic_flags),
            "confidence": self.confidence,
        }

    def to_prompt_context(self) -> Mapping[str, Any]:
        payload = self.to_json_dict()
        return {
            "schema_version": payload["schema_version"],
            "brand_id": payload["brand_id"],
            "client_segment": payload["client_segment"],
            "lead_stage": payload["lead_stage"],
            "topic_id": payload["topic_id"],
            "product_scope": payload["product_scope"],
            "filled_slots": payload["filled_slots"],
            "missing_slots": payload["missing_slots"],
            "next_best_question": payload["next_best_question"],
            "next_step_type": payload["next_step_type"],
            "handoff_reason": payload["handoff_reason"],
            "safety_blockers": payload["safety_blockers"],
            "semantic_flags": payload["semantic_flags"],
            "confidence": payload["confidence"],
        }


def build_lead_funnel_state(
    client_message: str,
    *,
    active_brand: str = "unknown",
    recent_messages: Sequence[str] = (),
    context: Mapping[str, Any] | None = None,
    topic_id: str = "",
    message_type: str = "",
    risk_level: str = "",
    route: str = "",
    safety_flags: Sequence[str] = (),
) -> LeadFunnelState:
    ctx = dict(context or {})
    brand = normalize_brand(active_brand or ctx.get("active_brand"))
    recent_client_text = _client_only_recent_text(recent_messages)
    text = "\n".join([recent_client_text, str(client_message or "")])
    normalized = normalize_text(text)
    current_normalized = normalize_text(client_message)
    blockers = detect_safety_blockers(current_normalized, route=route, risk_level=risk_level, safety_flags=safety_flags)
    client_fields = ctx.get("known_client_fields") if isinstance(ctx.get("known_client_fields"), Mapping) else {}
    dialog_fields = ctx.get("known_dialog_fields") if isinstance(ctx.get("known_dialog_fields"), Mapping) else {}
    identity = ctx.get("client_identity") if isinstance(ctx.get("client_identity"), Mapping) else {}
    read_only_customer = ctx.get("read_only_customer_context") if isinstance(ctx.get("read_only_customer_context"), Mapping) else {}

    slots = extract_lead_slots(
        client_message=client_message,
        recent_text=text,
        known_client_fields=client_fields,
        known_dialog_fields=dialog_fields,
        client_identity=identity,
        read_only_customer_context=read_only_customer,
    )
    product_scope = detect_product_scope(normalized, topic_id=topic_id)
    client_segment = detect_client_segment(client_fields, identity=identity, read_only_customer_context=read_only_customer)
    semantic_flags = list(build_semantic_flags(slots, normalized, current_normalized, blockers=blockers))

    if blockers:
        return LeadFunnelState(
            brand_id=brand,
            client_segment=client_segment,
            lead_stage="p0_manager_only",
            topic_id=topic_id,
            product_scope=product_scope,
            known_slots=slots,
            missing_slots=(),
            next_best_question="",
            next_step_type="manager_only_p0",
            handoff_reason=", ".join(blockers),
            safety_blockers=tuple(blockers),
            semantic_flags=tuple(semantic_flags),
            confidence=0.95,
        )

    missing, next_question, next_step = choose_next_question(slots, brand_id=brand, product_scope=product_scope, topic_id=topic_id)
    lead_stage = choose_lead_stage(
        brand_id=brand,
        product_scope=product_scope,
        missing_slots=missing,
        message_type=message_type,
        current_text=current_normalized,
        client_segment=client_segment,
    )
    return LeadFunnelState(
        brand_id=brand,
        client_segment=client_segment,
        lead_stage=lead_stage,
        topic_id=topic_id,
        product_scope=product_scope,
        known_slots=slots,
        missing_slots=tuple(missing),
        next_best_question=next_question,
        next_step_type=next_step,
        handoff_reason="" if lead_stage != "manager_handoff" else "needs_human_action_or_missing_fact",
        safety_blockers=(),
        semantic_flags=tuple(semantic_flags),
        confidence=0.82 if missing else 0.9,
    )


def lead_funnel_context_payload(state: LeadFunnelState) -> Mapping[str, Any]:
    return state.to_prompt_context()


def extract_lead_slots(
    *,
    client_message: str,
    recent_text: str,
    known_client_fields: Mapping[str, Any],
    known_dialog_fields: Mapping[str, Any],
    client_identity: Mapping[str, Any],
    read_only_customer_context: Mapping[str, Any],
) -> LeadSlots:
    known_from: dict[str, str] = {}
    confidence: dict[str, float] = {}
    current = normalize_text(client_message)
    recent = normalize_text(recent_text)

    def take(field: str, value: Any, source: str, score: float) -> str:
        text = str(value or "").strip()
        if text:
            known_from.setdefault(field, source)
            confidence.setdefault(field, score)
        return text

    grade = take("grade", extract_grade(current), "client_message", 0.95)
    if not grade:
        grade = take("grade", extract_grade(recent), "dialog_history", 0.84)
    if not grade:
        grade = take("grade", known_dialog_fields.get("grade"), "dialog_history", 0.8)
    if not grade:
        grade = take("grade", known_client_fields.get("grade"), "crm_readonly", 0.72)

    subject = take("subject", extract_subjects(current), "client_message", 0.95)
    if not subject:
        subject = take("subject", extract_subjects(recent), "dialog_history", 0.82)
    if not subject:
        subject = take("subject", known_dialog_fields.get("subject"), "dialog_history", 0.78)
    if not subject:
        subject = take("subject", known_client_fields.get("subject"), "crm_readonly", 0.7)

    fmt = take("format", extract_format(current), "client_message", 0.95)
    if not fmt:
        fmt = take("format", extract_format(recent), "dialog_history", 0.82)
    if not fmt:
        fmt = take("format", known_dialog_fields.get("format"), "dialog_history", 0.78)

    city, location = extract_city_location(current or recent)
    if city:
        take("city", city, "client_message" if city in current else "dialog_history", 0.86)
    if location:
        take("location", location, "client_message" if location.casefold() in current else "dialog_history", 0.86)

    goal = take("goal", extract_goal(current), "client_message", 0.9)
    if not goal:
        goal = take("goal", extract_goal(recent), "dialog_history", 0.76)

    product = take("product", extract_product(current), "client_message", 0.9)
    if not product:
        product = take("product", extract_product(recent), "dialog_history", 0.76)

    camp_direction = take("camp_direction", extract_camp_direction(current or recent), "client_message", 0.8)
    shift = take("shift", extract_shift(current or recent), "client_message", 0.78)
    student_name = take("student_name", known_client_fields.get("student_name"), "crm_readonly", 0.72)
    parent_name = take("parent_name", known_client_fields.get("parent_name"), "crm_readonly", 0.72)
    phone_known = bool(
        known_client_fields.get("phone")
        or client_identity.get("phone")
        or read_only_customer_context.get("phone")
        or client_identity.get("channel_user_id")
    )
    if phone_known:
        known_from.setdefault("phone_known", "telegram_or_crm_readonly")
        confidence.setdefault("phone_known", 0.8)

    return LeadSlots(
        grade=grade,
        subject=subject,
        format=fmt,
        city=city,
        location=location,
        goal=goal,
        product=product,
        camp_direction=camp_direction,
        shift=shift,
        student_name=student_name,
        parent_name=parent_name,
        phone_known=phone_known,
        known_from=known_from,
        confidence_by_field=confidence,
    )


def choose_next_question(
    slots: LeadSlots,
    *,
    brand_id: str,
    product_scope: str,
    topic_id: str,
) -> tuple[list[str], str, str]:
    missing: list[str] = []
    if brand_id == "unknown":
        return ["brand"], "Подскажите, пожалуйста, какой учебный центр или программа вам интересны?", "ask_brand"

    if product_scope in {"lvsh", "city_camp"}:
        if not slots.grade:
            return ["grade"], "В каком классе ребёнок?", "ask_camp_class"
        if product_scope == "unknown" and not slots.product:
            missing.append("product")
        return missing, "", "offer_manager_seat_check"

    topic = str(topic_id or "").casefold()
    if "tax" in topic or "matkap" in topic or "payment_status" in topic:
        return [], "", "offer_manager_check"

    if not slots.grade:
        missing.append("grade")
        return missing, "В каком классе ребёнок?", "ask_grade"
    if not slots.subject and product_scope not in {"documents", "payment", "trial"}:
        missing.append("subject")
        return missing, "Какой предмет или направление рассматриваете?", "ask_subject"
    if not slots.format and product_scope in {"regular_course", "online_course", "offline_course", "unknown"}:
        missing.append("format")
        return missing, "Удобнее смотреть онлайн или очный формат?", "ask_format"
    if not slots.goal and product_scope in {"regular_course", "online_course", "offline_course"}:
        return ["goal"], "Какая сейчас главная цель: подтянуть базу, подготовиться к экзамену или идти глубже?", "ask_goal"
    return [], "", "offer_group_check"


def choose_lead_stage(
    *,
    brand_id: str,
    product_scope: str,
    missing_slots: Sequence[str],
    message_type: str,
    current_text: str,
    client_segment: str,
) -> str:
    if brand_id == "unknown":
        return "first_contact"
    if has_any_marker(current_text, OFF_TOPIC_MARKERS):
        return "closed_or_waiting"
    if str(message_type or "").strip() in {"non_question", "wait_for_more"}:
        return "closed_or_waiting"
    if missing_slots:
        return "qualification_needed"
    if client_segment in {"known_customer", "staff_test"} and product_scope in {"payment", "documents"}:
        return "manager_handoff"
    return "next_step_offered"


def detect_safety_blockers(
    text: str,
    *,
    route: str = "",
    risk_level: str = "",
    safety_flags: Sequence[str] = (),
) -> list[str]:
    flags = {normalize_text(item) for item in safety_flags}
    blockers: list[str] = []
    if codes_from_text(text):
        blockers.append("p0_keyword")
    if "high_risk_manager_only" in flags or "p0" in " ".join(flags):
        blockers.append("p0_safety_flag")
    if str(route or "").strip() == "manager_only" and str(risk_level or "").strip().casefold() in {"high", "critical"}:
        blockers.append("high_risk_route")
    return list(dict.fromkeys(blockers))


def detect_client_segment(
    known_client_fields: Mapping[str, Any],
    *,
    identity: Mapping[str, Any],
    read_only_customer_context: Mapping[str, Any],
) -> str:
    if identity.get("debug_impersonation") or known_client_fields.get("debug_impersonation"):
        return "staff_test"
    if known_client_fields.get("student_name") or known_client_fields.get("parent_name"):
        return "known_customer"
    if read_only_customer_context and (
        read_only_customer_context.get("summary")
        or read_only_customer_context.get("local_runtime_context")
        or read_only_customer_context.get("amo_context")
        or read_only_customer_context.get("tallanto_context")
    ):
        return "known_customer"
    return "new_lead"


def detect_product_scope(text: str, *, topic_id: str = "") -> str:
    topic = str(topic_id or "").casefold()
    if has_any_marker(text, ("лвш", "выездн", "менделеево", "лагерь с прожив")):
        return "lvsh"
    if "camp" in topic or topic in {"theme:026_camp_general", "theme:027_camp_living_conditions", "theme:028_transport_logistics"}:
        return "city_camp"
    if has_any_marker(text, ("городск", "лш", "летн", "лагер", "смен")):
        return "city_camp"
    if has_marker(text, "интенсив"):
        return "intensive"
    if has_any_marker(text, ("пробн", "фрагмент занят")) or "trial" in topic:
        return "trial"
    if any(marker in topic for marker in ("payment", "pricing", "discount", "installment")):
        return "payment"
    if any(marker in topic for marker in ("documents", "tax", "matkap", "certificate")):
        return "documents"
    if has_marker(text, "онлайн"):
        return "online_course"
    if has_any_marker(text, ("очно", "сретенка", "красносельск", "мфти", "пацаева", "долгопруд")):
        return "offline_course"
    if any(marker in topic for marker in ("program", "schedule", "format", "trial")):
        return "regular_course"
    return "unknown"


def build_semantic_flags(
    slots: LeadSlots,
    text: str,
    current_text: str,
    *,
    blockers: Sequence[str],
) -> tuple[str, ...]:
    flags: list[str] = []
    if blockers:
        flags.append("p0_blocks_qualification")
    if slots.grade and slots.subject:
        flags.append("class_and_subject_known")
    if slots.student_name:
        flags.append("student_known_do_not_reask_name")
    if slots.phone_known:
        flags.append("phone_known_do_not_reask")
    if has_any_marker(current_text, OFF_TOPIC_MARKERS):
        flags.append("off_topic_redirect")
    if "?" not in text and not current_text.endswith("?"):
        flags.append("may_be_context_update")
    return tuple(dict.fromkeys(flags))


def extract_grade(text: str) -> str:
    match = re.search(r"\b(?P<grade>[1-9]|1[01])\s*(?:класс[ае]?|кл\.?)\b", text)
    if match:
        return match.group("grade")
    if has_marker(text, "огэ"):
        return "9"
    if has_marker(text, "егэ"):
        return "11"
    if anchored_bare_grade_enabled():
        grade, _quote = extract_anchored_bare_grade_with_quote(text)
        return grade
    return ""


def anchored_bare_grade_enabled() -> bool:
    return str(os.getenv(ANCHORED_BARE_GRADE_ENV) or "").strip().lower() in _TRUE_ENV_VALUES


def extract_anchored_bare_grade_with_quote(text: str) -> tuple[str, str]:
    normalized = normalize_text(text)
    candidates: list[tuple[str, tuple[int, int], str]] = []
    for match in _BARE_GRADE_CANDIDATE_RE.finditer(normalized):
        grade = match.group("grade")
        span = match.span("grade")
        if not _is_safe_bare_grade_candidate(normalized, span=span):
            continue
        phrase = _short_phrase_around_span(normalized, span=span)
        if not _phrase_has_bare_grade_anchor(phrase):
            continue
        candidates.append((grade, span, phrase))
    if len(candidates) != 1:
        return "", ""
    grade, _span, phrase = candidates[0]
    return grade, phrase


def _is_safe_bare_grade_candidate(text: str, *, span: tuple[int, int]) -> bool:
    start, end = span
    before = text[max(0, start - 24) : start]
    after = text[end : min(len(text), end + 32)]
    if _span_overlaps_any(span, _PHONE_RE.finditer(text)):
        return False
    if _span_overlaps_any(span, _TIME_RE.finditer(text)):
        return False
    if _span_overlaps_any(span, _DATE_RE.finditer(text)):
        return False
    if _range_touches_span(text, span=span):
        return False
    if _MONEY_AFTER_RE.search(after):
        return False
    if _AGE_AFTER_RE.search(after) or _AGE_BEFORE_RE.search(before):
        return False
    if _COUNT_AFTER_RE.search(after) or _COUNT_BEFORE_RE.search(before):
        return False
    if re.search(r"\bв\s*$", before, re.I):
        return False
    return True


def _span_overlaps_any(span: tuple[int, int], matches: Any) -> bool:
    start, end = span
    for match in matches:
        if start < match.end() and end > match.start():
            return True
    return False


def _range_touches_span(text: str, *, span: tuple[int, int]) -> bool:
    start, end = span
    left = text[max(0, start - 3) : start]
    right = text[end : min(len(text), end + 3)]
    return bool(re.search(r"\d\s*[-–—]\s*$", left) or re.search(r"^\s*[-–—]\s*\d", right))


def _short_phrase_around_span(text: str, *, span: tuple[int, int]) -> str:
    start, end = span
    left = max(text.rfind(separator, 0, start) for separator in (".", "!", "?", ";", "\n", ","))
    phrase_start = 0 if left < 0 else left + 1
    right_candidates = [text.find(separator, end) for separator in (".", "!", "?", ";", "\n", ",")]
    right_candidates = [candidate for candidate in right_candidates if candidate >= 0]
    phrase_end = min(right_candidates) if right_candidates else len(text)
    phrase = text[phrase_start:phrase_end].strip()
    if len(phrase) <= 140:
        return phrase
    window_start = max(0, start - 60)
    window_end = min(len(text), end + 60)
    return text[window_start:window_end].strip()


def _phrase_has_bare_grade_anchor(phrase: str) -> bool:
    if not phrase:
        return False
    subject = extract_subjects(phrase)
    if subject and "," not in subject:
        return True
    if extract_format(phrase):
        return True
    return bool(re.search(r"\b(?:класс\w*|кл\.?|огэ|егэ)\b", phrase, re.I))


def extract_subjects(text: str) -> str:
    subjects: list[str] = []
    for marker, subject in SUBJECT_MARKERS:
        if has_marker(text, marker) and not _marker_is_explicitly_negated_before_correction(text, marker):
            subjects.append(subject)
    return ", ".join(dict.fromkeys(subjects))


def _marker_is_explicitly_negated_before_correction(text: str, marker: str) -> bool:
    normalized = normalize_text(text)
    needle = normalize_text(marker)
    if not needle:
        return False
    return bool(
        re.search(
            rf"(?<![0-9a-zа-я])не\s+{re.escape(needle)}[0-9a-zа-я]*[^.?!\n]{{0,80}}(?<![0-9a-zа-я])а(?![0-9a-zа-я])",
            normalized,
        )
    )


def _client_only_recent_text(recent_messages: Sequence[str]) -> str:
    parts: list[str] = []
    for item in recent_messages:
        for raw_line in str(item or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            lowered = line.casefold()
            if lowered.startswith(("ответ:", "бот:", "assistant:", "bot:")):
                continue
            if lowered.startswith(("клиент:", "client:", "user:")) and ":" in line:
                line = line.split(":", 1)[1].strip()
            if line:
                parts.append(line)
    return "\n".join(parts)


def extract_format(text: str) -> str:
    normalized = normalize_text(text)
    online_hit = (
        re.search(r"(?<![a-zа-яё])онлайн(?:[а-яёa-z-]*)?\b", normalized)
        or has_marker(normalized, "дистанц")
        or has_marker(normalized, "вебинар")
    )
    offline_hit = (
        re.search(r"(?<![a-zа-яё])очн(?:о|ый|ая|ое|ые|ых|ым|ом|ую|ого|ому|ыми)?\b", normalized)
        or re.search(r"(?<![a-zа-яё])офлайн(?:[а-яёa-z-]*)?\b", normalized)
        or has_marker(normalized, "в классе")
    )
    online_negated = bool(re.search(r"\b(?:не|только\s+не)\s+онлайн\w*\b", normalized))
    offline_negated = bool(re.search(r"\b(?:не|только\s+не)\s+очн\w*\b|\b(?:не|только\s+не)\s+офлайн\w*\b", normalized))
    if (
        online_hit
        and offline_hit
        and not online_negated
        and not offline_negated
        and (
            has_marker(normalized, "или")
            or (
                anchored_bare_grade_enabled()
                and re.search(r"\bи\b", normalized)
                and _CHOICE_VERBS_RE.search(normalized)
            )
        )
    ):
        return ""
    if online_hit and not online_negated and not (offline_hit and not offline_negated):
        return "online"
    if offline_hit and not offline_negated:
        return "offline"
    if online_hit and not online_negated:
        return "online"
    return ""


def extract_city_location(text: str) -> tuple[str, str]:
    if has_any_marker(text, ("долгопруд", "мфти", "пацаева")):
        return "Долгопрудный", "МФТИ/Пацаева"
    if has_marker(text, "сретенка"):
        return "Москва", "Сретенка"
    if has_marker(text, "красносельск"):
        return "Москва", "Верхняя Красносельская"
    if has_marker(text, "москв"):
        return "Москва", ""
    if has_marker(text, "менделеево"):
        return "Менделеево", "Менделеево"
    return "", ""


def extract_goal(text: str) -> str:
    if has_marker(text, "олимпиад"):
        return "олимпиадная подготовка"
    if has_marker(text, "огэ"):
        return "подготовка к ОГЭ"
    if has_marker(text, "егэ"):
        return "подготовка к ЕГЭ"
    if has_any_marker(text, ("подтянуть", "пробел", "школьн")):
        return "подтянуть школьную базу"
    if has_any_marker(text, ("углуб", "сильн", "глубже")):
        return "углублённое обучение"
    return ""


def extract_product(text: str) -> str:
    if has_any_marker(text, ("лвш", "выездн", "менделеево")):
        return "ЛВШ"
    if has_any_marker(text, ("лагер", "лш")):
        return "летняя школа"
    if has_marker(text, "интенсив"):
        return "интенсив"
    if has_marker(text, "курс"):
        return "курс"
    return ""


def extract_camp_direction(text: str) -> str:
    if has_any_marker(text, ("выезд", "менделеево", "прожив")):
        return "выездная"
    if has_any_marker(text, ("город", "москва")):
        return "городская"
    return ""


def extract_shift(text: str) -> str:
    match = re.search(r"\b\d{1,2}\s*[-–—]\s*\d{1,2}\s+[а-я]+", text)
    return match.group(0) if match else ""


def normalize_brand(value: Any) -> str:
    text = normalize_text(value)
    if text in {"foton", "фотон"}:
        return "foton"
    if text in {"unpk", "унпк", "унпк мфти", "мфти"}:
        return "unpk"
    return "unknown"


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").casefold().replace("ё", "е").split())
