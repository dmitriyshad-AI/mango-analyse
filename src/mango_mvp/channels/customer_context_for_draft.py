from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from mango_mvp.customer_timeline.context_provider import (
    get_customer_context_for_phone,
    normalize_phone_for_match,
)


CUSTOMER_CONTEXT_FOR_DRAFT_SCHEMA_VERSION = "customer_context_for_draft_v1"

ROUTE_DRAFT_FOR_MANAGER = "draft_for_manager"
ROUTE_MANAGER_ONLY = "manager_only"

CHILD_CLARIFICATION_TEXT = "Подскажите, пожалуйста, про кого из детей идет речь?"
SAFE_DEFAULT_DRAFT_TEXT = "Здравствуйте! Спасибо за сообщение. Передам вопрос менеджеру, он вернется с проверенным ответом."

PAYMENT_OR_DOCUMENT_MARKERS = (
    "оплат",
    "платеж",
    "платёж",
    "счет",
    "счёт",
    "квитанц",
    "чек",
    "касс",
    "возврат",
    "рассроч",
    "маткапитал",
    "материнск",
    "налогов",
    "справк",
    "договор",
    "документ",
)

TALLANTO_MISSING_STATUSES = {
    "missing",
    "no_match",
    "not_found",
    "unmatched",
    "none",
    "нет",
    "не найден",
    "не найдено",
}

NEW_LEAD_MARKERS = (
    "new",
    "new_lead",
    "lead_new",
    "первич",
    "новый",
    "новая",
    "новое",
    "первое обращение",
)


@dataclass(frozen=True)
class CustomerDraftContext:
    warnings: tuple[str, ...] = ()
    required_clarifications: tuple[str, ...] = ()
    route: str = "draft_for_manager"
    crm_recommendations: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    safe_context: Mapping[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> Mapping[str, Any]:
        return {
            "schema_version": CUSTOMER_CONTEXT_FOR_DRAFT_SCHEMA_VERSION,
            "warnings": list(self.warnings),
            "required_clarifications": list(self.required_clarifications),
            "route": self.route,
            "crm_recommendations": [dict(item) for item in self.crm_recommendations],
            "safe_context": dict(self.safe_context),
        }


def build_customer_context_for_draft(
    source: Mapping[str, Any] | str,
    *,
    incoming_text: str = "",
    topic_id: str = "",
    route: str = ROUTE_DRAFT_FOR_MANAGER,
    amo_context: Mapping[str, Any] | None = None,
    tallanto_context: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    timeline_context: Mapping[str, Any] | None = None,
    timeline_db: Path | str | None = None,
    fallback_rows: Sequence[Mapping[str, Any]] | None = None,
    tenant_id: str = "foton",
    crm_recommendations: Sequence[Mapping[str, Any]] | None = None,
) -> CustomerDraftContext | dict[str, Any]:
    """Build read-only customer context for a manager draft.

    Backward-compatible mode: pass a mapping as the only argument and receive
    `CustomerDraftContext`. New T6 mode: pass a phone string plus prepared
    AMO/Tallanto/timeline snapshots and receive a prompt-ready dict.
    """

    if isinstance(source, Mapping) and _uses_legacy_source_api(
        incoming_text=incoming_text,
        topic_id=topic_id,
        route=route,
        amo_context=amo_context,
        tallanto_context=tallanto_context,
        timeline_context=timeline_context,
        timeline_db=timeline_db,
        fallback_rows=fallback_rows,
        crm_recommendations=crm_recommendations,
    ):
        return _build_legacy_customer_context_for_draft(source)
    return _build_prompt_customer_context_for_draft(
        str(source),
        incoming_text=incoming_text,
        topic_id=topic_id,
        route=route,
        amo_context=amo_context,
        tallanto_context=tallanto_context,
        timeline_context=timeline_context,
        timeline_db=timeline_db,
        fallback_rows=fallback_rows,
        tenant_id=tenant_id,
        crm_recommendations=crm_recommendations,
    )


def _build_legacy_customer_context_for_draft(source: Mapping[str, Any]) -> CustomerDraftContext:
    warnings: list[str] = []
    clarifications: list[str] = []
    route = ROUTE_DRAFT_FOR_MANAGER

    tallanto_ids = tuple(str(item).strip() for item in source.get("tallanto_student_ids") or () if str(item).strip())
    risk_flags = {str(item).strip() for item in source.get("risk_flags") or () if str(item).strip()}
    topic = str(source.get("topic") or "").lower()
    is_new_lead = bool(source.get("is_new_lead"))

    if len(tallanto_ids) > 1 or "multiple_tallanto" in risk_flags or "family_phone" in risk_flags:
        warnings.append("На телефоне несколько учеников или семейная история.")
        clarifications.append("Уточнить, про кого из детей идет речь.")
    if "no_reliable_tallanto" in risk_flags and is_new_lead:
        warnings.append("Клиент может быть новым лидом без карточки в Tallanto.")
    elif "no_reliable_tallanto" in risk_flags:
        warnings.append("Нет надежной связки с Tallanto, менеджеру нужно проверить вручную.")
    if any(marker in topic for marker in ("оплат", "возврат", "документ", "справ", "маткап", "налог")):
        route = ROUTE_MANAGER_ONLY
        warnings.append("Финансы или документы требуют проверки менеджером.")

    recommendations = tuple(build_crm_recommendation(item) for item in source.get("crm_recommendations") or ())
    return CustomerDraftContext(
        warnings=tuple(dict.fromkeys(warnings)),
        required_clarifications=tuple(dict.fromkeys(clarifications)),
        route=route,
        crm_recommendations=recommendations,
        safe_context={
            "customer_id": source.get("customer_id"),
            "amo_deal_id": source.get("amo_deal_id"),
            "has_tallanto_match": bool(tallanto_ids),
            "risk_flags": sorted(risk_flags),
        },
    )


def _build_prompt_customer_context_for_draft(
    phone: str,
    *,
    incoming_text: str = "",
    topic_id: str = "",
    route: str = ROUTE_DRAFT_FOR_MANAGER,
    amo_context: Mapping[str, Any] | None = None,
    tallanto_context: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    timeline_context: Mapping[str, Any] | None = None,
    timeline_db: Path | str | None = None,
    fallback_rows: Sequence[Mapping[str, Any]] | None = None,
    tenant_id: str = "foton",
    crm_recommendations: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    normalized_phone = normalize_phone_for_match(phone)
    amo = dict(amo_context or {})
    timeline = dict(timeline_context or {})
    timeline_warnings: list[str] = []
    if not timeline and timeline_db is not None:
        timeline = get_customer_context_for_phone(
            phone,
            timeline_db=timeline_db,
            fallback_rows=fallback_rows or (),
            tenant_id=tenant_id,
        )
        timeline_warnings = [safe_text(item) for item in timeline.get("warnings", []) if safe_text(item)]

    students = extract_tallanto_students(tallanto_context)
    tallanto_status = tallanto_match_status(tallanto_context)
    tallanto_found = has_tallanto_match(tallanto_context, students=students, match_status=tallanto_status)
    new_lead = is_new_lead(amo)
    family_phone_risk = has_family_phone_risk(amo, tallanto_context, students=students)
    child_clarification_required = family_phone_risk or len(students) > 1
    payment_or_documents_review = requires_payment_or_document_review(incoming_text, topic_id=topic_id)

    warnings: list[str] = list(timeline_warnings)
    safety_flags: list[str] = ["no_auto_send", "read_only_context", "requires_manager_approval"]
    manager_checklist: list[str] = []
    required_questions: list[str] = []

    if child_clarification_required:
        warnings.append("family_phone_or_multiple_students")
        safety_flags.extend(["family_phone", "child_clarification_required"])
        required_questions.append(CHILD_CLARIFICATION_TEXT)
        manager_checklist.append(CHILD_CLARIFICATION_TEXT)

    if not tallanto_found and new_lead:
        warnings.append("tallanto_not_found_for_new_lead")
        manager_checklist.append("Tallanto не найден: для нового лида это предупреждение, не блокировка.")
    elif not tallanto_found:
        warnings.append("tallanto_not_found_check_manually")
        manager_checklist.append("Проверить Tallanto вручную перед точным ответом по ученику.")

    if payment_or_documents_review:
        safety_flags.append("payment_or_documents_manager_review_required")
        manager_checklist.append("Проверить оплату, документы или справки вручную перед ответом.")

    resolved_route = normalize_route(route)
    if payment_or_documents_review and resolved_route not in {ROUTE_DRAFT_FOR_MANAGER, ROUTE_MANAGER_ONLY}:
        resolved_route = ROUTE_DRAFT_FOR_MANAGER

    recommendations = build_crm_text_recommendations(
        crm_recommendations or (),
        phone=normalized_phone,
        child_clarification_required=child_clarification_required,
        payment_or_documents_review=payment_or_documents_review,
        tallanto_missing=not tallanto_found,
        new_lead=new_lead,
    )

    return {
        "schema_version": CUSTOMER_CONTEXT_FOR_DRAFT_SCHEMA_VERSION,
        "phone": normalized_phone,
        "route": resolved_route,
        "hard_block": False,
        "blocked": False,
        "requires_manager_review": True,
        "requires_child_clarification": child_clarification_required,
        "payment_or_documents_review_required": payment_or_documents_review,
        "safe_for_auto_send": False,
        "safe_draft_text": build_safe_draft_text(
            child_clarification_required=child_clarification_required,
            payment_or_documents_review=payment_or_documents_review,
        ),
        "required_questions": tuple(required_questions),
        "manager_checklist": tuple(dedupe_preserve_order(manager_checklist)),
        "warnings": tuple(dedupe_preserve_order(warnings)),
        "safety_flags": tuple(dedupe_preserve_order(safety_flags)),
        "crm_recommendations": tuple(recommendations),
        "sources": {
            "amo": {
                "present": bool(amo),
                "read_only": True,
                "is_new_lead": new_lead,
            },
            "tallanto": {
                "present": tallanto_found,
                "read_only": True,
                "match_status": tallanto_status,
                "students_count": len(students),
                "missing_is_hard_block": False,
            },
            "timeline": {
                "present": bool(timeline.get("found")) if timeline else False,
                "read_only": True,
                "primary_read_enabled": False,
                "used_as_automatic_truth": False,
                "summary": safe_text(timeline.get("summary")) if timeline else "",
            },
        },
        "draft_requirements": {
            "must_ask_child_clarification": child_clarification_required,
            "child_clarification_question": CHILD_CLARIFICATION_TEXT if child_clarification_required else "",
            "payment_or_documents_require_manager_review": payment_or_documents_review,
            "crm_recommendations_text_only": True,
            "requires_manager_approval": True,
        },
        "safety": customer_context_for_draft_safety_contract(),
    }


def customer_context_for_draft_safety_contract() -> dict[str, bool]:
    return {
        "read_amo": True,
        "read_tallanto": True,
        "read_customer_timeline": True,
        "timeline_primary_read_enabled": False,
        "write_crm": False,
        "write_amo": False,
        "write_tallanto": False,
        "write_customer_timeline_db": False,
        "send_messenger": False,
        "send_email": False,
        "live_send": False,
        "network_calls": False,
        "subprocess_calls": False,
        "run_asr": False,
        "run_ra": False,
        "requires_manager_approval": True,
    }


def build_crm_recommendation(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        text = str(value.get("text") or value.get("summary") or "").strip()
        target = str(value.get("target") or "AMO").strip() or "AMO"
    else:
        text = str(value or "").strip()
        target = "AMO"
    return {
        "target": target,
        "action": "text_suggestion",
        "text": text,
        "requires_manager_approval": True,
        "live_write_enabled": False,
        "live_write": False,
    }


def _uses_legacy_source_api(
    *,
    incoming_text: str,
    topic_id: str,
    route: str,
    amo_context: Mapping[str, Any] | None,
    tallanto_context: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
    timeline_context: Mapping[str, Any] | None,
    timeline_db: Path | str | None,
    fallback_rows: Sequence[Mapping[str, Any]] | None,
    crm_recommendations: Sequence[Mapping[str, Any]] | None,
) -> bool:
    return (
        not incoming_text
        and not topic_id
        and route == ROUTE_DRAFT_FOR_MANAGER
        and amo_context is None
        and tallanto_context is None
        and timeline_context is None
        and timeline_db is None
        and fallback_rows is None
        and crm_recommendations is None
    )


def extract_tallanto_students(
    tallanto_context: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
) -> tuple[dict[str, Any], ...]:
    if tallanto_context is None:
        return ()
    if isinstance(tallanto_context, Mapping):
        for key in ("students", "student_matches", "matches", "children"):
            value = tallanto_context.get(key)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                return tuple(dict(item) for item in value if isinstance(item, Mapping))
        student = tallanto_context.get("student")
        if isinstance(student, Mapping):
            return (dict(student),)
        names = split_multi_value(tallanto_context.get("student_names") or tallanto_context.get("ФИО ученика Tallanto"))
        if names:
            return tuple({"name": name} for name in names)
        if any(safe_text(tallanto_context.get(key)) for key in ("student_id", "tallanto_id", "ID Tallanto")):
            return (dict(tallanto_context),)
        return ()
    if isinstance(tallanto_context, Sequence) and not isinstance(tallanto_context, (str, bytes, bytearray)):
        return tuple(dict(item) for item in tallanto_context if isinstance(item, Mapping))
    return ()


def has_tallanto_match(
    tallanto_context: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
    *,
    students: Sequence[Mapping[str, Any]],
    match_status: str,
) -> bool:
    if students:
        return True
    if tallanto_context is None:
        return False
    normalized_status = match_status.casefold()
    if normalized_status in TALLANTO_MISSING_STATUSES:
        return False
    if any(marker in normalized_status for marker in ("no_match", "not_found", "unmatched")):
        return False
    if isinstance(tallanto_context, Mapping):
        if any(safe_text(tallanto_context.get(key)) for key in ("tallanto_id", "ID Tallanto", "student_id")):
            return True
        if normalized_status:
            return True
    return False


def tallanto_match_status(tallanto_context: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None) -> str:
    if isinstance(tallanto_context, Mapping):
        return safe_text(
            tallanto_context.get("match_status")
            or tallanto_context.get("tallanto_match_status")
            or tallanto_context.get("Статус матчинга Tallanto")
            or tallanto_context.get("status")
        )
    if isinstance(tallanto_context, Sequence) and not isinstance(tallanto_context, (str, bytes, bytearray)):
        return "multiple_tallanto_matches" if len(tallanto_context) > 1 else "single_tallanto_match"
    return ""


def has_family_phone_risk(
    amo_context: Mapping[str, Any],
    tallanto_context: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
    *,
    students: Sequence[Mapping[str, Any]],
) -> bool:
    if len(students) > 1:
        return True
    for key in ("family_phone", "multiple_students", "several_students"):
        if boolish(amo_context.get(key)):
            return True
    for key in ("children_count", "students_count", "student_count"):
        if int_or_zero(amo_context.get(key)) > 1:
            return True
    haystack = " ".join(
        safe_text(value)
        for value in (
            amo_context.get("phone_kind"),
            amo_context.get("risk"),
            amo_context.get("risk_class"),
            tallanto_match_status(tallanto_context),
        )
    ).casefold()
    return any(marker in haystack for marker in ("family", "семейн", "multiple", "несколько"))


def is_new_lead(amo_context: Mapping[str, Any]) -> bool:
    if boolish(amo_context.get("is_new_lead")):
        return True
    haystack = " ".join(
        safe_text(amo_context.get(key))
        for key in ("lead_status", "status", "stage", "pipeline_stage", "deal_status")
    ).casefold()
    return any(marker in haystack for marker in NEW_LEAD_MARKERS)


def requires_payment_or_document_review(incoming_text: str, *, topic_id: str = "") -> bool:
    haystack = f"{incoming_text} {topic_id}".casefold()
    return any(marker in haystack for marker in PAYMENT_OR_DOCUMENT_MARKERS)


def build_crm_text_recommendations(
    raw_recommendations: Sequence[Mapping[str, Any]],
    *,
    phone: str,
    child_clarification_required: bool,
    payment_or_documents_review: bool,
    tallanto_missing: bool,
    new_lead: bool,
) -> list[dict[str, Any]]:
    recommendations: list[dict[str, Any]] = []
    for item in raw_recommendations:
        text = safe_text(item.get("text") or item.get("summary") or item.get("title"))
        if not text:
            continue
        recommendations.append(crm_text_suggestion(text, target=safe_text(item.get("target") or "AMO")))

    if child_clarification_required:
        recommendations.append(crm_text_suggestion("Проверить в AMO/Tallanto, к какому ребенку относится обращение."))
    if payment_or_documents_review:
        recommendations.append(crm_text_suggestion("Проверить оплату, документы или справки перед ответом клиенту."))
    if tallanto_missing and new_lead:
        recommendations.append(
            crm_text_suggestion("Tallanto не найден: если лид новый, обновлять CRM только вручную после проверки.")
        )
    elif tallanto_missing:
        recommendations.append(crm_text_suggestion("Проверить связь AMO и Tallanto вручную."))

    if not recommendations and phone:
        recommendations.append(crm_text_suggestion(f"Проверить карточку клиента по телефону {phone} перед ответом."))
    return dedupe_recommendations(recommendations)


def crm_text_suggestion(text: str, *, target: str = "AMO") -> dict[str, Any]:
    return {
        "target": safe_text(target) or "AMO",
        "action": "text_suggestion",
        "text": safe_text(text),
        "requires_manager_approval": True,
        "live_write_enabled": False,
    }


def build_safe_draft_text(*, child_clarification_required: bool, payment_or_documents_review: bool) -> str:
    if child_clarification_required:
        return f"Здравствуйте! {CHILD_CLARIFICATION_TEXT}"
    if payment_or_documents_review:
        return (
            "Здравствуйте! Спасибо за вопрос. Передам его менеджеру, он вернется "
            "с проверенным ответом после ручной проверки."
        )
    return SAFE_DEFAULT_DRAFT_TEXT


def normalize_route(route: str) -> str:
    value = safe_text(route)
    if value == ROUTE_MANAGER_ONLY:
        return ROUTE_MANAGER_ONLY
    return ROUTE_DRAFT_FOR_MANAGER


def split_multi_value(value: Any) -> tuple[str, ...]:
    text = safe_text(value)
    if not text:
        return ()
    return tuple(part for part in (safe_text(item) for item in re.split(r"[|;,]+", text)) if part)


def dedupe_recommendations(items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in items:
        key = (safe_text(item.get("target")), safe_text(item.get("action")), safe_text(item.get("text")))
        if key in seen or not key[2]:
            continue
        seen.add(key)
        result.append(dict(item))
    return result


def dedupe_preserve_order(values: Sequence[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = safe_text(value)
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return safe_text(value).casefold() in {"1", "true", "yes", "y", "да", "истина"}


def int_or_zero(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def safe_text(value: Any) -> str:
    return str(value or "").strip()
