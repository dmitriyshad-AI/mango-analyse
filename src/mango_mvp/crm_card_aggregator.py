from __future__ import annotations

import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from mango_mvp.quality.tenant_text_normalizer import normalize_manager_text


CRM_CARD_AGGREGATOR_SCHEMA_VERSION = "crm_card_aggregator_v1"
CRM_CARD_AGGREGATOR_FLAG = "CRM_CARD_AGGREGATOR_ENABLED"

CONTACT_CARD_FIELDS = (
    "Статус матчинга",
    "AI-приоритет",
    "AI-рекомендованный следующий шаг",
    "Последняя AI-сводка",
    "Авто история общения",
)

DEAL_CARD_REQUIRED_FIELDS = (
    "AI-сводка по сделке",
    "AI-история по сделке",
    "AI-рекомендованный следующий шаг",
    "AI-дата следующего касания",
    "AI-фактический статус сделки",
    "AI-приоритет сделки",
    "AI-актуальные возражения",
    "AI-основание рекомендации",
    "AI-качество привязки к сделке",
    "AI-предупреждение по сделке",
    "AI-Tallanto статус по сделке",
    "AI-дата обновления сделки",
)

DEAL_CARD_OPTIONAL_FIELDS = (
    "AI-бюджет диапазон",
    "AI-бюджет комментарий",
    "AI-чувствительность к цене",
    "AI-интерес к скидке",
)

MANUAL_HISTORY_FIELD = "История общения"
CONTACT_AMO_HISTORY_FIELD = "Авто история общения"
CONTACT_AUTO_HISTORY_LIMIT = 1600
DEAL_HISTORY_LIMIT = 4500
SHORT_FIELD_LIMIT = 350
DEFAULT_TEXT_LIMIT = 1300
OBJECTION_LIMIT = 90
COMPACTION_SUFFIX = " [сжато]"
HISTORY_TRUNCATION_SUFFIX = "…"

PAYMENT_SIGNAL_TYPES = {"paid_no_access", "tallanto_payment", "payment_without_access"}
HOT_SIGNAL_TYPES = {"hot_lead_silent_7d", "price_interest", "paid_no_access"}
IDENTITY_CONFLICT_MARKERS = ("ambiguous", "duplicate", "shared", "family")
SERVICE_SNAPSHOT_EVENT_TYPES = {"amo_contact_snapshot", "tallanto_student_snapshot", "amo_deal_stage"}


@dataclass(frozen=True)
class ManagerFacts:
    summary: str = ""
    history: str = ""
    chronology: str = ""
    objections: str = ""
    next_step: str = ""
    follow_up_date: str = ""
    priority: str = ""
    probability: str = ""
    recommended_product: str = ""
    products_interest: str = ""
    tallanto_history: str = ""
    budget_range: str = ""
    budget_comment: str = ""
    price_sensitivity: str = ""
    discount_interest: str = ""
    amo_contact_id: str = ""
    amo_lead_id: str = ""
    existing_amo_fields: Mapping[str, Any] | None = None


def card_aggregator_enabled() -> bool:
    return os.getenv(CRM_CARD_AGGREGATOR_FLAG, "0").strip().lower() in {"1", "true", "yes", "on"}


def manager_facts_from_row(row: Mapping[str, Any] | None) -> ManagerFacts:
    row = row or {}
    return ManagerFacts(
        summary=first_text(row, ("Краткое резюме последнего свежего звонка", "Последняя AI-сводка", "summary")),
        history=first_text(row, ("Краткая история общения", "history", "latest_history_summary")),
        chronology=first_text(row, ("Хронология общения (последние 5 касаний)", "chronology")),
        objections=first_text(row, ("Возражения", "AI-актуальные возражения", "objections")),
        next_step=first_text(row, ("Следующий шаг", "AI-рекомендованный следующий шаг", "next_step", "last_next_step_action")),
        follow_up_date=first_text(row, ("Рекомендуемая дата следующего контакта", "AI-дата следующего касания", "recommended_followup_date")),
        priority=first_text(row, ("Приоритет лида", "AI-приоритет", "lead_priority", "priority")),
        probability=first_text(row, ("Вероятность продажи, %", "sale_probability_pct", "sale_probability_percent")),
        recommended_product=first_text(row, ("Рекомендуемый продукт", "recommended_product")),
        products_interest=first_text(row, ("Продукты интереса", "interests_products", "products")),
        tallanto_history=first_text(row, ("История общения Tallanto", "AI-Tallanto статус по сделке", "tallanto_history")),
        budget_range=first_text(row, ("AI-бюджет диапазон", "budget_range")),
        budget_comment=first_text(row, ("AI-бюджет комментарий", "budget", "budget_comment")),
        price_sensitivity=first_text(row, ("AI-чувствительность к цене", "price_sensitivity")),
        discount_interest=first_text(row, ("AI-интерес к скидке", "discount_interest")),
        amo_contact_id=first_text(row, ("AMO contact IDs", "amo_contact_id", "selected_contact_id")),
        amo_lead_id=first_text(row, ("selected_deal_id", "AMO lead IDs", "amo_lead_id", "lead_id")),
        existing_amo_fields=row.get("existing_amo_fields") if isinstance(row.get("existing_amo_fields"), Mapping) else None,
    )


def build_crm_card_projection(
    profile: Mapping[str, Any],
    *,
    manager_facts: Mapping[str, Any] | ManagerFacts | None = None,
    selected_amo_lead_id: str | None = None,
    existing_amo_fields: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    facts = manager_facts if isinstance(manager_facts, ManagerFacts) else manager_facts_from_row(manager_facts)
    if existing_amo_fields is None:
        existing_amo_fields = facts.existing_amo_fields or {}
    if not profile.get("found", True):
        return _missing_profile_projection(profile, facts, existing_amo_fields)

    customer = _mapping(profile.get("customer"))
    timeline_items = [_mapping(item) for item in _sequence(profile.get("timeline", {}).get("items"))]
    signals = [_mapping(item) for item in _sequence(profile.get("signals"))]
    conflicts = [_mapping(item) for item in _sequence(profile.get("conflicts", {}).get("items"))]
    identity_links = [_mapping(item) for item in _sequence(profile.get("identity_links"))]
    opportunities = [_mapping(item) for item in _sequence(profile.get("opportunities"))]
    manager_projection = _mapping(profile.get("manager_projection"))
    manager_opportunities = [_mapping(item) for item in _sequence(manager_projection.get("opportunities"))]
    bot_items = [_mapping(item) for item in _sequence(profile.get("bot_context", {}).get("items"))]
    history_items = _history_events(timeline_items)
    latest_call = _latest_history_call(timeline_items)

    generated_at = _snapshot_time(profile, customer)
    identity_status = _safe_text(customer.get("identity_status")) or "unknown"
    identity_ambiguous = _identity_is_ambiguous(identity_status, identity_links, conflicts)
    blockers = _base_blockers(profile, identity_ambiguous=identity_ambiguous)
    lead_id = _safe_text(selected_amo_lead_id or facts.amo_lead_id or _manager_amo_id(manager_projection, "amo_lead_ids") or _identity_value(identity_links, "amo_lead_id"))
    contact_id = _safe_text(facts.amo_contact_id or _manager_amo_id(manager_projection, "amo_contact_ids") or _identity_value(identity_links, "amo_contact_id"))
    source_counts = _source_counts(timeline_items)
    what_collected = _what_collected(customer, timeline_items, signals, conflicts, bot_items)
    latest_summary = _latest_summary(latest_call, facts, history_items)
    next_step = _next_step(signals, facts, latest_call)
    objections = _objections(facts, latest_call)
    call_interests = _call_analysis_field(latest_call, "interests")
    call_target_product = _call_analysis_field(latest_call, "target_product")
    latest_call_key = _event_key(latest_call)
    auto_history = _contact_auto_history(
        latest_summary=latest_summary,
        latest_call=latest_call,
        timeline_items=history_items,
        signals=signals,
        facts=facts,
        objections=objections,
        interests=call_interests,
        target_product=call_target_product,
    )
    priority = _priority(signals, facts, history_items or timeline_items)
    match_status = _match_status(identity_status, identity_links, conflicts)
    contact_payload = {
        "Статус матчинга": match_status,
        "AI-приоритет": priority,
        "AI-рекомендованный следующий шаг": fit_text(next_step, 800),
        "Последняя AI-сводка": fit_text(latest_summary, 1200),
        "Авто история общения": fit_text(auto_history, CONTACT_AUTO_HISTORY_LIMIT),
    }
    contact_payload = _drop_empty(contact_payload)

    selected_opportunity = _select_opportunity(manager_opportunities or opportunities, lead_id)
    deal_blockers = list(blockers)
    if identity_ambiguous:
        deal_blockers.append("p9_ambiguous_identity_manual_review")
    if lead_id and selected_opportunity.get("opportunity_id") and selected_opportunity.get("source_system") != "amocrm_snapshot":
        deal_blockers.append("selected_deal_not_confirmed_by_amocrm_opportunity")
    if lead_id and not lead_id.isdigit():
        deal_blockers.append("amo_lead_id_masked_in_read_api")
    if not lead_id and not selected_opportunity:
        deal_blockers.append("amo_lead_id_not_available_in_profile")
    deal_events = _deal_events(history_items, selected_opportunity.get("opportunity_id"))
    history_scope = "история по сделке" if selected_opportunity.get("opportunity_id") else "история по клиенту, не по конкретной сделке"
    deal_payload = {
        "AI-сводка по сделке": fit_text(_deal_summary(selected_opportunity, history_items, latest_call, latest_summary, history_scope), 1600),
        "AI-история по сделке": fit_history_text(
            _deal_history(
                deal_events or history_items,
                history_scope=history_scope,
                compact_event_keys={latest_call_key} if latest_call_key else set(),
                compact_full_texts={_normalized_text(latest_summary)} if latest_summary else set(),
            ),
            DEAL_HISTORY_LIMIT,
        ),
        "AI-рекомендованный следующий шаг": fit_text(next_step, SHORT_FIELD_LIMIT),
        "AI-дата следующего касания": fit_text(facts.follow_up_date or _followup_from_signal(signals), SHORT_FIELD_LIMIT),
        "AI-фактический статус сделки": fit_text(_deal_status(selected_opportunity, facts, timeline_items), DEFAULT_TEXT_LIMIT),
        "AI-приоритет сделки": fit_text(_deal_priority(priority, signals), SHORT_FIELD_LIMIT),
        "AI-актуальные возражения": fit_text(objections or "Актуальные возражения в истории не выделены.", DEFAULT_TEXT_LIMIT),
        "AI-основание рекомендации": fit_text(_recommendation_reason(signals, history_items, facts), DEFAULT_TEXT_LIMIT),
        "AI-качество привязки к сделке": fit_text(_binding_quality(identity_status, identity_links, selected_opportunity, history_scope), DEFAULT_TEXT_LIMIT),
        "AI-предупреждение по сделке": fit_text(_warnings(deal_blockers, conflicts, signals), DEFAULT_TEXT_LIMIT),
        "AI-Tallanto статус по сделке": fit_text(_tallanto_status(timeline_items, facts), 1600),
        "AI-дата обновления сделки": generated_at,
        "AI-бюджет диапазон": facts.budget_range,
        "AI-бюджет комментарий": facts.budget_comment,
        "AI-чувствительность к цене": facts.price_sensitivity,
        "AI-интерес к скидке": facts.discount_interest,
    }
    deal_payload = _drop_empty(deal_payload)

    ready = not blockers and bool(contact_payload)
    if contact_id and not contact_id.isdigit():
        blockers.append("amo_contact_id_masked_in_read_api")
        ready = False
    elif not contact_id:
        blockers.append("amo_contact_id_not_available_in_profile")
        ready = False
    deal_ready = ready and not deal_blockers and bool(lead_id or selected_opportunity)

    return {
        "schema_version": CRM_CARD_AGGREGATOR_SCHEMA_VERSION,
        "customer_id": _safe_text(profile.get("customer_id") or customer.get("customer_id")),
        "snapshot_as_of": generated_at,
        "last_event_at": _safe_text(profile.get("last_event_at")),
        "identity_status": identity_status,
        "source_counts": dict(source_counts),
        "what_collected": what_collected,
        "what_already_in_amo": _what_already_in_amo(profile, existing_amo_fields, lead_id=lead_id, contact_id=contact_id),
        "contact_card": {
            "fields": contact_payload,
            "ready_for_amo": ready,
            "blockers": blockers,
        },
        "deal_card": {
            "fields": deal_payload,
            "ready_for_amo": deal_ready,
            "blockers": deal_blockers,
            "selected_amo_lead_id": lead_id,
            "selected_opportunity_id": _safe_text(selected_opportunity.get("opportunity_id")),
            "scope": history_scope,
        },
        "workbook": {
            "ready": "да" if ready and deal_ready else "нет",
            "blockers": " | ".join(_dedupe(blockers + deal_blockers)),
            "what_goes_to_amo": render_payload_preview(contact_payload, deal_payload),
        },
        "bot_safety": _bot_safety_summary(timeline_items, signals, bot_items),
        "safety": {
            "read_only": True,
            "write_amo": False,
            "write_tallanto": False,
            "uses_manager_review_text": True,
            "tallanto_amounts_bot_safe": False,
        },
    }


def apply_contact_card_payload(row: Mapping[str, Any]) -> dict[str, Any] | None:
    if not card_aggregator_enabled():
        return None
    payload = _json_payload(row.get("crm_card_contact_payload_json"))
    if payload is None:
        payload = {field: row.get(field) for field in CONTACT_CARD_FIELDS if _safe_text(row.get(field))}
    payload = {field: _safe_text(payload.get(field)) for field in CONTACT_CARD_FIELDS if _safe_text(payload.get(field))}
    return payload or None


def apply_deal_card_payload(payload: Mapping[str, Any], card_fields: Mapping[str, Any] | None) -> dict[str, str]:
    if not card_aggregator_enabled() or not card_fields:
        return {str(key): _safe_text(value) for key, value in payload.items()}
    merged = dict(payload)
    for field in (*DEAL_CARD_REQUIRED_FIELDS, *DEAL_CARD_OPTIONAL_FIELDS):
        if _safe_text(card_fields.get(field)):
            merged[field] = _safe_text(card_fields[field])
    return {str(key): _safe_text(value) for key, value in merged.items() if _safe_text(value)}


def contact_ready_blocker(row: Mapping[str, Any]) -> str:
    if not card_aggregator_enabled():
        return ""
    ready = _safe_text(row.get("crm_card_ready") or row.get("Готово"))
    blockers = _safe_text(row.get("crm_card_blockers") or row.get("Блокеры"))
    if ready and ready.casefold() not in {"да", "yes", "true", "1"}:
        return "crm_card_not_ready:" + (blockers or "manual_review_required")
    return ""


def render_payload_preview(contact_payload: Mapping[str, Any], deal_payload: Mapping[str, Any]) -> str:
    parts: list[str] = []
    if contact_payload:
        parts.append("Контакт:\n" + "\n".join(f"- {key}: {_safe_text(value)}" for key, value in contact_payload.items()))
    if deal_payload:
        parts.append("Сделка:\n" + "\n".join(f"- {key}: {_safe_text(value)}" for key, value in deal_payload.items()))
    return "\n\n".join(parts)


def fit_text(value: Any, limit: int) -> str:
    text = normalize_manager_text(value)
    if len(text) <= limit:
        return text
    budget = max(20, limit - len(COMPACTION_SUFFIX))
    chunk = text[:budget].rstrip()
    cut = max(chunk.rfind(" "), chunk.rfind(","), chunk.rfind(";"), chunk.rfind("."))
    if cut >= int(budget * 0.58):
        chunk = chunk[:cut]
    return chunk.rstrip(" ,;:.") + COMPACTION_SUFFIX


def fit_history_text(value: Any, limit: int) -> str:
    text = normalize_manager_text(value)
    if len(text) <= limit:
        return text
    budget = max(20, limit - len(HISTORY_TRUNCATION_SUFFIX))
    chunk = text[:budget].rstrip()
    cut = max(chunk.rfind("\n"), chunk.rfind(". "), chunk.rfind("; "), chunk.rfind(", "), chunk.rfind(" "))
    if cut >= int(budget * 0.58):
        chunk = chunk[:cut]
    return chunk.rstrip(" ,;:.") + HISTORY_TRUNCATION_SUFFIX


def compact_objection_explicit(value: Any, *, limit: int = OBJECTION_LIMIT) -> str:
    text = normalize_manager_text(value).strip(" .;:")
    if len(text) <= limit:
        return text
    return fit_text(text, limit)


def first_text(row: Mapping[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = _safe_text(row.get(key))
        if value:
            return value
    return ""


def _missing_profile_projection(
    profile: Mapping[str, Any],
    facts: ManagerFacts,
    existing_amo_fields: Mapping[str, Any],
) -> dict[str, Any]:
    fallback_summary = facts.summary or facts.history
    contact_payload = _drop_empty(
        {
            "Статус матчинга": "unmatched",
            "AI-приоритет": facts.priority or "review",
            "AI-рекомендованный следующий шаг": facts.next_step or "Проверить клиента вручную: профиль в customer_timeline не найден.",
            "Последняя AI-сводка": fallback_summary,
            "Авто история общения": fit_text(
                "\n\n".join(
                    part
                    for part in (
                        f"Сводка клиента:\n{fallback_summary}" if fallback_summary else "",
                        f"Возражения: {facts.objections}" if facts.objections else "",
                        f"Следующий шаг: {facts.next_step}" if facts.next_step else "",
                    )
                    if part
                ),
                CONTACT_AUTO_HISTORY_LIMIT,
            ),
        }
    )
    blockers = ["customer_profile_not_found"]
    return {
        "schema_version": CRM_CARD_AGGREGATOR_SCHEMA_VERSION,
        "customer_id": _safe_text(profile.get("customer_id")),
        "snapshot_as_of": _safe_text(profile.get("snapshot_as_of")),
        "last_event_at": _safe_text(profile.get("last_event_at")),
        "identity_status": "unmatched",
        "source_counts": {},
        "what_collected": fallback_summary,
        "what_already_in_amo": _what_already_in_amo(profile, existing_amo_fields),
        "contact_card": {"fields": contact_payload, "ready_for_amo": False, "blockers": blockers},
        "deal_card": {"fields": {}, "ready_for_amo": False, "blockers": blockers, "scope": "нет профиля"},
        "workbook": {
            "ready": "нет",
            "blockers": " | ".join(blockers),
            "what_goes_to_amo": render_payload_preview(contact_payload, {}),
        },
        "bot_safety": {"bot_safe_fields": [], "manager_only_fields": list(CONTACT_CARD_FIELDS)},
        "safety": {"read_only": True, "write_amo": False, "write_tallanto": False},
    }


def _snapshot_time(profile: Mapping[str, Any], customer: Mapping[str, Any]) -> str:
    return (
        _safe_text(profile.get("snapshot_as_of"))
        or _safe_text(profile.get("last_event_at"))
        or first_text(customer, ("last_seen_at", "updated_at", "created_at"))
    )


def _base_blockers(profile: Mapping[str, Any], *, identity_ambiguous: bool) -> list[str]:
    blockers: list[str] = []
    if identity_ambiguous:
        blockers.append("p9_ambiguous_identity_manual_review")
    readiness = _mapping(profile.get("readiness"))
    if int(readiness.get("open_conflicts") or 0) > 0:
        blockers.append("open_conflicts_require_manager_review")
    return _dedupe(blockers)


def _identity_is_ambiguous(
    identity_status: str,
    identity_links: Sequence[Mapping[str, Any]],
    conflicts: Sequence[Mapping[str, Any]],
) -> bool:
    if identity_status.casefold() == "ambiguous":
        return True
    for link in identity_links:
        if _safe_text(link.get("match_class")).casefold() in {"ambiguous", "duplicate"}:
            return True
    for conflict in conflicts:
        if _safe_text(conflict.get("status")).casefold() != "open":
            continue
        conflict_type = _safe_text(conflict.get("conflict_type")).casefold()
        summary = _safe_text(conflict.get("summary")).casefold()
        if any(marker in conflict_type or marker in summary for marker in IDENTITY_CONFLICT_MARKERS):
            return True
    return False


def _identity_value(identity_links: Sequence[Mapping[str, Any]], link_type: str) -> str:
    for link in identity_links:
        if _safe_text(link.get("link_type")) == link_type and _safe_text(link.get("match_class")) == "strong_unique":
            value = _safe_text(link.get("link_value"))
            if value:
                return value
    for link in identity_links:
        if _safe_text(link.get("link_type")) == link_type:
            value = _safe_text(link.get("link_value"))
            if value:
                return value
    return ""


def _manager_amo_id(manager_projection: Mapping[str, Any], key: str) -> str:
    values = _sequence(manager_projection.get(key))
    for value in values:
        text = _safe_text(value)
        if text:
            return text
    return ""


def _match_status(
    identity_status: str,
    identity_links: Sequence[Mapping[str, Any]],
    conflicts: Sequence[Mapping[str, Any]],
) -> str:
    if _identity_is_ambiguous(identity_status, identity_links, conflicts):
        return "ambiguous_manual_review"
    classes = {_safe_text(link.get("match_class")) for link in identity_links}
    if "strong_unique" in classes:
        return "strong_unique"
    if identity_status in {"partial", "unmatched"}:
        return identity_status
    return identity_status or "unknown"


def _source_counts(events: Sequence[Mapping[str, Any]]) -> Counter[str]:
    return Counter(_safe_text(item.get("source_system")) or "unknown" for item in events)


def _what_collected(
    customer: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    signals: Sequence[Mapping[str, Any]],
    conflicts: Sequence[Mapping[str, Any]],
    bot_items: Sequence[Mapping[str, Any]],
) -> str:
    source_counts = _source_counts(events)
    summary = _mapping(customer.get("summary"))
    parts = [
        f"Событий: {len(events)}",
        "Источники: " + ", ".join(f"{key}={value}" for key, value in source_counts.most_common()) if source_counts else "",
        f"Сигналов: {len(signals)}",
        f"Конфликтов: {len(conflicts)}",
        f"Manager-фрагментов: {len(bot_items)}",
        f"Бренд источника: {_safe_text(summary.get('brand'))}" if _safe_text(summary.get("brand")) else "",
    ]
    return "; ".join(part for part in parts if part)


def _latest_summary(latest_call: Mapping[str, Any], facts: ManagerFacts, history_items: Sequence[Mapping[str, Any]]) -> str:
    summary = _event_history_summary(latest_call)
    if summary and summary not in {"no_exact_phone_match"}:
        return summary
    if facts.summary or facts.history:
        return facts.summary or facts.history
    for event in sorted(history_items, key=lambda item: _safe_text(item.get("event_at")), reverse=True):
        summary = _event_history_summary(event)
        if summary and summary not in {"no_exact_phone_match"}:
            return summary
    return ""


def _next_step(signals: Sequence[Mapping[str, Any]], facts: ManagerFacts, latest_call: Mapping[str, Any]) -> str:
    call_next_step = _call_analysis_field(latest_call, "next_step")
    if call_next_step:
        return call_next_step
    for signal in sorted(signals, key=lambda item: _safe_text(item.get("created_at")), reverse=True):
        action = _safe_text(signal.get("recommended_action"))
        if _safe_text(signal.get("status") or "active") == "active" and action:
            return action
    return facts.next_step or "Проверить историю клиента и поставить ручной следующий шаг в AMO."


def _priority(signals: Sequence[Mapping[str, Any]], facts: ManagerFacts, events: Sequence[Mapping[str, Any]]) -> str:
    if facts.priority:
        return facts.priority
    signal_types = {_safe_text(item.get("signal_type")) for item in signals}
    severities = {_safe_text(item.get("severity")) for item in signals}
    if signal_types & HOT_SIGNAL_TYPES or severities & {"high", "critical"}:
        return "hot"
    if events:
        return "warm"
    return "review"


def _deal_priority(contact_priority: str, signals: Sequence[Mapping[str, Any]]) -> str:
    if any(_safe_text(item.get("signal_type")) == "paid_no_access" for item in signals):
        return "review"
    return contact_priority or "review"


def _objections(facts: ManagerFacts, latest_call: Mapping[str, Any]) -> str:
    raw = facts.objections
    if not raw:
        raw = _call_analysis_field(latest_call, "objections")
    parts = [compact_objection_explicit(part) for part in split_parts(raw)]
    return "; ".join(_dedupe(part for part in parts if part))


def _contact_auto_history(
    *,
    latest_summary: str,
    latest_call: Mapping[str, Any],
    timeline_items: Sequence[Mapping[str, Any]],
    signals: Sequence[Mapping[str, Any]],
    facts: ManagerFacts,
    objections: str,
    interests: str = "",
    target_product: str = "",
) -> str:
    blocks: list[str] = []
    if latest_summary:
        blocks.append("Последняя содержательная сводка: см. поле «Последняя AI-сводка».")
    facts_lines: list[str] = []
    if facts.recommended_product:
        facts_lines.append(f"Рекомендуемый продукт: {facts.recommended_product}")
    elif target_product:
        facts_lines.append(f"Целевой продукт: {target_product}")
    if facts.products_interest:
        facts_lines.append(f"Продукты интереса: {facts.products_interest}")
    elif interests:
        facts_lines.append(f"Интересы: {interests}")
    if objections:
        facts_lines.append(f"Возражения: {objections}")
    if facts.next_step:
        facts_lines.append(f"Следующий шаг: {facts.next_step}")
    if facts.follow_up_date:
        facts_lines.append(f"Рекомендуемая дата следующего контакта: {facts.follow_up_date}")
    if facts.priority:
        facts_lines.append(f"Приоритет лида: {facts.priority}")
    if facts.probability:
        facts_lines.append(f"Вероятность продажи, %: {facts.probability}")
    active_actions = [
        _safe_text(signal.get("recommended_action"))
        for signal in signals
        if _safe_text(signal.get("recommended_action")) and _safe_text(signal.get("status") or "active") == "active"
    ]
    if active_actions:
        facts_lines.append("Активные сигналы: " + "; ".join(_dedupe(active_actions[:3])))
    if facts_lines:
        blocks.append("\n".join(facts_lines))
    latest_call_key = _event_key(latest_call)
    chronology = _chronology_text(
        timeline_items,
        compact_event_keys={latest_call_key} if latest_call_key else set(),
        compact_full_texts={_normalized_text(latest_summary)} if latest_summary else set(),
    )
    if chronology:
        blocks.append("Хронология:\n" + chronology)
    tallanto = _tallanto_status(timeline_items, facts)
    if tallanto:
        blocks.append("Tallanto:\n" + tallanto)
    return fit_history_text("\n\n".join(blocks), CONTACT_AUTO_HISTORY_LIMIT)


def _chronology_text(
    events: Sequence[Mapping[str, Any]],
    *,
    limit: int = 5,
    compact_event_keys: set[str] | None = None,
    compact_full_texts: set[str] | None = None,
    compact_call_summaries: bool = False,
) -> str:
    lines: list[str] = []
    compact_event_keys = compact_event_keys or set()
    compact_full_texts = compact_full_texts or set()
    for event in sorted(events, key=lambda item: _safe_text(item.get("event_at")), reverse=True)[:limit]:
        event_at = _safe_text(event.get("event_at"))[:10] or "дата не указана"
        source = _safe_text(event.get("source_system") or event.get("event_type"))
        summary = _event_history_summary(event)
        if not summary:
            continue
        latest_compact = _event_key(event) in compact_event_keys or _normalized_text(summary) in compact_full_texts
        compact_summary = _safe_text(event.get("event_type")) == "mango_call" and (
            compact_call_summaries or latest_compact
        )
        brand = _brand_from_event(event)
        prefix = f"{event_at} {source}"
        if brand:
            prefix += f" [{brand}]"
        structured = _call_analysis_lines(event)
        suffix = "\n" + "\n".join(structured) if structured else ""
        if compact_summary:
            summary = _compact_event_reference(latest=latest_compact)
        lines.append(f"{prefix}: {summary}{suffix}")
    return "\n".join(lines)


def _history_events(events: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    result: list[Mapping[str, Any]] = []
    for event in events:
        event_type = _safe_text(event.get("event_type"))
        if event_type in SERVICE_SNAPSHOT_EVENT_TYPES:
            continue
        if event_type == "mango_call":
            if event.get("call_history_eligible") is True:
                result.append(event)
            continue
        result.append(event)
    return result


def _latest_history_call(events: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    calls = [
        event
        for event in events
        if _safe_text(event.get("event_type")) == "mango_call" and event.get("call_history_eligible") is True
    ]
    if not calls:
        return {}
    return sorted(calls, key=lambda item: _safe_text(item.get("event_at")), reverse=True)[0]


def _event_history_summary(event: Mapping[str, Any]) -> str:
    call_analysis = _mapping(event.get("call_analysis"))
    return _safe_text(call_analysis.get("history_summary") or event.get("summary") or event.get("text_preview"))


def _event_key(event: Mapping[str, Any]) -> str:
    if not event:
        return ""
    for key in ("event_id", "source_ref", "source_id"):
        value = _safe_text(event.get(key))
        if value:
            return f"{_safe_text(event.get('source_system') or event.get('event_type'))}:{value}"
    return stable_text_key(
        _safe_text(event.get("event_type")),
        _safe_text(event.get("event_at")),
        _event_history_summary(event),
    )


def stable_text_key(*parts: str) -> str:
    return "|".join(_normalized_text(part) for part in parts if _safe_text(part))


def _normalized_text(value: Any) -> str:
    return re.sub(r"\s+", " ", _safe_text(value)).strip().casefold()


def _compact_event_reference(*, latest: bool) -> str:
    if latest:
        return "последний содержательный звонок; полная сводка в поле «Последняя AI-сводка»"
    return "содержательный звонок; полный текст есть в контактной автоистории; ниже краткие детали"


def _call_analysis_lines(event: Mapping[str, Any]) -> list[str]:
    call_analysis = _mapping(event.get("call_analysis"))
    if not call_analysis:
        return []
    lines: list[str] = []
    objections = _join_values(call_analysis.get("objections"))
    next_step = _safe_text(call_analysis.get("next_step"))
    pain_points = _join_values(call_analysis.get("pain_points"))
    interests = _join_values(call_analysis.get("interests"))
    target_product = _safe_text(call_analysis.get("target_product"))
    budget = _join_values(call_analysis.get("budget"))
    if objections:
        lines.append(f"Возражения: {objections}")
    if next_step:
        lines.append(f"Следующий шаг: {next_step}")
    if pain_points:
        lines.append(f"Боли/ограничения: {pain_points}")
    if interests:
        lines.append(f"Интересы: {interests}")
    if target_product:
        lines.append(f"Целевой продукт: {target_product}")
    if budget:
        lines.append(f"Бюджет: {budget}")
    return lines


def _call_analysis_field(event: Mapping[str, Any], key: str) -> str:
    call_analysis = _mapping(event.get("call_analysis"))
    return _join_values(call_analysis.get(key))


def _join_values(value: Any) -> str:
    if isinstance(value, Mapping):
        parts = []
        for key, item in value.items():
            text = _join_values(item)
            if text:
                parts.append(f"{key}: {text}")
        return "; ".join(parts)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return "; ".join(_safe_text(item) for item in value if _safe_text(item))
    return _safe_text(value)


def _brand_from_event(event: Mapping[str, Any]) -> str:
    subject = _safe_text(event.get("subject")).casefold()
    summary = _safe_text(event.get("summary")).casefold()
    if "унпк" in subject or "унпк" in summary or "unpk" in subject or "unpk" in summary:
        return "УНПК"
    if "фотон" in subject or "фотон" in summary or "foton" in subject or "foton" in summary:
        return "Фотон"
    return ""


def _select_opportunity(opportunities: Sequence[Mapping[str, Any]], lead_id: str) -> Mapping[str, Any]:
    amo_opportunities = [item for item in opportunities if _safe_text(item.get("opportunity_type")) == "amo_deal"]
    if lead_id:
        for item in amo_opportunities:
            haystack = json.dumps(item, ensure_ascii=False)
            if lead_id in haystack:
                return item
    return amo_opportunities[0] if amo_opportunities else {}


def _deal_events(events: Sequence[Mapping[str, Any]], opportunity_id: Any) -> list[Mapping[str, Any]]:
    opp = _safe_text(opportunity_id)
    if not opp:
        return []
    return [event for event in events if _safe_text(event.get("opportunity_id")) == opp]


def _deal_summary(
    opportunity: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    latest_call: Mapping[str, Any],
    latest_summary: str,
    history_scope: str,
) -> str:
    title = _safe_text(opportunity.get("title")) or "сделка не выбрана"
    if latest_summary:
        last_at = _safe_text(latest_call.get("event_at"))[:10] if latest_call else ""
        last_part = (
            f"Последний содержательный звонок {last_at}: полная сводка вынесена в поле «Последняя AI-сводка»."
            if last_at
            else "Последняя содержательная сводка вынесена в поле «Последняя AI-сводка»."
        )
    else:
        last_part = "Содержательных событий не найдено."
    return f"Сделка: {title}. Основа: {history_scope}. {last_part}"


def _deal_history(
    events: Sequence[Mapping[str, Any]],
    *,
    history_scope: str,
    compact_event_keys: set[str] | None = None,
    compact_full_texts: set[str] | None = None,
) -> str:
    chronology = _chronology_text(
        events,
        limit=10,
        compact_event_keys=compact_event_keys,
        compact_full_texts=compact_full_texts,
        compact_call_summaries=True,
    )
    if not chronology:
        return f"{history_scope}: релевантные события не найдены."
    return f"{history_scope}:\n{chronology}"


def _deal_status(opportunity: Mapping[str, Any], facts: ManagerFacts, events: Sequence[Mapping[str, Any]]) -> str:
    stage = _latest_amo_deal_stage(events)
    if opportunity:
        status = (
            f"AMO opportunity: {_safe_text(opportunity.get('title')) or 'без названия'}; "
            f"статус {_safe_text(opportunity.get('status')) or 'не указан'}."
        )
        return f"{status} Последний этап AMO: {stage}." if stage else status
    if facts.amo_lead_id:
        base = f"AMO lead {facts.amo_lead_id}: статус нужно сверить в AMO snapshot."
        return f"{base} Последний этап AMO: {stage}." if stage else base
    if stage:
        return f"Последний этап AMO: {stage}."
    return "AMO сделка не подтверждена в read_api profile."


def _latest_amo_deal_stage(events: Sequence[Mapping[str, Any]]) -> str:
    for event in sorted(events, key=lambda item: _safe_text(item.get("event_at")), reverse=True):
        if _safe_text(event.get("event_type")) != "amo_deal_stage":
            continue
        before = _safe_text(event.get("stage_before"))
        after = _safe_text(event.get("stage_after"))
        summary = _safe_text(event.get("summary") or event.get("text_preview"))
        if after and before:
            return f"{before} → {after}"
        if after:
            return after
        if summary:
            return summary
    return ""


def _followup_from_signal(signals: Sequence[Mapping[str, Any]]) -> str:
    for signal in signals:
        expires = _safe_text(signal.get("expires_at"))
        if expires:
            return expires[:10]
    return ""


def _recommendation_reason(
    signals: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    facts: ManagerFacts,
) -> str:
    if signals:
        top = signals[0]
        return (
            f"Основание: активный сигнал {_safe_text(top.get('signal_type'))}; "
            f"{_safe_text(top.get('evidence_text')) or 'evidence в профиле'}."
        )
    if facts.next_step:
        return "Основание: fallback из старого analyze-поля next_step."
    return f"Основание: {len(events)} событий в customer_timeline; нужен ручной контроль менеджера."


def _binding_quality(
    identity_status: str,
    identity_links: Sequence[Mapping[str, Any]],
    opportunity: Mapping[str, Any],
    history_scope: str,
) -> str:
    classes = Counter(_safe_text(item.get("match_class")) or "unknown" for item in identity_links)
    quality = ", ".join(f"{key}: {value}" for key, value in classes.most_common()) or "identity_links отсутствуют"
    opp = _safe_text(opportunity.get("opportunity_id")) or "нет opportunity_id"
    return f"Identity: {identity_status}; links: {quality}; opportunity: {opp}; {history_scope}."


def _warnings(
    blockers: Sequence[str],
    conflicts: Sequence[Mapping[str, Any]],
    signals: Sequence[Mapping[str, Any]],
) -> str:
    warnings: list[str] = []
    if blockers:
        warnings.append("Блокеры: " + "; ".join(_dedupe(blockers)))
    for conflict in conflicts[:3]:
        if _safe_text(conflict.get("status")).casefold() == "open":
            warnings.append("Открытый конфликт: " + (_safe_text(conflict.get("summary")) or _safe_text(conflict.get("conflict_type"))))
    for signal in signals[:3]:
        if _safe_text(signal.get("signal_type")) == "paid_no_access":
            warnings.append("Есть сигнал оплаты без подтвержденного доступа; перед действием сверить Tallanto.")
    return " ".join(_dedupe(warnings)) or "Критичных предупреждений агрегатор не выявил."


def _tallanto_status(events: Sequence[Mapping[str, Any]], facts: ManagerFacts) -> str:
    if facts.tallanto_history:
        return facts.tallanto_history
    tallanto = [
        _tallanto_snapshot_text(event)
        for event in events
        if _safe_text(event.get("event_type")).startswith("tallanto_") or _safe_text(event.get("source_system")) == "tallanto_snapshot"
    ]
    return "; ".join(_dedupe(item for item in tallanto if item))[:1600]


def _tallanto_snapshot_text(event: Mapping[str, Any]) -> str:
    text = _safe_text(event.get("summary") or event.get("text_preview"))
    normalized = text.casefold()
    if normalized == "exact_phone_single":
        return "Tallanto: найден один ученик по телефону."
    if normalized == "no_exact_phone_match":
        return "Tallanto: точного совпадения по телефону нет."
    return text


def _what_already_in_amo(
    profile: Mapping[str, Any],
    existing_amo_fields: Mapping[str, Any],
    *,
    lead_id: str = "",
    contact_id: str = "",
) -> str:
    if existing_amo_fields:
        return "\n".join(f"{key}: {_safe_text(value)}" for key, value in existing_amo_fields.items() if _safe_text(value))
    timeline = _mapping(profile.get("timeline"))
    events = [_mapping(item) for item in _sequence(timeline.get("items"))]
    amo_events = [event for event in events if _safe_text(event.get("event_type")).startswith("amo_") or "amo" in _safe_text(event.get("source_system"))]
    if not amo_events and not lead_id and not contact_id:
        return "AMO-сущность не найдена в read_api profile."
    parts = []
    if contact_id:
        parts.append(f"AMO contact id: {contact_id}")
    if lead_id:
        parts.append(f"AMO lead id: {lead_id}")
    if amo_events:
        parts.append("AMO events: " + "; ".join(_safe_text(event.get("summary") or event.get("subject")) for event in amo_events[:5]))
    return "\n".join(part for part in parts if part)


def _bot_safety_summary(
    events: Sequence[Mapping[str, Any]],
    signals: Sequence[Mapping[str, Any]],
    bot_items: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    bot_safe_chunks = [item for item in bot_items if item.get("allowed_for_bot") is True and item.get("requires_manager_review") is False]
    manager_only_chunks = [item for item in bot_items if item not in bot_safe_chunks]
    return {
        "bot_safe_chunk_count": len(bot_safe_chunks),
        "manager_only_chunk_count": len(manager_only_chunks) + len(events) + len(signals),
        "bot_safe_fields": [],
        "manager_only_fields": list(CONTACT_CARD_FIELDS) + list(DEAL_CARD_REQUIRED_FIELDS),
        "money_fields_manager_only": True,
    }


def split_parts(value: Any) -> list[str]:
    text = normalize_manager_text(value)
    if not text:
        return []
    return [part.strip(" .;,:") for part in re.split(r"\s+\|\s+|\n|;", text) if part.strip(" .;,:")]


def _json_payload(value: Any) -> dict[str, Any] | None:
    if isinstance(value, Mapping):
        return dict(value)
    text = _safe_text(value)
    if not text:
        return None
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        return None
    return dict(decoded) if isinstance(decoded, Mapping) else None


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)) else ()


def _drop_empty(payload: Mapping[str, Any]) -> dict[str, str]:
    return {str(key): _safe_text(value) for key, value in payload.items() if _safe_text(value)}


def _dedupe(values: Sequence[str] | Any) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _safe_text(value)
        key = text.casefold()
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


__all__ = [
    "CRM_CARD_AGGREGATOR_FLAG",
    "CRM_CARD_AGGREGATOR_SCHEMA_VERSION",
    "CONTACT_CARD_FIELDS",
    "DEAL_CARD_OPTIONAL_FIELDS",
    "DEAL_CARD_REQUIRED_FIELDS",
    "ManagerFacts",
    "apply_contact_card_payload",
    "apply_deal_card_payload",
    "build_crm_card_projection",
    "card_aggregator_enabled",
    "compact_objection_explicit",
    "contact_ready_blocker",
    "fit_text",
    "manager_facts_from_row",
    "render_payload_preview",
]
