from __future__ import annotations

import os
from typing import Any, Mapping, Sequence

from mango_mvp.amocrm_runtime.config import get_settings
from mango_mvp.amocrm_runtime.tallanto_api import TallantoApiClient, TallantoApiError, build_tallanto_api_config

settings = get_settings()


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _contact_phones(contact: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for field_name in ("phone_mobile", "phone_work", "phone_home", "phone_other", "phone"):
        raw_value = contact.get(field_name)
        if raw_value is None:
            continue
        if isinstance(raw_value, list):
            values.extend(_safe_text(item) for item in raw_value)
        else:
            values.append(_safe_text(raw_value))
    return [value for value in values if value]


def _compact_contact(contact: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": _safe_text(contact.get("id")),
        "name": " ".join(
            value
            for value in (
                _safe_text(contact.get("last_name")),
                _safe_text(contact.get("first_name")),
                _safe_text(contact.get("middle_name")),
            )
            if value
        )
        or _safe_text(contact.get("name")),
        "branch": _safe_text(contact.get("filial")),
        "assigned_user_name": _safe_text(contact.get("assigned_user_name")),
        "email": _safe_text(contact.get("email1")),
        "phones": _contact_phones(contact),
        "amo_id": _safe_text(contact.get("amo_id")),
        "type_client": _safe_text(contact.get("type_client")),
    }


def _compact_items(records: list[dict[str, Any]], *, allowed_fields: tuple[str, ...], limit: int = 20) -> list[dict[str, Any]]:
    compacted: list[dict[str, Any]] = []
    for record in records[:limit]:
        item = {field_name: _safe_text(record.get(field_name)) for field_name in allowed_fields}
        compacted.append({key: value for key, value in item.items() if value})
    return compacted


def brand_scope_from_filial(value: Any) -> str:
    filial = _safe_text(value).replace("ё", "е").casefold()
    if not filial:
        return "unknown"
    if any(token in filial for token in ("shd", "шд", "жако")):
        return "skip_shd"
    if any(token in filial for token in ("foton", "фотон")):
        return "foton"
    if any(token in filial for token in ("mfti", "мфти", "pacaeva", "пацаева")):
        return "unpk"
    if any(token in filial for token in ("onlajn", "online", "онлайн")):
        return "shared"
    return "unknown"


def _live_card_brand_failclosed_enabled() -> bool:
    value = os.getenv("CRM_LIVE_CARD_BRAND_FAILCLOSED")
    if value is None:
        return True
    return value.strip().casefold() not in {"0", "false", "no", "off", "нет"}


def build_live_tallanto_context(
    *,
    phone: str,
    tallanto_id: str | None = None,
    tallanto_match_status: str | None = None,
    active_brand: str | None = None,
    max_related_records: int = 40,
) -> dict[str, Any]:
    mode = _safe_text(settings.crm_tallanto_mode).lower()
    if mode != "http":
        return {
            "enabled": False,
            "status": "disabled",
            "reason": f"crm_tallanto_mode={mode or 'unset'}",
        }

    try:
        client = TallantoApiClient(build_tallanto_api_config())
    except TallantoApiError as exc:
        return {
            "enabled": True,
            "status": "error",
            "error": str(exc),
        }

    matched_via = "phone"
    payload: dict[str, Any]
    live_card_only = os.getenv("TALLANTO_BATCH_FETCH", "1") == "1"
    try:
        if _safe_text(tallanto_id) and _safe_text(tallanto_match_status) in {"exact_phone_single", "manual_confirmed", "id_confirmed"}:
            payload = client.build_contact_context_by_contact_id(
                _safe_text(tallanto_id),
                max_related_records=max_related_records,
                live_card_only=live_card_only,
            )
            matched_via = "tallanto_id"
            if int(payload.get("contacts_found") or 0) == 0:
                payload = client.build_contact_context(
                    phone,
                    max_contacts=5,
                    max_related_records=max_related_records,
                    live_card_only=live_card_only,
                )
                matched_via = "phone_fallback"
        else:
            payload = client.build_contact_context(
                phone,
                max_contacts=5,
                max_related_records=max_related_records,
                live_card_only=live_card_only,
            )
    except TallantoApiError as exc:
        return {
            "enabled": True,
            "status": "error",
            "error": str(exc),
        }

    compact_contexts: list[dict[str, Any]] = []
    live_contexts: list[dict[str, Any]] = []
    for context in payload.get("contexts") or []:
        if not isinstance(context, dict):
            continue
        contact = context.get("contact") if isinstance(context.get("contact"), dict) else {}
        opportunities = [
            item for item in (context.get("opportunities") or []) if isinstance(item, dict)
        ]
        requests = [
            item for item in (context.get("requests") or []) if isinstance(item, dict)
        ]
        finances = [
            item for item in (context.get("finances") or []) if isinstance(item, dict)
        ]
        course_relations = [
            item for item in (context.get("course_relations") or []) if isinstance(item, dict)
        ]
        class_relations = [
            item for item in (context.get("class_relations") or []) if isinstance(item, dict)
        ]
        abonements = [
            item for item in (context.get("abonements") or []) if isinstance(item, dict)
        ]
        classes = [
            item for item in (context.get("classes") or []) if isinstance(item, dict)
        ]
        live_contexts.append(
            {
                "contact": contact,
                "finances": finances,
                "abonements": abonements,
                "class_relations": class_relations,
                "classes": classes,
            }
        )
        compact_contexts.append(
            {
                "contact": _compact_contact(contact),
                "opportunity_count": len(opportunities),
                "request_count": len(requests),
                "finance_count": len(finances),
                "course_relation_count": len(course_relations),
                "class_relation_count": len(class_relations),
                "opportunities": _compact_items(
                    opportunities,
                    allowed_fields=(
                        "id",
                        "name",
                        "sales_stage",
                        "next_step",
                        "date_closed",
                        "system_date_closed",
                        "assigned_user_name",
                        "filial",
                    ),
                ),
                "requests": _compact_items(
                    requests,
                    allowed_fields=(
                        "id",
                        "name",
                        "status",
                        "date_next_contact",
                        "assigned_user_name",
                        "filial",
                        "source_contact",
                    ),
                ),
                "finances": _compact_items(
                    finances,
                    allowed_fields=(
                        "id",
                        "name",
                        "date_entered",
                        "payment_summa",
                        "payment_status",
                    ),
                ),
                "course_relations": _compact_items(
                    course_relations,
                    allowed_fields=(
                        "id",
                        "course_id",
                        "contact_id",
                    ),
                ),
                "class_relations": _compact_items(
                    class_relations,
                    allowed_fields=(
                        "id",
                        "class_id",
                        "contact_id",
                    ),
                ),
                "abonement_count": len(abonements),
                "class_count": len(classes),
            }
        )
    live_card = build_tallanto_live_card(
        live_contexts,
        active_brand=active_brand,
        matched_via=matched_via,
    )

    return {
        "enabled": True,
        "status": "ok",
        "matched_via": matched_via,
        "contacts_found": len(compact_contexts),
        "contexts": compact_contexts,
        "live_card": live_card,
    }


def build_tallanto_live_card(
    contexts: Sequence[Mapping[str, Any]],
    *,
    active_brand: str | None = None,
    matched_via: str = "phone",
) -> dict[str, Any]:
    if len(contexts) != 1:
        return _no_card(
            "multiple_contacts" if len(contexts) > 1 else "no_contact",
            matched_via=matched_via,
            contacts_found=len(contexts),
        )
    context = contexts[0]
    contact = context.get("contact") if isinstance(context.get("contact"), Mapping) else {}
    scope = brand_scope_from_filial(contact.get("filial") or contact.get("branch"))
    skipped = {"filial_shd": 0, "inactive_class": 0}
    if scope == "skip_shd":
        skipped["filial_shd"] = 1
        return _no_card("filial_shd", matched_via=matched_via, contacts_found=1, skipped=skipped)
    active = _safe_text(active_brand).casefold()
    fail_closed = _live_card_brand_failclosed_enabled()
    if fail_closed and not active:
        return _no_card("brand_unverified", matched_via=matched_via, contacts_found=1, brand_scope=scope, skipped=skipped)
    if active and scope not in {active, "shared"}:
        return _no_card("brand_mismatch", matched_via=matched_via, contacts_found=1, brand_scope=scope, skipped=skipped)
    if scope == "unknown" and active:
        return _no_card("brand_unknown", matched_via=matched_via, contacts_found=1, brand_scope=scope, skipped=skipped)
    brand = active if active and scope == "shared" else scope
    if brand == "shared":
        brand = "unknown"

    finances = [item for item in context.get("finances") or [] if isinstance(item, Mapping)]
    abonements = [item for item in context.get("abonements") or [] if isinstance(item, Mapping)]
    classes = [item for item in context.get("classes") or [] if isinstance(item, Mapping)]
    payments = _payment_items(finances)
    balance = _balance_items(abonements)
    schedule, enrollment, inactive_count = _schedule_and_enrollment(classes)
    skipped["inactive_class"] = inactive_count
    return {
        "schema_version": "tallanto_live_card_v1",
        "status": "ok",
        "matched": "single",
        "matched_via": matched_via,
        "brand": brand,
        "brand_scope": scope,
        "payments": payments,
        "balance": balance,
        "schedule": schedule,
        "enrollment": enrollment,
        "ttl_seconds": {
            "payments": 900,
            "balance": 900,
            "schedule": 21600,
            "enrollment": 3600,
        },
        "skipped": skipped,
        "_provenance": {
            "source": "Tallanto",
            "contact_count": 1,
            "finance_count": len(finances),
            "abonement_count": len(abonements),
            "class_count": len(classes),
        },
        "_filtered_at_source": True,
    }


def _no_card(
    reason: str,
    *,
    matched_via: str,
    contacts_found: int,
    brand_scope: str = "",
    skipped: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    result = {
        "schema_version": "tallanto_live_card_v1",
        "status": "no_card",
        "reason": reason,
        "matched_via": matched_via,
        "contacts_found": contacts_found,
        "skipped": dict(skipped or {}),
    }
    if brand_scope:
        result["brand_scope"] = brand_scope
    return result


def _payment_items(records: Sequence[Mapping[str, Any]], *, limit: int = 5) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for record in sorted(records, key=lambda item: _safe_text(item.get("date_entered")), reverse=True):
        item = {
            "date": _safe_text(record.get("date_entered") or record.get("payment_date") or record.get("date")),
            "status": _safe_text(record.get("payment_status") or record.get("status")),
            "sum": _safe_text(record.get("payment_summa") or record.get("summa") or record.get("amount")),
        }
        cleaned = {key: value for key, value in item.items() if value}
        if cleaned:
            items.append(cleaned)
        if len(items) >= limit:
            break
    return items


def _balance_items(records: Sequence[Mapping[str, Any]], *, limit: int = 5) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for record in records:
        if not _record_is_active(record):
            continue
        visits_left = _first_value(record, ("num_visit_left", "number_visit_left", "visits_left"))
        if visits_left == "":
            continue
        item = {
            "visits_left": visits_left,
            "status": _safe_text(record.get("status")),
            "valid_until": _safe_text(record.get("date_finish") or record.get("valid_until")),
        }
        items.append({key: value for key, value in item.items() if value})
        if len(items) >= limit:
            break
    return items


def _schedule_and_enrollment(records: Sequence[Mapping[str, Any]], *, limit: int = 8) -> tuple[list[dict[str, str]], list[dict[str, str]], int]:
    schedule: list[dict[str, str]] = []
    enrollment: list[dict[str, str]] = []
    inactive = 0
    for record in records:
        if not _record_is_active(record):
            inactive += 1
            continue
        class_id = _safe_text(record.get("id"))
        title = _safe_text(record.get("name") or record.get("title"))
        remaining = _remaining_seats(record)
        schedule_item = {
            "class_id": class_id,
            "title": title,
            "date_start": _safe_text(record.get("date_start")),
            "time_start": _safe_text(record.get("time_start") or record.get("start_time")),
            "time_finish": _safe_text(record.get("time_finish") or record.get("finish_time")),
            "room": _safe_text(record.get("auditory") or record.get("auditorium") or record.get("room") or record.get("cabinet")),
        }
        schedule.append({key: value for key, value in schedule_item.items() if value})
        enrollment_item = {
            "class_id": class_id,
            "title": title,
            "remaining_seats": remaining,
            "status": _safe_text(record.get("status")),
        }
        enrollment.append({key: value for key, value in enrollment_item.items() if value})
        if len(schedule) >= limit:
            break
    return schedule, enrollment, inactive


def _record_is_active(record: Mapping[str, Any]) -> bool:
    status = _safe_text(record.get("status")).casefold()
    if not status:
        return True
    return status not in {"notactive", "inactive", "closed", "archive", "archived", "0"}


def _remaining_seats(record: Mapping[str, Any]) -> str:
    value = _first_value(record, ("remaining_seats", "seats_left", "free_places"))
    if value in {"", "10000"}:
        return ""
    return value


def _first_value(record: Mapping[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = _safe_text(record.get(key))
        if value != "":
            return value
    return ""


__all__ = ["build_live_tallanto_context", "build_tallanto_live_card", "brand_scope_from_filial"]
