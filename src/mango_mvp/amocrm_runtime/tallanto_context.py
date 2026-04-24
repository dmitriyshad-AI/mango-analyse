from __future__ import annotations

from typing import Any

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


def build_live_tallanto_context(
    *,
    phone: str,
    tallanto_id: str | None = None,
    tallanto_match_status: str | None = None,
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
    try:
        if _safe_text(tallanto_id) and _safe_text(tallanto_match_status) in {"exact_phone_single", "manual_confirmed", "id_confirmed"}:
            payload = client.build_contact_context_by_contact_id(
                _safe_text(tallanto_id),
                max_related_records=max_related_records,
            )
            matched_via = "tallanto_id"
            if int(payload.get("contacts_found") or 0) == 0:
                payload = client.build_contact_context(
                    phone,
                    max_contacts=5,
                    max_related_records=max_related_records,
                )
                matched_via = "phone_fallback"
        else:
            payload = client.build_contact_context(
                phone,
                max_contacts=5,
                max_related_records=max_related_records,
            )
    except TallantoApiError as exc:
        return {
            "enabled": True,
            "status": "error",
            "error": str(exc),
        }

    compact_contexts: list[dict[str, Any]] = []
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
            }
        )

    return {
        "enabled": True,
        "status": "ok",
        "matched_via": matched_via,
        "contacts_found": len(compact_contexts),
        "contexts": compact_contexts,
    }


__all__ = ["build_live_tallanto_context"]
