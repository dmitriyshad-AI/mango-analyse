from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from mango_mvp.channels.dialogue_contract_pipeline import DIALOGUE_CONTRACT_PIPELINE_ENV
from mango_mvp.channels.new_lead_funnel import LeadFunnelState, build_lead_funnel_state, lead_funnel_context_payload
from mango_mvp.channels.subscription_llm import AUTONOMY_MATRIX_SAFE_TOPIC_IDS, SubscriptionDraftResult
from mango_mvp.channels.telegram_pilot_context_builder import build_telegram_pilot_context_from_snapshot


def known_dialog_fields_from_messages(messages: Sequence[str], *, active_brand: str = "") -> dict[str, str]:
    client_parts: list[str] = []
    for item in messages:
        for raw_line in str(item or "").splitlines():
            line = raw_line.strip()
            lowered = line.casefold()
            if lowered.startswith("ответ:"):
                continue
            if lowered.startswith("клиент:"):
                line = line.split(":", 1)[1].strip()
            if line:
                client_parts.append(line)
    text = "\n".join(client_parts)
    normalized = text.casefold().replace("ё", "е")
    result: dict[str, str] = {}
    grade = re.search(r"\b(?P<grade>[1-9]|1[01])\s*(?:класс|кл\.?)\b", normalized)
    if grade:
        result["grade"] = grade.group("grade")
    subjects: list[str] = []
    for marker, canonical in (
        ("математ", "математика"),
        ("физик", "физика"),
        ("информат", "информатика"),
        ("программирован", "программирование"),
        ("русск", "русский язык"),
        ("англий", "английский язык"),
        ("хими", "химия"),
        ("биолог", "биология"),
    ):
        if marker in normalized:
            subjects.append(canonical)
    if subjects:
        result["subject"] = ", ".join(dict.fromkeys(subjects))
    if "онлайн" in normalized:
        result["format"] = "онлайн"
    elif "очно" in normalized or "офлайн" in normalized:
        result["format"] = "очно"
    if active_brand:
        result["active_brand"] = active_brand
    return result


def merge_known_field_aliases(target: dict[str, str], source: Mapping[str, Any]) -> None:
    aliases = {
        "parent_name": ("parent_name", "parent", "parent_full_name", "fio_parent", "parent_fio"),
        "student_name": ("student_name", "student", "student_full_name", "fio_student", "student_fio", "child_name"),
        "phone": ("phone", "normalized_phone", "client_phone"),
        "grade": ("grade", "class", "student_grade", "klass"),
        "subject": ("subject", "course_subject", "interest_subject"),
        "known_course": ("known_course", "current_course", "course"),
        "current_group": ("current_group", "group", "tallanto_group"),
    }
    for normalized, keys in aliases.items():
        for key in keys:
            value = str(source.get(key) or "").strip()
            if value:
                target.setdefault(normalized, value)
                break


def known_client_fields_from_sources(
    *,
    debug_phone: str = "",
    debug_client: Mapping[str, Any] | None = None,
    crm_context: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    result: dict[str, str] = {}
    crm_payload = crm_context or {}
    merge_known_field_aliases(result, debug_client or {})
    if debug_phone:
        result.setdefault("phone", debug_phone)
    local = crm_payload.get("local_runtime_context") if isinstance(crm_payload.get("local_runtime_context"), Mapping) else {}
    if local:
        merge_known_field_aliases(result, local)
        result.update(
            {
                key: value
                for key, value in known_dialog_fields_from_messages([str(local.get("history_summary") or "")]).items()
                if value and key not in result
            }
        )
    amo = crm_payload.get("amo_context") if isinstance(crm_payload.get("amo_context"), Mapping) else {}
    tallanto = crm_payload.get("tallanto_context") if isinstance(crm_payload.get("tallanto_context"), Mapping) else {}
    if amo.get("status") == "ok":
        result.setdefault("amo_context", "found")
    if tallanto.get("status") == "ok":
        result.setdefault("tallanto_context", "found")
    return {key: str(value)[:180] for key, value in result.items() if str(value or "").strip()}


def debug_client_label(client: Mapping[str, Any]) -> str:
    parent = str(client.get("parent_name") or "").strip()
    student = str(client.get("student_name") or "").strip()
    if parent and student:
        return f"{parent}, ученик {student}"
    return parent or student


def debug_customer_summary(phone: str, client: Mapping[str, Any]) -> str:
    if not phone:
        return ""
    label = debug_client_label(client)
    if label:
        return f"Тестовый режим сотрудника: отвечать как известному клиенту с телефона {phone}. Клиент: {label}."
    return f"Тестовый режим сотрудника: отвечать как известному клиенту с телефона {phone}. Клиент не найден в локальной тестовой карте."


def build_known_context_summary(known_client_fields: Mapping[str, Any], known_dialog_fields: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for label, fields in (
        ("Из CRM/локального контекста известно", known_client_fields),
        ("Из текущего диалога известно", known_dialog_fields),
    ):
        public = {
            key: value
            for key, value in fields.items()
            if key in {"parent_name", "student_name", "grade", "subject", "format", "known_course", "current_group", "active_brand"}
        }
        if public:
            parts.append(f"{label}: " + ", ".join(f"{key}={value}" for key, value in public.items()))
    return "; ".join(parts)[:700]


def build_pilot_context_payload(
    *,
    current_text: str,
    snapshot_path: Path | str,
    active_brand: str,
    recent_messages: Sequence[str] = (),
    dialogue_memory: Mapping[str, Any] | None = None,
    session_id: str,
    channel: str,
    channel_thread_id: str,
    channel_user_id: str,
    dialogue_contract_pipeline_enabled: bool = True,
    sends_client_replies: bool = True,
    debug_impersonation_enabled: bool = True,
    debug_phone: str = "",
    debug_client: Mapping[str, Any] | None = None,
    crm_context: Mapping[str, Any] | None = None,
    current_message_id: str = "",
) -> Mapping[str, Any]:
    client_identity: dict[str, Any] = {
        "channel": channel,
        "channel_thread_id": str(channel_thread_id),
        "channel_user_id": str(channel_user_id),
    }
    if debug_phone:
        client_identity.update(
            {
                "phone": debug_phone,
                "debug_impersonation": True,
                **dict(debug_client or {}),
            }
        )
    crm_payload = dict(crm_context or {})
    customer_summary = debug_customer_summary(debug_phone, debug_client or {})
    if crm_payload.get("summary"):
        customer_summary = "\n".join(item for item in (customer_summary, str(crm_payload["summary"])) if item)
    known_client_fields = known_client_fields_from_sources(
        debug_phone=debug_phone,
        debug_client=debug_client or {},
        crm_context=crm_payload,
    )
    known_dialog_fields = known_dialog_fields_from_messages(
        [*tuple(recent_messages)[-10:], current_text],
        active_brand=active_brand,
    )
    known_context_summary_text = build_known_context_summary(known_client_fields, known_dialog_fields)
    rop_policy = {
        "bot_permission": "bot_answer_self_for_pilot",
        "autonomy_policy": {
            "allow_autonomous": True,
            "allowed_topic_ids": sorted(AUTONOMY_MATRIX_SAFE_TOPIC_IDS),
            "default": "draft_for_manager_or_manager_only",
            "fact_requirement": "client_safe_fact_verified",
            "p0_overrides_autonomy": True,
        },
    }
    pilot_context = build_telegram_pilot_context_from_snapshot(
        current_text,
        snapshot_path=Path(snapshot_path),
        active_brand=active_brand,
        rop_policy=rop_policy,
        recent_messages=tuple(recent_messages)[-10:],
        client_identity=client_identity,
        customer_summary=customer_summary,
        amo_context=crm_payload.get("amo_context") if isinstance(crm_payload.get("amo_context"), Mapping) else None,
        tallanto_context=crm_payload.get("tallanto_context")
        if isinstance(crm_payload.get("tallanto_context"), Mapping)
        else None,
        timeline_context=crm_payload.get("timeline_context")
        if isinstance(crm_payload.get("timeline_context"), Mapping)
        else None,
        risk_flags=tuple(crm_payload.get("risk_flags") or ()),
        known_slots=known_dialog_fields,
        dialogue_memory=dialogue_memory or {},
        session_id=session_id,
        current_message_id=current_message_id,
    )
    payload = dict(pilot_context.to_prompt_context())
    payload["active_brand"] = active_brand
    payload["snapshot_path"] = str(snapshot_path)
    payload["knowledge_snapshot_path"] = str(snapshot_path)
    payload[DIALOGUE_CONTRACT_PIPELINE_ENV] = dialogue_contract_pipeline_enabled
    if known_client_fields:
        payload["known_client_fields"] = known_client_fields
    if known_dialog_fields:
        payload["known_dialog_fields"] = known_dialog_fields
    if known_context_summary_text:
        payload["known_context_summary"] = known_context_summary_text
    if crm_payload:
        payload["read_only_customer_context"] = crm_payload
    payload["public_pilot_mode"] = {
        "enabled": True,
        "sends_client_replies": bool(sends_client_replies),
        "debug_impersonation_enabled": bool(debug_impersonation_enabled),
        "brand_isolation_required": True,
        "no_crm_tallanto_write": True,
        "crm_tallanto_read_only": True,
        "do_not_disclose_crm_tallanto_private_data": True,
    }
    return payload


def build_funnel_state(
    *,
    current_text: str,
    active_brand: str,
    recent_messages: Sequence[str] = (),
    context: Mapping[str, Any],
    result: SubscriptionDraftResult | None = None,
) -> LeadFunnelState:
    return build_lead_funnel_state(
        current_text,
        active_brand=active_brand,
        recent_messages=tuple(recent_messages)[-10:],
        context=context,
        topic_id=str((result.topic_id if result else context.get("topic_id")) or ""),
        message_type=str((result.message_type if result else context.get("message_type")) or ""),
        risk_level=str((result.risk_level if result else context.get("risk_level")) or ""),
        route=str((result.route if result else context.get("route")) or ""),
        safety_flags=tuple(result.safety_flags if result else context.get("safety_flags") or ()),
    )


def attach_funnel_state_to_context(context: Mapping[str, Any], funnel_state: LeadFunnelState) -> Mapping[str, Any]:
    payload = dict(context)
    funnel_payload = lead_funnel_context_payload(funnel_state)
    payload["funnel_state"] = funnel_payload
    payload["known_slots"] = dict(funnel_payload.get("filled_slots") or {})
    payload["missing_slots"] = list(funnel_payload.get("missing_slots") or [])
    payload["next_best_question"] = str(funnel_payload.get("next_best_question") or "")
    payload["next_step_type"] = str(funnel_payload.get("next_step_type") or "")
    payload["lead_stage"] = str(funnel_payload.get("lead_stage") or "")
    payload["client_segment"] = str(funnel_payload.get("client_segment") or "")
    payload["semantic_flags"] = list(funnel_payload.get("semantic_flags") or [])
    return payload
