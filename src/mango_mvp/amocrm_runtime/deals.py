from __future__ import annotations

import concurrent.futures
import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from sqlalchemy.orm import Session

from mango_mvp.amocrm_runtime.amo_integration import (
    fetch_contact,
    fetch_lead,
    fetch_leads_batch,
    fetch_lead_notes,
    fetch_lead_tasks,
    fetch_pipelines_with_statuses,
    fetch_recent_leads,
    fetch_related_leads,
    fetch_users,
    search_contacts_by_phone,
    send_lead_custom_field_update,
)
from mango_mvp.amocrm_runtime.config import get_settings
from mango_mvp.amocrm_runtime.db import SessionLocal
from mango_mvp.amocrm_runtime.deal_dossier import build_deal_dossier
from mango_mvp.amocrm_runtime.deal_llm import DealLLMAnalyzer, DealLLMError
from mango_mvp.amocrm_runtime.phone_context import PhoneContext, get_phone_context
from mango_mvp.utils.phone import normalize_phone

settings = get_settings()
READ_ONLY_QUEUE_WORKERS = 6

TERMINAL_SUCCESS_STATUS_IDS = {142}
TERMINAL_LOST_STATUS_IDS = {143}
DEFAULT_TARGET_PIPELINE_NAMES = {"Лиды", "Сделки B2C"}
SOFT_DELAY_MARKERS = (
    "подума",
    "не сейчас",
    "позже",
    "верн",
    "после экзамен",
    "после каникул",
    "после отпуска",
    "к осени",
    "в сентябре",
    "в августе",
    "в мае",
    "после егэ",
    "после огэ",
    "ждет счет",
    "ждёт счет",
    "ждет распис",
    "ждёт распис",
)
HARD_DECLINE_MARKERS = (
    "неинтерес",
    "не актуаль",
    "неактуаль",
    "не подходит",
    "не нужен",
    "не нужна",
    "не будем",
    "отказ",
    "выбрали другое",
    "выбрали другую",
    "купили у конкур",
    "уже купили",
    "нецелев",
    "тема закрыта",
    "окончательно",
)
ALTERNATIVE_OFFER_MARKERS = (
    "дорого",
    "бюджет",
    "скид",
    "рассроч",
    "формат",
    "онлайн",
    "офлайн",
    "другой продукт",
    "другой курс",
    "лагер",
    "индивидуаль",
    "другая программ",
)
ACTIVE_CLIENT_LOSS_REASON_MARKERS = (
    "действующий клиент",
    "действующий ученик",
    "текущий клиент",
    "текущий ученик",
    "продолжает обучение",
    "уже учится",
)
DEFAULT_AMO_LEAD_FIELD_MAP = {
    "close_verdict": "AI-вердикт по закрытию",
    "premature_close_risk": "AI-risk: premature close",
    "close_reason_summary": "AI-основание вердикта",
    "recommended_next_step": "AI-рекомендованный следующий шаг",
    "follow_up_due_at": "AI-дата следующего касания",
    "deal_summary": "AI-сводка по сделке",
}
VERDICT_DISPLAY_MAP = {
    "closed_valid": "Закрыта корректно",
    "closed_too_early": "Закрыта слишком рано",
    "follow_up_needed": "Нужен follow-up",
    "reopen_recommended": "Вернуть в работу",
    "alternative_offer_needed": "Нужен альтернативный оффер",
    "manual_review": "Нужна ручная проверка",
}
RISK_DISPLAY_MAP = {
    "no_risk": "Нет риска",
    "low": "Низкий",
    "medium": "Средний",
    "high": "Высокий",
    "critical": "Критический",
    "manual_review": "Нужна ручная проверка",
}
CLOSED_DEAL_ACTIONABLE_VERDICTS = {
    "closed_too_early",
    "follow_up_needed",
    "reopen_recommended",
    "alternative_offer_needed",
}
CLOSED_DEAL_WRITEBACK_KEYS = (
    "close_verdict",
    "premature_close_risk",
    "close_reason_summary",
    "recommended_next_step",
    "follow_up_due_at",
    "deal_summary",
)
OPEN_DEAL_WRITEBACK_KEYS = (
    "recommended_next_step",
    "follow_up_due_at",
    "deal_summary",
)


@dataclass
class LeadCandidate:
    contact_id: int
    lead_id: int
    score: int
    confidence: float
    reason: str
    lead: dict[str, Any]


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _to_dt(value: Any) -> Optional[datetime]:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    text = _safe_text(value)
    if not text:
        return None
    normalized = text.replace("T", " ").replace("Z", "+00:00")
    for fmt in (
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%d.%m.%Y %H:%M",
        "%d.%m.%Y",
    ):
        try:
            parsed = datetime.strptime(normalized, fmt)
            return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)
        except ValueError:
            continue
    try:
        parsed = datetime.fromisoformat(normalized)
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)
    except ValueError:
        return None


def _iso_or_none(value: Any) -> Optional[str]:
    dt = _to_dt(value)
    return dt.isoformat() if dt is not None else None


def _normalize_name_tokens(value: str) -> set[str]:
    return {chunk.casefold() for chunk in _safe_text(value).replace("_", " ").split() if chunk.strip()}


def _user_matches_managers(user_name: str, managers: list[str]) -> bool:
    user_tokens = _normalize_name_tokens(user_name)
    if not user_tokens:
        return False
    for manager in managers:
        if user_tokens & _normalize_name_tokens(manager):
            return True
    return False


def _field_values_text(field_values: Any) -> str:
    if not isinstance(field_values, list):
        return ""
    values: list[str] = []
    for item in field_values:
        if not isinstance(item, dict):
            continue
        text = _safe_text(item.get("value"))
        if text:
            values.append(text)
    return " | ".join(values)


def _extract_custom_field(entity: dict[str, Any], field_name: str) -> str:
    target = field_name.casefold()
    for item in entity.get("custom_fields_values") or []:
        if not isinstance(item, dict):
            continue
        if _safe_text(item.get("field_name")).casefold() == target:
            return _field_values_text(item.get("values"))
    return ""


def _build_pipeline_meta(pipelines: list[dict[str, Any]]) -> tuple[dict[int, dict[str, Any]], dict[tuple[int, int], dict[str, Any]]]:
    pipeline_map: dict[int, dict[str, Any]] = {}
    status_map: dict[tuple[int, int], dict[str, Any]] = {}
    for pipeline in pipelines:
        pipeline_id = int(pipeline.get("id") or 0)
        if not pipeline_id:
            continue
        pipeline_map[pipeline_id] = pipeline
        for status in (pipeline.get("_embedded") or {}).get("statuses") or []:
            status_id = int(status.get("id") or 0)
            if status_id:
                status_map[(pipeline_id, status_id)] = status
    return pipeline_map, status_map


def _default_target_pipeline_ids(pipelines: list[dict[str, Any]]) -> set[int]:
    explicit = {int(item) for item in settings.crm_amo_target_pipeline_ids if str(item).isdigit()}
    if explicit:
        return explicit
    result: set[int] = set()
    for pipeline in pipelines:
        if pipeline.get("is_archive"):
            continue
        if _safe_text(pipeline.get("name")) in DEFAULT_TARGET_PIPELINE_NAMES:
            result.add(int(pipeline["id"]))
    return result


def _is_closed_status(status_id: int) -> bool:
    return status_id in TERMINAL_SUCCESS_STATUS_IDS or status_id in TERMINAL_LOST_STATUS_IDS


def _is_lost_status(status_id: int) -> bool:
    return status_id in TERMINAL_LOST_STATUS_IDS


def _is_open_status(status_id: int) -> bool:
    return not _is_closed_status(status_id)


def _loss_reason_summary(lead: dict[str, Any]) -> str:
    embedded_loss_reason = (lead.get("_embedded") or {}).get("loss_reason") or []
    if embedded_loss_reason and isinstance(embedded_loss_reason[0], dict):
        text = _safe_text(embedded_loss_reason[0].get("name"))
        if text:
            return text
    custom_reason = _extract_custom_field(lead, "Причина отказа (лид)")
    if custom_reason:
        return custom_reason
    return ""


def _loss_reason_is_active_client(value: Any) -> bool:
    text = _safe_text(value).casefold()
    if not text:
        return False
    return any(marker in text for marker in ACTIVE_CLIENT_LOSS_REASON_MARKERS)


def _contact_phones(contact: dict[str, Any]) -> list[str]:
    phones: list[str] = []
    for item in contact.get("custom_fields_values") or []:
        if not isinstance(item, dict):
            continue
        field_code = _safe_text(item.get("field_code")).upper()
        field_name = _safe_text(item.get("field_name")).casefold()
        if field_code != "PHONE" and "тел" not in field_name and "phone" not in field_name:
            continue
        for value_item in item.get("values") or []:
            if not isinstance(value_item, dict):
                continue
            normalized = normalize_phone(value_item.get("value"))
            if normalized:
                phones.append(normalized)
    unique: list[str] = []
    seen: set[str] = set()
    for phone in phones:
        if phone in seen:
            continue
        seen.add(phone)
        unique.append(phone)
    return unique


def _candidate_score(
    *,
    lead: dict[str, Any],
    phone_context: PhoneContext,
    pipeline_map: dict[int, dict[str, Any]],
    status_map: dict[tuple[int, int], dict[str, Any]],
    user_map: dict[int, str],
    target_pipeline_ids: set[int],
    reference_dt: Optional[datetime],
    contact_id: int,
) -> LeadCandidate:
    lead_id = int(lead.get("id") or 0)
    pipeline_id = int(lead.get("pipeline_id") or 0)
    status_id = int(lead.get("status_id") or 0)
    responsible_user_id = int(lead.get("responsible_user_id") or 0)
    reasons: list[str] = []
    score = 0

    pipeline = pipeline_map.get(pipeline_id) or {}
    status = status_map.get((pipeline_id, status_id)) or {}
    pipeline_name = _safe_text(pipeline.get("name"))
    status_name = _safe_text(status.get("name"))

    if pipeline_id in target_pipeline_ids:
        score += 25
        reasons.append(f"pipeline:{pipeline_name or pipeline_id}")
    elif not pipeline.get("is_archive"):
        score += 10
        reasons.append("pipeline:active_non_target")

    if _is_lost_status(status_id):
        score += 20
        reasons.append(f"status:lost:{status_name or status_id}")
    elif _is_closed_status(status_id):
        score += 8
        reasons.append(f"status:closed:{status_name or status_id}")
    else:
        score += 16
        reasons.append(f"status:open:{status_name or status_id}")

    if reference_dt is not None:
        for candidate_dt, label in (
            (_to_dt(lead.get("updated_at")), "updated"),
            (_to_dt(lead.get("closed_at")), "closed"),
            (_to_dt(lead.get("created_at")), "created"),
        ):
            if candidate_dt is None:
                continue
            delta_days = abs((reference_dt - candidate_dt).total_seconds()) / 86400
            if delta_days <= 3:
                score += 18
                reasons.append(f"time:{label}:<=3d")
                break
            if delta_days <= 14:
                score += 12
                reasons.append(f"time:{label}:<=14d")
                break
            if delta_days <= 45:
                score += 6
                reasons.append(f"time:{label}:<=45d")
                break

    if responsible_user_id and _user_matches_managers(user_map.get(responsible_user_id, ""), phone_context.manager_history):
        score += 12
        reasons.append("manager_match")

    closest_task_at = _to_dt(lead.get("closest_task_at"))
    if closest_task_at is not None and reference_dt is not None:
        delta_days = abs((reference_dt - closest_task_at).total_seconds()) / 86400
        if delta_days <= 7:
            score += 8
            reasons.append("task_near_call")

    latest_call = phone_context.call_rows[0] if phone_context.call_rows else {}
    call_type = _safe_text(latest_call.get("Тип звонка"))
    if call_type == "sales_call" and pipeline_name in DEFAULT_TARGET_PIPELINE_NAMES:
        score += 6
        reasons.append("sales_context_match")
    elif call_type == "service_call" and pipeline_name == "Сделки B2C":
        score -= 4
        reasons.append("service_context_penalty")

    confidence = max(0.05, min(0.99, score / 100.0))
    return LeadCandidate(
        contact_id=contact_id,
        lead_id=lead_id,
        score=score,
        confidence=confidence,
        reason="; ".join(reasons),
        lead=lead,
    )


def _collect_text_corpus(
    *,
    phone_context: PhoneContext,
    lead: dict[str, Any],
    notes: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
) -> str:
    chunks: list[str] = [
        phone_context.history_summary,
        phone_context.chronology,
        phone_context.interest_summary,
        phone_context.objections_summary,
        phone_context.recommended_next_step,
        _loss_reason_summary(lead),
        _extract_custom_field(lead, "Причина отказа (лид)"),
    ]
    if phone_context.contact_row:
        chunks.extend(
            [
                _safe_text(phone_context.contact_row.get("Краткое резюме последнего свежего звонка")),
                _safe_text(phone_context.contact_row.get("Следующий шаг")),
                _safe_text(phone_context.contact_row.get("Возражения")),
            ]
        )
    for row in phone_context.call_rows[:8]:
        chunks.extend(
            [
                _safe_text(row.get("Краткое резюме разговора")),
                _safe_text(row.get("Следующий шаг")),
                _safe_text(row.get("Возражения")),
            ]
        )
    for note in notes[:10]:
        params = note.get("params") or {}
        if isinstance(params, dict):
            chunks.append(_safe_text(params.get("text")))
    for task in tasks[:10]:
        chunks.append(_safe_text(task.get("text")))
        result = task.get("result")
        if isinstance(result, dict):
            chunks.append(_safe_text(result.get("text")))
    return "\n".join(chunk for chunk in chunks if chunk).casefold()


def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)


def _determine_follow_up_due(phone_context: PhoneContext, latest_call_at: Optional[datetime]) -> Optional[str]:
    if phone_context.follow_up_due_at:
        return phone_context.follow_up_due_at
    if latest_call_at is None:
        return None
    return (latest_call_at + timedelta(days=2)).date().isoformat()


def _determine_next_step(verdict: str, phone_context: PhoneContext) -> str:
    if phone_context.recommended_next_step:
        return phone_context.recommended_next_step
    defaults = {
        "reopen_recommended": "Вернуть сделку в работу, назначить задачу ответственному и связаться с клиентом в течение 24 часов.",
        "closed_too_early": "Поставить follow-up и уточнить актуальность интереса, срок и альтернативный формат предложения.",
        "follow_up_needed": "Сделать повторное касание и уточнить решение клиента и срок возврата к обсуждению.",
        "alternative_offer_needed": "Подготовить альтернативный оффер по формату, продукту или бюджету и вернуться к клиенту.",
        "manual_review": "Проверить вручную выбор сделки и основания закрытия перед записью в amoCRM.",
        "closed_valid": "",
    }
    return defaults.get(verdict, "")


def _summarize_lead(lead: dict[str, Any], pipeline_name: str, status_name: str, phone_context: PhoneContext) -> str:
    latest_summary = _safe_text(phone_context.contact_row.get("Краткое резюме последнего свежего звонка") if phone_context.contact_row else "")
    return (
        f"Сделка '{_safe_text(lead.get('name')) or lead.get('id')}' в воронке '{pipeline_name}' со статусом '{status_name}'. "
        f"По телефону {phone_context.phone} в Mango analyse найдено {len(phone_context.call_rows)} звонков. "
        f"Последний содержательный контекст: {latest_summary or phone_context.history_summary[:300]}"
    ).strip()


def _risk_bucket(score: int) -> str:
    if score >= 85:
        return "critical"
    if score >= 65:
        return "high"
    if score >= 40:
        return "medium"
    if score >= 20:
        return "low"
    return "no_risk"


def _analysis_from_selected_lead(
    session: Session,
    *,
    phone_context: PhoneContext,
    candidate: LeadCandidate,
    contact: dict[str, Any],
    pipelines: list[dict[str, Any]],
    users: list[dict[str, Any]],
    lead: Optional[dict[str, Any]] = None,
    notes: Optional[list[dict[str, Any]]] = None,
    tasks: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    lead = lead or fetch_lead(session, lead_id=candidate.lead_id, with_fields="contacts")
    notes = list(notes) if notes is not None else list(fetch_lead_notes(session, lead_id=candidate.lead_id))
    tasks = list(tasks) if tasks is not None else list(fetch_lead_tasks(session, lead_id=candidate.lead_id))
    pipeline_map, status_map = _build_pipeline_meta(pipelines)
    user_map = {int(item.get("id") or 0): _safe_text(item.get("name")) for item in users if int(item.get("id") or 0)}

    pipeline_id = int(lead.get("pipeline_id") or 0)
    status_id = int(lead.get("status_id") or 0)
    pipeline_name = _safe_text((pipeline_map.get(pipeline_id) or {}).get("name"))
    status_name = _safe_text((status_map.get((pipeline_id, status_id)) or {}).get("name"))
    latest_call_at = _to_dt(phone_context.last_call_at)
    closed_at = _to_dt(lead.get("closed_at"))
    loss_reason_summary = _loss_reason_summary(lead)
    corpus = _collect_text_corpus(phone_context=phone_context, lead=lead, notes=notes, tasks=tasks)

    has_soft_delay = _contains_any(corpus, SOFT_DELAY_MARKERS)
    has_hard_decline = _contains_any(corpus, HARD_DECLINE_MARKERS)
    has_alternative_offer = _contains_any(corpus, ALTERNATIVE_OFFER_MARKERS)
    has_next_step = bool(phone_context.recommended_next_step)
    has_follow_up_due = bool(phone_context.follow_up_due_at)
    warm_or_hot = phone_context.current_sales_temperature in {"warm", "hot"}
    ambiguous_signal = has_hard_decline and (has_soft_delay or has_alternative_offer or has_next_step)
    no_tasks = len(tasks) == 0 and lead.get("closest_task_at") in (None, "")
    close_too_fast = False
    if closed_at is not None and latest_call_at is not None and closed_at >= latest_call_at:
        delta_days = (closed_at - latest_call_at).total_seconds() / 86400
        close_too_fast = delta_days <= 7
    else:
        delta_days = None

    risk_score = 0
    reasons: list[str] = []
    if has_soft_delay:
        risk_score += 35
        reasons.append("В звонках есть сигналы отложенного интереса: клиент не сказал окончательное 'нет'.")
    if has_next_step:
        risk_score += 18
        reasons.append("В истории уже был согласованный следующий шаг.")
    if has_follow_up_due:
        risk_score += 10
        reasons.append("По клиенту уже извлечена дата следующего касания.")
    if warm_or_hot:
        risk_score += 12
        reasons.append(f"Текущая температура клиента по звонкам: {phone_context.current_sales_temperature}.")
    if close_too_fast:
        risk_score += 18
        reasons.append("Сделка закрыта вскоре после разговора, без длительного окна follow-up.")
    if no_tasks:
        risk_score += 10
        reasons.append("По сделке не видно качественного follow-up через задачи.")
    if has_alternative_offer:
        risk_score += 14
        reasons.append("По разговору виден потенциал альтернативного оффера, а не окончательной потери клиента.")
    if has_hard_decline:
        risk_score -= 55
        reasons.append("В истории есть признаки жёсткого окончательного отказа.")

    if _is_lost_status(status_id) and _loss_reason_is_active_client(loss_reason_summary):
        close_verdict = "closed_valid"
        premature_close_risk = "no_risk"
        reasons.insert(
            0,
            "Сделка закрыта по причине 'Действующий клиент': клиент продолжает обучение, переоткрытие не требуется.",
        )
    elif candidate.confidence < 0.58:
        close_verdict = "manual_review"
        premature_close_risk = "manual_review"
        reasons.insert(0, "Матчинг сделки недостаточно уверенный: лучше проверить вручную.")
    elif ambiguous_signal and _is_lost_status(status_id):
        close_verdict = "manual_review"
        premature_close_risk = "manual_review"
        reasons.insert(0, "В истории одновременно есть признаки интереса и жёсткого отказа: нужен ручной разбор.")
    elif status_id in TERMINAL_SUCCESS_STATUS_IDS:
        close_verdict = "closed_valid"
        premature_close_risk = "no_risk"
        reasons = reasons or ["Сделка закрыта как успешная."]
    elif _is_lost_status(status_id):
        if has_hard_decline and risk_score < 20:
            close_verdict = "closed_valid"
            premature_close_risk = "no_risk"
        elif risk_score >= 70:
            close_verdict = "reopen_recommended"
            premature_close_risk = _risk_bucket(risk_score)
        elif has_alternative_offer and risk_score >= 40:
            close_verdict = "alternative_offer_needed"
            premature_close_risk = _risk_bucket(risk_score)
        elif risk_score >= 55:
            close_verdict = "closed_too_early"
            premature_close_risk = _risk_bucket(risk_score)
        elif risk_score >= 35:
            close_verdict = "follow_up_needed"
            premature_close_risk = _risk_bucket(risk_score)
        else:
            close_verdict = "closed_valid"
            premature_close_risk = _risk_bucket(risk_score)
    else:
        if warm_or_hot or has_next_step:
            close_verdict = "follow_up_needed"
            premature_close_risk = "low"
            reasons = reasons or ["Сделка еще живая: по истории нужен follow-up, а не закрытие."]
        else:
            close_verdict = "manual_review"
            premature_close_risk = "manual_review"
            reasons = reasons or ["Недостаточно сигналов для автоматического вердикта."]

    close_reason_summary = " ".join(reasons).strip()
    follow_up_due_at = _determine_follow_up_due(phone_context, latest_call_at)
    recommended_next_step = _determine_next_step(close_verdict, phone_context)
    deal_summary = _summarize_lead(lead, pipeline_name, status_name, phone_context)

    return {
        "matched_contact_id": int(contact.get("id") or 0),
        "matched_lead_id": int(lead.get("id") or 0),
        "match_confidence": round(candidate.confidence, 3),
        "match_reason": candidate.reason,
        "call_count_for_lead": len(phone_context.call_rows),
        "first_call_at": phone_context.first_call_at,
        "last_call_at": phone_context.last_call_at,
        "all_call_ids": phone_context.call_ids,
        "manager_history": phone_context.manager_history,
        "interest_summary": phone_context.interest_summary,
        "objections_summary": phone_context.objections_summary,
        "current_sales_temperature": phone_context.current_sales_temperature,
        "recommended_next_step": recommended_next_step,
        "follow_up_due_at": follow_up_due_at,
        "premature_close_risk": premature_close_risk,
        "close_verdict": close_verdict,
        "close_reason_summary": close_reason_summary,
        "deal_summary": deal_summary,
        "manager_action_summary": recommended_next_step,
        "phone": phone_context.phone,
        "pipeline_id": pipeline_id,
        "pipeline_name": pipeline_name,
        "status_id": status_id,
        "status_name": status_name,
        "loss_reason_summary": loss_reason_summary,
        "lead_name": _safe_text(lead.get("name")),
        "lead_responsible_user_id": int(lead.get("responsible_user_id") or 0),
        "lead_responsible_user_name": user_map.get(int(lead.get("responsible_user_id") or 0), ""),
        "lead_created_at": _iso_or_none(lead.get("created_at")),
        "lead_updated_at": _iso_or_none(lead.get("updated_at")),
        "lead_closed_at": _iso_or_none(lead.get("closed_at")),
        "notes_count": len(notes),
        "tasks_count": len(tasks),
        "close_too_fast": close_too_fast,
        "latest_call_type": _safe_text(phone_context.call_rows[0].get("Тип звонка") if phone_context.call_rows else ""),
        "latest_call_summary": _safe_text(phone_context.contact_row.get("Краткое резюме последнего свежего звонка") if phone_context.contact_row else "") or _safe_text(phone_context.call_rows[0].get("Краткое резюме разговора") if phone_context.call_rows else ""),
        "history_summary": phone_context.history_summary,
        "chronology": phone_context.chronology,
        "tallanto_id": phone_context.tallanto_id,
        "tallanto_match_status": phone_context.tallanto_match_status,
        "analysis_source": "heuristic",
        "analysis_mode": "heuristic",
        "confidence": round(candidate.confidence, 3),
        "needs_manual_review": close_verdict == "manual_review",
        "evidence_signals": [],
        "conflict_flags": [],
    }


def _analysis_mode() -> str:
    mode = _safe_text(settings.crm_analysis_mode).lower()
    if mode in {"llm_shadow", "llm_primary", "heuristic"}:
        return mode
    return "heuristic"


def _comparison_summary(heuristic: dict[str, Any], llm: dict[str, Any]) -> dict[str, Any]:
    heuristic_verdict = _safe_text(heuristic.get("close_verdict"))
    llm_verdict = _safe_text(llm.get("close_verdict"))
    heuristic_risk = _safe_text(heuristic.get("premature_close_risk"))
    llm_risk = _safe_text(llm.get("premature_close_risk"))
    verdict_changed = heuristic_verdict != llm_verdict
    risk_changed = heuristic_risk != llm_risk
    severe_verdict_conflict = {
        heuristic_verdict,
        llm_verdict,
    } >= {"closed_valid", "reopen_recommended"} or (
        heuristic_verdict == "closed_valid"
        and llm_verdict in {"closed_too_early", "follow_up_needed", "alternative_offer_needed"}
    ) or (
        llm_verdict == "closed_valid"
        and heuristic_verdict in {"closed_too_early", "follow_up_needed", "alternative_offer_needed"}
    )
    return {
        "heuristic_verdict": heuristic_verdict,
        "llm_verdict": llm_verdict,
        "heuristic_risk": heuristic_risk,
        "llm_risk": llm_risk,
        "verdict_changed": verdict_changed,
        "risk_changed": risk_changed,
        "severe_conflict": bool(severe_verdict_conflict),
    }


def _writeback_blockers(
    *,
    analysis: dict[str, Any],
    mode: str,
    comparison: Optional[dict[str, Any]] = None,
) -> list[str]:
    blockers: list[str] = []
    if mode == "llm_shadow":
        blockers.append("shadow_mode")
    if _safe_text(analysis.get("close_verdict")) == "manual_review":
        blockers.append("manual_review_verdict")
    if bool(analysis.get("needs_manual_review")):
        blockers.append("needs_manual_review")
    try:
        match_confidence = float(analysis.get("match_confidence") or 0)
    except (TypeError, ValueError):
        match_confidence = 0.0
    if match_confidence < 0.72:
        blockers.append("low_match_confidence")
    if analysis.get("analysis_source") == "llm":
        try:
            llm_confidence = float(analysis.get("confidence") or 0)
        except (TypeError, ValueError):
            llm_confidence = 0.0
        if llm_confidence < 0.65:
            blockers.append("low_llm_confidence")
        if _has_blocking_conflict_flags(analysis.get("conflict_flags") or []):
            blockers.append("llm_conflict_flags")
    if comparison and comparison.get("severe_conflict") and _should_block_on_heuristic_llm_conflict(
        analysis=analysis,
        mode=mode,
    ):
        blockers.append("heuristic_llm_conflict")
    deduped: list[str] = []
    seen: set[str] = set()
    for item in blockers:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


NON_BLOCKING_CONFLICT_MARKERS = {
    "loss_reason_conflicts_with_call_history",
    "closed_without_executed_follow_up",
    "agreed_next_step_but_lead_closed",
    "после закрытия были содержательные касания",
    "причина закрытия",
    "не подтверждается содержанием звонков",
    "сделка закрыта без выполненного follow-up",
    "согласован следующий шаг, но сделка закрыта",
}

BLOCKING_CONFLICT_MARKERS = {
    "llm_runtime_error",
    "manual_review",
    "ambiguous",
    "ambiguity",
    "insufficient",
    "insufficient_data",
    "missing_critical",
    "multiple_candidate",
    "multiple equally probable",
    "match_ambiguity",
    "uncertain_match",
    "нет уверенного матча",
    "неоднознач",
    "недостаточно данных",
    "критически не хватает данных",
    "несколько одинаково вероятных сделок",
    "противоречивые источники",
    "нельзя уверенно выбрать",
}


def _has_blocking_conflict_flags(flags: list[Any]) -> bool:
    for raw_flag in flags:
        flag = _safe_text(raw_flag).casefold()
        if not flag:
            continue
        if any(marker in flag for marker in NON_BLOCKING_CONFLICT_MARKERS):
            continue
        if any(marker in flag for marker in BLOCKING_CONFLICT_MARKERS):
            return True
    return False


def _should_block_on_heuristic_llm_conflict(*, analysis: dict[str, Any], mode: str) -> bool:
    if mode == "llm_shadow":
        return True
    if analysis.get("analysis_source") != "llm":
        return True
    try:
        llm_confidence = float(analysis.get("confidence") or 0)
    except (TypeError, ValueError):
        llm_confidence = 0.0
    if bool(analysis.get("needs_manual_review")):
        return True
    return llm_confidence < 0.82


def _finalize_analysis(
    *,
    heuristic_analysis: dict[str, Any],
    llm_analysis: Optional[dict[str, Any]],
) -> tuple[dict[str, Any], Optional[dict[str, Any]], Optional[dict[str, Any]]]:
    mode = _analysis_mode()
    comparison = _comparison_summary(heuristic_analysis, llm_analysis) if llm_analysis else None

    if mode == "heuristic" or llm_analysis is None:
        final = dict(heuristic_analysis)
    else:
        final = {
            **heuristic_analysis,
            **llm_analysis,
            "analysis_source": "llm",
            "analysis_mode": mode,
        }

    blockers = _writeback_blockers(analysis=final, mode=mode, comparison=comparison)
    final["writeback_allowed"] = not blockers
    final["writeback_blockers"] = blockers
    if comparison is not None:
        final["heuristic_llm_comparison"] = comparison
    return final, llm_analysis, comparison


def _build_dossier_and_analysis(
    session: Session,
    *,
    phone_context: PhoneContext,
    candidate: LeadCandidate,
    contact: dict[str, Any],
    pipelines: list[dict[str, Any]],
    users: list[dict[str, Any]],
    active_brand: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any], Optional[dict[str, Any]], Optional[dict[str, Any]], dict[str, Any]]:
    lead = fetch_lead(session, lead_id=candidate.lead_id, with_fields="contacts")
    notes = fetch_lead_notes(session, lead_id=candidate.lead_id)
    tasks = fetch_lead_tasks(session, lead_id=candidate.lead_id)
    heuristic_analysis = _analysis_from_selected_lead(
        session,
        phone_context=phone_context,
        candidate=candidate,
        contact=contact,
        pipelines=pipelines,
        users=users,
        lead=lead,
        notes=notes,
        tasks=tasks,
    )
    pipeline_map, status_map = _build_pipeline_meta(pipelines)
    pipeline_id = int(lead.get("pipeline_id") or 0)
    status_id = int(lead.get("status_id") or 0)
    user_map = {int(item.get("id") or 0): _safe_text(item.get("name")) for item in users if int(item.get("id") or 0)}
    dossier = build_deal_dossier(
        phone_context=phone_context,
        contact=contact,
        lead=lead,
        notes=notes,
        tasks=tasks,
        pipeline_name=_safe_text((pipeline_map.get(pipeline_id) or {}).get("name")),
        status_name=_safe_text((status_map.get((pipeline_id, status_id)) or {}).get("name")),
        user_map=user_map,
        active_brand=active_brand,
        transcript_excerpt_chars=settings.crm_analysis_transcript_excerpt_chars,
        max_transcript_calls=settings.crm_analysis_max_transcript_calls,
    )
    llm_analysis: Optional[dict[str, Any]] = None
    if _analysis_mode() in {"llm_shadow", "llm_primary"}:
        analyzer = DealLLMAnalyzer()
        try:
            llm_analysis = analyzer.analyze(dossier=dossier, heuristic_analysis=heuristic_analysis)
        except DealLLMError as exc:
            llm_analysis = {
                "analysis_schema_version": "deal_llm_v1",
                "close_verdict": "manual_review",
                "premature_close_risk": "manual_review",
                "close_reason_summary": f"LLM анализ завершился ошибкой: {exc}",
                "recommended_next_step": "Проверить сделку вручную, LLM verdict временно недоступен.",
                "follow_up_due_at": heuristic_analysis.get("follow_up_due_at"),
                "deal_summary": heuristic_analysis.get("deal_summary"),
                "manager_action_summary": "",
                "confidence": 0.0,
                "needs_manual_review": True,
                "evidence_signals": [],
                "conflict_flags": ["llm_runtime_error"],
                "llm_provider": settings.crm_analysis_provider,
                "llm_model": settings.crm_analysis_model,
                "llm_prompt_version": "deal_llm_v1",
            }
    final_analysis, normalized_llm_analysis, comparison = _finalize_analysis(
        heuristic_analysis=heuristic_analysis,
        llm_analysis=llm_analysis,
    )
    return final_analysis, heuristic_analysis, normalized_llm_analysis, comparison, dossier


def resolve_target_lead(
    session: Session,
    *,
    phone: str,
    call_at: Optional[str] = None,
) -> dict[str, Any]:
    normalized_phone = normalize_phone(phone)
    if not normalized_phone:
        return {
            "phone": phone,
            "status": "manual_review",
            "summary": "Телефон не удалось нормализовать.",
            "candidates": [],
            "selected": None,
        }

    phone_context = get_phone_context(normalized_phone)
    contacts = search_contacts_by_phone(session, phone=normalized_phone, limit=10)
    if not contacts:
        return {
            "phone": normalized_phone,
            "status": "manual_review",
            "summary": "Контакт в amoCRM по этому телефону не найден.",
            "candidates": [],
            "selected": None,
        }

    pipelines = fetch_pipelines_with_statuses(session)
    users = fetch_users(session)
    pipeline_map, status_map = _build_pipeline_meta(pipelines)
    user_map = {int(item.get("id") or 0): _safe_text(item.get("name")) for item in users if int(item.get("id") or 0)}
    target_pipeline_ids = _default_target_pipeline_ids(pipelines)
    reference_dt = _to_dt(call_at) or _to_dt(phone_context.last_call_at if phone_context else None)

    candidates: list[LeadCandidate] = []
    for contact in contacts:
        contact_id = int(contact.get("id") or 0)
        embedded_leads = (contact.get("_embedded") or {}).get("leads") or []
        lead_ids = [int(item.get("id") or 0) for item in embedded_leads if int(item.get("id") or 0)]
        leads: list[dict[str, Any]] = []
        if lead_ids:
            if os.getenv("AMO_LEADS_BATCH_FETCH", "1") == "1":
                leads = fetch_leads_batch(session, lead_ids=lead_ids, with_fields="contacts")
            else:
                for lead_id in lead_ids:
                    leads.append(fetch_lead(session, lead_id=lead_id, with_fields="contacts"))
        else:
            leads = fetch_related_leads(session, contact_id=contact_id)
        if phone_context is None:
            continue
        for lead in leads:
            candidates.append(
                _candidate_score(
                    lead=lead,
                    phone_context=phone_context,
                    pipeline_map=pipeline_map,
                    status_map=status_map,
                    user_map=user_map,
                    target_pipeline_ids=target_pipeline_ids,
                    reference_dt=reference_dt,
                    contact_id=contact_id,
                )
            )

    candidates.sort(key=lambda item: (item.score, item.confidence), reverse=True)
    selected = candidates[0] if candidates else None
    ambiguous = False
    if len(candidates) >= 2 and selected is not None:
        ambiguous = (selected.score - candidates[1].score) < 8
    manual_review = selected is None or selected.confidence < 0.58 or ambiguous

    return {
        "phone": normalized_phone,
        "status": "manual_review" if manual_review else "matched",
        "summary": "Сделка выбрана автоматически." if not manual_review else "Нужна ручная проверка выбора сделки.",
        "contact_candidates": [
            {
                "contact_id": int(contact.get("id") or 0),
                "name": _safe_text(contact.get("name")),
                "phones": _contact_phones(contact),
            }
            for contact in contacts
        ],
        "candidates": [
            {
                "contact_id": item.contact_id,
                "lead_id": item.lead_id,
                "lead_name": _safe_text(item.lead.get("name")),
                "pipeline_id": int(item.lead.get("pipeline_id") or 0),
                "status_id": int(item.lead.get("status_id") or 0),
                "pipeline_name": _safe_text((pipeline_map.get(int(item.lead.get("pipeline_id") or 0)) or {}).get("name")),
                "status_name": _safe_text((status_map.get((int(item.lead.get("pipeline_id") or 0), int(item.lead.get("status_id") or 0))) or {}).get("name")),
                "confidence": round(item.confidence, 3),
                "score": item.score,
                "reason": item.reason,
            }
            for item in candidates[:10]
        ],
        "selected": {
            "contact_id": selected.contact_id,
            "lead_id": selected.lead_id,
            "confidence": round(selected.confidence, 3),
            "score": selected.score,
            "reason": selected.reason,
        }
        if selected is not None
        else None,
    }


def analyze_by_phone(
    session: Session,
    *,
    phone: str,
    call_at: Optional[str] = None,
) -> dict[str, Any]:
    resolved = resolve_target_lead(session, phone=phone, call_at=call_at)
    if resolved.get("selected") is None:
        return {
            **resolved,
            "analysis": {
                "phone": normalize_phone(phone),
                "close_verdict": "manual_review",
                "premature_close_risk": "manual_review",
                "close_reason_summary": resolved.get("summary") or "Нужна ручная проверка.",
            },
        }

    normalized_phone = normalize_phone(phone)
    phone_context = get_phone_context(normalized_phone or "")
    if phone_context is None:
        return {
            **resolved,
            "analysis": {
                "phone": normalized_phone,
                "close_verdict": "manual_review",
                "premature_close_risk": "manual_review",
                "close_reason_summary": "В Mango analyse нет истории звонков по этому телефону.",
            },
        }

    pipelines = fetch_pipelines_with_statuses(session)
    users = fetch_users(session)
    selected_meta = resolved["selected"]
    contact = fetch_contact(session, contact_id=int(selected_meta["contact_id"]))
    lead = fetch_lead(session, lead_id=int(selected_meta["lead_id"]), with_fields="contacts")
    candidate = LeadCandidate(
        contact_id=int(selected_meta["contact_id"]),
        lead_id=int(selected_meta["lead_id"]),
        score=int(selected_meta["score"]),
        confidence=float(selected_meta["confidence"]),
        reason=_safe_text(selected_meta.get("reason")),
        lead=lead,
    )
    analysis, heuristic_analysis, llm_analysis, comparison, dossier = _build_dossier_and_analysis(
        session,
        phone_context=phone_context,
        candidate=candidate,
        contact=contact,
        pipelines=pipelines,
        users=users,
    )
    return {
        **resolved,
        "analysis_mode": _analysis_mode(),
        "analysis": analysis,
        "heuristic_analysis": heuristic_analysis,
        "llm_analysis": llm_analysis,
        "comparison": comparison,
        "dossier": dossier,
    }


def _lead_field_map() -> dict[str, str]:
    if settings.crm_amo_lead_field_map:
        try:
            parsed = json.loads(settings.crm_amo_lead_field_map)
            if isinstance(parsed, dict):
                return {str(k): _safe_text(v) for k, v in parsed.items() if _safe_text(v)}
        except json.JSONDecodeError:
            pass
    return dict(DEFAULT_AMO_LEAD_FIELD_MAP)


def _display_verdict(value: Any) -> str:
    normalized = _safe_text(value).casefold()
    return VERDICT_DISPLAY_MAP.get(normalized, _safe_text(value))


def _display_risk(value: Any) -> str:
    normalized = _safe_text(value).casefold()
    return RISK_DISPLAY_MAP.get(normalized, _safe_text(value))


def _summarize_context_for_deal_field(analysis: dict[str, Any], *, is_open_deal: bool) -> str:
    latest_summary = _safe_text(analysis.get("latest_call_summary"))
    history_summary = _safe_text(analysis.get("history_summary"))
    chronology = _safe_text(analysis.get("chronology"))
    objections = _safe_text(analysis.get("objections_summary"))
    next_step = _safe_text(analysis.get("recommended_next_step"))

    parts: list[str] = []
    primary_context = latest_summary or history_summary
    if primary_context:
        parts.append(f"Контекст: {primary_context}")
    if chronology:
        parts.append(f"Последние касания: {chronology}")
    if objections:
        parts.append(f"Возражения: {objections}")
    if next_step:
        parts.append(f"Следующий шаг: {next_step}")
    if is_open_deal:
        parts.append("Полная история общения — в карточке контакта.")
    return " ".join(part.strip() for part in parts if part.strip()).strip()


def _logical_writeback_keys_for_analysis(analysis: dict[str, Any]) -> tuple[str, ...]:
    status_id = int(analysis.get("status_id") or 0)
    verdict = _safe_text(analysis.get("close_verdict")).casefold()
    if _is_open_status(status_id):
        return OPEN_DEAL_WRITEBACK_KEYS
    if verdict in CLOSED_DEAL_ACTIONABLE_VERDICTS:
        return CLOSED_DEAL_WRITEBACK_KEYS
    return tuple()


def _prepare_writeback_payload(analysis: dict[str, Any]) -> dict[str, Any]:
    field_map = _lead_field_map()
    payload: dict[str, Any] = {}
    logical_keys = _logical_writeback_keys_for_analysis(analysis)
    if not logical_keys:
        return payload

    is_open_deal = _is_open_status(int(analysis.get("status_id") or 0))
    for logical_key in logical_keys:
        field_name = _safe_text(field_map.get(logical_key))
        if not field_name:
            continue
        if logical_key == "close_verdict":
            value = _display_verdict(analysis.get(logical_key))
        elif logical_key == "premature_close_risk":
            value = _display_risk(analysis.get(logical_key))
        elif logical_key == "deal_summary":
            value = _summarize_context_for_deal_field(analysis, is_open_deal=is_open_deal) or analysis.get(logical_key)
        else:
            value = analysis.get(logical_key)
        if value in (None, ""):
            continue
        payload[field_name] = value
    if settings.crm_analysis_write_ai_office_field:
        ai_office_field = _safe_text(field_map.get("ai_office")) or "AI office"
        ai_office_value = analysis.get("ai_office")
        if ai_office_value in (None, ""):
            ai_office_value = _build_ai_office_service_note(analysis)
        if ai_office_value not in (None, ""):
            payload[ai_office_field] = ai_office_value
    return payload


def _build_ai_office_service_note(analysis: dict[str, Any]) -> str:
    parts: list[str] = []
    matched_lead_id = analysis.get("matched_lead_id")
    matched_contact_id = analysis.get("matched_contact_id")
    if matched_lead_id:
        parts.append(f"lead_id={matched_lead_id}")
    if matched_contact_id:
        parts.append(f"contact_id={matched_contact_id}")
    if analysis.get("match_confidence") not in (None, ""):
        parts.append(f"match_confidence={analysis.get('match_confidence')}")
    if analysis.get("pipeline_name"):
        parts.append(f"pipeline={analysis.get('pipeline_name')}")
    if analysis.get("status_name"):
        parts.append(f"status={analysis.get('status_name')}")
    if analysis.get("phone"):
        parts.append(f"phone={analysis.get('phone')}")
    if analysis.get("current_sales_temperature"):
        parts.append(f"temperature={analysis.get('current_sales_temperature')}")
    if analysis.get("call_count_for_lead") not in (None, ""):
        parts.append(f"call_count={analysis.get('call_count_for_lead')}")
    if analysis.get("tallanto_id"):
        parts.append(f"tallanto_id={analysis.get('tallanto_id')}")
    if analysis.get("match_reason"):
        parts.append(f"match_reason={analysis.get('match_reason')}")
    if analysis.get("close_verdict"):
        parts.append(f"verdict={analysis.get('close_verdict')}")
    if analysis.get("premature_close_risk"):
        parts.append(f"risk={analysis.get('premature_close_risk')}")
    if analysis.get("follow_up_due_at"):
        parts.append(f"follow_up_due_at={analysis.get('follow_up_due_at')}")
    return " | ".join(str(part).strip() for part in parts if str(part).strip())


def _entity_custom_field_value_map(entity: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in entity.get("custom_fields_values") or []:
        if not isinstance(item, dict):
            continue
        field_name = _safe_text(item.get("field_name"))
        if not field_name:
            continue
        values: list[str] = []
        for value_item in item.get("values") or []:
            if not isinstance(value_item, dict):
                continue
            value = _safe_text(value_item.get("value"))
            if value:
                values.append(value)
        result[field_name] = " | ".join(values).strip()
    return result


def _filter_payload_for_safe_mode(
    *,
    lead: dict[str, Any],
    payload: dict[str, Any],
) -> tuple[dict[str, Any], list[str], list[str]]:
    current_values = _entity_custom_field_value_map(lead)
    allowed_payload: dict[str, Any] = {}
    skipped_nonempty: list[str] = []
    unchanged_fields: list[str] = []
    for field_name, new_value in payload.items():
        normalized_new = _safe_text(new_value)
        current_value = _safe_text(current_values.get(field_name))
        if not current_value:
            allowed_payload[field_name] = new_value
            continue
        if current_value == normalized_new:
            unchanged_fields.append(field_name)
            continue
        skipped_nonempty.append(field_name)
    return allowed_payload, skipped_nonempty, unchanged_fields


def write_analysis_to_lead(session: Session, *, analysis: dict[str, Any]) -> dict[str, Any]:
    lead_id = int(analysis.get("matched_lead_id") or 0)
    if not lead_id:
        raise ValueError("matched_lead_id is required for write-back")
    blockers = [
        _safe_text(item)
        for item in (analysis.get("writeback_blockers") or [])
        if _safe_text(item)
    ]
    if blockers:
        raise ValueError(f"write-back is blocked for this analysis: {', '.join(blockers)}")
    payload = _prepare_writeback_payload(analysis)
    if not payload:
        return {
            "mode": "amo_api",
            "entity_type": "lead",
            "entity_id": int(lead_id),
            "status": "skipped",
            "reason": "analysis_state_has_no_actionable_deal_payload",
            "updated_fields": [],
            "skipped_fields": [],
        }

    skipped_nonempty: list[str] = []
    unchanged_fields: list[str] = []
    if settings.crm_amo_deal_writeback_safe_mode:
        current_lead = fetch_lead(session, lead_id=lead_id, with_fields="contacts")
        payload, skipped_nonempty, unchanged_fields = _filter_payload_for_safe_mode(
            lead=current_lead,
            payload=payload,
        )
        if not payload:
            return {
                "mode": "amo_api",
                "entity_type": "lead",
                "entity_id": int(lead_id),
                "status": "skipped",
                "reason": "safe_mode_prevented_overwrite",
                "updated_fields": [],
                "skipped_fields": skipped_nonempty,
                "unchanged_fields": unchanged_fields,
            }

    result = send_lead_custom_field_update(session, lead_id=lead_id, field_payload=payload)
    result["status"] = "written"
    if skipped_nonempty:
        result["skipped_fields"] = skipped_nonempty
    if unchanged_fields:
        result["unchanged_fields"] = unchanged_fields
    return result


def _choose_best_phone_context_for_contact(session: Session, contact_id: int) -> tuple[Optional[str], Optional[PhoneContext], dict[str, Any]]:
    contact = fetch_contact(session, contact_id=contact_id)
    best_phone: Optional[str] = None
    best_context: Optional[PhoneContext] = None
    for phone in _contact_phones(contact):
        phone_context = get_phone_context(phone)
        if phone_context is None:
            continue
        if best_context is None:
            best_phone = phone
            best_context = phone_context
            continue
        best_last = _to_dt(best_context.last_call_at)
        candidate_last = _to_dt(phone_context.last_call_at)
        if candidate_last and (best_last is None or candidate_last > best_last):
            best_phone = phone
            best_context = phone_context
            continue
        if candidate_last == best_last and len(phone_context.call_rows) > len(best_context.call_rows):
            best_phone = phone
            best_context = phone_context
    return best_phone, best_context, contact


def _queue_dir() -> Path:
    path = Path(settings.crm_amo_deal_queue_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _queue_row(analysis: dict[str, Any]) -> dict[str, Any]:
    return {
        "Телефон": analysis.get("phone") or "",
        "ID контакта amoCRM": analysis.get("matched_contact_id") or "",
        "ID сделки amoCRM": analysis.get("matched_lead_id") or "",
        "Сделка": analysis.get("lead_name") or "",
        "Воронка": analysis.get("pipeline_name") or "",
        "Статус": analysis.get("status_name") or "",
        "AI-вердикт": analysis.get("close_verdict") or "",
        "AI-risk": analysis.get("premature_close_risk") or "",
        "Источник анализа": analysis.get("analysis_source") or "",
        "Режим анализа": analysis.get("analysis_mode") or "",
        "LLM confidence": analysis.get("confidence") or "",
        "Match confidence": analysis.get("match_confidence") or "",
        "Следующий шаг": analysis.get("recommended_next_step") or "",
        "Дата следующего касания": analysis.get("follow_up_due_at") or "",
        "Основание": analysis.get("close_reason_summary") or "",
        "Краткая история": analysis.get("history_summary") or "",
        "Tallanto ID": analysis.get("tallanto_id") or "",
        "Writeback allowed": "Да" if analysis.get("writeback_allowed") else "Нет",
        "Writeback blockers": " | ".join(str(item) for item in (analysis.get("writeback_blockers") or []) if _safe_text(item)),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def build_recent_closed_queue(
    session: Session,
    *,
    days_back: Optional[int] = None,
    apply_writeback: bool = False,
    max_leads: Optional[int] = None,
) -> dict[str, Any]:
    days = max(1, int(days_back or settings.crm_amo_recent_closed_days))
    max_items = max(1, int(max_leads)) if max_leads is not None else None
    now = datetime.now(timezone.utc)
    pipelines = fetch_pipelines_with_statuses(session)
    pipeline_map, status_map = _build_pipeline_meta(pipelines)
    target_pipeline_ids = _default_target_pipeline_ids(pipelines)
    users = fetch_users(session)
    closed_from_ts = int((now - timedelta(days=days)).timestamp())
    leads = fetch_recent_leads(session, closed_from_ts=closed_from_ts)

    results: list[dict[str, Any]] = []
    contact_cache: dict[int, dict[str, Any]] = {}
    phone_context_cache: dict[int, tuple[Optional[str], Optional[PhoneContext], dict[str, Any]]] = {}
    pending_analyses: list[tuple[PhoneContext, LeadCandidate, dict[str, Any]]] = []
    eligible_leads: list[tuple[dict[str, Any], int, int, list[dict[str, Any]]]] = []
    unique_contact_ids: list[int] = []
    seen_contact_ids: set[int] = set()

    for lead in leads:
        pipeline_id = int(lead.get("pipeline_id") or 0)
        status_id = int(lead.get("status_id") or 0)
        if pipeline_id not in target_pipeline_ids:
            continue
        if not _is_lost_status(status_id):
            continue
        closed_at = _to_dt(lead.get("closed_at")) or _to_dt(lead.get("updated_at")) or _to_dt(lead.get("created_at"))
        if closed_at is None or closed_at < (now - timedelta(days=days)):
            continue

        contact_refs = (lead.get("_embedded") or {}).get("contacts") or []
        if not contact_refs:
            lead = fetch_lead(session, lead_id=int(lead.get("id") or 0), with_fields="contacts")
            contact_refs = (lead.get("_embedded") or {}).get("contacts") or []

        eligible_leads.append((lead, pipeline_id, status_id, contact_refs))
        for item in contact_refs:
            contact_id = int(item.get("id") or 0)
            if contact_id and contact_id not in seen_contact_ids:
                seen_contact_ids.add(contact_id)
                unique_contact_ids.append(contact_id)

    def _prefetch_phone_context(contact_id: int) -> tuple[int, tuple[Optional[str], Optional[PhoneContext], dict[str, Any]]]:
        worker_session = SessionLocal()
        try:
            return contact_id, _choose_best_phone_context_for_contact(worker_session, contact_id)
        finally:
            worker_session.close()

    if unique_contact_ids:
        if apply_writeback:
            for contact_id in unique_contact_ids:
                phone_context_cache[contact_id] = _choose_best_phone_context_for_contact(session, contact_id)
        else:
            max_workers = min(READ_ONLY_QUEUE_WORKERS, len(unique_contact_ids))
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(_prefetch_phone_context, contact_id) for contact_id in unique_contact_ids]
                for future in concurrent.futures.as_completed(futures):
                    contact_id, cached_context = future.result()
                    phone_context_cache[contact_id] = cached_context

    for lead, pipeline_id, status_id, contact_refs in eligible_leads:
        best_phone: Optional[str] = None
        best_context: Optional[PhoneContext] = None
        best_contact: Optional[dict[str, Any]] = None
        best_contact_id: Optional[int] = None
        for item in contact_refs:
            contact_id = int(item.get("id") or 0)
            if not contact_id:
                continue
            cached_context = phone_context_cache.get(contact_id)
            if cached_context is None:
                cached_context = _choose_best_phone_context_for_contact(session, contact_id)
                phone_context_cache[contact_id] = cached_context
            phone, context, contact = cached_context
            contact_cache[contact_id] = contact
            if context is None:
                continue
            if best_context is None:
                best_phone, best_context, best_contact, best_contact_id = phone, context, contact, contact_id
                continue
            current_last = _to_dt(best_context.last_call_at)
            candidate_last = _to_dt(context.last_call_at)
            if candidate_last and (current_last is None or candidate_last > current_last):
                best_phone, best_context, best_contact, best_contact_id = phone, context, contact, contact_id

        if best_context is None or best_phone is None or best_contact is None or best_contact_id is None:
            results.append(
                {
                    "phone": "",
                    "matched_contact_id": int(contact_refs[0].get("id") or 0) if contact_refs else 0,
                    "matched_lead_id": int(lead.get("id") or 0),
                    "match_confidence": 0.2,
                    "match_reason": "Не удалось связать сделку с историей звонков по телефону.",
                    "close_verdict": "manual_review",
                    "premature_close_risk": "manual_review",
                    "close_reason_summary": "У сделки нет контактного телефона с найденной историей в Mango analyse.",
                    "recommended_next_step": "Проверить вручную контактные данные и релевантность сделки.",
                    "follow_up_due_at": None,
                    "deal_summary": f"Сделка {_safe_text(lead.get('name')) or lead.get('id')} требует ручной проверки из-за отсутствия phone match.",
                    "pipeline_name": _safe_text((pipeline_map.get(pipeline_id) or {}).get("name")),
                    "status_name": _safe_text((status_map.get((pipeline_id, status_id)) or {}).get("name")),
                    "lead_name": _safe_text(lead.get("name")),
                    "history_summary": "",
                    "tallanto_id": "",
                }
            )
            if max_items is not None and len(results) >= max_items:
                break
            continue

        pending_analyses.append(
            (
                best_context,
                LeadCandidate(
                    contact_id=best_contact_id,
                    lead_id=int(lead.get("id") or 0),
                    score=95,
                    confidence=0.95,
                    reason="lead_selected_from_recent_closed_queue",
                    lead=lead,
                ),
                best_contact,
            )
        )
        if max_items is not None and (len(results) + len(pending_analyses)) >= max_items:
            break

    def _analyze_with_fresh_session(
        phone_context: PhoneContext,
        candidate: LeadCandidate,
        contact: dict[str, Any],
    ) -> dict[str, Any]:
        worker_session = SessionLocal()
        try:
            analysis, heuristic_analysis, llm_analysis, comparison, dossier = _build_dossier_and_analysis(
                worker_session,
                phone_context=phone_context,
                candidate=candidate,
                contact=contact,
                pipelines=pipelines,
                users=users,
            )
            analysis["heuristic_analysis"] = heuristic_analysis
            analysis["llm_analysis"] = llm_analysis
            analysis["comparison"] = comparison
            analysis["dossier"] = dossier
            return analysis
        finally:
            worker_session.close()

    if apply_writeback or len(pending_analyses) <= 1:
        for phone_context, candidate, contact in pending_analyses:
            analysis, heuristic_analysis, llm_analysis, comparison, dossier = _build_dossier_and_analysis(
                session,
                phone_context=phone_context,
                candidate=candidate,
                contact=contact,
                pipelines=pipelines,
                users=users,
            )
            analysis["heuristic_analysis"] = heuristic_analysis
            analysis["llm_analysis"] = llm_analysis
            analysis["comparison"] = comparison
            analysis["dossier"] = dossier
            if apply_writeback and analysis.get("writeback_allowed"):
                analysis["writeback_result"] = write_analysis_to_lead(session, analysis=analysis)
            results.append(analysis)
    else:
        max_workers = min(READ_ONLY_QUEUE_WORKERS, len(pending_analyses))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_analyze_with_fresh_session, phone_context, candidate, contact)
                for phone_context, candidate, contact in pending_analyses
            ]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = _queue_dir() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    reopen = [row for row in results if row.get("close_verdict") == "reopen_recommended"]
    follow_up = [row for row in results if row.get("close_verdict") in {"closed_too_early", "follow_up_needed", "alternative_offer_needed"}]
    manual = [row for row in results if row.get("close_verdict") == "manual_review"]

    all_rows = [_queue_row(row) for row in results]
    _write_csv(run_dir / "all_results.csv", all_rows)
    _write_csv(run_dir / "reopen_candidates.csv", [_queue_row(row) for row in reopen])
    _write_csv(run_dir / "follow_up_candidates.csv", [_queue_row(row) for row in follow_up])
    _write_csv(run_dir / "manual_review.csv", [_queue_row(row) for row in manual])
    (run_dir / "all_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "days_back": days,
        "max_leads": max_items,
        "target_pipeline_ids": sorted(target_pipeline_ids),
        "analyzed": len(results),
        "reopen_candidates": len(reopen),
        "follow_up_candidates": len(follow_up),
        "manual_review": len(manual),
        "writeback_applied": apply_writeback,
        "analysis_mode": _analysis_mode(),
        "files": {
            "all_results_csv": str(run_dir / "all_results.csv"),
            "reopen_candidates_csv": str(run_dir / "reopen_candidates.csv"),
            "follow_up_candidates_csv": str(run_dir / "follow_up_candidates.csv"),
            "manual_review_csv": str(run_dir / "manual_review.csv"),
            "all_results_json": str(run_dir / "all_results.json"),
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (_queue_dir() / "latest_run.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def get_latest_queue_snapshot() -> dict[str, Any]:
    path = _queue_dir() / "latest_run.json"
    if not path.exists():
        return {
            "status": "empty",
            "summary": "Очередь сделок еще не строилась.",
        }
    return json.loads(path.read_text(encoding="utf-8"))
