from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from mango_mvp.amocrm_runtime.phone_context import PhoneContext
from mango_mvp.amocrm_runtime.tallanto_context import build_live_tallanto_context

_SQLITE_ARTIFACT_CACHE: dict[tuple[str, str], dict[str, Any]] = {}


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _parse_dt(value: Any) -> tuple[int, str]:
    candidate = _safe_text(value)
    if not candidate:
        return (0, "")
    normalized = candidate.replace("T", " ")
    for fmt in (
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%d.%m.%Y %H:%M",
        "%d.%m.%Y",
    ):
        try:
            parsed = datetime.strptime(normalized, fmt)
            return (int(parsed.timestamp()), candidate)
        except ValueError:
            continue
    return (0, candidate)


def _json_loads(raw: Any) -> dict[str, Any]:
    text = _safe_text(raw)
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _truncate(text: Any, max_chars: int) -> str:
    value = _safe_text(text)
    if not value or len(value) <= max_chars:
        return value
    if max_chars < 120:
        return value[:max_chars]
    head = max_chars // 2
    tail = max_chars - head - 1
    return f"{value[:head].rstrip()}…{value[-tail:].lstrip()}"


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = _safe_text(value)
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def _note_text(note: dict[str, Any]) -> str:
    params = note.get("params") if isinstance(note.get("params"), dict) else {}
    candidates = [
        note.get("text"),
        params.get("text"),
        params.get("note"),
        params.get("message"),
        params.get("service"),
    ]
    for candidate in candidates:
        text = _safe_text(candidate)
        if text:
            return text
    return ""


def _task_text(task: dict[str, Any]) -> str:
    candidates = [
        task.get("text"),
        task.get("result"),
        task.get("result", {}).get("text") if isinstance(task.get("result"), dict) else "",
    ]
    for candidate in candidates:
        text = _safe_text(candidate)
        if text:
            return text
    return ""


def _fetch_call_artifact(source_db_path: str, source_filename: str) -> dict[str, Any]:
    db_path = _safe_text(source_db_path)
    filename = _safe_text(source_filename)
    if not db_path or not filename:
        return {}
    key = (db_path, filename)
    cached = _SQLITE_ARTIFACT_CACHE.get(key)
    if cached is not None:
        return dict(cached)

    path = Path(db_path)
    if not path.exists() or not path.is_file():
        _SQLITE_ARTIFACT_CACHE[key] = {}
        return {}

    try:
        connection = sqlite3.connect(path)
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            """
            SELECT source_filename,
                   started_at,
                   manager_name,
                   duration_sec,
                   transcript_text,
                   transcript_variants_json,
                   analysis_json,
                   resolve_status,
                   analysis_status
              FROM call_records
             WHERE source_filename = ?
             ORDER BY CASE WHEN analysis_status = 'done' THEN 0 ELSE 1 END,
                      CASE WHEN resolve_status IN ('done', 'skipped') THEN 0 ELSE 1 END,
                      id ASC
             LIMIT 1
            """,
            (filename,),
        ).fetchone()
    except sqlite3.Error:
        row = None
    finally:
        try:
            connection.close()
        except Exception:
            pass

    if row is None:
        _SQLITE_ARTIFACT_CACHE[key] = {}
        return {}

    artifact = {str(k): row[k] for k in row.keys()}
    _SQLITE_ARTIFACT_CACHE[key] = artifact
    return dict(artifact)


def _analysis_summary_from_json(raw_analysis: Any) -> str:
    analysis = _json_loads(raw_analysis)
    if not analysis:
        return ""
    for key in (
        "history_summary",
        "summary",
        "deal_summary",
        "close_reason_summary",
    ):
        text = _safe_text(analysis.get(key))
        if text:
            return text
    return ""


def _variant_overview(raw_variants: Any) -> dict[str, Any]:
    payload = _json_loads(raw_variants)
    if not payload:
        return {}
    variant_names: list[str] = []
    transcript_samples: dict[str, str] = {}
    for key, value in payload.items():
        if not isinstance(value, dict):
            continue
        transcript_text = _safe_text(value.get("transcript_text"))
        if transcript_text:
            variant_names.append(key)
            transcript_samples[key] = _truncate(transcript_text, 400)
    if not variant_names:
        return {}
    return {
        "available_variants": sorted(variant_names),
        "variant_transcript_samples": transcript_samples,
    }


def build_deal_dossier(
    *,
    phone_context: PhoneContext,
    contact: dict[str, Any],
    lead: dict[str, Any],
    notes: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
    pipeline_name: str,
    status_name: str,
    user_map: dict[int, str],
    active_brand: str | None = None,
    transcript_excerpt_chars: int = 2200,
    max_transcript_calls: int = 8,
) -> dict[str, Any]:
    contact_row = phone_context.contact_row or {}
    call_rows = list(phone_context.call_rows)
    call_rows.sort(key=lambda item: _parse_dt(item.get("Дата и время звонка", ""))[0], reverse=True)
    tallanto_live = build_live_tallanto_context(
        phone=phone_context.phone,
        tallanto_id=phone_context.tallanto_id,
        tallanto_match_status=phone_context.tallanto_match_status,
        active_brand=active_brand,
    )

    call_history: list[dict[str, Any]] = []
    transcript_context: list[dict[str, Any]] = []
    call_ids: list[str] = []
    source_db_paths: list[str] = []

    for index, row in enumerate(call_rows):
        source_filename = _safe_text(row.get("Имя исходного файла"))
        source_db_path = _safe_text(row.get("Источник лучшего статуса"))
        artifact = _fetch_call_artifact(source_db_path, source_filename)
        transcript_text = _safe_text(artifact.get("transcript_text"))
        raw_analysis = artifact.get("analysis_json")
        summary = _safe_text(row.get("Краткое резюме разговора")) or _analysis_summary_from_json(raw_analysis)
        call_id = _safe_text(row.get("ID звонка"))
        if call_id:
            call_ids.append(call_id)
        if source_db_path:
            source_db_paths.append(source_db_path)
        call_entry = {
            "call_id": call_id,
            "started_at": _safe_text(row.get("Дата и время звонка")) or _safe_text(artifact.get("started_at")),
            "manager_name": _safe_text(row.get("Менеджер")) or _safe_text(artifact.get("manager_name")),
            "direction": _safe_text(row.get("Направление звонка")),
            "duration_sec": _safe_text(row.get("Длительность, сек")) or _safe_text(artifact.get("duration_sec")),
            "call_type": _safe_text(row.get("Тип звонка")),
            "fresh_period": _safe_text(row.get("Свежий период")),
            "resolve_status": _safe_text(row.get("Статус Resolve")),
            "analyze_status": _safe_text(row.get("Статус Analyze")),
            "summary": summary,
            "products": _safe_text(row.get("Продукты интереса")),
            "subjects": _safe_text(row.get("Предметы интереса")),
            "objections": _safe_text(row.get("Возражения")),
            "next_step": _safe_text(row.get("Следующий шаг")),
            "follow_up_due_at": _safe_text(row.get("Рекомендуемая дата следующего контакта")),
            "probability_percent": _safe_text(row.get("Вероятность продажи, %")),
            "source_filename": source_filename,
            "source_db_path": source_db_path,
        }
        call_history.append(call_entry)
        if index < max_transcript_calls:
            transcript_entry = {
                **call_entry,
                "transcript_excerpt": _truncate(transcript_text, transcript_excerpt_chars),
                "full_transcript_available": bool(transcript_text),
                "analysis_excerpt": _truncate(_analysis_summary_from_json(raw_analysis), 500),
                "variant_overview": _variant_overview(artifact.get("transcript_variants_json")),
            }
            transcript_context.append(transcript_entry)

    normalized_notes = []
    for item in notes[:50]:
        text = _note_text(item)
        normalized_notes.append(
            {
                "id": int(item.get("id") or 0),
                "created_at": _safe_text(item.get("created_at")),
                "updated_at": _safe_text(item.get("updated_at")),
                "note_type": _safe_text(item.get("note_type")) or _safe_text(item.get("entity_type")),
                "text": _truncate(text, 1500),
            }
        )

    normalized_tasks = []
    for item in tasks[:50]:
        normalized_tasks.append(
            {
                "id": int(item.get("id") or 0),
                "created_at": _safe_text(item.get("created_at")),
                "updated_at": _safe_text(item.get("updated_at")),
                "complete_till": _safe_text(item.get("complete_till")),
                "is_completed": bool(item.get("is_completed")),
                "text": _truncate(_task_text(item), 700),
                "responsible_user_id": int(item.get("responsible_user_id") or 0),
                "responsible_user_name": user_map.get(int(item.get("responsible_user_id") or 0), ""),
            }
        )

    lead_custom_fields: dict[str, str] = {}
    for item in lead.get("custom_fields_values") or []:
        if not isinstance(item, dict):
            continue
        field_name = _safe_text(item.get("field_name"))
        if not field_name:
            continue
        values: list[str] = []
        for value_item in item.get("values") or []:
            if not isinstance(value_item, dict):
                continue
            text = _safe_text(value_item.get("value"))
            if text:
                values.append(text)
        if values:
            lead_custom_fields[field_name] = " | ".join(_dedupe(values))

    return {
        "dossier_schema_version": "deal_dossier_v1",
        "phone": phone_context.phone,
        "phone_context_source_dir": phone_context.source_dir,
        "contact": {
            "id": int(contact.get("id") or 0),
            "name": _safe_text(contact.get("name")),
            "responsible_user_id": int(contact.get("responsible_user_id") or 0),
            "responsible_user_name": user_map.get(int(contact.get("responsible_user_id") or 0), ""),
        },
        "lead": {
            "id": int(lead.get("id") or 0),
            "name": _safe_text(lead.get("name")),
            "pipeline_id": int(lead.get("pipeline_id") or 0),
            "pipeline_name": pipeline_name,
            "status_id": int(lead.get("status_id") or 0),
            "status_name": status_name,
            "responsible_user_id": int(lead.get("responsible_user_id") or 0),
            "responsible_user_name": user_map.get(int(lead.get("responsible_user_id") or 0), ""),
            "created_at": _safe_text(lead.get("created_at")),
            "updated_at": _safe_text(lead.get("updated_at")),
            "closed_at": _safe_text(lead.get("closed_at")),
            "loss_reason": _safe_text(((lead.get("_embedded") or {}).get("loss_reason") or [{}])[0].get("name")),
            "custom_fields": lead_custom_fields,
        },
        "contact_rollup": {
            "total_calls_history": _safe_text(contact_row.get("Всего звонков в истории")),
            "fully_analyzed_calls": _safe_text(contact_row.get("Звонков с полным анализом")),
            "unfinished_calls": _safe_text(contact_row.get("Незакрытых звонков в истории")),
            "full_history_analyzed": _safe_text(contact_row.get("Полная история проанализирована")),
            "first_call_at": phone_context.first_call_at,
            "last_call_at": phone_context.last_call_at,
            "fresh_calls_count": _safe_text(contact_row.get("Свежих звонков за период")),
            "latest_fresh_call_at": _safe_text(contact_row.get("Последний свежий звонок")),
            "latest_fresh_call_analyzed": _safe_text(contact_row.get("Последний свежий звонок проанализирован")),
            "latest_fresh_manager": _safe_text(contact_row.get("Менеджер последнего свежего звонка")),
            "latest_fresh_summary": _safe_text(contact_row.get("Краткое резюме последнего свежего звонка")),
            "latest_fresh_type": _safe_text(contact_row.get("Тип последнего свежего звонка")),
            "history_summary": phone_context.history_summary,
            "chronology": phone_context.chronology,
            "interest_summary": phone_context.interest_summary,
            "objections_summary": phone_context.objections_summary,
            "current_sales_temperature": phone_context.current_sales_temperature,
            "recommended_next_step": phone_context.recommended_next_step,
            "follow_up_due_at": phone_context.follow_up_due_at,
            "parent_fio": _safe_text(contact_row.get("ФИО родителя")),
            "child_fio": _safe_text(contact_row.get("ФИО ребенка")),
            "email": _safe_text(contact_row.get("Email")),
            "recommended_product": _safe_text(contact_row.get("Рекомендуемый продукт")),
            "tallanto_id": phone_context.tallanto_id,
            "tallanto_match_status": phone_context.tallanto_match_status,
            "tallanto_parent_fio": _safe_text(contact_row.get("ФИО родителя Tallanto")),
            "tallanto_contact": _safe_text(contact_row.get("Контакт Tallanto")),
            "tallanto_owner": _safe_text(contact_row.get("Ответственный Tallanto")),
            "tallanto_student_type": _safe_text(contact_row.get("Тип ученика Tallanto")),
            "tallanto_branch": _safe_text(contact_row.get("Филиал Tallanto")),
        },
        "manager_history": phone_context.manager_history,
        "all_call_ids": _dedupe(call_ids),
        "source_db_paths": _dedupe(source_db_paths),
        "call_history": call_history,
        "transcript_context": transcript_context,
        "notes": normalized_notes,
        "tasks": normalized_tasks,
        "tallanto_live": tallanto_live,
    }
