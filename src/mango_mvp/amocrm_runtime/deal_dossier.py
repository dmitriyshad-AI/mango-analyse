from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from mango_mvp.amocrm_runtime.phone_context import PhoneContext
from mango_mvp.amocrm_runtime.tallanto_context import build_live_tallanto_context
from mango_mvp.services.dialogue_contract import (
    DialogueContractError,
    SOURCE_TRANSCRIPT_FALLBACK,
    UNTRUSTED_SUMMARY,
    build_dialogue_input,
    call_record_view,
    guard_stored_analysis,
)

MISSING_CALL_ARTIFACT_SUMMARY = (
    "Исходные данные звонка недоступны. Содержание и роли сторон требуют ручной проверки."
)

# Everything in a dossier entry that states what a *side* of the call did: an
# agreement, an objection, a next step, a deadline, a probability.  With the
# sides unproven none of it may reach an AMO draft, so it is dropped rather
# than shown with a caveat nobody reads.
ROLE_DEPENDENT_CALL_FIELDS = (
    "products",
    "subjects",
    "objections",
    "next_step",
    "follow_up_due_at",
    "probability_percent",
)

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


def _fetch_call_artifact(
    source_db_path: str,
    source_filename: str,
    call_record_id: Any = None,
    *,
    connection: sqlite3.Connection | None = None,
) -> dict[str, Any]:
    db_path = _safe_text(source_db_path)
    filename = _safe_text(source_filename)
    if not db_path or not filename:
        return {}

    path = Path(db_path)
    if not path.exists() or not path.is_file():
        return {}

    owned_connection = connection is None
    current: sqlite3.Connection | None = None
    try:
        current = connection or sqlite3.connect(path)
        current.row_factory = sqlite3.Row
        columns = {
            str(item[1]) for item in current.execute("PRAGMA table_info(call_records)")
        }
        recording_column = (
            "source_recording_id"
            if "source_recording_id" in columns
            else "NULL AS source_recording_id"
        )
        record_id = _safe_text(call_record_id)
        where = "id = ? AND source_filename = ?" if record_id else "source_filename = ?"
        params: tuple[Any, ...] = (record_id, filename) if record_id else (filename,)
        rows = current.execute(
            f"""
            SELECT id,
                   source_call_id,
                   {recording_column},
                   source_filename,
                   started_at,
                   manager_name,
                   duration_sec,
                   transcript_text,
                   transcript_variants_json,
                   analysis_json,
                   resolve_status,
                   analysis_status
              FROM call_records
             WHERE {where}
             ORDER BY CASE WHEN analysis_status = 'done' THEN 0 ELSE 1 END,
                      CASE WHEN resolve_status IN ('done', 'skipped') THEN 0 ELSE 1 END,
                      id ASC
            """,
            params,
        ).fetchall()
        # The filename is a legacy compatibility key, not an identity.  A
        # non-unique basename must fail closed instead of selecting a neighbour.
        row = rows[0] if len(rows) == 1 else None
    except sqlite3.Error:
        row = None
    finally:
        try:
            if owned_connection and current is not None:
                current.close()
        except Exception:
            pass

    if row is None:
        return {}

    artifact = {str(k): row[k] for k in row.keys()}
    return dict(artifact)


def _call_artifact_fingerprint(artifact: dict[str, Any]) -> str:
    raw = json.dumps(
        artifact, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def call_source_snapshot_is_current(snapshot: Any) -> bool:
    if not isinstance(snapshot, list):
        return False
    for item in snapshot:
        if not isinstance(item, dict):
            return False
        artifact = _fetch_call_artifact(
            _safe_text(item.get("source_db_path")),
            _safe_text(item.get("source_filename")),
            item.get("call_record_id"),
        )
        expected = _safe_text(item.get("source_fingerprint"))
        if not artifact or not expected or _call_artifact_fingerprint(artifact) != expected:
            return False
    return True


def _guarded_call_analysis(artifact: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """The stored analysis as the shared role guard allows it to be read.

    The dossier feeds an AMO draft a human will send, so it may not be the one
    reader that revives a cleaned field.  It applies the same projector as
    Analyse, Excel, AI Office and both Google paths.  Without the source row the
    export cannot prove that its old values passed the current guard, so absence
    is untrusted rather than permission to reuse them.
    """
    if not artifact:
        return {}, True
    guarded = guard_stored_analysis(
        call_record_view(artifact), _json_loads(artifact.get("analysis_json"))
    )
    flags = guarded.get("quality_flags")
    untrusted = bool(
        isinstance(flags, dict)
        and (
            flags.get("role_attribution_untrusted")
            or flags.get("analysis_contract_invalid")
        )
    )
    return guarded, untrusted


def _analysis_summary(analysis: dict[str, Any]) -> str:
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


def _analysis_call_fields(analysis: dict[str, Any]) -> dict[str, str]:
    """Project the current guarded analysis; never revive an older export row."""
    fields = analysis.get("display_fields")
    if not isinstance(fields, dict):
        fields = analysis.get("structured_fields")
    fields = fields if isinstance(fields, dict) else {}
    interests = fields.get("interests") if isinstance(fields.get("interests"), dict) else {}
    next_step = fields.get("next_step") if isinstance(fields.get("next_step"), dict) else {}

    def joined(value: Any) -> str:
        if isinstance(value, list):
            return " | ".join(_dedupe([_safe_text(item) for item in value]))
        return _safe_text(value)

    return {
        "products": joined(interests.get("products")),
        "subjects": joined(interests.get("subjects")),
        "objections": joined(fields.get("objections")),
        "next_step": _safe_text(next_step.get("action")),
        "follow_up_due_at": _safe_text(next_step.get("due")),
        "probability_percent": _safe_text(analysis.get("follow_up_score")),
        "lead_priority": _safe_text(fields.get("lead_priority")),
    }


def _safe_transcript_excerpt(
    artifact: dict[str, Any], transcript_text: str, roles_untrusted: bool, max_chars: int
) -> str:
    if not roles_untrusted:
        return _truncate(transcript_text, max_chars)
    try:
        dialogue = build_dialogue_input(call_record_view(artifact))
        if dialogue.source == SOURCE_TRANSCRIPT_FALLBACK:
            return ""
        return dialogue.render(max_chars=max_chars)
    except DialogueContractError:
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
    call_source_snapshot: list[dict[str, str]] = []

    for index, row in enumerate(call_rows):
        source_filename = _safe_text(row.get("Имя исходного файла"))
        source_db_path = _safe_text(row.get("Источник лучшего статуса"))
        call_record_id = _safe_text(row.get("ID звонка"))
        artifact = _fetch_call_artifact(
            source_db_path, source_filename, call_record_id
        )
        if artifact:
            call_source_snapshot.append(
                {
                    "source_db_path": source_db_path,
                    "source_filename": source_filename,
                    "call_record_id": _safe_text(artifact.get("id")),
                    "source_fingerprint": _call_artifact_fingerprint(artifact),
                }
            )
        transcript_text = _safe_text(artifact.get("transcript_text"))
        analysis, roles_untrusted = _guarded_call_analysis(artifact)
        analysis_summary = _analysis_summary(analysis)
        analysis_fields = _analysis_call_fields(analysis)
        summary = (
            MISSING_CALL_ARTIFACT_SUMMARY
            if not artifact
            else UNTRUSTED_SUMMARY
            if roles_untrusted
            else analysis_summary
        )
        call_id = call_record_id
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
            **analysis_fields,
            "source_filename": source_filename,
            "source_db_path": source_db_path,
        }
        if roles_untrusted:
            # Fail-closed: the recording never proved who the manager was, so
            # the dossier states nothing about what either side did.
            for field in ROLE_DEPENDENT_CALL_FIELDS:
                call_entry[field] = ""
            call_entry["role_attribution_untrusted"] = True
            call_entry["call_artifact_missing"] = not bool(artifact)
        call_history.append(call_entry)
        if index < max_transcript_calls:
            transcript_excerpt = _safe_transcript_excerpt(
                artifact, transcript_text, roles_untrusted, transcript_excerpt_chars
            )
            transcript_entry = {
                **call_entry,
                "transcript_excerpt": transcript_excerpt,
                "full_transcript_available": bool(transcript_text),
                "analysis_excerpt": _truncate(analysis_summary, 500),
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

    trusted_calls = [
        item for item in call_history if not item.get("role_attribution_untrusted")
    ]
    latest_trusted = trusted_calls[0] if trusted_calls else {}
    trusted_products = _dedupe(
        [
            value
            for item in trusted_calls
            for value in _safe_text(item.get("products")).split(" | ")
        ]
    )
    trusted_subjects = _dedupe(
        [
            value
            for item in trusted_calls
            for value in _safe_text(item.get("subjects")).split(" | ")
        ]
    )
    trusted_objections = _dedupe(
        [
            value
            for item in trusted_calls
            for value in _safe_text(item.get("objections")).split(" | ")
        ]
    )
    trusted_summaries = [
        _safe_text(item.get("summary"))
        for item in reversed(trusted_calls)
        if _safe_text(item.get("summary"))
    ]
    trusted_chronology = [
        f"{_safe_text(item.get('started_at'))}: {_safe_text(item.get('summary'))}"
        for item in reversed(trusted_calls)
        if _safe_text(item.get("started_at")) and _safe_text(item.get("summary"))
    ]
    contact_rollup = {
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
        "latest_fresh_summary": _safe_text(latest_trusted.get("summary")),
        "latest_fresh_type": _safe_text(latest_trusted.get("call_type")),
        "history_summary": "\n".join(trusted_summaries),
        "chronology": "\n".join(trusted_chronology),
        "interest_summary": " | ".join(_dedupe([*trusted_products, *trusted_subjects])),
        "objections_summary": " | ".join(trusted_objections),
        "current_sales_temperature": _safe_text(latest_trusted.get("lead_priority")),
        "recommended_next_step": _safe_text(latest_trusted.get("next_step")),
        "follow_up_due_at": _safe_text(latest_trusted.get("follow_up_due_at")) or None,
        "parent_fio": _safe_text(contact_row.get("ФИО родителя")),
        "child_fio": _safe_text(contact_row.get("ФИО ребенка")),
        "email": _safe_text(contact_row.get("Email")),
        "recommended_product": trusted_products[0] if trusted_products else "",
        "tallanto_id": phone_context.tallanto_id,
        "tallanto_match_status": phone_context.tallanto_match_status,
        "tallanto_parent_fio": _safe_text(contact_row.get("ФИО родителя Tallanto")),
        "tallanto_contact": _safe_text(contact_row.get("Контакт Tallanto")),
        "tallanto_owner": _safe_text(contact_row.get("Ответственный Tallanto")),
        "tallanto_student_type": _safe_text(contact_row.get("Тип ученика Tallanto")),
        "tallanto_branch": _safe_text(contact_row.get("Филиал Tallanto")),
    }
    any_call_untrusted = any(
        bool(item.get("role_attribution_untrusted")) for item in call_history
    )
    all_calls_untrusted = bool(call_history) and all(
        bool(item.get("role_attribution_untrusted")) for item in call_history
    )
    if any_call_untrusted:
        for field in (
            "latest_fresh_summary", "latest_fresh_type", "history_summary", "chronology",
            "interest_summary", "objections_summary", "current_sales_temperature",
            "recommended_next_step", "follow_up_due_at", "parent_fio", "child_fio",
            "email", "recommended_product",
        ):
            contact_rollup[field] = ""
        contact_rollup["role_attribution_untrusted"] = True
        contact_rollup["role_attribution_mixed"] = not all_calls_untrusted

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
        "contact_rollup": contact_rollup,
        "manager_history": phone_context.manager_history,
        "all_call_ids": _dedupe(call_ids),
        "source_db_paths": _dedupe(source_db_paths),
        "call_source_snapshot": call_source_snapshot,
        "call_source_count": len(call_rows),
        "call_history": call_history,
        "transcript_context": transcript_context,
        "notes": normalized_notes,
        "tasks": normalized_tasks,
        "tallanto_live": tallanto_live,
    }
