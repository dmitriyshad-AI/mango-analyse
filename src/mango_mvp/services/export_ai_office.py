from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional

import requests
from sqlalchemy import select
from sqlalchemy.orm import Session

from mango_mvp.config import Settings
from mango_mvp.models import CallRecord
from mango_mvp.services.analyze import AnalyzeService


class AIOfficeExportError(RuntimeError):
    pass


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _clean_list(value: Any) -> list[str]:
    items: list[str] = []
    seen: set[str] = set()
    for raw in _as_list(value):
        text = _clean_text(raw)
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        items.append(text)
    return items


def _iso_datetime(value: Optional[datetime]) -> str | None:
    if value is None:
        return None
    dt = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _normalize_api_base_url(base_url: str) -> str:
    normalized = (base_url or "").strip().rstrip("/")
    if not normalized:
        raise AIOfficeExportError("AI_OFFICE_API_BASE_URL is not configured.")
    if normalized.endswith("/api"):
        return normalized
    return f"{normalized}/api"


def _build_insights_url(base_url: str, project_id: str) -> str:
    api_base = _normalize_api_base_url(base_url)
    project = str(project_id or "").strip()
    if not project:
        raise AIOfficeExportError("AI Office project_id is required.")
    return f"{api_base}/projects/{project}/calls/insights"


def _parse_analysis(call: CallRecord, settings: Settings) -> Dict[str, Any]:
    raw = (call.analysis_json or "").strip()
    if not raw:
        raise AIOfficeExportError(f"Call #{call.id} has empty analysis_json.")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AIOfficeExportError(f"Call #{call.id} has invalid analysis_json.") from exc
    if not isinstance(payload, dict):
        raise AIOfficeExportError(f"Call #{call.id} analysis_json must decode to an object.")
    return AnalyzeService(settings).migrate_analysis_payload(call, payload)


def build_call_insight_payload(call: CallRecord, analysis: Dict[str, Any]) -> Dict[str, Any]:
    normalized = analysis if isinstance(analysis, dict) else {}
    blocks = _as_dict(normalized.get("structured_fields"))
    if not blocks:
        blocks = _as_dict(normalized.get("crm_blocks"))

    people = _as_dict(blocks.get("people"))
    contacts = _as_dict(blocks.get("contacts"))
    student = _as_dict(blocks.get("student"))
    interests = _as_dict(blocks.get("interests"))
    commercial = _as_dict(blocks.get("commercial"))
    next_step = _as_dict(blocks.get("next_step"))

    history_summary = (
        _clean_text(normalized.get("history_summary"))
        or _clean_text(normalized.get("history_short"))
        or _clean_text(normalized.get("summary"))
    )
    if not history_summary:
        raise AIOfficeExportError(f"Call #{call.id} analysis does not contain history summary.")

    lead_priority = _clean_text(blocks.get("lead_priority"))
    if lead_priority not in {"hot", "warm", "cold"}:
        lead_priority = None

    payload = {
        "schema_version": "call_insight_v1",
        "source": {
            "system": "mango_analyse",
            "call_record_id": str(call.id),
            "source_call_id": _clean_text(call.source_call_id),
            "source_file": _clean_text(call.source_file),
            "source_filename": _clean_text(call.source_filename),
            "started_at": _iso_datetime(call.started_at),
            "duration_sec": float(call.duration_sec) if call.duration_sec is not None else None,
            "direction": _clean_text(call.direction),
            "manager_name": _clean_text(call.manager_name),
            "phone": _clean_text(call.phone),
        },
        "processing": {
            "transcription_status": _clean_text(call.transcription_status),
            "resolve_status": _clean_text(call.resolve_status),
            "analysis_status": _clean_text(call.analysis_status),
            "resolve_quality_score": call.resolve_quality_score,
        },
        "identity_hints": {
            "phone": _clean_text(call.phone) or _clean_text(contacts.get("phone_from_filename")),
            "parent_fio": _clean_text(people.get("parent_fio")),
            "child_fio": _clean_text(people.get("child_fio")),
            "email": _clean_text(contacts.get("email")),
            "grade_current": _clean_text(student.get("grade_current")),
            "school": _clean_text(student.get("school")),
            "preferred_channel": _clean_text(contacts.get("preferred_channel")),
        },
        "call_summary": {
            "history_summary": history_summary,
            "history_short": _clean_text(normalized.get("history_short")) or _clean_text(normalized.get("summary")),
            "evidence": [
                {
                    "speaker": _clean_text(_as_dict(item).get("speaker")),
                    "ts": _clean_text(_as_dict(item).get("ts")),
                    "text": _clean_text(_as_dict(item).get("text")) or "",
                }
                for item in _as_list(normalized.get("evidence"))
                if _clean_text(_as_dict(item).get("text"))
            ],
        },
        "sales_insight": {
            "interests": {
                "products": _clean_list(interests.get("products")),
                "format": _clean_list(interests.get("format")),
                "subjects": _clean_list(interests.get("subjects")),
                "exam_targets": _clean_list(interests.get("exam_targets")),
            },
            "commercial": {
                "price_sensitivity": _clean_text(commercial.get("price_sensitivity")),
                "budget": _clean_text(commercial.get("budget")) or _clean_text(normalized.get("budget")),
                "discount_interest": commercial.get("discount_interest"),
            },
            "objections": _clean_list(blocks.get("objections")) or _clean_list(normalized.get("objections")),
            "next_step": {
                "action": _clean_text(next_step.get("action")) or _clean_text(normalized.get("next_step")),
                "due": _clean_text(next_step.get("due")) or _clean_text(normalized.get("timeline")),
            },
            "lead_priority": lead_priority,
            "follow_up_score": int(normalized.get("follow_up_score"))
            if normalized.get("follow_up_score") is not None
            else None,
            "follow_up_reason": _clean_text(normalized.get("follow_up_reason")),
            "personal_offer": _clean_text(normalized.get("personal_offer")),
            "pain_points": _clean_list(normalized.get("pain_points")),
            "tags": _clean_list(normalized.get("tags")),
        },
        "quality_flags": _as_dict(normalized.get("quality_flags")),
        "raw_analysis": normalized,
    }
    return payload


def build_call_insight_payload_for_record(call: CallRecord, settings: Settings) -> Dict[str, Any]:
    analysis = _parse_analysis(call, settings)
    return build_call_insight_payload(call, analysis)


def _post_call_insight(
    *,
    settings: Settings,
    project_id: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    api_key = _clean_text(settings.ai_office_api_key)
    if not api_key:
        raise AIOfficeExportError("AI_OFFICE_API_KEY is not configured.")

    url = _build_insights_url(settings.ai_office_api_base_url or "", project_id)
    try:
        response = requests.post(
            url,
            json=payload,
            headers={
                "Accept": "application/json",
                "X-API-Key": api_key,
            },
            timeout=max(1, int(settings.ai_office_timeout_sec)),
        )
    except requests.RequestException as exc:
        raise AIOfficeExportError(f"Failed to reach AI Office: {exc}") from exc

    try:
        data = response.json()
    except ValueError:
        data = {"detail": response.text.strip()}

    if response.status_code == 201:
        return {
            "status": "created",
            "http_status": response.status_code,
            "response": data,
        }
    if response.status_code == 409:
        return {
            "status": "duplicate",
            "http_status": response.status_code,
            "response": data,
        }
    raise AIOfficeExportError(
        f"AI Office returned HTTP {response.status_code}: {data.get('detail') or response.text.strip()}"
    )


def push_call_insights(
    session: Session,
    settings: Settings,
    *,
    project_id: str,
    ids: Optional[Iterable[int]] = None,
    limit: int = 100,
    only_done: bool = True,
    dry_run: bool = False,
) -> Dict[str, Any]:
    requested_ids = [int(item) for item in ids] if ids is not None else []
    query = select(CallRecord).where(CallRecord.analysis_json.is_not(None))
    if only_done:
        query = query.where(CallRecord.analysis_status == "done")

    if requested_ids:
        query = query.where(CallRecord.id.in_(requested_ids))
        calls = session.scalars(query.order_by(CallRecord.id.asc())).all()
        call_by_id = {int(call.id): call for call in calls}
        ordered_calls = [call_by_id[call_id] for call_id in requested_ids if call_id in call_by_id]
        missing_ids = [call_id for call_id in requested_ids if call_id not in call_by_id]
    else:
        ordered_calls = session.scalars(
            query.order_by(CallRecord.id.asc()).limit(max(0, int(limit)))
        ).all()
        missing_ids = []

    created = 0
    duplicates = 0
    failed = 0
    items: list[Dict[str, Any]] = []

    for call in ordered_calls:
        try:
            payload = build_call_insight_payload_for_record(call, settings)
            if dry_run:
                result = {"status": "dry_run", "http_status": None, "response": None}
            else:
                result = _post_call_insight(settings=settings, project_id=project_id, payload=payload)

            status_value = result["status"]
            if status_value == "created":
                created += 1
            elif status_value == "duplicate":
                duplicates += 1

            items.append(
                {
                    "call_id": int(call.id),
                    "source_call_id": _clean_text(call.source_call_id),
                    "source_filename": _clean_text(call.source_filename),
                    "status": status_value,
                    "http_status": result["http_status"],
                }
            )
        except Exception as exc:  # noqa: BLE001
            failed += 1
            items.append(
                {
                    "call_id": int(call.id),
                    "source_call_id": _clean_text(call.source_call_id),
                    "source_filename": _clean_text(call.source_filename),
                    "status": "failed",
                    "error": str(exc),
                }
            )

    return {
        "project_id": str(project_id),
        "selected": len(ordered_calls),
        "created": created,
        "duplicates": duplicates,
        "failed": failed,
        "missing_ids": missing_ids,
        "dry_run": bool(dry_run),
        "items": items,
    }
