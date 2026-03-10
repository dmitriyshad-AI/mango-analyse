from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from mango_mvp.clients.amocrm import AmoCRMClient
from mango_mvp.config import Settings
from mango_mvp.models import CallRecord


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _clean_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    result: List[str] = []
    seen: set[str] = set()
    for item in value:
        text = _clean_text(item)
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def _get_block(analysis: Dict[str, Any], *path: str) -> Any:
    current: Any = analysis
    for idx, key in enumerate(path):
        if not isinstance(current, dict):
            return None
        if (
            idx == 0
            and key == "crm_blocks"
            and key not in current
            and "structured_fields" in current
        ):
            key = "structured_fields"
        current = current.get(key)
    return current


def _analysis_interests(analysis: Dict[str, Any]) -> List[str]:
    legacy = _clean_list(analysis.get("interests"))
    products = _clean_list(_get_block(analysis, "crm_blocks", "interests", "products"))
    formats = _clean_list(_get_block(analysis, "crm_blocks", "interests", "format"))
    subjects = _clean_list(_get_block(analysis, "crm_blocks", "interests", "subjects"))
    exams = _clean_list(_get_block(analysis, "crm_blocks", "interests", "exam_targets"))
    return _clean_list(legacy + products + formats + subjects + exams)


def _analysis_student_grade(analysis: Dict[str, Any]) -> str | None:
    return _clean_text(analysis.get("student_grade")) or _clean_text(
        _get_block(analysis, "crm_blocks", "student", "grade_current")
    )


def _analysis_target_product(analysis: Dict[str, Any]) -> str | None:
    target = _clean_text(analysis.get("target_product"))
    if target:
        return target
    products = _clean_list(_get_block(analysis, "crm_blocks", "interests", "products"))
    return products[0] if products else None


def _analysis_budget(analysis: Dict[str, Any]) -> str | None:
    return _clean_text(analysis.get("budget")) or _clean_text(
        _get_block(analysis, "crm_blocks", "commercial", "budget")
    )


def _analysis_timeline(analysis: Dict[str, Any]) -> str | None:
    return _clean_text(analysis.get("timeline")) or _clean_text(
        _get_block(analysis, "crm_blocks", "next_step", "due")
    )


def _analysis_next_step(analysis: Dict[str, Any]) -> str | None:
    return _clean_text(analysis.get("next_step")) or _clean_text(
        _get_block(analysis, "crm_blocks", "next_step", "action")
    )


def _analysis_history(analysis: Dict[str, Any]) -> str:
    return (
        _clean_text(analysis.get("history_summary"))
        or _clean_text(analysis.get("history_short"))
        or _clean_text(analysis.get("summary"))
        or ""
    )


def _build_custom_fields(settings: Settings, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    fields: List[Dict[str, Any]] = []
    interests = _analysis_interests(analysis)
    student_grade = _analysis_student_grade(analysis)
    target_product = _analysis_target_product(analysis)
    budget = _analysis_budget(analysis)
    timeline = _analysis_timeline(analysis)
    next_step = _analysis_next_step(analysis)
    mapping = [
        (settings.amocrm_interests_field_id, ", ".join(interests)),
        (settings.amocrm_student_grade_field_id, student_grade),
        (settings.amocrm_target_product_field_id, target_product),
        (settings.amocrm_personal_offer_field_id, analysis.get("personal_offer")),
        (settings.amocrm_budget_field_id, budget),
        (settings.amocrm_timeline_field_id, timeline),
        (settings.amocrm_next_step_field_id, next_step),
        (
            settings.amocrm_followup_score_field_id,
            str(analysis.get("follow_up_score"))
            if analysis.get("follow_up_score") is not None
            else None,
        ),
    ]
    for field_id, value in mapping:
        if not field_id or value is None or value == "":
            continue
        fields.append({"field_id": field_id, "values": [{"value": value}]})
    return fields


def _build_note_text(call: CallRecord, analysis: Dict[str, Any]) -> str:
    interests = _analysis_interests(analysis)
    student_grade = _analysis_student_grade(analysis) or ""
    target_product = _analysis_target_product(analysis) or ""
    budget = _analysis_budget(analysis) or ""
    timeline = _analysis_timeline(analysis) or ""
    next_step = _analysis_next_step(analysis) or ""
    history_short = _analysis_history(analysis)
    subjects = _clean_list(_get_block(analysis, "crm_blocks", "interests", "subjects"))
    formats = _clean_list(_get_block(analysis, "crm_blocks", "interests", "format"))
    objections = _clean_list(_get_block(analysis, "crm_blocks", "objections")) or _clean_list(
        analysis.get("objections")
    )
    lead_priority = _clean_text(_get_block(analysis, "crm_blocks", "lead_priority")) or ""

    lines = [
        f"Call file: {call.source_filename}",
        f"Phone: {call.phone or 'unknown'}",
        f"Manager: {call.manager_name or 'unknown'}",
        "",
        f"Summary: {history_short}",
        f"Interests: {', '.join(interests)}",
        f"Subjects: {', '.join(subjects)}",
        f"Formats: {', '.join(formats)}",
        f"Student grade: {student_grade}",
        f"Target product: {target_product}",
        f"Personal offer: {analysis.get('personal_offer') or ''}",
        f"Pain points: {', '.join(analysis.get('pain_points') or [])}",
        f"Objections: {', '.join(objections)}",
        f"Budget: {budget}",
        f"Timeline: {timeline}",
        f"Next step: {next_step}",
        f"Lead priority: {lead_priority}",
        f"Follow-up score: {analysis.get('follow_up_score')}",
        f"Reason: {analysis.get('follow_up_reason') or ''}",
    ]
    return "\n".join(lines).strip()


class AmoCRMSyncService:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._client = AmoCRMClient(settings) if not settings.sync_dry_run else None

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(timezone.utc)

    def _retry_delay(self, attempts: int) -> timedelta:
        base = max(1, self._settings.retry_base_delay_sec)
        multiplier = max(1, 2 ** max(0, attempts - 1))
        return timedelta(seconds=base * multiplier)

    def run(self, session: Session, limit: int) -> Dict[str, int]:
        now = self._utc_now()
        max_attempts = max(1, self._settings.sync_max_attempts)
        calls = session.scalars(
            select(CallRecord)
            .where(CallRecord.analysis_status == "done")
            .where(CallRecord.dead_letter_stage.is_(None))
            .where(CallRecord.sync_status.in_(["pending", "failed"]))
            .where(CallRecord.sync_attempts < max_attempts)
            .where(or_(CallRecord.next_retry_at.is_(None), CallRecord.next_retry_at <= now))
            .order_by(CallRecord.id.asc())
            .limit(limit)
        ).all()
        success = 0
        failed = 0
        skipped = 0

        for call in calls:
            call.sync_attempts = int(call.sync_attempts or 0) + 1
            attempt = call.sync_attempts
            try:
                if not call.analysis_json:
                    raise RuntimeError("analysis_json is empty")
                analysis: Dict[str, Any] = json.loads(call.analysis_json)
                if self._settings.sync_dry_run:
                    call.sync_status = "done"
                    call.next_retry_at = None
                    call.dead_letter_stage = None
                    call.last_error = "dry_run"
                    success += 1
                    session.add(call)
                    continue

                if not call.phone:
                    raise RuntimeError("no phone in call record")

                assert self._client is not None
                contact = self._client.find_contact_by_phone(call.phone)
                if not contact:
                    raise RuntimeError(f"contact not found for {call.phone}")

                contact_id = int(contact["id"])
                note_text = _build_note_text(call, analysis)
                self._client.add_contact_note(contact_id, note_text)
                fields = _build_custom_fields(self._settings, analysis)
                self._client.update_contact_fields(contact_id, fields)

                score = int(analysis.get("follow_up_score") or 0)
                next_step = _analysis_next_step(analysis) or "Follow up"
                if (
                    score >= self._settings.follow_up_task_threshold
                    and self._settings.amocrm_task_type_id
                ):
                    due = int((datetime.now(timezone.utc) + timedelta(days=1)).timestamp())
                    self._client.create_task(
                        contact_id=contact_id,
                        text=f"[AI] {next_step}",
                        complete_till_unix=due,
                        task_type_id=self._settings.amocrm_task_type_id,
                        responsible_user_id=self._settings.amocrm_task_responsible_user_id,
                    )

                call.amocrm_contact_id = contact_id
                call.sync_status = "done"
                call.next_retry_at = None
                call.dead_letter_stage = None
                call.last_error = None
                success += 1
            except Exception as exc:  # noqa: BLE001
                call.last_error = f"sync: {exc}"
                if attempt >= max_attempts:
                    call.sync_status = "dead"
                    call.dead_letter_stage = "sync"
                    call.next_retry_at = None
                else:
                    call.sync_status = "failed"
                    call.next_retry_at = self._utc_now() + self._retry_delay(attempt)
                failed += 1
            session.add(call)

        session.commit()
        return {
            "processed": len(calls),
            "success": success,
            "failed": failed,
            "skipped": skipped,
        }
