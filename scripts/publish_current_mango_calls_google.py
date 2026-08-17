#!/usr/bin/env python3
"""Plan the safe current-day Google projection for Mango calls.

This module never writes Google. External publication belongs exclusively to
``publish_live_mango_calls_google.py``.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import sqlite3
import stat
import sys
import tempfile
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Optional, Sequence
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mango_mvp.productization.mango_calls_service_contract import (  # noqa: E402
    ControlledEnumerationBinding,
    has_dual_asr_or_exception,
    parse_aware_datetime,
    read_stable_regular_bytes,
    resolve_row_is_complete,
    sha256_file,
    validate_quarantine_items_payload,
    validate_ready_manifest_payload,
)
from mango_mvp.productization.owner_only_io import (  # noqa: E402
    CLOUD_PATH_MARKERS,
    path_has_cloud_marker,
    read_stable_regular_bytes_with_path,
)
from mango_mvp.productization.ready_publication import (  # noqa: E402
    ready_publication_lock,
    recover_ready_generation,
)
from mango_mvp.services.dialogue_contract import (  # noqa: E402
    call_record_view,
    guard_stored_analysis,
    json_object,
    manager_result_ru,
)
from mango_mvp.services.export_excel import call_to_row  # noqa: E402


MOSCOW = ZoneInfo("Europe/Moscow")
SCHEMA = "mango_calls_current_google_v1"
SAFE_PLAN_SCHEMA = "mango_calls_current_google_safe_plan_v2"
CONFIG_SCHEMA = "mango_calls_current_google_config_v1"
LEGACY_WRITE_MIGRATION = (
    "publish_current_mango_calls_google.py is plan-only; "
    "use publish_live_mango_calls_google.py for every Google write"
)
NEUTRAL_SUMMARY = (
    "Смысловой анализ не завершён; строка ожидает повторной обработки."
)
MISSING_SUMMARY = (
    "Смысловой анализ завершён, но краткое содержание отсутствует; нужна проверка."
)
OVERSIZED_SUMMARY = (
    "Краткое содержание превышает допустимый предел Analyze; нужна проверка."
)
MAX_MANAGER_VALUE = 2_000
MAX_ANALYZE_SUMMARY = 32_000
TRANSCRIPT_LINK_HEADER = "Путь/ссылка на полную расшифровку в закрытой папке"
HEADERS = (
    "call_key",
    "Дата и время",
    "Менеджер",
    "Направление",
    "Клиент",
    "Телефон",
    "Длительность",
    "Тип разговора",
    "Краткое содержание",
    "Результат",
    "Интерес клиента",
    "Главное возражение",
    "Следующий шаг",
    "Срок",
    TRANSCRIPT_LINK_HEADER,
    "Нужна проверка",
    "Причина проверки",
    "Комментарий РОПа",
    "Решение РОПа",
)
ROP_HEADERS = ("Комментарий РОПа", "Решение РОПа")
MANAGED_HEADERS = HEADERS[: -len(ROP_HEADERS)]
FORBIDDEN_LOCAL_PATH_MARKERS = CLOUD_PATH_MARKERS
FORBIDDEN_FIELD_RE = re.compile(
    r"(?:transcript|dialogue|расшифровк|sqlite|analysis_json|resolve_json|source_file)",
    re.IGNORECASE,
)
DRIVE_FILE_RE = re.compile(
    r"^https://drive\.google\.com/(?:file/d/|open\?id=)(?P<id>[A-Za-z0-9_-]+)"
)
CALL_TYPE_RU = {
    "sales_call": "Продажа / подбор обучения",
    "service_call": "Сервисный вопрос",
    "existing_client_progress": "Текущий клиент / продолжение",
    "technical_call": "Технический вопрос",
    "non_conversation": "Разговор не состоялся",
}
BLOCKING_PROCESSING_REASONS = (
    "Распознавание не завершено",
    "Вторая расшифровка GigaAM не готова",
    "Разделение ролей не завершено",
    "Смысловой анализ не завершён",
)


def clean_text(value: Any, *, maximum: int = MAX_MANAGER_VALUE) -> str:
    text = " ".join(str(value or "").split())
    if len(text) > maximum:
        raise RuntimeError("manager-facing value is unexpectedly long")
    if text.startswith(("=", "+", "-", "@")):
        text = "'" + text
    return text


def manager_summary(value: Any) -> tuple[str, str]:
    text = " ".join(str(value or "").split())
    if len(text) > MAX_ANALYZE_SUMMARY:
        return OVERSIZED_SUMMARY, "Краткое содержание превышает допустимый предел Analyze"
    return clean_text(text, maximum=MAX_ANALYZE_SUMMARY), ""


def exact_acl_ok(
    permissions: Any,
    *,
    owner_email: str,
    allowed_emails: Iterable[str],
    expected_roles: Optional[Mapping[str, str]] = None,
) -> bool:
    expected = {email.strip().casefold() for email in allowed_emails if email.strip()}
    owner = owner_email.strip().casefold()
    expected.add(owner)
    if not owner or not isinstance(permissions, Sequence) or isinstance(
        permissions, (str, bytes)
    ):
        return False
    actual: set[str] = set()
    owner_seen = False
    normalized_roles = {
        str(email).strip().casefold(): str(role).strip()
        for email, role in (expected_roles or {}).items()
        if str(email).strip()
    }
    for permission in permissions:
        if not isinstance(permission, Mapping) or permission.get("type") != "user":
            return False
        email = str(permission.get("emailAddress") or "").strip().casefold()
        role = str(permission.get("role") or "")
        if not email or role not in {"owner", "organizer", "writer", "reader"}:
            return False
        if normalized_roles and normalized_roles.get(email) != role:
            return False
        actual.add(email)
        owner_seen = owner_seen or (email == owner and role in {"owner", "organizer"})
    return (
        owner_seen
        and actual == expected
        and (not normalized_roles or set(normalized_roles) == expected)
    )


def validate_private_link(
    call_key: str,
    raw_link: Any,
    link_evidence: Mapping[str, Any],
    *,
    expected_emails: set[str],
) -> str:
    link = clean_text(raw_link, maximum=2_000)
    if not link:
        return ""
    if not DRIVE_FILE_RE.match(link):
        raise RuntimeError("transcript link must be a Google Drive file URL or empty")
    evidence = link_evidence.get(call_key)
    if not isinstance(evidence, Mapping) or evidence.get("acl_readback_ok") is not True:
        raise RuntimeError("transcript link has no private ACL readback")
    if str(evidence.get("url") or "") != link:
        raise RuntimeError("transcript link evidence URL mismatch")
    emails = {
        str(item).strip().casefold()
        for item in evidence.get("allowed_emails", ())
        if str(item).strip()
    }
    if emails != expected_emails:
        raise RuntimeError("transcript link ACL differs from spreadsheet ACL")
    return link


def _ready_rows(path: Path) -> list[Mapping[str, Any]]:
    before = path.stat()
    with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True, timeout=60) as con:
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA query_only=ON")
        if str(con.execute("PRAGMA integrity_check").fetchone()[0]) != "ok":
            raise RuntimeError("ready SQLite integrity check failed")
        rows = [dict(row) for row in con.execute("SELECT * FROM call_records")]
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise RuntimeError("ready SQLite changed while reading")
    return rows


def load_manager_rows(
    *,
    ready_db: Path,
    ready_manifest: Path,
    day: date,
    owner_email: str,
    allowed_emails: Iterable[str],
    link_evidence: Optional[Mapping[str, Any]] = None,
    ready_manifest_payload: Optional[Mapping[str, Any]] = None,
    controlled_binding: Optional[ControlledEnumerationBinding] = None,
) -> list[Mapping[str, Any]]:
    manifest = dict(
        ready_manifest_payload
        if ready_manifest_payload is not None
        else stable_json_object(ready_manifest, label="ready manifest")
    )
    errors = validate_ready_manifest_payload(
        manifest,
        require_closure=False,
        required_day=day,
        controlled_binding=controlled_binding,
    )
    if errors:
        raise RuntimeError("ready manifest rejected: " + ",".join(errors))
    before = ready_db.stat()
    if sha256_file(ready_db) != manifest.get("sha256") or before.st_size != manifest.get(
        "size_bytes"
    ):
        raise RuntimeError("ready SQLite does not match manifest")
    rows = _ready_rows(ready_db)
    if ready_db.stat().st_mtime_ns != before.st_mtime_ns:
        raise RuntimeError("ready SQLite changed after validation")
    expected_emails = {
        owner_email.strip().casefold(),
        *(email.strip().casefold() for email in allowed_emails if email.strip()),
    }
    links = link_evidence or {}
    verdicts = manifest.get("daily_verdicts")
    day_verdict = (
        verdicts.get(day.isoformat()) if isinstance(verdicts, Mapping) else None
    )
    quarantine_items = (
        list(day_verdict.get("quarantine_items") or ())
        if isinstance(day_verdict, Mapping)
        else []
    )
    quarantine_keys = {
        str(item.get("call_key") or "")
        for item in quarantine_items
        if isinstance(item, Mapping)
    }
    result: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        try:
            started = parse_aware_datetime(row.get("started_at"))
        except (TypeError, ValueError):
            continue
        if started.astimezone(MOSCOW).date() != day:
            continue
        call_key = clean_text(row.get("source_call_id") or row.get("id"), maximum=256)
        if not call_key or call_key in seen:
            raise RuntimeError("ready generation has duplicate or empty call_key")
        seen.add(call_key)
        if call_key in quarantine_keys:
            continue
        stored_analysis = json_object(row.get("analysis_json"))
        analysis_complete = row.get("analysis_status") == "done" and bool(stored_analysis)
        # This script only plans and prepares, but its plan is still read by a
        # human and copied into Google.  Almost every stored payload here
        # predates the role guard, so it goes through exactly the same
        # fail-closed projector as Analyse, Excel, AI Office and the live
        # publisher: re-reading an old payload is not a reason to believe it.
        analysis = (
            guard_stored_analysis(call_record_view(row), stored_analysis)
            if analysis_complete
            else {}
        )
        business_analysis = analysis
        projection_record = dict(row)
        projection_record["started_at"] = started
        projection_record.setdefault("source_filename", "")
        normalized = (
            call_to_row(
                SimpleNamespace(**projection_record),
                dict(analysis),
            )
            if analysis_complete
            else {}
        )
        summary_issue = ""
        if analysis_complete:
            summary, summary_issue = manager_summary(
                normalized.get("history_summary")
            )
            summary = summary or MISSING_SUMMARY
        else:
            summary = NEUTRAL_SUMMARY
        issues: list[str] = []
        if row.get("transcription_status") != "done":
            issues.append("Распознавание не завершено")
        if not has_dual_asr_or_exception(row):
            issues.append("Вторая расшифровка GigaAM не готова")
        if not resolve_row_is_complete(row):
            issues.append("Разделение ролей не завершено")
        if row.get("analysis_status") != "done" or not analysis:
            issues.append("Смысловой анализ не завершён")
        elif summary == MISSING_SUMMARY:
            issues.append("Краткое содержание отсутствует")
        if summary_issue:
            issues.append(summary_issue)
        direction_value = str(row.get("direction") or "")
        if direction_value == "inbound":
            direction = "Входящий"
        elif direction_value in {"outbound", "outgoing"}:
            direction = "Исходящий"
        else:
            direction = "Не определено"
            issues.append("Направление не определено")
        quality = business_analysis.get("quality_flags")
        if isinstance(quality, Mapping) and quality.get(
            "transcript_quality_requires_manual_review"
        ):
            issues.append("Качество расшифровки требует проверки")
        if analysis_complete and normalized.get("needs_review"):
            issues.append(
                clean_text(normalized.get("review_reasons"))
                or "Смысловой анализ запросил ручную проверку"
            )
        explicit_result = manager_result_ru(business_analysis) if analysis_complete else ""
        if explicit_result == "—":
            explicit_result = ""
        if analysis_complete and not explicit_result:
            if clean_text(normalized.get("call_type")) == "non_conversation":
                explicit_result = "Разговор не состоялся"
            elif clean_text(normalized.get("next_step_action")):
                explicit_result = "Следующий шаг выделен анализом"
            else:
                explicit_result = "Итог не зафиксирован"
                issues.append("Итог разговора требует ручной проверки")
        link = validate_private_link(
            call_key,
            (links.get(call_key) or {}).get("url")
            if isinstance(links.get(call_key), Mapping)
            else "",
            links,
            expected_emails=expected_emails,
        )
        manager_row = {
            "call_key": call_key,
            "Дата и время": started.astimezone(MOSCOW).strftime("%d.%m.%Y %H:%M:%S"),
            "Менеджер": clean_text(row.get("manager_name")) or "Не определён",
            "Направление": direction,
            "Клиент": "ФИО не подтверждено",
            "Телефон": clean_text(row.get("phone")),
            "Длительность": round(float(row.get("duration_sec") or 0), 1),
            "Тип разговора": CALL_TYPE_RU.get(
                clean_text(normalized.get("call_type")),
                clean_text(normalized.get("call_type")),
            ),
            "Краткое содержание": summary,
            "Результат": explicit_result,
            "Интерес клиента": clean_text(
                normalized.get("interests_products")
                or normalized.get("recommended_product")
            ),
            "Главное возражение": clean_text(normalized.get("objections")),
            "Следующий шаг": clean_text(normalized.get("next_step_action")),
            "Срок": clean_text(normalized.get("next_step_due_raw")),
            TRANSCRIPT_LINK_HEADER: link,
            "Нужна проверка": "Да" if issues else "Нет",
            "Причина проверки": "; ".join(dict.fromkeys(issues)),
        }
        if set(manager_row) != set(MANAGED_HEADERS):
            raise RuntimeError("manager projection schema mismatch")
        serialized = json.dumps(manager_row, ensure_ascii=False, default=str)
        if FORBIDDEN_FIELD_RE.search(serialized):
            # The approved link header is the only occurrence of the Russian
            # word; inspect keys separately and allow that one explicit field.
            forbidden_keys = [
                key
                for key in manager_row
                if key != TRANSCRIPT_LINK_HEADER
                and FORBIDDEN_FIELD_RE.search(key)
            ]
            if forbidden_keys or any(
                token in serialized.casefold()
                for token in ("analysis_json", "resolve_json", ".sqlite", "/users/")
            ):
                raise RuntimeError("manager projection contains forbidden diagnostics")
        result.append(manager_row)
    result.sort(key=lambda item: (str(item["Дата и время"]), str(item["call_key"])))
    return result


def _safe_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def validate_safe_rows(rows: Any) -> list[Mapping[str, Any]]:
    if not isinstance(rows, list):
        raise RuntimeError("safe Google plan rows must be a list")
    validated: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for raw in rows:
        if not isinstance(raw, Mapping) or set(raw) != set(MANAGED_HEADERS):
            raise RuntimeError("safe Google plan row schema mismatch")
        key = str(raw.get("call_key") or "").strip()
        if not key or len(key) > 256 or key in seen:
            raise RuntimeError("safe Google plan has duplicate or invalid call_key")
        seen.add(key)
        normalized: dict[str, Any] = {}
        for header in MANAGED_HEADERS:
            value = raw.get(header, "")
            if isinstance(value, (dict, list, tuple, set)):
                raise RuntimeError("safe Google plan contains a structured diagnostic value")
            if header == "Длительность":
                try:
                    numeric = float(value or 0)
                except (TypeError, ValueError) as exc:
                    raise RuntimeError("safe Google plan duration is invalid") from exc
                if numeric < 0 or numeric > 86_400:
                    raise RuntimeError("safe Google plan duration is invalid")
                normalized[header] = numeric
                continue
            text = str(value or "")
            maximum = (
                MAX_ANALYZE_SUMMARY
                if header == "Краткое содержание"
                else MAX_MANAGER_VALUE
            )
            if len(text) > maximum:
                raise RuntimeError("safe Google plan contains an oversized value")
            if text.startswith(("=", "+", "-", "@")):
                text = "'" + text
            normalized[header] = text
        link = str(normalized[TRANSCRIPT_LINK_HEADER])
        if link and not re.fullmatch(
            r"https://drive\.google\.com/(?:file/d/|open\?id=)[A-Za-z0-9_-]+(?:/view)?(?:\?.*)?",
            link,
        ):
            raise RuntimeError("safe Google plan contains an unapproved transcript link")
        serialized = json.dumps(normalized, ensure_ascii=False, sort_keys=True)
        if any(
            token in serialized.casefold()
            for token in (
                "analysis_json",
                "resolve_json",
                "transcript_text",
                "dialogue_lines",
                ".sqlite",
                ".mp3",
                "/users/",
                "prompt_version",
                "model_revision",
            )
        ):
            raise RuntimeError("safe Google plan contains forbidden diagnostics")
        validated.append(normalized)
    return validated


def processing_ready_row(row: Mapping[str, Any]) -> bool:
    reasons = str(row.get("Причина проверки") or "")
    return not any(reason in reasons for reason in BLOCKING_PROCESSING_REASONS)


def build_safe_plan(
    *,
    day: date,
    rows: Sequence[Mapping[str, Any]],
    ready_manifest: Mapping[str, Any],
    now: Optional[datetime] = None,
    controlled_binding: Optional[ControlledEnumerationBinding] = None,
) -> Mapping[str, Any]:
    errors = validate_ready_manifest_payload(
        ready_manifest,
        require_closure=False,
        required_day=day,
        controlled_binding=controlled_binding,
    )
    if errors:
        raise RuntimeError("ready manifest rejected: " + ",".join(errors))
    safe_rows = validate_safe_rows(list(rows))
    verdicts = ready_manifest.get("daily_verdicts")
    verdict = verdicts.get(day.isoformat()) if isinstance(verdicts, Mapping) else None
    if not isinstance(verdict, Mapping) or verdict.get("consistency_ok") is not True:
        raise RuntimeError("ready manifest has no green Stage10 verdict for requested day")
    pending_unique = int(verdict.get("pending_unique") or 0)
    generated = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    stage10_counts = {
        field: int(verdict.get(field) or 0)
        for field in (
            "mango_unique",
            "ready_unique",
            "quarantine_unique",
            "pending_unique",
            "unexplained_missing",
            "quarantine_without_reason",
        )
    }
    processing_ready_unique = sum(processing_ready_row(row) for row in safe_rows)
    row_completion_ok = processing_ready_unique == len(safe_rows)
    stage10_counts["processing_ready_unique"] = processing_ready_unique
    if not (
        stage10_counts["ready_unique"]
        <= len(safe_rows)
        <= stage10_counts["mango_unique"] - stage10_counts["quarantine_unique"]
    ):
        raise RuntimeError("manager rows do not match target-day Stage10 balance")
    quarantine_items = list(verdict.get("quarantine_items") or ())
    quarantine_errors = validate_quarantine_items_payload(
        quarantine_items,
        day=day,
        expected_count=stage10_counts["quarantine_unique"],
        expected_without_reason=stage10_counts["quarantine_without_reason"],
    )
    if quarantine_errors:
        raise RuntimeError("ready manifest quarantine details are invalid")
    if {str(item["call_key"]) for item in quarantine_items} & {
        str(row["call_key"]) for row in safe_rows
    }:
        raise RuntimeError("ready and quarantine manager rows overlap")
    return {
        "schema_version": SAFE_PLAN_SCHEMA,
        "moscow_day": day.isoformat(),
        "generated_at_utc": generated.isoformat(),
        "expires_at_utc": (generated + timedelta(minutes=60)).isoformat(),
        "source_ready_sha256": str(ready_manifest.get("sha256") or ""),
        "stage10_sha256": _safe_json_sha256(verdict),
        "consistency_ok": True,
        "closure_ok": verdict.get("closure_ok") is True and row_completion_ok,
        "row_completion_ok": row_completion_ok,
        "pending_unique": pending_unique,
        "stage10_counts": stage10_counts,
        "rows": safe_rows,
        "quarantine_items": quarantine_items,
    }


def validate_safe_plan_payload(
    payload: Any, *, expected_day: date, now: Optional[datetime] = None
) -> list[Mapping[str, Any]]:
    if not isinstance(payload, Mapping) or payload.get("schema_version") != SAFE_PLAN_SCHEMA:
        raise RuntimeError("safe Google plan schema is invalid")
    if payload.get("moscow_day") != expected_day.isoformat():
        raise RuntimeError("safe Google plan is for another Moscow day")
    try:
        generated = parse_aware_datetime(payload.get("generated_at_utc"))
        expires = parse_aware_datetime(payload.get("expires_at_utc"))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("safe Google plan timestamp is invalid") from exc
    if generated.astimezone(MOSCOW).date() != expected_day:
        raise RuntimeError("safe Google plan timestamp is outside its Moscow day")
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    if expires <= generated or expires - generated > timedelta(minutes=60):
        raise RuntimeError("safe Google plan expiry window is invalid")
    if current > expires:
        raise RuntimeError("safe Google plan has expired")
    if payload.get("consistency_ok") is not True:
        raise RuntimeError("safe Google plan consistency gate is not green")
    for field in ("source_ready_sha256", "stage10_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(payload.get(field) or "")):
            raise RuntimeError(f"safe Google plan {field} is invalid")
    try:
        pending = int(payload.get("pending_unique") or 0)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("safe Google plan pending count is invalid") from exc
    if pending < 0 or (payload.get("closure_ok") is True and pending != 0):
        raise RuntimeError("safe Google plan closure/pending state is inconsistent")
    counts = payload.get("stage10_counts")
    if not isinstance(counts, Mapping):
        raise RuntimeError("safe Google plan Stage10 counts are missing")
    count_fields = (
        "mango_unique",
        "ready_unique",
        "quarantine_unique",
        "pending_unique",
        "unexplained_missing",
        "quarantine_without_reason",
    )
    try:
        normalized_counts = {field: int(counts[field]) for field in count_fields}
        processing_ready_unique = int(counts["processing_ready_unique"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("safe Google plan Stage10 counts are invalid") from exc
    if (
        any(value < 0 for value in normalized_counts.values())
        or normalized_counts["pending_unique"] != pending
        or normalized_counts["quarantine_without_reason"] != 0
        or normalized_counts["mango_unique"]
        != sum(
            normalized_counts[field]
            for field in (
                "ready_unique",
                "quarantine_unique",
                "pending_unique",
                "unexplained_missing",
            )
        )
    ):
        raise RuntimeError("safe Google plan Stage10 balance is invalid")
    safe_rows = validate_safe_rows(payload.get("rows"))
    derived_processing_ready = sum(processing_ready_row(row) for row in safe_rows)
    expected_row_completion = processing_ready_unique == len(safe_rows)
    if (
        processing_ready_unique != derived_processing_ready
        or processing_ready_unique < 0
        or processing_ready_unique > len(safe_rows)
        or payload.get("row_completion_ok") is not expected_row_completion
        or (
            payload.get("closure_ok") is True
            and processing_ready_unique != len(safe_rows)
        )
    ):
        raise RuntimeError("safe Google plan row completion state is inconsistent")
    quarantine_items = payload.get("quarantine_items")
    quarantine_errors = validate_quarantine_items_payload(
        quarantine_items,
        day=expected_day,
        expected_count=normalized_counts["quarantine_unique"],
        expected_without_reason=normalized_counts["quarantine_without_reason"],
    )
    if quarantine_errors:
        raise RuntimeError("safe Google plan quarantine details are invalid")
    if not (
        normalized_counts["ready_unique"]
        <= len(safe_rows)
        <= normalized_counts["mango_unique"]
        - normalized_counts["quarantine_unique"]
    ):
        raise RuntimeError("safe Google plan rows do not match Stage10 balance")
    if {str(item["call_key"]) for item in quarantine_items} & {
        str(row["call_key"]) for row in safe_rows
    }:
        raise RuntimeError("safe Google plan ready/quarantine rows overlap")
    return safe_rows


class GoogleGateway:
    """Read-only Google metadata used by the single live publisher subclass."""

    def __init__(self, session: Any, spreadsheet_id: str) -> None:
        self.session = session
        self.spreadsheet_id = spreadsheet_id

    @staticmethod
    def _json(response: Any) -> Mapping[str, Any]:
        if not 200 <= int(response.status_code) < 300:
            raise RuntimeError(f"Google HTTP {response.status_code}")
        payload = response.json()
        if not isinstance(payload, Mapping):
            raise RuntimeError("Google returned invalid JSON")
        return payload

    def _permissions_for(self, file_id: str) -> Sequence[Mapping[str, Any]]:
        result: list[Mapping[str, Any]] = []
        page_token = ""
        seen_tokens: set[str] = set()
        while True:
            params = {
                "fields": (
                    "nextPageToken,permissions("
                    "id,type,role,emailAddress,allowFileDiscovery,permissionDetails)"
                ),
                "pageSize": 100,
                "supportsAllDrives": "true",
            }
            if page_token:
                params["pageToken"] = page_token
            response = self.session.get(
                f"https://www.googleapis.com/drive/v3/files/{file_id}/permissions",
                params=params,
                timeout=60,
            )
            payload = self._json(response)
            permissions = payload.get("permissions")
            if not isinstance(permissions, list):
                raise RuntimeError("Google permissions pagination is incomplete")
            result.extend(item for item in permissions if isinstance(item, Mapping))
            next_token = str(payload.get("nextPageToken") or "")
            if not next_token:
                return result
            if next_token in seen_tokens:
                raise RuntimeError("Google permissions pagination loop")
            seen_tokens.add(next_token)
            page_token = next_token

    def permissions(self) -> Sequence[Mapping[str, Any]]:
        return self._permissions_for(self.spreadsheet_id)

    def sheets(self) -> Sequence[Mapping[str, Any]]:
        response = self.session.get(
            f"https://sheets.googleapis.com/v4/spreadsheets/{self.spreadsheet_id}",
            params={"fields": "sheets.properties(sheetId,title)"},
            timeout=60,
        )
        return [
            item.get("properties") or {}
            for item in self._json(response).get("sheets") or ()
            if isinstance(item, Mapping)
        ]



def publish_current(*_args: Any, **_kwargs: Any) -> Mapping[str, Any]:
    """Refuse the retired preliminary-sheet writer before touching its gateway."""
    raise RuntimeError(LEGACY_WRITE_MIGRATION)


def owner_json(path: Optional[Path]) -> Mapping[str, Any]:
    if path is None:
        return {}
    assert_private_local_path(path, label="evidence/state JSON")
    try:
        raw = read_stable_regular_bytes(
            path, label="evidence_state_json", owner_only_mode=0o600
        )
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("evidence/state JSON is invalid") from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError("evidence/state JSON must be an object")
    return payload


def stable_json_object(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        raw = read_stable_regular_bytes(path, label=label.replace(" ", "_"))
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} is invalid") from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"{label} must be an object")
    return payload


def assert_private_local_path(path: Path, *, label: str) -> None:
    resolved = path.expanduser().resolve(strict=False)
    lowered = str(resolved).casefold()
    if any(marker in lowered for marker in FORBIDDEN_LOCAL_PATH_MARKERS):
        raise RuntimeError(f"{label} must stay outside cloud-synced folders")
    try:
        resolved.relative_to(ROOT)
    except ValueError:
        return
    raise RuntimeError(f"{label} must stay outside the repository")


def atomic_owner_json(path: Path, payload: Mapping[str, Any]) -> None:
    assert_private_local_path(path, label="owner JSON")
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    if stat.S_IMODE(path.parent.stat().st_mode) != 0o700:
        raise RuntimeError("owner JSON parent must be owner-only 0700")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        os.chmod(path, 0o600)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def publication_lock(path: Path) -> Iterable[None]:
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    parent = os.lstat(path.parent)
    if (
        not stat.S_ISDIR(parent.st_mode)
        or stat.S_ISLNK(parent.st_mode)
        or parent.st_uid != os.getuid()
        or stat.S_IMODE(parent.st_mode) != 0o700
    ):
        raise RuntimeError("Google publication lock directory is unsafe")
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise RuntimeError("Google publication lock is unsafe") from exc
    with os.fdopen(descriptor, "a+", encoding="utf-8") as handle:
        opened = os.fstat(handle.fileno())
        current = os.lstat(path)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_uid != os.getuid()
            or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino)
        ):
            raise RuntimeError("Google publication lock is unsafe")
        if stat.S_IMODE(opened.st_mode) != 0o600:
            os.fchmod(handle.fileno(), 0o600)
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another Google current publisher is active") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def validate_credentials(path: Path) -> Mapping[str, Any]:
    candidate = path.expanduser().absolute()
    try:
        raw, resolved = read_stable_regular_bytes_with_path(
            candidate,
            label="google_credentials",
            owner_only_mode=0o600,
        )
    except RuntimeError as exc:
        raise RuntimeError("Google credentials must be owner-only 0600") from exc
    if resolved == ROOT or ROOT in resolved.parents or path_has_cloud_marker(resolved):
        raise RuntimeError("Google credentials must stay outside repository and cloud folders")
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, Mapping) or not str(payload.get("client_email") or "").strip():
        raise RuntimeError("Google credentials are invalid")
    return payload


def load_google_config(path: Path) -> Mapping[str, Any]:
    payload = owner_json(path)
    if payload.get("schema_version") != CONFIG_SCHEMA:
        raise RuntimeError("Google current config schema is invalid")
    required = ("spreadsheet_id", "owner_email", "service_account_email", "credentials")
    if any(not str(payload.get(field) or "").strip() for field in required):
        raise RuntimeError("Google current config is incomplete")
    raw_permissions = payload.get("allowed_permissions")
    if not isinstance(raw_permissions, list) or not raw_permissions:
        raise RuntimeError("Google current config permissions are missing")
    roles: dict[str, str] = {}
    for item in raw_permissions:
        if not isinstance(item, Mapping):
            raise RuntimeError("Google current config permission is invalid")
        email = str(item.get("email") or "").strip().casefold()
        role = str(item.get("role") or "").strip()
        if not email or role not in {"owner", "writer", "reader"} or email in roles:
            raise RuntimeError("Google current config permission is invalid")
        roles[email] = role
    owner = str(payload["owner_email"]).strip().casefold()
    service = str(payload["service_account_email"]).strip().casefold()
    if roles.get(owner) != "owner" or roles.get(service) != "writer" or owner == service:
        raise RuntimeError("Google current config owner/service roles are invalid")
    pilot = date.fromisoformat(str(payload.get("pilot_started_day") or ""))
    credentials = Path(str(payload["credentials"])).expanduser()
    credentials_payload = validate_credentials(credentials)
    if str(credentials_payload.get("client_email") or "").casefold() != service:
        raise RuntimeError("Google service account does not match owner config")
    return {
        **dict(payload),
        "owner_email": owner,
        "service_account_email": service,
        "expected_roles": roles,
        "allowed_emails": tuple(sorted(set(roles) - {owner})),
        "pilot_started_day": pilot,
        "credentials": credentials,
        "credentials_info": dict(credentials_payload),
    }


def authorized_session(credentials_info: Any) -> Any:
    from google.auth.transport.requests import AuthorizedSession
    from google.oauth2.service_account import Credentials

    if isinstance(credentials_info, Path):
        credentials_info = validate_credentials(credentials_info)
    if not isinstance(credentials_info, Mapping):
        raise RuntimeError("Google credentials are invalid")
    scopes = (
        "https://www.googleapis.com/auth/drive.metadata.readonly",
        "https://www.googleapis.com/auth/spreadsheets",
    )
    return AuthorizedSession(
        Credentials.from_service_account_info(dict(credentials_info), scopes=scopes)
    )



def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a safe preliminary Mango calls Google plan."
    )
    parser.add_argument("--ready-db", type=Path)
    parser.add_argument("--ready-manifest", type=Path)
    parser.add_argument("--day", type=date.fromisoformat, default=datetime.now(MOSCOW).date())
    parser.add_argument("--owner-email")
    parser.add_argument("--allowed-email", action="append", default=[])
    parser.add_argument("--link-evidence", type=Path)
    parser.add_argument("--pilot-start-day", type=date.fromisoformat)
    parser.add_argument("--retention-approved", action="store_true")
    parser.add_argument("--plan-out", type=Path)
    parser.add_argument("--safe-plan", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--state", type=Path)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Retired legacy option; always refuses with migration guidance.",
    )
    parser.add_argument("--confirmation", default="")
    parser.add_argument(
        "--approved-plan-sha256",
        help="Owner-approved SHA-256 of the exact short-lived safe plan.",
    )
    args = parser.parse_args(argv)
    if args.execute:
        raise RuntimeError(LEGACY_WRITE_MIGRATION)
    if not args.execute:
        if not args.ready_db or not args.owner_email:
            raise RuntimeError("dry-run requires ready DB and owner email")
        manifest_path = args.ready_manifest or args.ready_db.with_suffix(
            ".manifest.json"
        )
        with ready_publication_lock(args.ready_db):
            recover_ready_generation(args.ready_db, lock_held=True)
            manifest_payload = stable_json_object(
                manifest_path, label="ready manifest"
            )
            links = owner_json(args.link_evidence)
            rows = load_manager_rows(
                ready_db=args.ready_db,
                ready_manifest=manifest_path,
                day=args.day,
                owner_email=args.owner_email,
                allowed_emails=args.allowed_email,
                link_evidence=links,
                ready_manifest_payload=manifest_payload,
            )
            safe_plan = build_safe_plan(
                day=args.day,
                rows=rows,
                ready_manifest=manifest_payload,
            )
        if args.plan_out:
            atomic_owner_json(args.plan_out, safe_plan)
        report = {
            "schema_version": SCHEMA,
            "status": "dry_run",
            "day": args.day.isoformat(),
            "rows": len(rows),
            "quarantine_rows": len(safe_plan["quarantine_items"]),
            "safe_plan_sha256": _safe_json_sha256(safe_plan),
            "plan_written": bool(args.plan_out),
            "full_transcript_fields_written": 0,
            "external_write": False,
        }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
