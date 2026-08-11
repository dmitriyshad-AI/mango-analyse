#!/usr/bin/env python3
"""Safe current-day Google projection for Mango calls.

Dry-run is the default.  The module deliberately has no transcript column and
updates existing rows by header name plus call_key, preserving ROP-owned cells.
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
    parse_aware_datetime,
    read_stable_regular_bytes,
    sha256_file,
    validate_ready_manifest_payload,
)
from mango_mvp.services.export_excel import call_to_row  # noqa: E402


MOSCOW = ZoneInfo("Europe/Moscow")
SCHEMA = "mango_calls_current_google_v1"
SAFE_PLAN_SCHEMA = "mango_calls_current_google_safe_plan_v1"
CONFIG_SCHEMA = "mango_calls_current_google_config_v1"
CURRENT_TITLE = "Сегодня — предварительно"
REVIEW_TITLE = "Требует проверки"
SUMMARY_TITLE = "Сводка"
BANNER_PREFIX = "ПРЕДВАРИТЕЛЬНО, ДЕНЬ НЕ ЗАКРЫТ"
CONFIRMATION = "UPDATE_MANGO_CURRENT_SHEET"
NEUTRAL_SUMMARY = (
    "Смысловой анализ не завершён; строка ожидает повторной обработки."
)
MISSING_SUMMARY = (
    "Смысловой анализ завершён, но краткое содержание отсутствует; нужна проверка."
)
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
REVIEW_HEADERS = MANAGED_HEADERS
SUMMARY_HEADERS = ("Показатель", "Значение")
FORBIDDEN_LOCAL_PATH_MARKERS = (
    "yandex.disk",
    "yandexdisk",
    "cloudstorage",
    "mobile documents",
    "dropbox",
    "onedrive",
    "google drive",
)
FORBIDDEN_FIELD_RE = re.compile(
    r"(?:transcript|dialogue|расшифровк|sqlite|analysis_json|resolve_json|source_file)",
    re.IGNORECASE,
)
PUBLIC_LINK_RE = re.compile(r"^https?://", re.IGNORECASE)
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


def json_object(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    try:
        payload = json.loads(str(value or "{}"))
    except (TypeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, Mapping) else {}


def clean_text(value: Any, *, maximum: int = 2_000) -> str:
    text = " ".join(str(value or "").split())
    if len(text) > maximum:
        raise RuntimeError("manager-facing value is unexpectedly long")
    if text.startswith(("=", "+", "-", "@")):
        text = "'" + text
    return text


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
) -> list[Mapping[str, Any]]:
    manifest = dict(
        ready_manifest_payload
        if ready_manifest_payload is not None
        else stable_json_object(ready_manifest, label="ready manifest")
    )
    errors = validate_ready_manifest_payload(manifest, require_closure=False)
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
        analysis = json_object(row.get("analysis_json"))
        analysis_complete = row.get("analysis_status") == "done" and bool(analysis)
        business_analysis = analysis if analysis_complete else {}
        normalized = (
            call_to_row(
                SimpleNamespace(
                    id=row.get("id") or call_key,
                    started_at=started,
                    phone=row.get("phone"),
                    manager_name=row.get("manager_name"),
                    duration_sec=row.get("duration_sec"),
                    source_filename="",
                    source_file="",
                ),
                dict(business_analysis),
            )
            if analysis_complete
            else {}
        )
        summary = (
            clean_text(normalized.get("history_summary")) or MISSING_SUMMARY
            if analysis_complete
            else NEUTRAL_SUMMARY
        )
        issues: list[str] = []
        if row.get("transcription_status") != "done":
            issues.append("Распознавание не завершено")
        if row.get("resolve_status") not in {"done", "skipped"}:
            issues.append("Разделение ролей не завершено")
        if row.get("analysis_status") != "done" or not analysis:
            issues.append("Смысловой анализ не завершён")
        elif summary == MISSING_SUMMARY:
            issues.append("Краткое содержание отсутствует")
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
            issues.append("Смысловой анализ запросил ручную проверку")
        explicit_result = clean_text(
            business_analysis.get("result") or business_analysis.get("call_result")
        )
        if analysis_complete and not explicit_result:
            issues.append("Результат разговора не выделен текущей схемой Analyze")
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
            if len(text) > 2_000:
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


def build_safe_plan(
    *,
    day: date,
    rows: Sequence[Mapping[str, Any]],
    ready_manifest: Mapping[str, Any],
    now: Optional[datetime] = None,
) -> Mapping[str, Any]:
    errors = validate_ready_manifest_payload(ready_manifest, require_closure=False)
    if errors:
        raise RuntimeError("ready manifest rejected: " + ",".join(errors))
    if ready_manifest.get("consistency_ok") is not True:
        raise RuntimeError("ready manifest consistency gate is not green")
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
        )
    }
    return {
        "schema_version": SAFE_PLAN_SCHEMA,
        "moscow_day": day.isoformat(),
        "generated_at_utc": generated.isoformat(),
        "expires_at_utc": (generated + timedelta(minutes=60)).isoformat(),
        "source_ready_sha256": str(ready_manifest.get("sha256") or ""),
        "stage10_sha256": _safe_json_sha256(verdict),
        "consistency_ok": True,
        "closure_ok": verdict.get("closure_ok") is True,
        "pending_unique": pending_unique,
        "stage10_counts": stage10_counts,
        "rows": safe_rows,
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
    )
    try:
        normalized_counts = {field: int(counts[field]) for field in count_fields}
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("safe Google plan Stage10 counts are invalid") from exc
    if (
        any(value < 0 for value in normalized_counts.values())
        or normalized_counts["pending_unique"] != pending
        or normalized_counts["mango_unique"]
        != sum(normalized_counts[field] for field in count_fields[1:])
    ):
        raise RuntimeError("safe Google plan Stage10 balance is invalid")
    return validate_safe_rows(payload.get("rows"))


def header_map(
    header: Sequence[Any], required_headers: Sequence[str] = HEADERS
) -> Mapping[str, int]:
    normalized = [str(value or "").strip() for value in header]
    duplicates = {value for value in normalized if value and normalized.count(value) > 1}
    if duplicates:
        raise RuntimeError("Google sheet has duplicate headers")
    missing = [name for name in required_headers if name not in normalized]
    if missing:
        raise RuntimeError("Google sheet is missing required headers")
    return {name: normalized.index(name) for name in required_headers}


def plan_named_upsert(
    existing: Sequence[Sequence[Any]],
    desired: Sequence[Mapping[str, Any]],
    *,
    required_headers: Sequence[str],
    managed_headers: Sequence[str],
    manual_headers: Sequence[str] = (),
    clear_absent: bool = False,
) -> Mapping[str, Any]:
    if not existing:
        existing = [list(required_headers)]
    mapping = header_map(existing[0], required_headers)
    key_column = mapping["call_key"]
    rows_by_key: dict[str, int] = {}
    existing_by_key: dict[str, Sequence[Any]] = {}
    manual_before: dict[str, tuple[Any, ...]] = {}
    for offset, row in enumerate(existing[1:], start=3):
        key = str(row[key_column] if key_column < len(row) else "").strip()
        if not key:
            continue
        if key in rows_by_key:
            raise RuntimeError("Google sheet has duplicate call_key")
        rows_by_key[key] = offset
        existing_by_key[key] = row
        manual_before[key] = tuple(
            row[mapping[name]] if mapping[name] < len(row) else ""
            for name in manual_headers
        )
    next_row = max(3, len(existing) + 2)
    updates: list[Mapping[str, Any]] = []
    desired_keys = {str(row["call_key"]) for row in desired}
    if clear_absent:
        for key in sorted(set(existing_by_key) - desired_keys):
            for name in managed_headers:
                if name == "call_key":
                    continue
                prior_row = existing_by_key[key]
                prior = prior_row[mapping[name]] if mapping[name] < len(prior_row) else ""
                if str(prior) != "":
                    updates.append(
                        {
                            "row": rows_by_key[key],
                            "column": mapping[name] + 1,
                            "header": name,
                            "value": "",
                        }
                    )
    for desired_row in desired:
        key = str(desired_row["call_key"])
        target_row = rows_by_key.get(key)
        if target_row is None:
            target_row = next_row
            next_row += 1
            rows_by_key[key] = target_row
        for name in managed_headers:
            prior_row = existing_by_key.get(key)
            if prior_row is not None:
                prior_value = (
                    prior_row[mapping[name]] if mapping[name] < len(prior_row) else ""
                )
                if str(prior_value) == str(desired_row.get(name, "")):
                    continue
            updates.append(
                {
                    "row": target_row,
                    "column": mapping[name] + 1,
                    "header": name,
                    "value": desired_row.get(name, ""),
                }
            )
    return {
        "updates": updates,
        "manual_before": manual_before,
        "desired_keys": sorted(desired_keys),
    }


def plan_upsert(
    existing: Sequence[Sequence[Any]], desired: Sequence[Mapping[str, Any]]
) -> Mapping[str, Any]:
    return plan_named_upsert(
        existing,
        desired,
        required_headers=HEADERS,
        managed_headers=MANAGED_HEADERS,
        manual_headers=ROP_HEADERS,
        clear_absent=True,
    )


def column_name(number: int) -> str:
    if number < 1:
        raise ValueError("column number must be positive")
    result = ""
    while number:
        number, remainder = divmod(number - 1, 26)
        result = chr(65 + remainder) + result
    return result


def quote_title(title: str) -> str:
    return "'" + title.replace("'", "''") + "'"


def verify_named_readback(
    values: Sequence[Sequence[Any]],
    desired: Sequence[Mapping[str, Any]],
    manual_before: Mapping[str, Sequence[Any]],
    *,
    required_headers: Sequence[str],
    managed_headers: Sequence[str],
    manual_headers: Sequence[str] = (),
    require_absent_cleared: bool = False,
) -> None:
    mapping = header_map(values[0] if values else (), required_headers)
    key_column = mapping["call_key"]
    rows: dict[str, Sequence[Any]] = {}
    for row in values[1:]:
        key = str(row[key_column] if key_column < len(row) else "").strip()
        if not key:
            continue
        if key in rows:
            raise RuntimeError("Google readback has duplicate call_key")
        rows[key] = row
    for wanted in desired:
        key = str(wanted["call_key"])
        actual = rows.get(key)
        if actual is None:
            raise RuntimeError("Google readback is missing an updated row")
        for name in managed_headers:
            value = actual[mapping[name]] if mapping[name] < len(actual) else ""
            if str(value) != str(wanted.get(name, "")):
                raise RuntimeError("Google managed-cell readback mismatch")
        if manual_headers and key in manual_before:
            manual = tuple(
                actual[mapping[name]] if mapping[name] < len(actual) else ""
                for name in manual_headers
            )
            if tuple(map(str, manual)) != tuple(map(str, manual_before[key])):
                raise RuntimeError("Google ROP-owned cells changed")
    if require_absent_cleared:
        desired_keys = {str(item["call_key"]) for item in desired}
        for key, actual in rows.items():
            if key in desired_keys:
                continue
            if any(
                str(actual[mapping[name]] if mapping[name] < len(actual) else "")
                for name in managed_headers
                if name != "call_key"
            ):
                raise RuntimeError("Google stale managed row was not cleared")


def verify_readback(
    values: Sequence[Sequence[Any]],
    desired: Sequence[Mapping[str, Any]],
    manual_before: Mapping[str, Sequence[Any]],
) -> None:
    verify_named_readback(
        values,
        desired,
        manual_before,
        required_headers=HEADERS,
        managed_headers=MANAGED_HEADERS,
        manual_headers=ROP_HEADERS,
        require_absent_cleared=True,
    )


def retention_allows(
    *, day: date, pilot_started_day: date, retention_approved: bool
) -> bool:
    return retention_approved or 0 <= (day - pilot_started_day).days < 14


class GoogleGateway:
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

    def file_permissions(self, file_id: str) -> Sequence[Mapping[str, Any]]:
        return self._permissions_for(file_id)

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

    def protections(self, title: str) -> Sequence[Mapping[str, Any]]:
        response = self.session.get(
            f"https://sheets.googleapis.com/v4/spreadsheets/{self.spreadsheet_id}",
            params={
                "fields": (
                    "sheets(properties(sheetId,title),"
                    "protectedRanges(range,warningOnly,editors(users,groups,domainUsersCanEdit)))"
                )
            },
            timeout=60,
        )
        for sheet in self._json(response).get("sheets") or ():
            if not isinstance(sheet, Mapping):
                continue
            properties = sheet.get("properties")
            if isinstance(properties, Mapping) and properties.get("title") == title:
                ranges = sheet.get("protectedRanges") or ()
                if not isinstance(ranges, Sequence) or isinstance(ranges, (str, bytes)):
                    raise RuntimeError("Google protection readback is invalid")
                return [item for item in ranges if isinstance(item, Mapping)]
        raise RuntimeError("Google sheet protection readback target is missing")

    def column_hidden(self, title: str, column_index: int) -> bool:
        response = self.session.get(
            f"https://sheets.googleapis.com/v4/spreadsheets/{self.spreadsheet_id}",
            params={
                "ranges": (
                    f"{quote_title(title)}!{column_name(column_index + 1)}:"
                    f"{column_name(column_index + 1)}"
                ),
                "includeGridData": "true",
                "fields": (
                    "sheets(properties(title),"
                    "data(startColumn,columnMetadata(hiddenByUser)))"
                ),
            },
            timeout=60,
        )
        for sheet in self._json(response).get("sheets") or ():
            if not isinstance(sheet, Mapping):
                continue
            properties = sheet.get("properties")
            if not isinstance(properties, Mapping) or properties.get("title") != title:
                continue
            for data in sheet.get("data") or ():
                if not isinstance(data, Mapping) or int(data.get("startColumn", -1)) != column_index:
                    continue
                metadata = data.get("columnMetadata")
                return bool(
                    isinstance(metadata, Sequence)
                    and not isinstance(metadata, (str, bytes))
                    and metadata
                    and isinstance(metadata[0], Mapping)
                    and metadata[0].get("hiddenByUser") is True
                )
        return False

    def read_values(self, title: str) -> Sequence[Sequence[Any]]:
        response = self.session.get(
            f"https://sheets.googleapis.com/v4/spreadsheets/{self.spreadsheet_id}/values/{quote_title(title)}!A2:ZZ",
            params={"majorDimension": "ROWS"},
            timeout=60,
        )
        return self._json(response).get("values") or ()

    def read_banner(self, title: str) -> str:
        response = self.session.get(
            f"https://sheets.googleapis.com/v4/spreadsheets/{self.spreadsheet_id}/values/"
            f"{quote_title(title)}!A1",
            timeout=60,
        )
        values = self._json(response).get("values") or ()
        return str(values[0][0]) if values and values[0] else ""

    def batch_sheet_requests(self, requests: Sequence[Mapping[str, Any]]) -> None:
        response = self.session.post(
            f"https://sheets.googleapis.com/v4/spreadsheets/{self.spreadsheet_id}:batchUpdate",
            json={"requests": list(requests)},
            timeout=60,
        )
        self._json(response)

    def write_values(self, data: Sequence[Mapping[str, Any]]) -> None:
        response = self.session.post(
            f"https://sheets.googleapis.com/v4/spreadsheets/{self.spreadsheet_id}/values:batchUpdate",
            json={"valueInputOption": "RAW", "data": list(data)},
            timeout=120,
        )
        self._json(response)

    def clear_values(self, title: str) -> None:
        response = self.session.post(
            f"https://sheets.googleapis.com/v4/spreadsheets/{self.spreadsheet_id}/values/"
            f"{quote_title(title)}!A3:ZZ:clear",
            json={},
            timeout=60,
        )
        self._json(response)


def _sheet_map(gateway: Any) -> dict[str, int]:
    return {
        str(item.get("title")): int(item.get("sheetId"))
        for item in gateway.sheets()
    }


def _ensure_sheet(
    gateway: Any,
    *,
    sheets: Mapping[str, int],
    title: str,
    headers: Sequence[str],
    day: date,
    owner_email: str,
    rop_editors: Sequence[str],
    protect_rop: bool,
) -> tuple[dict[str, int], bool]:
    created = title not in sheets
    if created:
        gateway.batch_sheet_requests([{"addSheet": {"properties": {"title": title}}}])
        current = _sheet_map(gateway)
        if title not in current:
            raise RuntimeError("Google sheet creation readback failed")
        last_column = column_name(len(headers))
        banner = f"{BANNER_PREFIX} — {day.isoformat()}"
        gateway.write_values(
            [
                {
                    "range": f"{quote_title(title)}!A1:{last_column}2",
                    "majorDimension": "ROWS",
                    "values": [
                        [banner, *([""] * (len(headers) - 1))],
                        list(headers),
                    ],
                }
            ]
        )
        requests: list[Mapping[str, Any]] = []
        if "call_key" in headers:
            key_index = list(headers).index("call_key")
            requests.append(
                {
                    "updateDimensionProperties": {
                        "range": {
                            "sheetId": current[title],
                            "dimension": "COLUMNS",
                            "startIndex": key_index,
                            "endIndex": key_index + 1,
                        },
                        "properties": {"hiddenByUser": True},
                        "fields": "hiddenByUser",
                    }
                }
            )
        if protect_rop:
            start = list(headers).index(ROP_HEADERS[0])
            requests.append(
                {
                    "addProtectedRange": {
                        "protectedRange": {
                            "range": {
                                "sheetId": current[title],
                                "startRowIndex": 1,
                                "startColumnIndex": start,
                                "endColumnIndex": start + len(ROP_HEADERS),
                            },
                            "description": "Поля РОПа; служба Mango не редактирует",
                            "warningOnly": False,
                            "editors": {
                                "users": sorted(
                                    {
                                        owner_email.strip(),
                                        *(value.strip() for value in rop_editors if value.strip()),
                                    }
                                )
                            },
                        }
                    }
                }
            )
        if requests:
            gateway.batch_sheet_requests(requests)
        sheets = current
    values = gateway.read_values(title)
    header_map(values[0] if values else (), headers)
    if "call_key" in headers:
        key_index = list(headers).index("call_key")
        if not hasattr(gateway, "column_hidden") or gateway.column_hidden(
            title, key_index
        ) is not True:
            raise RuntimeError("Google call_key hidden-column readback failed")
    if protect_rop:
        if not hasattr(gateway, "protections"):
            raise RuntimeError("Google ROP protection readback is unavailable")
        start = list(headers).index(ROP_HEADERS[0])
        expected_editors = {
            owner_email.strip().casefold(),
            *(value.strip().casefold() for value in rop_editors if value.strip()),
        }
        protection_ok = False
        for protection in gateway.protections(title):
            protected_range = protection.get("range")
            editors = protection.get("editors")
            if not isinstance(protected_range, Mapping) or not isinstance(editors, Mapping):
                continue
            users = {
                str(value).strip().casefold()
                for value in editors.get("users", ())
                if str(value).strip()
            }
            protection_ok = protection_ok or bool(
                protection.get("warningOnly") is not True
                and int(protected_range.get("sheetId", -1)) == int(sheets[title])
                and int(protected_range.get("startRowIndex", -1)) == 1
                and int(protected_range.get("startColumnIndex", -1)) == start
                and int(protected_range.get("endColumnIndex", -1))
                == start + len(ROP_HEADERS)
                and users == expected_editors
                and not editors.get("groups")
                and editors.get("domainUsersCanEdit") is not True
            )
        if not protection_ok:
            raise RuntimeError("Google ROP protection readback failed")
    if hasattr(gateway, "read_banner"):
        banner = str(gateway.read_banner(title) or "")
        if (
            title in {REVIEW_TITLE, SUMMARY_TITLE}
            and (not banner.startswith(BANNER_PREFIX) or day.isoformat() not in banner)
        ):
            gateway.write_values(
                [
                    {
                        "range": f"{quote_title(title)}!A1",
                        "majorDimension": "ROWS",
                        "values": [[f"{BANNER_PREFIX} — {day.isoformat()}"]],
                    }
                ]
            )
            banner = str(gateway.read_banner(title) or "")
        if not banner.startswith(BANNER_PREFIX) or day.isoformat() not in banner:
            raise RuntimeError("Google preliminary banner readback failed")
    return dict(sheets), created


def _cell_updates(title: str, updates: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [
        {
            "range": (
                f"{quote_title(title)}!"
                f"{column_name(int(item['column']))}{int(item['row'])}"
            ),
            "majorDimension": "ROWS",
            "values": [[item["value"]]],
        }
        for item in updates
    ]


def _summary_rows(
    desired_rows: Sequence[Mapping[str, Any]],
    stage10_summary: Optional[Mapping[str, Any]] = None,
) -> list[list[Any]]:
    stage10 = stage10_summary or {}
    review = sum(str(row.get("Нужна проверка") or "") == "Да" for row in desired_rows)
    analyzed = sum(
        str(row.get("Краткое содержание") or "") != NEUTRAL_SUMMARY
        for row in desired_rows
    )
    return [
        ["Найдено Mango", int(stage10.get("mango_unique", len(desired_rows)))],
        ["Полностью готово", int(stage10.get("ready_unique", len(desired_rows)))],
        ["Ожидает обработки", int(stage10.get("pending_unique", 0))],
        ["Карантин", int(stage10.get("quarantine_unique", 0))],
        ["Непонятные пропуски", int(stage10.get("unexplained_missing", 0))],
        ["Требуют проверки", review],
        ["Смысловой анализ завершён", analyzed],
        ["Баланс подтверждён", "Да" if stage10.get("consistency_ok", True) else "Нет"],
        ["День закрыт", "Да" if stage10.get("closure_ok", False) else "Нет"],
    ]


def _validate_live_links(
    gateway: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    owner_email: str,
    allowed_emails: Sequence[str],
    expected_roles: Optional[Mapping[str, str]],
) -> None:
    for row in rows:
        link = str(row.get(TRANSCRIPT_LINK_HEADER) or "")
        if not link:
            continue
        match = DRIVE_FILE_RE.match(link)
        if match is None or not hasattr(gateway, "file_permissions"):
            raise RuntimeError("private transcript link cannot be verified live")
        if not exact_acl_ok(
            gateway.file_permissions(match.group("id")),
            owner_email=owner_email,
            allowed_emails=allowed_emails,
            expected_roles=expected_roles,
        ):
            raise RuntimeError("private transcript link ACL is not the exact whitelist")


def publish_current(
    gateway: Any,
    *,
    day: date,
    desired_rows: Sequence[Mapping[str, Any]],
    owner_email: str,
    allowed_emails: Iterable[str],
    pilot_started_day: date,
    retention_approved: bool,
    prior_day: Optional[date],
    stage10_summary: Optional[Mapping[str, Any]] = None,
    expected_roles: Optional[Mapping[str, str]] = None,
    service_account_email: str = "",
) -> Mapping[str, Any]:
    if not retention_allows(
        day=day,
        pilot_started_day=pilot_started_day,
        retention_approved=retention_approved,
    ):
        raise RuntimeError("Google pilot retention decision is overdue")
    desired_rows = validate_safe_rows(list(desired_rows))
    allowed = tuple(allowed_emails)
    if not exact_acl_ok(
        gateway.permissions(),
        owner_email=owner_email,
        allowed_emails=allowed,
        expected_roles=expected_roles,
    ):
        raise RuntimeError("Google spreadsheet ACL is not the exact approved whitelist")
    _validate_live_links(
        gateway,
        desired_rows,
        owner_email=owner_email,
        allowed_emails=allowed,
        expected_roles=expected_roles,
    )
    sheets = _sheet_map(gateway)
    rotated_from: Optional[str] = None
    if prior_day and prior_day != day and CURRENT_TITLE in sheets:
        archive_title = f"Звонки {prior_day.isoformat()} — предварительно"
        if archive_title in sheets:
            if not hasattr(gateway, "read_banner") or day.isoformat() not in str(
                gateway.read_banner(CURRENT_TITLE) or ""
            ):
                raise RuntimeError("Google day rotation target already exists")
        else:
            gateway.batch_sheet_requests(
                [
                    {
                        "updateSheetProperties": {
                            "properties": {
                                "sheetId": sheets[CURRENT_TITLE],
                                "title": archive_title,
                            },
                            "fields": "title",
                        }
                    }
                ]
            )
        rotated_from = prior_day.isoformat()
        sheets = _sheet_map(gateway)
    rop_editors = tuple(
        email for email in allowed if email.casefold() != service_account_email.casefold()
    )
    sheets, _ = _ensure_sheet(
        gateway,
        sheets=sheets,
        title=CURRENT_TITLE,
        headers=HEADERS,
        day=day,
        owner_email=owner_email,
        rop_editors=rop_editors,
        protect_rop=True,
    )
    sheets, _ = _ensure_sheet(
        gateway,
        sheets=sheets,
        title=REVIEW_TITLE,
        headers=REVIEW_HEADERS,
        day=day,
        owner_email=owner_email,
        rop_editors=(),
        protect_rop=False,
    )
    sheets, _ = _ensure_sheet(
        gateway,
        sheets=sheets,
        title=SUMMARY_TITLE,
        headers=SUMMARY_HEADERS,
        day=day,
        owner_email=owner_email,
        rop_editors=(),
        protect_rop=False,
    )
    existing = gateway.read_values(CURRENT_TITLE)
    planned = plan_upsert(existing, desired_rows)
    data = _cell_updates(CURRENT_TITLE, planned["updates"])
    if data:
        gateway.write_values(data)
    readback = gateway.read_values(CURRENT_TITLE)
    verify_readback(readback, desired_rows, planned["manual_before"])
    review_rows = [
        row for row in desired_rows if str(row.get("Нужна проверка") or "") == "Да"
    ]
    review_existing = gateway.read_values(REVIEW_TITLE)
    review_plan = plan_named_upsert(
        review_existing,
        review_rows,
        required_headers=REVIEW_HEADERS,
        managed_headers=REVIEW_HEADERS,
        clear_absent=True,
    )
    review_data = _cell_updates(REVIEW_TITLE, review_plan["updates"])
    if review_data:
        gateway.write_values(review_data)
    verify_named_readback(
        gateway.read_values(REVIEW_TITLE),
        review_rows,
        {},
        required_headers=REVIEW_HEADERS,
        managed_headers=REVIEW_HEADERS,
        require_absent_cleared=True,
    )

    summary_values = gateway.read_values(SUMMARY_TITLE)
    header_map(summary_values[0] if summary_values else (), SUMMARY_HEADERS)
    wanted_summary = _summary_rows(desired_rows, stage10_summary)
    existing_summary = [list(row[:2]) for row in summary_values[1:]]
    summary_changed = existing_summary != wanted_summary
    if summary_changed:
        if hasattr(gateway, "clear_values"):
            gateway.clear_values(SUMMARY_TITLE)
        gateway.write_values(
            [
                {
                    "range": f"{quote_title(SUMMARY_TITLE)}!A3:B{2 + len(wanted_summary)}",
                    "majorDimension": "ROWS",
                    "values": wanted_summary,
                }
            ]
        )
    summary_readback = [list(row[:2]) for row in gateway.read_values(SUMMARY_TITLE)[1:]]
    if summary_readback != wanted_summary:
        raise RuntimeError("Google summary readback mismatch")
    if not exact_acl_ok(
        gateway.permissions(),
        owner_email=owner_email,
        allowed_emails=allowed,
        expected_roles=expected_roles,
    ):
        raise RuntimeError("Google spreadsheet ACL changed during update")
    return {
        "schema_version": SCHEMA,
        "status": "updated" if data else "unchanged",
        "day": day.isoformat(),
        "rows": len(desired_rows),
        "managed_cell_updates": len(data) + len(review_data) + (
            len(wanted_summary) * 2 if summary_changed else 0
        ),
        "rotated_from": rotated_from,
        "current_title": CURRENT_TITLE,
        "review_title": REVIEW_TITLE,
        "summary_title": SUMMARY_TITLE,
        "retention_approved": retention_approved,
        "full_transcript_fields_written": 0,
    }


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
    info = path.lstat()
    if (
        not stat.S_ISREG(info.st_mode)
        or path.is_symlink()
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) != 0o600
    ):
        raise RuntimeError("Google credentials must be owner-only 0600")
    resolved = path.resolve()
    if resolved == ROOT or ROOT in resolved.parents or any(
        marker in part.casefold()
        for part in resolved.parts
        for marker in ("yandex.disk", "icloud", "mobile documents", "dropbox", "onedrive")
    ):
        raise RuntimeError("Google credentials must stay outside repository and cloud folders")
    payload = json.loads(path.read_text(encoding="utf-8"))
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
    }


def authorized_session(credentials: Path) -> Any:
    from google.auth.transport.requests import AuthorizedSession
    from google.oauth2.service_account import Credentials

    scopes = (
        "https://www.googleapis.com/auth/drive.metadata.readonly",
        "https://www.googleapis.com/auth/spreadsheets",
    )
    return AuthorizedSession(
        Credentials.from_service_account_file(str(credentials), scopes=scopes)
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Update safe preliminary Mango calls Google sheet.")
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
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirmation", default="")
    parser.add_argument(
        "--approved-plan-sha256",
        help="Owner-approved SHA-256 of the exact short-lived safe plan.",
    )
    args = parser.parse_args(argv)
    if not args.execute:
        if not args.ready_db or not args.owner_email:
            raise RuntimeError("dry-run requires ready DB and owner email")
        manifest_path = args.ready_manifest or args.ready_db.with_suffix(
            ".manifest.json"
        )
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
            "safe_plan_sha256": _safe_json_sha256(safe_plan),
            "plan_written": bool(args.plan_out),
            "full_transcript_fields_written": 0,
            "external_write": False,
        }
    else:
        if (
            args.confirmation != CONFIRMATION
            or not args.config
            or not args.safe_plan
            or not args.state
            or not re.fullmatch(r"[0-9a-f]{64}", args.approved_plan_sha256 or "")
        ):
            raise RuntimeError(
                "execute requires explicit confirmation, owner config, safe plan, "
                "its approved SHA-256 and state"
            )
        if args.ready_db or args.ready_manifest or args.link_evidence:
            raise RuntimeError("execute must not read ready DB, manifest or link evidence")
        config = load_google_config(args.config)
        safe_plan_payload = owner_json(args.safe_plan)
        safe_plan_sha256 = _safe_json_sha256(safe_plan_payload)
        if safe_plan_sha256 != args.approved_plan_sha256:
            raise RuntimeError("safe Google plan does not match the approved SHA-256")
        rows = validate_safe_plan_payload(safe_plan_payload, expected_day=args.day)
        state = owner_json(args.state) if args.state.exists() else {}
        prior = (
            date.fromisoformat(str(state["active_day"]))
            if state.get("active_day")
            else None
        )
        lock_path = args.state.with_suffix(args.state.suffix + ".lock")
        with publication_lock(lock_path):
            atomic_owner_json(
                args.state,
                {
                    "schema_version": SCHEMA,
                    "status": "write_uncertain",
                    "active_day": prior.isoformat() if prior else None,
                    "target_day": args.day.isoformat(),
                    "spreadsheet_id": config["spreadsheet_id"],
                    "safe_plan_sha256": safe_plan_sha256,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            report = publish_current(
                GoogleGateway(
                    authorized_session(config["credentials"]),
                    str(config["spreadsheet_id"]),
                ),
                day=args.day,
                desired_rows=rows,
                owner_email=str(config["owner_email"]),
                allowed_emails=config["allowed_emails"],
                pilot_started_day=config["pilot_started_day"],
                retention_approved=bool(
                    config.get("retention_policy_approved")
                    or args.retention_approved
                ),
                prior_day=prior,
                stage10_summary={
                    **dict(safe_plan_payload.get("stage10_counts") or {}),
                    "consistency_ok": safe_plan_payload.get("consistency_ok") is True,
                    "closure_ok": safe_plan_payload.get("closure_ok") is True,
                },
                expected_roles=config["expected_roles"],
                service_account_email=str(config["service_account_email"]),
            )
            atomic_owner_json(
                args.state,
                {
                    "schema_version": SCHEMA,
                    "status": "success",
                    "active_day": args.day.isoformat(),
                    "spreadsheet_id": config["spreadsheet_id"],
                    "safe_plan_sha256": safe_plan_sha256,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
            )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
