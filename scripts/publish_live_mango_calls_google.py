#!/usr/bin/env python3
"""Deterministic publisher for the production Mango calls Google sheet.

Dry-run is the default.  The script owns only the Google projection and the
compatibility ``sync_status`` marker; it never runs or claims pipeline stages.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence
from urllib.parse import quote
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mango_mvp.productization.contracts import stable_event_key  # noqa: E402

try:  # noqa: E402
    from scripts.publish_current_mango_calls_google import (
        GoogleGateway,
        atomic_owner_json,
        authorized_session,
        owner_json,
        publication_lock,
        validate_credentials,
    )
except ImportError:  # pragma: no cover - direct ``python scripts/...`` execution
    from publish_current_mango_calls_google import (  # type: ignore
        GoogleGateway,
        atomic_owner_json,
        authorized_session,
        owner_json,
        publication_lock,
        validate_credentials,
    )


CONFIG_SCHEMA = "mango_calls_live_google_config_v1"
STATE_SCHEMA = "mango_calls_live_google_state_v1"
PROJECTION_VERSION = "mango_calls_live_google_projection_v3"
CONFIRMATION = "PUBLISH_MANGO_CALLS_LIVE"
BOOTSTRAP_CONFIRMATION = "BOOTSTRAP_MANGO_CALLS_LIVE"
MOSCOW = ZoneInfo("Europe/Moscow")
MAX_CELL_CHARS = 50_000
HASH_RE = re.compile(r"^[0-9a-f]{64}$")
CALL_KEY_RE = re.compile(r"^mango:mango_office:[^\r\n]{1,512}$")
STATE_STATUSES = {"reserved", "verified"}
LIVE_HEADERS = (
    "№",
    "Дата и время (МСК)",
    "Менеджер",
    "Направление",
    "Длительность",
    "Категория",
    "Телефон клиента",
    "Нужна проверка",
    "Тема",
    "Конспект разговора",
    "Результат",
    "Возражение / причина",
    "Следующий шаг",
    "Срок",
    "Что проверить РОПу",
    "Полная расшифровка",
)
LINE_RE = re.compile(r"^\[([^]]+)]\s+([^:]+):\s*(.*)$")
PHYSICAL_RE = re.compile(
    r"^(?:CHANNEL_(?:LEFT|RIGHT)|Дорожка\s+(?:левая|правая))\s*:\s*",
    re.IGNORECASE,
)
TECHNICAL_RE = re.compile(r"^(?:source_call_id|sha(?:-?256)?)\s*:", re.IGNORECASE)
DURATION_RE = re.compile(r"^\s*(\d+)\s*мин\s*(\d+)\s*с\s*$", re.IGNORECASE)


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def json_object(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    try:
        parsed = json.loads(str(value or "{}"))
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, Mapping) else {}


def validation_error_code(prefix: str, exc: Exception) -> str:
    message = str(exc).lower()
    known = (
        ("unknown or unmapped role", "dialogue_role_unknown"),
        ("malformed line", "dialogue_line_malformed"),
        ("full transcript is empty", "transcript_empty"),
        ("50000", "cell_too_long"),
        ("duration", "duration_invalid"),
        ("source_call_id is empty", "source_call_id_empty"),
        ("call_key is invalid", "source_call_id_invalid"),
        ("analysis_json", "analysis_json_invalid"),
    )
    suffix = next((code for marker, code in known if marker in message), "validation_error")
    return f"{prefix}_{suffix}"


def validated_call_key(value: Any) -> str:
    source_call_id = str(value or "").strip()
    if not source_call_id:
        raise ValueError("source_call_id is empty")
    call_key = stable_event_key("mango", "mango_office", source_call_id)
    if not CALL_KEY_RE.fullmatch(call_key):
        raise ValueError("call_key is invalid")
    return call_key


def required_json_object(value: Any, label: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        if not value:
            raise ValueError(f"{label} is empty")
        return value
    if value is None or not str(value).strip():
        raise ValueError(f"{label} is empty")
    try:
        parsed = json.loads(str(value))
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is invalid") from exc
    if not isinstance(parsed, Mapping):
        raise ValueError(f"{label} is not an object")
    if not parsed:
        raise ValueError(f"{label} is empty")
    return parsed


def parse_utc(value: Any) -> datetime:
    parsed = datetime.fromisoformat(str(value or "").strip().replace("Z", "+00:00"))
    return (
        parsed.replace(tzinfo=timezone.utc)
        if parsed.tzinfo is None
        else parsed.astimezone(timezone.utc)
    )


def half_up_seconds(value: Any) -> int:
    seconds = float(value)
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("duration must be finite and non-negative")
    return int(math.floor(seconds + 0.5))


def format_duration(value: Any) -> str:
    minutes, seconds = divmod(half_up_seconds(value), 60)
    return f"{minutes} мин {seconds} с"


def row_height(summary: Any) -> int:
    paragraphs = str(summary or "").splitlines() or [""]
    logical = sum(max(1, math.ceil(len(paragraph) / 85)) for paragraph in paragraphs)
    return min(260, max(42, 12 + 18 * logical))


def list_text(value: Any) -> str:
    if isinstance(value, list):
        return "; ".join(str(item).strip() for item in value if str(item).strip())
    return str(value or "").strip()


def normalize_summary_time(value: Any, started_utc: datetime) -> str:
    """Convert only a generated leading call timestamp from UTC to Moscow."""
    summary = str(value or "").strip()
    moscow = started_utc.astimezone(MOSCOW)
    for pattern in ("%d.%m.%Y %H:%M:%S", "%d.%m.%Y %H:%M", "%Y-%m-%d %H:%M:%S"):
        utc_prefix = started_utc.strftime(pattern)
        if summary.startswith(utc_prefix):
            return moscow.strftime(pattern) + summary[len(utc_prefix):]
    return summary


def role_name(label: Any, mapping: Mapping[str, Any]) -> Optional[str]:
    low = str(label or "").strip().lower()
    if low in {"менеджер", "manager"} or low.startswith("менеджер "):
        return "Менеджер"
    if low in {"клиент", "client"} or low.startswith("клиент "):
        return "Клиент"
    physical = (
        "left"
        if low in {"дорожка левая", "channel_left", "left"}
        else "right"
        if low in {"дорожка правая", "channel_right", "right"}
        else None
    )
    if not physical:
        return None
    return {"manager": "Менеджер", "client": "Клиент"}.get(
        str(mapping.get(physical) or "").lower()
    )


def render_transcript(record: Mapping[str, Any]) -> str:
    variants = json_object(record.get("transcript_variants_json"))
    mapping = json_object(variants.get("role_mapping"))
    lines = variants.get("dialogue_lines")
    groups: list[list[str]] = []
    if isinstance(lines, list):
        for raw in lines:
            match = LINE_RE.match(str(raw or "").strip())
            if not match:
                raise ValueError("dialogue_lines contains a malformed line")
            timestamp, label, content = match.groups()
            role, content = role_name(label, mapping), content.strip()
            if role is None:
                raise ValueError("dialogue_lines contains an unknown or unmapped role")
            if groups and groups[-1][1] == role:
                groups[-1][2] = f"{groups[-1][2]} {content}".strip()
            else:
                groups.append([timestamp, role, content])
    if groups:
        return "\n".join(f"[{ts}] {role}: {text}" for ts, role, text in groups)
    full = variants.get("full")
    final = full.get("final") if isinstance(full, Mapping) else None
    fallback = str(final or record.get("transcript_text") or "").strip()
    cleaned: list[str] = []
    for raw_line in fallback.splitlines():
        line = raw_line.strip()
        if TECHNICAL_RE.match(line):
            continue
        line = PHYSICAL_RE.sub("", line).strip()
        if line:
            cleaned.append(line)
    fallback = " ".join(cleaned).strip()
    return f"[00:00.0] Не определено: {fallback}" if fallback else ""


def render_legacy_identity_transcript(record: Mapping[str, Any]) -> str:
    """Reproduce the old lenient projection only to identify an existing row."""
    variants = json_object(record.get("transcript_variants_json"))
    mapping = json_object(variants.get("role_mapping"))
    lines = variants.get("dialogue_lines")
    groups: list[list[str]] = []
    if isinstance(lines, list):
        for raw in lines:
            match = LINE_RE.match(str(raw or "").strip())
            if not match:
                continue
            timestamp, label, content = match.groups()
            role = role_name(label, mapping) or "Не определено"
            content = content.strip()
            if groups and groups[-1][1] == role:
                groups[-1][2] = f"{groups[-1][2]} {content}".strip()
            else:
                groups.append([timestamp, role, content])
    if groups:
        return "\n".join(f"[{ts}] {role}: {text}" for ts, role, text in groups)
    return render_transcript({**dict(record), "transcript_variants_json": {**variants, "dialogue_lines": []}})


def manager_display(value: Any, mapping: Mapping[str, Any]) -> str:
    raw = str(value or "").strip()
    display = str(mapping.get(raw) or raw).strip()
    technical = display.lower().startswith(("mango_", "unknown", "неизвест", "тест"))
    return display if re.search(r"[А-Яа-яЁё]", display) and not technical else "Не определён"


def call_projection(record: Mapping[str, Any], manager_map: Mapping[str, Any]) -> dict[str, Any]:
    call_key = validated_call_key(record.get("source_call_id"))
    started = parse_utc(record.get("started_at"))
    duration = float(record.get("duration_sec"))
    analysis = required_json_object(record.get("analysis_json"), "analysis_json")
    fields = json_object(analysis.get("structured_fields"))
    interests = json_object(fields.get("interests"))
    next_step = json_object(fields.get("next_step"))
    flags = json_object(analysis.get("quality_flags"))
    call_type = str(flags.get("call_type") or "").lower()
    products = interests.get("products")
    topic = (
        str(analysis.get("target_product") or "").strip()
        or (str(products[0]).strip() if isinstance(products, list) and products else "")
        or list_text(analysis.get("interests"))
        or "—"
    )
    summary = normalize_summary_time((
        str(analysis.get("summary") or "").strip()
        or str(analysis.get("history_short") or "").strip()
        or str(analysis.get("history_summary") or "").strip()
        or "—"
    ), started)
    objections = list_text(fields.get("objections")) or list_text(analysis.get("objections")) or "—"
    action = str(next_step.get("action") or "").strip()
    if not action and isinstance(analysis.get("next_step"), str):
        action = str(analysis.get("next_step") or "").strip()
    due = str(next_step.get("due") or "").strip() or str(analysis.get("timeline") or "").strip()
    review = list_text(analysis.get("review_reasons")) or list_text(flags.get("review_reasons")) or "—"
    transcript = render_transcript(record)
    if not transcript:
        raise ValueError("full transcript is empty")
    values = [
        started.astimezone(MOSCOW).strftime("%Y-%m-%d %H:%M:%S"),
        manager_display(record.get("manager_name"), manager_map),
        {"outbound": "Исходящий", "inbound": "Входящий"}.get(
            str(record.get("direction") or "").lower(), "Не определено"
        ),
        format_duration(duration),
        {
            "sales_call": "Продажа",
            "service_call": "Сервис",
            "existing_client_progress": "Сервис",
        }.get(call_type, "Не определено"),
        str(record.get("phone") or ""),
        "Да",
        topic,
        summary,
        str(analysis.get("follow_up_reason") or "").strip() or "—",
        objections,
        action or "—",
        due or "—",
        review,
        transcript,
    ]
    if any(len(str(value)) > MAX_CELL_CHARS for value in values):
        raise ValueError("cell exceeds 50000 characters")
    source = {
        "projection_version": PROJECTION_VERSION,
        "call_key": call_key,
        "started_epoch": int(started.timestamp()),
        "tail": values,
    }
    return {
        "id": int(record["id"]),
        "call_key": call_key,
        "source_fingerprint": canonical_hash(source),
        "started_epoch": int(started.timestamp()),
        "duration_sec": duration,
        "phone": str(record.get("phone") or ""),
        "transcript_sha": hashlib.sha256(transcript.encode("utf-8")).hexdigest(),
        "tail": values,
        "sync_status": str(record.get("sync_status") or ""),
    }


def call_identity(record: Mapping[str, Any]) -> dict[str, Any]:
    call_key = validated_call_key(record.get("source_call_id"))
    analysis_status = str(record.get("analysis_status") or "")
    if analysis_status != "done":
        try:
            started_epoch = int(parse_utc(record.get("started_at")).timestamp())
        except (TypeError, ValueError, OverflowError):
            started_epoch = 0
        try:
            duration = float(record.get("duration_sec"))
        except (TypeError, ValueError, OverflowError):
            duration = 0.0
        transcript = ""
    else:
        started_epoch = int(parse_utc(record.get("started_at")).timestamp())
        duration = float(record.get("duration_sec"))
        try:
            transcript = render_transcript(record)
        except ValueError:
            transcript = render_legacy_identity_transcript(record)
    return {
        "id": int(record["id"]),
        "call_key": call_key,
        "started_epoch": started_epoch,
        "duration_sec": duration,
        "phone": str(record.get("phone") or ""),
        "transcript_sha": (
            hashlib.sha256(transcript.encode("utf-8")).hexdigest() if transcript else ""
        ),
        "analysis_status": analysis_status,
        "sync_status": str(record.get("sync_status") or ""),
    }


def sheet_duration(value: Any) -> tuple[str, float]:
    match = DURATION_RE.match(str(value or ""))
    if match:
        return "rounded", float(int(match.group(1)) * 60 + int(match.group(2)))
    return "legacy", float(str(value or "").replace(",", "."))


def normalized_phone(value: Any) -> str:
    raw = str(value or "").strip()
    return raw[1:] if raw.startswith("'+") else raw


def sheet_identity(row: Sequence[Any]) -> dict[str, Any]:
    if len(row) < 16:
        raise ValueError("physical row has fewer than 16 columns")
    raw_time = str(row[1] or "").strip()
    precision = "second"
    parsed = None
    for pattern, candidate_precision in (
        ("%Y-%m-%d %H:%M:%S", "second"),
        ("%d.%m.%Y %H:%M:%S", "second"),
        ("%Y-%m-%d %H:%M", "minute"),
        ("%d.%m.%Y %H:%M", "minute"),
    ):
        try:
            parsed = datetime.strptime(raw_time, pattern)
            precision = (
                "second_or_minute"
                if candidate_precision == "second" and parsed.second == 0
                else candidate_precision
            )
            break
        except ValueError:
            continue
    if parsed is None:
        raise ValueError("physical row date format is unsupported")
    utc = parsed.replace(tzinfo=MOSCOW).astimezone(timezone.utc)
    mode, duration = sheet_duration(row[4])
    transcript = str(row[15] or "")
    if not transcript:
        raise ValueError("physical row transcript is empty")
    return {
        "utc": int(utc.timestamp()),
        "time_precision": precision,
        "phone": normalized_phone(row[6]),
        "duration_mode": mode,
        "duration": duration,
        "transcript_sha": hashlib.sha256(transcript.encode("utf-8")).hexdigest(),
    }


def physical_row_hash(row: Sequence[Any]) -> str:
    return canonical_hash(list(row[:16]) + [""] * max(0, 16 - len(row)))


def identity_matches(identity: Mapping[str, Any], call: Mapping[str, Any]) -> bool:
    precision = identity.get("time_precision")
    same_time = identity["utc"] == call["started_epoch"]
    if precision in {"minute", "second_or_minute"} and not same_time:
        same_time = identity["utc"] // 60 == call["started_epoch"] // 60
    if not same_time or identity["phone"] != normalized_phone(call["phone"]):
        return False
    if identity["transcript_sha"] != call["transcript_sha"]:
        return False
    if identity["duration_mode"] == "rounded":
        return int(identity["duration"]) == half_up_seconds(call["duration_sec"])
    return abs(float(identity["duration"]) - float(call["duration_sec"])) <= 0.001


def is_blank_sheet_value(value: Any) -> bool:
    return value is None or value == ""


def normalize_values(values: Sequence[Sequence[Any]]) -> tuple[list[Any], list[list[Any]]]:
    if not values:
        raise ValueError("Google sheet is empty")
    header = list(values[0]) + [""] * max(0, 25 - len(values[0]))
    if tuple(header[:16]) != LIVE_HEADERS or any(
        not is_blank_sheet_value(value) for value in header[16:25]
    ):
        raise ValueError("Google header contract mismatch")
    rows: list[list[Any]] = []
    blank_seen = False
    for raw in values[1:]:
        row = list(raw) + [""] * max(0, 25 - len(raw))
        filled = any(not is_blank_sheet_value(value) for value in row[:16])
        if not filled:
            blank_seen = True
            if any(not is_blank_sheet_value(value) for value in row[16:25]):
                raise ValueError("Q:Y contain data")
            continue
        if blank_seen:
            raise ValueError("Google data rows are not contiguous")
        if any(not is_blank_sheet_value(value) for value in row[16:25]):
            raise ValueError("Q:Y contain data")
        rows.append(row[:16])
    return header, rows


def load_calls(
    db_path: Path, manager_map: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Mapping[str, Any]]]:
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    connection.execute("PRAGMA busy_timeout=30000")
    if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
        raise RuntimeError("working SQLite quick_check failed")
    if str(connection.execute("PRAGMA journal_mode").fetchone()[0]).lower() != "wal":
        raise RuntimeError("working SQLite is not WAL")
    rows = connection.execute(
        "SELECT id,source_call_id,started_at,phone,manager_name,direction,duration_sec,"
        "analysis_json,transcript_variants_json,transcript_text,analysis_status,sync_status "
        "FROM call_records ORDER BY id"
    ).fetchall()
    connection.close()
    calls: dict[str, Any] = {}
    identities: dict[str, Any] = {}
    errors: dict[str, Mapping[str, Any]] = {}
    for raw in rows:
        record = dict(raw)
        try:
            identity = call_identity(record)
        except (ValueError, TypeError, OverflowError) as exc:
            errors[str(record.get("id"))] = {
                "call_key": "",
                "code": validation_error_code("identity", exc),
            }
            continue
        if identity["call_key"] in identities:
            raise RuntimeError("duplicate stable call_key")
        identities[identity["call_key"]] = identity
        if identity["analysis_status"] != "done":
            continue
        try:
            call = call_projection(record, manager_map)
        except (ValueError, TypeError, OverflowError) as exc:
            errors[str(record.get("id"))] = {
                "call_key": identity["call_key"],
                "code": validation_error_code("projection", exc),
            }
            continue
        calls[call["call_key"]] = call
    return calls, identities, errors


def default_state(destination_id: str) -> dict[str, Any]:
    return {
        "schema_version": STATE_SCHEMA,
        "destination_id": destination_id,
        "entries": {},
        "data_errors": {},
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def bootstrap_entries(
    rows: Sequence[Sequence[Any]],
    call_to_row: Mapping[str, int],
    identities: Mapping[str, Mapping[str, Any]],
    calls: Mapping[str, Mapping[str, Any]],
    *,
    now: str,
) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    numbers: set[int] = set()
    for call_key, index in call_to_row.items():
        raw_number = float(rows[index][0])
        number = int(raw_number)
        if raw_number != number or number <= 0 or number in numbers:
            raise RuntimeError("display numbers are invalid or duplicated")
        numbers.add(number)
        identity = identities[call_key]
        current = calls.get(call_key)
        exact_current = bool(
            current
            and physical_row_hash(rows[index])
            == physical_row_hash(desired_row(current, number))
        )
        entries[call_key] = {
            "display_number": number,
            "status": "verified",
            "projection_version": (
                PROJECTION_VERSION if exact_current else "legacy_bootstrap"
            ),
            "source_fingerprint": (
                current["source_fingerprint"] if current else canonical_hash(identity)
            ),
            "started_epoch": int(identity["started_epoch"]),
            "planned_row_sha256": None,
            "last_verified_row_sha256": physical_row_hash(rows[index]),
            "attempts": 0,
            "verified_at": now,
        }
    return entries


def load_state(path: Path, destination_id: str, *, required: bool) -> dict[str, Any]:
    if not path.exists():
        if required:
            raise RuntimeError("publisher state is missing; bootstrap is required")
        return default_state(destination_id)
    state = dict(owner_json(path))
    if state.get("schema_version") != STATE_SCHEMA or state.get("destination_id") != destination_id:
        raise RuntimeError("publisher state contract mismatch")
    if not isinstance(state.get("entries"), Mapping):
        raise RuntimeError("publisher state entries are invalid")
    entries = {str(key): dict(value) for key, value in state["entries"].items()}
    seen_numbers: set[int] = set()
    for call_key, entry in entries.items():
        if not CALL_KEY_RE.fullmatch(call_key):
            raise RuntimeError("publisher state call_key is invalid")
        if str(entry.get("status") or "") not in STATE_STATUSES:
            raise RuntimeError("publisher state status is invalid")
        number = entry.get("display_number")
        if isinstance(number, bool) or not isinstance(number, int) or not 0 < number < 1_000_000:
            raise RuntimeError("publisher state display_number is invalid")
        if number in seen_numbers:
            raise RuntimeError("publisher state display_number is duplicated")
        seen_numbers.add(number)
        attempts = entry.get("attempts", 0)
        if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 0:
            raise RuntimeError("publisher state attempts is invalid")
        epoch = entry.get("started_epoch")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= 0:
            raise RuntimeError("publisher state started_epoch is invalid")
        for field in ("source_fingerprint", "planned_row_sha256", "last_verified_row_sha256"):
            value = entry.get(field)
            if value is not None and not HASH_RE.fullmatch(str(value)):
                raise RuntimeError(f"publisher state {field} is invalid")
        if entry["status"] == "reserved" and not entry.get("planned_row_sha256"):
            raise RuntimeError("reserved publisher state has no planned row hash")
        if entry["status"] == "verified" and (
            entry.get("planned_row_sha256") is not None
            or not entry.get("last_verified_row_sha256")
        ):
            raise RuntimeError("verified publisher state is inconsistent")
    state["entries"] = entries
    if not isinstance(state.get("data_errors", {}), Mapping):
        raise RuntimeError("publisher state data_errors are invalid")
    state["data_errors"] = dict(state.get("data_errors") or {})
    return state


def reconcile(
    rows: Sequence[Sequence[Any]], identities: Mapping[str, Mapping[str, Any]],
    state: Mapping[str, Any]
) -> tuple[dict[str, int], dict[int, str]]:
    entries = state.get("entries") if isinstance(state.get("entries"), Mapping) else {}
    hash_index: dict[str, set[str]] = {}
    for call_key, raw in entries.items():
        if not isinstance(raw, Mapping):
            continue
        for field in ("planned_row_sha256", "last_verified_row_sha256"):
            value = str(raw.get(field) or "")
            if value:
                hash_index.setdefault(value, set()).add(str(call_key))
    business_index: dict[tuple[str, str], set[str]] = {}
    for call_key, identity in identities.items():
        transcript_sha = str(identity.get("transcript_sha") or "")
        if transcript_sha:
            business_index.setdefault(
                (normalized_phone(identity.get("phone")), transcript_sha), set()
            ).add(call_key)
    call_to_row: dict[str, int] = {}
    row_to_call: dict[int, str] = {}
    for index, row in enumerate(rows):
        row_hash = physical_row_hash(row)
        hash_candidates = set(hash_index.get(row_hash, set()))
        candidates = hash_candidates
        if not candidates:
            try:
                identity = sheet_identity(row)
                candidates = {
                    key
                    for key in business_index.get(
                        (identity["phone"], identity["transcript_sha"]), set()
                    )
                    if identity_matches(identity, identities[key])
                }
            except (TypeError, ValueError) as exc:
                display_number = row[0] if row else ""
                raise RuntimeError(
                    f"{exc} at Google row {index + 2} (№ {display_number})"
                ) from exc
        if len(candidates) != 1:
            display_number = row[0] if row else ""
            raise RuntimeError(
                "unidentified_or_ambiguous_physical_row "
                f"at Google row {index + 2} (№ {display_number})"
            )
        call_key = next(iter(candidates))
        if call_key in call_to_row:
            first_index = call_to_row[call_key]
            display_number = row[0] if row else ""
            raise RuntimeError(
                "duplicate_physical_call at Google rows "
                f"{first_index + 2} and {index + 2} (№ {display_number})"
            )
        call_to_row[call_key] = index
        row_to_call[index] = call_key
    return call_to_row, row_to_call


def desired_row(call: Mapping[str, Any], display_number: int) -> list[Any]:
    if not 0 < display_number < 1_000_000:
        raise ValueError("display_number is outside the safe Q range")
    return [display_number, *call["tail"]]


def reserve(
    state: dict[str, Any], calls: Mapping[str, Mapping[str, Any]], rows: Sequence[Sequence[Any]],
    call_to_row: Mapping[str, int], *, limit: int
) -> tuple[dict[str, Any], list[str]]:
    entries: dict[str, dict[str, Any]] = state["entries"]
    used_numbers = {int(entry["display_number"]) for entry in entries.values() if entry.get("display_number")}
    due: list[tuple[int, int, str]] = []
    for call_key, call in calls.items():
        entry = entries.get(call_key)
        if entry:
            number = int(entry["display_number"])
            planned = desired_row(call, number)
            current = rows[call_to_row[call_key]] if call_key in call_to_row else None
            if (
                current is None
                or physical_row_hash(current) != physical_row_hash(planned)
                or entry.get("source_fingerprint") != call["source_fingerprint"]
                or entry.get("projection_version") != PROJECTION_VERSION
            ):
                priority = (
                    0
                    if entry.get("status") == "reserved"
                    else 1
                    if current is None
                    else 3
                )
                due.append((priority, int(call["id"]), call_key))
        elif call_key not in call_to_row:
            due.append((2, int(call["id"]), call_key))
    selected = [item[2] for item in sorted(due)[:limit]]
    next_number = max(used_numbers, default=0) + 1
    now = datetime.now(timezone.utc).isoformat()
    for call_key in selected:
        call = calls[call_key]
        entry = entries.get(call_key, {})
        if not entry.get("display_number"):
            while next_number in used_numbers:
                next_number += 1
            entry["display_number"] = next_number
            used_numbers.add(next_number)
            next_number += 1
        planned = desired_row(call, int(entry["display_number"]))
        entry.update(
            {
                "status": "reserved",
                "projection_version": PROJECTION_VERSION,
                "source_fingerprint": call["source_fingerprint"],
                "started_epoch": int(call["started_epoch"]),
                "planned_row_sha256": physical_row_hash(planned),
                "attempts": int(entry.get("attempts") or 0) + 1,
                "reserved_at": now,
            }
        )
        entries[call_key] = entry
    state["updated_at"] = now
    return state, selected


def applied_reservations(
    state: Mapping[str, Any], calls: Mapping[str, Mapping[str, Any]],
    rows: Sequence[Sequence[Any]], call_to_row: Mapping[str, int],
) -> list[str]:
    """Return prior unknown-result writes whose exact row is now visible."""
    recovered: list[str] = []
    for call_key, entry in state["entries"].items():
        if entry.get("status") != "reserved" or call_key not in call_to_row or call_key not in calls:
            continue
        call = calls[call_key]
        row = rows[call_to_row[call_key]]
        if (
            physical_row_hash(row) == entry.get("planned_row_sha256")
            and call["source_fingerprint"] == entry.get("source_fingerprint")
        ):
            recovered.append(call_key)
    return recovered


def google_cell(value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"userEnteredValue": {"boolValue": value}}
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return {"userEnteredValue": {"numberValue": value}}
    return {"userEnteredValue": {"stringValue": str(value or "")}}


def google_row(values: Sequence[Any]) -> dict[str, Any]:
    return {"values": [google_cell(value) for value in values]}


def height_requests(
    sheet_id: int,
    heights: Sequence[int],
    current_heights: Optional[Sequence[int]] = None,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    start = 0
    while start < len(heights):
        if (
            current_heights is not None
            and start < len(current_heights)
            and current_heights[start] == heights[start]
        ):
            start += 1
            continue
        end = start + 1
        while (
            end < len(heights)
            and heights[end] == heights[start]
            and (
                current_heights is None
                or end >= len(current_heights)
                or current_heights[end] != heights[end]
            )
        ):
            end += 1
        result.append(
            {
                "updateDimensionProperties": {
                    "range": {
                        "sheetId": sheet_id,
                        "dimension": "ROWS",
                        "startIndex": start + 1,
                        "endIndex": end + 1,
                    },
                    "properties": {"pixelSize": heights[start]},
                    "fields": "pixelSize",
                }
            }
        )
        start = end
    return result


def ledger_started_epoch(
    call_key: str, identities: Mapping[str, Mapping[str, Any]],
    entries: Mapping[str, Mapping[str, Any]],
) -> int:
    identity_epoch = int((identities.get(call_key) or {}).get("started_epoch") or 0)
    return identity_epoch if identity_epoch > 0 else int(entries[call_key]["started_epoch"])


def build_batch(
    *, sheet_id: int, rows: Sequence[Sequence[Any]], row_to_call: Mapping[int, str],
    calls: Mapping[str, Mapping[str, Any]], identities: Mapping[str, Mapping[str, Any]],
    state: Mapping[str, Any], selected: Sequence[str], summary_width_px: int,
    current_heights: Optional[Sequence[int]] = None,
) -> tuple[list[dict[str, Any]], list[list[Any]]]:
    entries: Mapping[str, Mapping[str, Any]] = state["entries"]
    selected_set = set(selected)
    requests: list[dict[str, Any]] = []
    physical_keys = [row_to_call[index] for index in range(len(rows))]
    physical_rows = {key: list(rows[index]) for index, key in row_to_call.items()}
    for index, call_key in enumerate(physical_keys):
        if call_key not in selected_set:
            continue
        row = desired_row(calls[call_key], int(entries[call_key]["display_number"]))
        requests.append(
            {
                "updateCells": {
                    "range": {
                        "sheetId": sheet_id,
                        "startRowIndex": index + 1,
                        "endRowIndex": index + 2,
                        "startColumnIndex": 0,
                        "endColumnIndex": 16,
                    },
                    "rows": [google_row(row)],
                    "fields": "userEnteredValue",
                }
            }
        )
    appended = [key for key in selected if key not in physical_keys]
    if appended:
        requests.append(
            {
                "appendCells": {
                    "sheetId": sheet_id,
                    "rows": [
                        google_row(desired_row(calls[key], int(entries[key]["display_number"])))
                        for key in appended
                    ],
                    "fields": "userEnteredValue",
                }
            }
        )
        physical_keys.extend(appended)
    q_values = [
        ledger_started_epoch(key, identities, entries) * 1_000_000
        + int(entries[key]["display_number"])
        for key in physical_keys
    ]
    if any(value >= 2**53 for value in q_values):
        raise ValueError("Q sort key is outside exact IEEE-754 integer range")
    requests.extend(
        [
            {
                "updateCells": {
                    "start": {"sheetId": sheet_id, "rowIndex": 1, "columnIndex": 16},
                    "rows": [google_row([value]) for value in q_values],
                    "fields": "userEnteredValue",
                }
            },
            {
                "sortRange": {
                    "range": {
                        "sheetId": sheet_id,
                        "startRowIndex": 1,
                        "endRowIndex": len(physical_keys) + 1,
                        "startColumnIndex": 0,
                        "endColumnIndex": 17,
                    },
                    "sortSpecs": [{"dimensionIndex": 16, "sortOrder": "DESCENDING"}],
                }
            },
            {
                "repeatCell": {
                    "range": {
                        "sheetId": sheet_id,
                        "startRowIndex": 1,
                        "endRowIndex": len(physical_keys) + 1,
                        "startColumnIndex": 16,
                        "endColumnIndex": 17,
                    },
                    "cell": {},
                    "fields": "userEnteredValue",
                }
            },
        ]
    )
    sorted_keys = sorted(
        physical_keys,
        key=lambda key: (
            ledger_started_epoch(key, identities, entries),
            int(entries[key]["display_number"]),
        ),
        reverse=True,
    )
    final_rows = [
        desired_row(calls[key], int(entries[key]["display_number"]))
        if key in selected_set
        else physical_rows[key]
        for key in sorted_keys
    ]
    data_range = {
        "sheetId": sheet_id,
        "startRowIndex": 1,
        "endRowIndex": len(final_rows) + 1,
    }
    requests.extend(
        [
            {
                "repeatCell": {
                    "range": {**data_range, "startColumnIndex": 9, "endColumnIndex": 10},
                    "cell": {"userEnteredFormat": {"wrapStrategy": "WRAP", "verticalAlignment": "TOP"}},
                    "fields": "userEnteredFormat.wrapStrategy,userEnteredFormat.verticalAlignment",
                }
            },
            {
                "repeatCell": {
                    "range": {**data_range, "startColumnIndex": 15, "endColumnIndex": 16},
                    "cell": {"userEnteredFormat": {"wrapStrategy": "CLIP", "verticalAlignment": "TOP"}},
                    "fields": "userEnteredFormat.wrapStrategy,userEnteredFormat.verticalAlignment",
                }
            },
            {
                "updateDimensionProperties": {
                    "range": {
                        "sheetId": sheet_id,
                        "dimension": "COLUMNS",
                        "startIndex": 9,
                        "endIndex": 10,
                    },
                    "properties": {"pixelSize": summary_width_px},
                    "fields": "pixelSize",
                }
            },
        ]
    )
    requests.extend(
        height_requests(
            sheet_id,
            [row_height(row[9]) for row in final_rows],
            current_heights,
        )
    )
    return requests, final_rows


class LiveGoogleGateway(GoogleGateway):
    def values(self, title: str) -> Sequence[Sequence[Any]]:
        range_name = quote(f"'{title}'!A1:Y")
        response = self.session.get(
            f"https://sheets.googleapis.com/v4/spreadsheets/{self.spreadsheet_id}/values/{range_name}",
            params={"majorDimension": "ROWS", "valueRenderOption": "UNFORMATTED_VALUE"},
            timeout=120,
        )
        return self._json(response).get("values") or ()

    def layout(self, title: str, last_row: int) -> Mapping[str, Any]:
        response = self.session.get(
            f"https://sheets.googleapis.com/v4/spreadsheets/{self.spreadsheet_id}",
            params={
                "ranges": f"'{title}'!A2:P{last_row}",
                "includeGridData": "true",
                "fields": (
                    "sheets(data(startRow,startColumn,columnMetadata(pixelSize),"
                    "rowMetadata(pixelSize),rowData(values(userEnteredValue(formulaValue),"
                    "userEnteredFormat(wrapStrategy,verticalAlignment)))))"
                ),
            },
            timeout=120,
        )
        return self._json(response)

    def batch_sheet_requests(self, requests: Sequence[Mapping[str, Any]]) -> None:
        response = self.session.post(
            f"https://sheets.googleapis.com/v4/spreadsheets/{self.spreadsheet_id}:batchUpdate",
            json={"requests": list(requests)},
            timeout=180,
        )
        self._json(response)


def validate_sheet_target(
    sheets: Sequence[Mapping[str, Any]], *, title: str, sheet_id: int
) -> None:
    title_matches = [item for item in sheets if str(item.get("title") or "") == title]
    id_matches = [item for item in sheets if int(item.get("sheetId", -1)) == sheet_id]
    if (
        len(title_matches) != 1
        or len(id_matches) != 1
        or int(title_matches[0].get("sheetId", -1)) != sheet_id
        or str(id_matches[0].get("title") or "") != title
    ):
        raise RuntimeError("Google sheet title/id target mismatch")


def verify_layout(payload: Mapping[str, Any], rows: Sequence[Sequence[Any]], width: int) -> None:
    sheets = payload.get("sheets") or ()
    data = (sheets[0].get("data") or ())[0] if sheets else {}
    columns = data.get("columnMetadata") or ()
    if len(columns) < 10 or int(columns[9].get("pixelSize") or 0) != width:
        raise RuntimeError("summary column width readback mismatch")
    metadata = data.get("rowMetadata") or ()
    row_data = data.get("rowData") or ()
    if len(metadata) < len(rows) or len(row_data) < len(rows):
        raise RuntimeError("layout readback is incomplete")
    for index, row in enumerate(rows):
        if int(metadata[index].get("pixelSize") or 0) != row_height(row[9]):
            raise RuntimeError(
                f"row height readback mismatch at Google row {index + 2} (№ {row[0]})"
            )
        values = row_data[index].get("values") or ()
        if len(values) < 16:
            raise RuntimeError("J/P format readback is incomplete")
        for column_index, cell in enumerate(values[:16]):
            entered = cell.get("userEnteredValue") or {}
            if "formulaValue" in entered:
                column = chr(ord("A") + column_index)
                raise RuntimeError(
                    "Google formulas are forbidden in A:P at "
                    f"{column}{index + 2} (№ {row[0]})"
                )
        j_format = values[9].get("userEnteredFormat") or {}
        p_format = values[15].get("userEnteredFormat") or {}
        if j_format.get("wrapStrategy") != "WRAP" or j_format.get("verticalAlignment") != "TOP":
            raise RuntimeError(
                f"summary format readback mismatch at Google row {index + 2} (№ {row[0]})"
            )
        if p_format.get("wrapStrategy") != "CLIP" or p_format.get("verticalAlignment") != "TOP":
            raise RuntimeError(
                f"transcript format readback mismatch at Google row {index + 2} (№ {row[0]})"
            )


def layout_row_heights(payload: Mapping[str, Any], row_count: int) -> list[int]:
    sheets = payload.get("sheets") or ()
    data = (sheets[0].get("data") or ())[0] if sheets else {}
    metadata = data.get("rowMetadata") or ()
    if len(metadata) < row_count:
        raise RuntimeError("layout readback is incomplete")
    return [int(metadata[index].get("pixelSize") or 0) for index in range(row_count)]


def load_config(path: Path, *, execute: bool) -> dict[str, Any]:
    payload = dict(owner_json(path))
    if payload.get("schema_version") != CONFIG_SCHEMA:
        raise RuntimeError("publisher config schema mismatch")
    required = (
        "spreadsheet_id", "sheet_id", "sheet_title", "working_db", "manager_identity",
        "credentials", "state", "lock", "summary_width_px",
    )
    if any(payload.get(field) in (None, "") for field in required):
        raise RuntimeError("publisher config is incomplete")
    payload["credentials_info"] = dict(validate_credentials(Path(str(payload["credentials"]))))
    if execute:
        expected = str(payload.get("expected_code_sha") or "")
        head = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True).strip()
        if len(expected) != 40 or head != expected or dirty:
            raise RuntimeError("publisher code revision mismatch or dirty worktree")
    return payload


def verify_sheet_snapshot(
    *, rows: Sequence[Sequence[Any]], call_to_row: Mapping[str, int],
    identities: Mapping[str, Mapping[str, Any]], state: Mapping[str, Any],
    exact_keys: Sequence[str] = (), unchanged_hashes: Optional[Mapping[str, str]] = None,
    recoverable_keys: Sequence[str] = (),
) -> None:
    entries: Mapping[str, Mapping[str, Any]] = state["entries"]
    missing_entries = set(call_to_row) - set(entries)
    if missing_entries:
        raise RuntimeError("physical Google row is missing from publisher state")
    missing_from_google = set(entries) - set(call_to_row)
    missing_verified = {
        key for key in missing_from_google if entries[key].get("status") == "verified"
    }
    unrecoverable = missing_verified - set(recoverable_keys)
    if unrecoverable:
        raise RuntimeError("verified Google row is missing and source is not publishable")
    actual_order = [key for key, _ in sorted(call_to_row.items(), key=lambda item: item[1])]
    expected_order = sorted(
        actual_order,
        key=lambda key: (
            ledger_started_epoch(key, identities, entries),
            int(entries[key]["display_number"]),
        ),
        reverse=True,
    )
    if actual_order != expected_order:
        raise RuntimeError("Google newest-first readback mismatch")
    for call_key in exact_keys:
        if call_key not in call_to_row:
            raise RuntimeError("Google exact row is missing")
        planned = entries[call_key].get("planned_row_sha256")
        if not planned or physical_row_hash(rows[call_to_row[call_key]]) != planned:
            raise RuntimeError("Google exact row readback mismatch")
    for call_key, before_hash in (unchanged_hashes or {}).items():
        if call_key not in call_to_row:
            raise RuntimeError("unselected Google row disappeared")
        if physical_row_hash(rows[call_to_row[call_key]]) != before_hash:
            raise RuntimeError("unselected Google row changed concurrently")


def write_sync_done(
    db_path: Path, expected: Mapping[str, Mapping[str, Any]],
    manager_map: Mapping[str, Any],
) -> None:
    if not expected:
        return
    connection = sqlite3.connect(db_path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout=30000")
    try:
        connection.execute("BEGIN IMMEDIATE")
        for call_key, proof in expected.items():
            row = connection.execute(
                "SELECT id,source_call_id,started_at,phone,manager_name,direction,duration_sec,"
                "analysis_json,transcript_variants_json,transcript_text,analysis_status,sync_status "
                "FROM call_records WHERE id=?",
                (int(proof["id"]),),
            ).fetchone()
            if row is None:
                raise RuntimeError("verified source row disappeared")
            current = call_projection(dict(row), manager_map)
            if (
                current["call_key"] != call_key
                or current["source_fingerprint"] != proof["source_fingerprint"]
            ):
                raise RuntimeError("source changed before sync commit")
            if current["sync_status"] == "done":
                continue
            cursor = connection.execute(
                "UPDATE call_records SET sync_status='done', sync_attempts=sync_attempts+1 "
                "WHERE id=? AND analysis_status='done' AND sync_status=?",
                (int(current["id"]), current["sync_status"]),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("sync commit ownership check failed")
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def finalize_verified(
    state: dict[str, Any], selected: Sequence[str], calls: Mapping[str, Mapping[str, Any]],
    physical_rows: Mapping[str, Sequence[Any]], state_path: Path,
) -> dict[str, Mapping[str, Any]]:
    now = datetime.now(timezone.utc).isoformat()
    verified: dict[str, Mapping[str, Any]] = {}
    for call_key in selected:
        entry = state["entries"][call_key]
        current = calls[call_key]
        row = physical_rows.get(call_key)
        if (
            row is None
            or current["source_fingerprint"] != entry.get("source_fingerprint")
            or physical_row_hash(row) != entry.get("planned_row_sha256")
        ):
            raise RuntimeError("source or readback changed before verified commit")
        entry.update(
            {
                "status": "verified",
                "last_verified_row_sha256": entry["planned_row_sha256"],
                "planned_row_sha256": None,
                "verified_at": now,
            }
        )
        verified[call_key] = {
            "id": int(current["id"]),
            "source_fingerprint": current["source_fingerprint"],
        }
    state["updated_at"] = now
    atomic_owner_json(state_path, state)
    return verified


def read_manager_map(path: Path) -> Mapping[str, Any]:
    payload = owner_json(path)
    mapping = payload.get("mapping")
    return dict(mapping) if isinstance(mapping, Mapping) else {}


def data_error_counts(errors: Mapping[str, Mapping[str, Any]]) -> Mapping[str, int]:
    result: dict[str, int] = {}
    for error in errors.values():
        code = str(error.get("code") or "unknown")
        result[code] = result.get(code, 0) + 1
    return result


def sync_proofs(
    calls: Mapping[str, Mapping[str, Any]], rows: Sequence[Sequence[Any]],
    call_to_row: Mapping[str, int], state: Mapping[str, Any], keys: Sequence[str],
) -> dict[str, Mapping[str, Any]]:
    proofs: dict[str, Mapping[str, Any]] = {}
    for call_key in keys:
        call = calls.get(call_key)
        entry = state["entries"].get(call_key)
        if not call or not entry or call_key not in call_to_row:
            raise RuntimeError("verified call cannot be synchronized")
        row_hash = physical_row_hash(rows[call_to_row[call_key]])
        if (
            entry.get("status") != "verified"
            or entry.get("source_fingerprint") != call["source_fingerprint"]
            or entry.get("last_verified_row_sha256") != row_hash
        ):
            raise RuntimeError("verified call proof is inconsistent")
        proofs[call_key] = {
            "id": int(call["id"]),
            "source_fingerprint": call["source_fingerprint"],
        }
    return proofs


def run(argv: Optional[Sequence[str]] = None) -> Mapping[str, Any]:
    parser = argparse.ArgumentParser(description="Publish the production Mango calls Google sheet")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirmation", default="")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    if args.bootstrap and args.execute:
        raise RuntimeError("bootstrap and execute are mutually exclusive")
    if args.bootstrap and args.confirmation != BOOTSTRAP_CONFIRMATION:
        raise RuntimeError("bootstrap requires explicit confirmation")
    if args.execute and args.confirmation != CONFIRMATION:
        raise RuntimeError("execute requires explicit confirmation")
    config = load_config(args.config, execute=args.execute or args.bootstrap)
    state_path = Path(str(config["state"]))
    db_path = Path(str(config["working_db"]))
    manager_path = Path(str(config["manager_identity"]))
    sheet_title = str(config["sheet_title"])
    sheet_id = int(config["sheet_id"])
    destination = (
        f"google_sheets:v1:{config['spreadsheet_id']}:{sheet_id}:mango:production"
    )
    with publication_lock(Path(str(config["lock"]))):
        manager_map = read_manager_map(manager_path)
        calls, identities, data_errors = load_calls(db_path, manager_map)
        state = load_state(state_path, destination, required=args.execute)
        gateway = LiveGoogleGateway(
            authorized_session(config["credentials_info"]), str(config["spreadsheet_id"])
        )
        validate_sheet_target(gateway.sheets(), title=sheet_title, sheet_id=sheet_id)
        _header, rows = normalize_values(gateway.values(sheet_title))
        call_to_row, row_to_call = reconcile(rows, identities, state)
        initial_layout = gateway.layout(sheet_title, len(rows) + 1)
        verify_layout(initial_layout, rows, int(config["summary_width_px"]))
        current_heights = layout_row_heights(initial_layout, len(rows))

        if args.bootstrap:
            if state_path.exists() and state["entries"]:
                raise RuntimeError("bootstrap refuses to replace a non-empty publisher state")
            now = datetime.now(timezone.utc).isoformat()
            entries = bootstrap_entries(
                rows, call_to_row, identities, calls, now=now
            )
            state["entries"] = entries
            state["data_errors"] = data_errors
            state["updated_at"] = now
            verify_sheet_snapshot(
                rows=rows, call_to_row=call_to_row, identities=identities, state=state,
                recoverable_keys=calls,
            )
            atomic_owner_json(state_path, state)
            load_state(state_path, destination, required=True)
            return {
                "status": "bootstrapped",
                "google_rows": len(rows),
                "matched": len(call_to_row),
                "data_errors": len(data_errors),
                "data_error_codes": data_error_counts(data_errors),
                "external_write": False,
            }

        if not args.execute:
            entries = state["entries"]
            audit_state = state
            if not entries:
                audit_state = dict(state)
                audit_state["entries"] = bootstrap_entries(
                    rows,
                    call_to_row,
                    identities,
                    calls,
                    now=datetime.now(timezone.utc).isoformat(),
                )
                entries = audit_state["entries"]
            verify_sheet_snapshot(
                rows=rows,
                call_to_row=call_to_row,
                identities=identities,
                state=audit_state,
                recoverable_keys=calls,
            )
            stale = sum(
                1
                for key, call in calls.items()
                if key in call_to_row
                and key in entries
                and physical_row_hash(rows[call_to_row[key]])
                != physical_row_hash(desired_row(call, int(entries[key]["display_number"])))
            )
            return {
                "status": "shadow_ok",
                "google_rows": len(rows),
                "analysis_done": len(calls),
                "matched": len(call_to_row),
                "missing": sum(key not in call_to_row for key in calls),
                "stale": stale,
                "data_errors": len(data_errors),
                "data_error_codes": data_error_counts(data_errors),
                "external_write": False,
            }

        verify_sheet_snapshot(
            rows=rows, call_to_row=call_to_row, identities=identities, state=state,
            recoverable_keys=calls,
        )
        recovered = applied_reservations(state, calls, rows, call_to_row)
        if recovered:
            verify_sheet_snapshot(
                rows=rows, call_to_row=call_to_row, identities=identities,
                state=state, exact_keys=recovered, recoverable_keys=calls,
            )
            verify_layout(
                gateway.layout(sheet_title, len(rows) + 1), rows,
                int(config["summary_width_px"]),
            )
            _fresh_header, fresh_rows = normalize_values(gateway.values(sheet_title))
            fresh_call_to_row, row_to_call = reconcile(fresh_rows, identities, state)
            verify_sheet_snapshot(
                rows=fresh_rows, call_to_row=fresh_call_to_row, identities=identities,
                state=state, exact_keys=recovered, recoverable_keys=calls,
            )
            rows, call_to_row = fresh_rows, fresh_call_to_row
            physical = {key: rows[index] for key, index in call_to_row.items()}
            manager_map = read_manager_map(manager_path)
            calls, identities, data_errors = load_calls(db_path, manager_map)
            state["data_errors"] = data_errors
            recovered_proofs = finalize_verified(
                state, recovered, calls, physical, state_path
            )
            write_sync_done(db_path, recovered_proofs, manager_map)

        configured_limit = config.get("batch_limit")
        limit = int(
            args.limit
            if args.limit is not None
            else configured_limit
            if configured_limit is not None
            else 25
        )
        if not 1 <= limit <= 25:
            raise RuntimeError("batch limit must be between 1 and 25")
        state["data_errors"] = data_errors
        state, selected = reserve(state, calls, rows, call_to_row, limit=limit)
        if not selected:
            pending = [
                key for key, call in calls.items()
                if key in call_to_row
                and state["entries"].get(key, {}).get("status") == "verified"
                and call["sync_status"] != "done"
            ]
            if pending:
                verify_layout(
                    gateway.layout(sheet_title, len(rows) + 1), rows,
                    int(config["summary_width_px"]),
                )
                _sync_header, sync_rows = normalize_values(gateway.values(sheet_title))
                sync_call_to_row, _sync_row_to_call = reconcile(sync_rows, identities, state)
                verify_sheet_snapshot(
                    rows=sync_rows, call_to_row=sync_call_to_row,
                    identities=identities, state=state, recoverable_keys=calls,
                )
                rows, call_to_row = sync_rows, sync_call_to_row
            proofs = sync_proofs(calls, rows, call_to_row, state, pending)
            atomic_owner_json(state_path, state)
            write_sync_done(db_path, proofs, manager_map)
            return {
                "status": "no_change",
                "google_rows": len(rows),
                "data_errors": len(data_errors),
                "data_error_codes": data_error_counts(data_errors),
                "external_write": False,
            }

        selected_set = set(selected)
        unchanged = {
            key: physical_row_hash(rows[index])
            for key, index in call_to_row.items()
            if key not in selected_set
        }
        atomic_owner_json(state_path, state)
        requests, expected_rows = build_batch(
            sheet_id=sheet_id,
            rows=rows,
            row_to_call=row_to_call,
            calls=calls,
            identities=identities,
            state=state,
            selected=selected,
            summary_width_px=int(config["summary_width_px"]),
            current_heights=current_heights,
        )
        gateway.batch_sheet_requests(requests)
        _post_header, post_rows = normalize_values(gateway.values(sheet_title))
        post_call_to_row, _post_row_to_call = reconcile(post_rows, identities, state)
        if len(post_rows) != len(expected_rows):
            raise RuntimeError("Google row count readback mismatch")
        verify_sheet_snapshot(
            rows=post_rows, call_to_row=post_call_to_row, identities=identities,
            state=state, exact_keys=selected, unchanged_hashes=unchanged,
            recoverable_keys=calls,
        )
        if [physical_row_hash(row) for row in post_rows] != [
            physical_row_hash(row) for row in expected_rows
        ]:
            raise RuntimeError("Google full row readback mismatch")
        verify_layout(
            gateway.layout(sheet_title, len(post_rows) + 1), post_rows,
            int(config["summary_width_px"]),
        )
        _final_header, final_rows = normalize_values(gateway.values(sheet_title))
        final_call_to_row, _final_row_to_call = reconcile(final_rows, identities, state)
        verify_sheet_snapshot(
            rows=final_rows, call_to_row=final_call_to_row, identities=identities,
            state=state, exact_keys=selected, unchanged_hashes=unchanged,
            recoverable_keys=calls,
        )
        if [physical_row_hash(row) for row in final_rows] != [
            physical_row_hash(row) for row in expected_rows
        ]:
            raise RuntimeError("Google final full readback mismatch")
        physical = {key: final_rows[index] for key, index in final_call_to_row.items()}
        latest_manager_map = read_manager_map(manager_path)
        latest_calls, _latest_identities, latest_errors = load_calls(
            db_path, latest_manager_map
        )
        state["data_errors"] = latest_errors
        verified = finalize_verified(
            state, selected, latest_calls, physical, state_path
        )
        write_sync_done(db_path, verified, latest_manager_map)
        return {
            "status": "published",
            "published": len(selected),
            "google_rows": len(final_rows),
            "requests": len(requests),
            "data_errors": len(latest_errors),
            "data_error_codes": data_error_counts(latest_errors),
            "external_write": True,
        }


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        report = run(argv)
    except Exception as exc:  # noqa: BLE001 - structured fail-closed CLI
        print(
            json.dumps(
                {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 1
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
