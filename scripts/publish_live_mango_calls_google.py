#!/usr/bin/env python3
"""Deterministic publisher for the production Mango calls Google sheet.

Dry-run is the default.  The script owns only the Google projection and the
compatibility ``sync_status`` marker; it never runs or claims pipeline stages.

Write policy, explicitly:

* Google and the working SQLite are written only under ``--execute`` with the
  exact confirmation phrase; a dry run reports ``external_write: false`` and
  issues zero ``batchUpdate`` calls.
* The owner-only ``state`` sidecar (local file, mode 0600) is the single
  exception: a dry run may record durable incidents there, because losing the
  reason a call could not be published is worse than an owner-local write.  The
  sidecar holds hashes, codes and timestamps — no transcript and no personal
  data — so it is a journal, not a publication.
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
from contextlib import contextmanager
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
from mango_mvp.quality.tenant_text_normalizer import (  # noqa: E402
    TENANT_TEXT_ENGINE_VERSION,
    tenant_ruleset_version,
)
from mango_mvp.services.dialogue_contract import (  # noqa: E402
    ANALYSIS_REASON_RU,
    ANALYSIS_SCHEMA_VERSION_V3,
    CALLS_TENANT_ID,
    CLAIM_CONTRACT_VERSION,
    CONTRACT_VERSION as DIALOGUE_CONTRACT_VERSION,
    ROLE_GUARD_VERSION,
    ROLE_REASON_RU,
    TIMEZONE_CONTRACT_VERSION,
    DialogueContractError,
    build_dialogue_input,
    claim_review_sentence_ru,
    guard_stored_analysis,
    json_object,
    manager_claim_evidence_ru,
    manager_result_ru,
    moscow_datetime,
    safe_error_text,
)


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
# v2 adds durable classed ``incidents`` in place of the transient
# ``data_errors`` map; ``load_state`` still reads a v1 sidecar and migrates it.
STATE_SCHEMA = "mango_calls_live_google_state_v2"
STATE_SCHEMA_LEGACY = "mango_calls_live_google_state_v1"
# v5 = physical-side dialogue and the shared fail-closed analysis projector, plus
# an oversize transcript that shortens instead of dropping the row and a review
# flag that can no longer contradict its own reason column.
# v6 = the contract fingerprint below is part of every row's identity.
# v7 = pending/quarantined calls share the same closed publication path.
PROJECTION_VERSION = "mango_calls_live_google_projection_v8"
# Every code-side contract whose change alters what a published row *means*
# while the stored columns stay byte-identical: a new dialogue/role contract, a
# new claim contract, a new normalizer ruleset, a new timezone rule, a new
# projection.  Without it a version bump left the sheet "already correct".
CONTRACT_FINGERPRINT = {
    "dialogue": DIALOGUE_CONTRACT_VERSION,
    "role_guard": ROLE_GUARD_VERSION,
    "claim": CLAIM_CONTRACT_VERSION,
    "timezone": TIMEZONE_CONTRACT_VERSION,
    "normalizer_engine": TENANT_TEXT_ENGINE_VERSION,
    "normalizer_ruleset": tenant_ruleset_version(CALLS_TENANT_ID),
    "normalizer_tenant": CALLS_TENANT_ID,
    "projection": PROJECTION_VERSION,
}
# ТЗ-05 §D5: one computed report over the state that already exists — no new
# table and no new state machine.  Every finished analysis must land in exactly
# one of these, or the run refuses to touch Google.
BALANCE_CATEGORIES = (
    "verified_current",
    "published",
    "reserved",
    "quarantined",
    "failed_with_incident",
)
# The projection outputs a published row is actually built from.  Hashing these
# instead of the whole ``analysis_json`` keeps a re-analysis that changed
# nothing but ``analyzed_at`` out of the "this row is stale" set.
ANALYSIS_RESULT_KEYS = (
    "analysis_schema_version",
    "history_summary",
    "history_summary_meta",
    "structured_fields",
    "claim_evidence",
    "normalized_facts",
    "role_attribution",
    "needs_review",
    "review_reasons",
    "review_reasons_ru",
    "neutral_topics",
    "target_product",
    "interests",
    "objections",
    "next_step",
    "timeline",
    "follow_up_reason",
    "summary",
    "manager_brief",
    "history_short",
)
CONFIRMATION = "PUBLISH_MANGO_CALLS_LIVE"
BOOTSTRAP_CONFIRMATION = "BOOTSTRAP_MANGO_CALLS_LIVE"
MOSCOW = ZoneInfo("Europe/Moscow")
MAX_CELL_CHARS = 50_000
# A conversation longer than one Google cell is shortened to whole replies, with
# the gap marked and the row explicitly sent to review.  The note below is part
# of the cell, so the budget for replies is what is left after it.
TRANSCRIPT_OVERSIZE_NOTE = (
    "[ Полная расшифровка не помещается в ячейку Google. "
    "Целиком она хранится в рабочей базе звонков и в файле расшифровки; "
    "здесь показаны только целые реплики с начала и конца разговора. ]"
)
TRANSCRIPT_UNRENDERABLE_NOTE = (
    "[ Расшифровка не помещается в ячейку Google даже одной репликой. "
    "Полный текст остался в рабочей базе звонков и в файле расшифровки. ]"
)
TRANSCRIPT_OVERSIZE_REVIEW_RU = (
    "полная расшифровка не поместилась в ячейку, показана часть реплик"
)
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
    "Основание ключевых выводов",
    "Что проверить РОПу",
    "Полная расшифровка",
)
MANAGED_COLUMN_COUNT = len(LIVE_HEADERS)
TRANSCRIPT_COLUMN_INDEX = LIVE_HEADERS.index("Полная расшифровка")
REVIEW_COLUMN_INDEX = LIVE_HEADERS.index("Что проверить РОПу")
SORT_KEY_COLUMN_INDEX = MANAGED_COLUMN_COUNT
LINE_RE = re.compile(r"^\[([^]]+)]\s+([^:]+):\s*(.*)$")
# The pre-contract projection: it never stripped MANAGER:/CLIENT: prefixes, so
# rows written before this release stay identifiable byte for byte.
LEGACY_PHYSICAL_RE = re.compile(
    r"^(?:CHANNEL_(?:LEFT|RIGHT)|Дорожка\s+(?:левая|правая))\s*:\s*", re.IGNORECASE
)
LEGACY_TECHNICAL_RE = re.compile(r"^(?:source_call_id|sha(?:-?256)?)\s*:", re.IGNORECASE)
DURATION_RE = re.compile(r"^\s*(\d+)\s*мин\s*(\d+)\s*с\s*$", re.IGNORECASE)
INCIDENT_SLA_HOURS = 24
LEGACY_ERROR_FIELD = "data_errors"
INCIDENT_CODE_RE = re.compile(r"^[a-z0-9_]{3,64}$")
# One class per kind of scan.  A scan refreshes only its own class, so an empty
# result from one of them can never close an incident the other still owns.
INCIDENT_CLASS_DATA = "data"
INCIDENT_CLASS_RECONCILE = "reconcile"
INCIDENT_CLASSES = (INCIDENT_CLASS_DATA, INCIDENT_CLASS_RECONCILE)
UNKNOWN_REVIEW_REASON_RU = "техническая проверка звонка не пройдена"
SAFE_PENDING_RESULT_RU = "Анализ не завершён"
SAFE_ERROR_RESULT_RU = "Ошибка анализа"
SAFE_PENDING_SUMMARY_RU = "Смысловой анализ ещё не завершён; выводы пока не публикуются."
SAFE_ERROR_SUMMARY_RU = "Данные анализа повреждены или устарели; выводы скрыты."
SAFE_PENDING_TRANSCRIPT_RU = "[ Смысловой анализ ещё не завершён. ]"
SAFE_ERROR_TRANSCRIPT_RU = "[ Данные анализа требуют проверки; содержательные выводы скрыты. ]"


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class DurableIncidentError(RuntimeError):
    """A fatal condition that must survive the crash as a stored incident.

    Printing it on stderr and dying loses it the moment the terminal scrolls;
    the owner needs to see on the next run that this call has been broken since
    a specific moment, so the incident is written before the exception escapes.
    """

    incident_class = INCIDENT_CLASS_DATA

    def __init__(self, message: str, *, code: str, fingerprint: str) -> None:
        super().__init__(message)
        self.code = code
        self.fingerprint = fingerprint

    def incident_errors(self) -> dict[str, Mapping[str, Any]]:
        return {
            f"{self.incident_class}:{self.fingerprint[:32]}": {
                "call_digest": "",
                "code": self.code,
            }
        }


class ReconcileError(DurableIncidentError):
    """A physical Google row that cannot be tied to exactly one stored call."""

    incident_class = INCIDENT_CLASS_RECONCILE


class IdentityCollisionError(DurableIncidentError):
    """Two stored calls claim the same stable identity: nothing can be matched."""


def validation_error_code(prefix: str, exc: Exception) -> str:
    message = str(exc).lower()
    known = (
        ("malformed line", "dialogue_line_malformed"),
        ("empty text", "dialogue_line_empty_text"),
        ("timecodes go backwards", "dialogue_order_invalid"),
        ("dialogue_lines is not a list", "dialogue_lines_invalid"),
        ("transcript_variants_json", "variants_json_invalid"),
        ("full transcript is empty", "transcript_empty"),
        ("50000", "cell_too_long"),
        ("duration", "duration_invalid"),
        ("source_call_id is empty", "source_call_id_empty"),
        ("call_key is invalid", "source_call_id_invalid"),
        ("analysis_json", "analysis_json_invalid"),
        ("analysis schema is stale", "analysis_schema_stale"),
        ("analysis contract is stale", "analysis_contract_invalid"),
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
    moscow = moscow_datetime(started_utc)
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
    """The exact transcript cell the sheet holds; roles stay neutral until proven.

    Identity matching hashes the *published* cell, so this must be byte-equal to
    what ``call_projection`` writes — including the oversize shortening.
    """
    return transcript_cell(build_dialogue_input(record))[0]


def transcript_cell(dialogue: Any) -> tuple[str, bool]:
    """Fit the conversation into one Google cell without losing the call.

    A call whose transcript is longer than the cell limit used to raise, which
    dropped the entire row: no date, no manager, no summary, no reason — the
    call simply was not in the report.  That is the worst possible outcome for
    the longest conversations, which are exactly the ones worth reading.

    So the row survives.  Whole replies are kept from both ends of the
    conversation inside the hard limit, the gap carries a visible marker, and a
    note says plainly where the full text is.  Google is *not* claimed to hold
    anything beyond what is in the cell.  The second value is ``True`` when the
    cell is not the whole conversation, and the caller turns that into a review
    reason — a shortened transcript is never published as a complete one.
    """
    full = dialogue.render()
    if len(full) <= MAX_CELL_CHARS:
        return full, False
    budget = MAX_CELL_CHARS - len(TRANSCRIPT_OVERSIZE_NOTE) - 1
    try:
        shortened = dialogue.render(max_chars=budget)
    except DialogueContractError:
        # Not even one whole reply fits: a reply is never cut in the middle, so
        # the cell carries the note alone and the row still exists.
        return TRANSCRIPT_UNRENDERABLE_NOTE, True
    return f"{TRANSCRIPT_OVERSIZE_NOTE}\n{shortened}", True


def render_legacy_identity_transcript(record: Mapping[str, Any]) -> str:
    """Reproduce the old lenient projection only to identify an existing row.

    This is byte-compatible with the projection that actually wrote the live
    sheet, including its fallback branch: it strips only the physical channel
    prefixes the old regex knew and therefore keeps ``MANAGER:``/``CLIENT:``.
    It must never raise — an unidentifiable row is worse than a stale one.
    """
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
    full = variants.get("full")
    final = full.get("final") if isinstance(full, Mapping) else None
    cleaned: list[str] = []
    for raw_line in str(final or record.get("transcript_text") or "").strip().splitlines():
        line = raw_line.strip()
        if LEGACY_TECHNICAL_RE.match(line):
            continue
        line = LEGACY_PHYSICAL_RE.sub("", line).strip()
        if line:
            cleaned.append(line)
    fallback = " ".join(cleaned).strip()
    return f"[00:00.0] Не определено: {fallback}" if fallback else ""


# Exactly the non-analysis stored columns ``call_projection`` reads.  The
# business output of Analyse is hashed separately below; excluding the raw JSON
# keeps a telemetry-only ``analyzed_at`` change from rewriting Google.
SOURCE_FINGERPRINT_COLUMNS = (
    "source_call_id",
    "started_at",
    "duration_sec",
    "direction",
    "phone",
    "manager_name",
    "transcript_variants_json",
    "transcript_text",
    "analysis_status",
)


def source_field_digests(record: Mapping[str, Any]) -> dict[str, str]:
    """Per-field SHA-256 of the raw stored values: irreversible, no PII stored.

    Hashing the value with its JSON type keeps ``None``, ``""`` and ``0``
    distinct, so a field being cleared is a change like any other.
    """
    return {name: canonical_hash(record.get(name)) for name in SOURCE_FINGERPRINT_COLUMNS}


def manager_display(value: Any, mapping: Mapping[str, Any]) -> str:
    raw = str(value or "").strip()
    display = str(mapping.get(raw) or raw).strip()
    technical = display.lower().startswith(("mango_", "unknown", "неизвест", "тест"))
    return display if re.search(r"[А-Яа-яЁё]", display) and not technical else "Не определён"


def review_sentences_ru(analysis: Mapping[str, Any], flags: Mapping[str, Any]) -> str:
    """Whole Russian sentences for the sales head; a raw code never leaks.

    ``review_reasons_ru`` is what the fail-closed projection already produced;
    an older payload only has codes, and an unmapped code degrades to one safe
    sentence rather than to English the reader cannot act on.
    """
    ready = [str(item).strip() for item in (analysis.get("review_reasons_ru") or []) if str(item).strip()]
    if ready:
        return "; ".join(dict.fromkeys(ready))
    sentences: list[str] = []
    for source in (analysis.get("review_reasons"), flags.get("review_reasons")):
        for raw in source if isinstance(source, list) else []:
            code = str(raw).strip()
            if not code:
                continue
            sentence = (
                ANALYSIS_REASON_RU.get(code)
                or ROLE_REASON_RU.get(code)
                or claim_review_sentence_ru(code)
            )
            if sentence is None:
                # Already a human sentence (an older payload stored those), or
                # an unknown code we refuse to print as-is.
                sentence = code if re.search(r"[А-Яа-яЁё]", code) else UNKNOWN_REVIEW_REASON_RU
            if sentence not in sentences:
                sentences.append(sentence)
    return "; ".join(sentences)


def result_text_ru(analysis: Mapping[str, Any]) -> str:
    """Compatibility wrapper around the one shared v3 result formatter."""
    return manager_result_ru(analysis)


def projection_result(
    record: Mapping[str, Any], values: Sequence[Any], transcript: str,
    analysis: Mapping[str, Any], *, analysis_done: bool,
) -> dict[str, Any]:
    started = parse_utc(record.get("started_at"))
    duration = float(record.get("duration_sec") or 0)
    meta = json_object(analysis.get("analysis_meta"))
    source = {
        "projection_version": PROJECTION_VERSION,
        "call_key": validated_call_key(record.get("source_call_id")),
        "started_epoch": int(started.timestamp()),
        "raw_source_sha256": source_field_digests(record),
        "tail_sha256": canonical_hash(values),
        "analysis_schema_version": str(analysis.get("analysis_schema_version") or ""),
        "analysis_result_sha256": canonical_hash(
            {name: analysis.get(name) for name in ANALYSIS_RESULT_KEYS}
        ),
        "analysis_attempts_sha256": hashlib.sha256(
            str(record.get("analysis_attempts_json") or "").encode("utf-8")
        ).hexdigest(),
        "analysis_runtime_identity": {
            name: str(meta.get(name) or "")
            for name in (
                "analysis_provider",
                "analysis_model",
                "analysis_prompt_version",
                "analysis_prompt_profile",
                "analysis_input_sha256",
                "analysis_source_sha256",
                "analysis_prompt_sha256",
                "dialogue_canonical_sha256",
                "manager_output_sha256",
            )
        },
        "contract_versions": CONTRACT_FINGERPRINT,
    }
    usage = json_object(meta.get("token_usage"))
    raw_ledger = record.get("analysis_attempts_json")
    ledger_authoritative = raw_ledger is not None and str(raw_ledger).strip() != ""
    ledger_valid = True
    if ledger_authoritative:
        try:
            raw_attempts = json.loads(str(raw_ledger))
        except (TypeError, json.JSONDecodeError):
            raw_attempts = []
            ledger_valid = False
        if not isinstance(raw_attempts, list):
            raw_attempts = []
            ledger_valid = False
    else:
        raw_attempts = meta.get("model_attempts")
        if not isinstance(raw_attempts, list):
            raw_attempts = []
    model_attempts = []
    for attempt in raw_attempts if isinstance(raw_attempts, list) else []:
        if not isinstance(attempt, Mapping):
            ledger_valid = False
            continue
        attempt_usage = json_object(attempt.get("token_usage"))
        model_attempts.append(
            {
                "attempt_id": str(attempt.get("attempt_id") or ""),
                "state": str(attempt.get("state") or ""),
                "stage": str(attempt.get("stage") or "analyze"),
                "provider": str(attempt.get("provider") or ""),
                "model": str(attempt.get("model") or ""),
                "profile": str(attempt.get("profile") or ""),
                "prompt_version": str(attempt.get("prompt_version") or ""),
                "model_called": bool(attempt.get("model_called")),
                "cache_hit": bool(attempt.get("cache_hit")),
                "token_usage": {
                    "source": str(attempt_usage.get("source") or "unavailable"),
                    "prompt_tokens": attempt_usage.get("prompt_tokens"),
                    "completion_tokens": attempt_usage.get("completion_tokens"),
                    "total_tokens": attempt_usage.get("total_tokens"),
                },
            }
        )
    # A non-empty dedicated DB ledger is the current source of truth.  Missing
    # IDs/states make it corrupt; they must not silently downgrade to legacy.
    strict_ledger = ledger_authoritative
    if strict_ledger:
        attempt_ids = [str(item.get("attempt_id") or "") for item in model_attempts]
        ledger_valid = ledger_valid and bool(attempt_ids) and all(attempt_ids)
        ledger_valid = ledger_valid and len(attempt_ids) == len(set(attempt_ids))
        ledger_valid = ledger_valid and all(
            item.get("state") in {"completed", "failed"} for item in model_attempts
        )
    model_call_count = sum(bool(item.get("model_called")) for item in model_attempts)
    cache_hit_count = sum(bool(item.get("cache_hit")) for item in model_attempts)
    if ledger_authoritative and not model_attempts:
        meta_attempts = meta.get("model_attempts")
        meta_reports_work = bool(
            int(meta.get("model_call_count") or 0)
            or int(meta.get("cache_hit_count") or 0)
            or (isinstance(meta_attempts, list) and meta_attempts)
            or any(
                isinstance(usage.get(name), (int, float)) and usage.get(name) > 0
                for name in ("prompt_tokens", "completion_tokens", "total_tokens")
            )
        )
        if meta_reports_work:
            ledger_valid = False
    return {
        "id": int(record["id"]),
        "call_key": source["call_key"],
        "source_fingerprint": canonical_hash(source),
        "started_epoch": source["started_epoch"],
        "duration_sec": duration,
        "phone": str(record.get("phone") or ""),
        "transcript_sha": hashlib.sha256(transcript.encode("utf-8")).hexdigest(),
        "tail": list(values),
        "sync_status": str(record.get("sync_status") or ""),
        "analysis_done": analysis_done,
        "analysis_provider": str(
            meta.get("analysis_provider")
            or (model_attempts[0].get("provider") if model_attempts else "")
            or "unknown"
        ),
        "analysis_model": str(
            meta.get("analysis_model")
            or (model_attempts[0].get("model") if model_attempts else "")
            or "unknown"
        ),
        "analysis_prompt_version": str(
            meta.get("analysis_prompt_version")
            or (model_attempts[-1].get("prompt_version") if model_attempts else "")
            or "unknown"
        ),
        "model_call_count": int(
            model_call_count
            if ledger_authoritative
            else (
                meta.get("model_call_count")
                if meta.get("model_call_count") is not None
                else model_call_count
            )
        ),
        "cache_hit_count": int(
            cache_hit_count
            if ledger_authoritative
            else (
                meta.get("cache_hit_count")
                if meta.get("cache_hit_count") is not None
                else cache_hit_count
            )
        ),
        "model_attempts": model_attempts,
        "attempt_ledger_source": "database" if ledger_authoritative else "legacy_meta",
        "attempt_ledger_valid": ledger_valid,
        "indeterminate_attempt_count": sum(
            item.get("state") in {"reserved", "indeterminate"}
            for item in model_attempts
        ),
        "token_usage": {
            "source": str(usage.get("source") or "unavailable"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
        },
    }


def analysis_cost_summary(calls: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Closed balance of exact, partial and unknown provider usage."""
    summary = {
        "analysis_done": 0,
        "failed_analysis_calls": 0,
        "cost_tracked_calls": 0,
        "model_calls": 0,
        "cache_hits": 0,
        "invalid_attempt_ledger_calls": 0,
        "indeterminate_attempts": 0,
        "provider_usage_calls": 0,
        "provider_exact_calls": 0,
        "provider_partial_calls": 0,
        "usage_unavailable_calls": 0,
        "exact_usage_model_calls": 0,
        "partial_usage_model_calls": 0,
        "unavailable_usage_model_calls": 0,
        "cache_only_calls": 0,
        "skipped_untrusted_role_calls": 0,
        "skipped_deterministic_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "by_provider_model_prompt": {},
    }
    for call in calls.values():
        attempts_present = bool(call.get("model_attempts"))
        model_calls_present = int(call.get("model_call_count") or 0) > 0
        if not call.get("analysis_done") and not attempts_present and not model_calls_present:
            continue
        summary["cost_tracked_calls"] += 1
        if call.get("analysis_done"):
            summary["analysis_done"] += 1
        else:
            summary["failed_analysis_calls"] += 1
        summary["model_calls"] += int(call.get("model_call_count") or 0)
        summary["cache_hits"] += int(call.get("cache_hit_count") or 0)
        if not bool(call.get("attempt_ledger_valid", True)):
            summary["invalid_attempt_ledger_calls"] += 1
        summary["indeterminate_attempts"] += int(
            call.get("indeterminate_attempt_count") or 0
        )
        usage = json_object(call.get("token_usage"))
        source = str(usage.get("source") or "unavailable")
        model_calls = int(call.get("model_call_count") or 0)
        provider = str(call.get("analysis_provider") or "unknown")
        model = str(call.get("analysis_model") or "unknown")
        final_prompt = str(call.get("analysis_prompt_version") or "unknown")

        def identity_bucket(
            prompt_version: str, *, attempt_provider: str = "", attempt_model: str = ""
        ) -> dict[str, int]:
            identity = "|".join(
                (
                    attempt_provider or provider,
                    attempt_model or model,
                    prompt_version or "unknown",
                )
            )
            return summary["by_provider_model_prompt"].setdefault(
                identity,
                {
                    "model_calls": 0,
                    "exact_usage_model_calls": 0,
                    "partial_usage_model_calls": 0,
                    "unavailable_usage_model_calls": 0,
                },
            )

        def classify_attempt(
            attempt_usage: Mapping[str, Any], prompt_version: str, *, add_tokens: bool,
            attempt_provider: str = "", attempt_model: str = "",
        ) -> None:
            attempt_source = str(attempt_usage.get("source") or "unavailable")
            bucket = identity_bucket(
                prompt_version,
                attempt_provider=attempt_provider,
                attempt_model=attempt_model,
            )
            bucket["model_calls"] += 1
            complete_counts = all(
                isinstance(attempt_usage.get(key), int)
                and not isinstance(attempt_usage.get(key), bool)
                and attempt_usage.get(key) >= 0
                for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            )
            exact = attempt_source in {"provider", "provider_exact"} and complete_counts
            if exact:
                summary["exact_usage_model_calls"] += 1
                bucket["exact_usage_model_calls"] += 1
            elif attempt_source in {"provider", "provider_partial", "provider_exact"}:
                summary["partial_usage_model_calls"] += 1
                bucket["partial_usage_model_calls"] += 1
            else:
                summary["unavailable_usage_model_calls"] += 1
                bucket["unavailable_usage_model_calls"] += 1
            if add_tokens:
                for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    value = attempt_usage.get(key)
                    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                        summary[key] += value

        attempts = [
            attempt
            for attempt in call.get("model_attempts")
            if isinstance(attempt, Mapping) and bool(attempt.get("model_called"))
        ] if isinstance(call.get("model_attempts"), list) else []
        if attempts:
            for attempt in attempts:
                classify_attempt(
                    json_object(attempt.get("token_usage")),
                    str(attempt.get("prompt_version") or final_prompt),
                    add_tokens=True,
                    attempt_provider=str(attempt.get("provider") or ""),
                    attempt_model=str(attempt.get("model") or ""),
                )
        elif model_calls:
            # Legacy payloads have only one aggregate usage object.  It can
            # classify every recorded call, but its token total is added once.
            for index in range(model_calls):
                classify_attempt(usage, final_prompt, add_tokens=index == 0)
        else:
            identity_bucket(final_prompt)
        if source in {"provider", "provider_exact", "provider_partial"}:
            summary["provider_usage_calls"] += 1
            complete_counts = all(
                isinstance(usage.get(key), int)
                and not isinstance(usage.get(key), bool)
                and usage.get(key) >= 0
                for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            )
            exact_usage = source in {"provider", "provider_exact"} and complete_counts
            if exact_usage:
                summary["provider_exact_calls"] += 1
            else:
                summary["provider_partial_calls"] += 1
        elif source == "skipped_untrusted_role":
            summary["skipped_untrusted_role_calls"] += 1
        elif source == "skipped_deterministic":
            summary["skipped_deterministic_calls"] += 1
        elif source == "cache_hit":
            summary["cache_only_calls"] += 1
        else:
            summary["usage_unavailable_calls"] += 1
    classified = sum(
        summary[key]
        for key in (
            "provider_usage_calls", "usage_unavailable_calls", "cache_only_calls",
            "skipped_untrusted_role_calls", "skipped_deterministic_calls",
        )
    )
    summary["classified_calls"] = classified
    summary["model_usage_balanced"] = summary["model_calls"] == sum(
        summary[key]
        for key in (
            "exact_usage_model_calls",
            "partial_usage_model_calls",
            "unavailable_usage_model_calls",
        )
    )
    summary["identity_balanced"] = all(
        item["model_calls"]
        == item["exact_usage_model_calls"]
        + item["partial_usage_model_calls"]
        + item["unavailable_usage_model_calls"]
        for item in summary["by_provider_model_prompt"].values()
    )
    summary["balanced"] = (
        classified == summary["cost_tracked_calls"]
        and summary["model_usage_balanced"]
        and summary["identity_balanced"]
        and summary["invalid_attempt_ledger_calls"] == 0
        and summary["indeterminate_attempts"] == 0
    )
    return summary


def require_closed_analysis_cost(summary: Mapping[str, Any]) -> None:
    """Never write externally while a paid attempt is missing or ambiguous."""
    if (
        not bool(summary.get("balanced"))
        or int(summary.get("invalid_attempt_ledger_calls") or 0)
        or int(summary.get("indeterminate_attempts") or 0)
    ):
        raise RuntimeError("analysis cost ledger does not close; external write is blocked")


def safe_call_projection(
    record: Mapping[str, Any], manager_map: Mapping[str, Any], *, failed: bool = False,
) -> dict[str, Any]:
    """Visible closed row for pending, malformed or stale analysis."""
    started = parse_utc(record.get("started_at"))
    try:
        duration = float(record.get("duration_sec") or 0)
        if not math.isfinite(duration) or duration < 0:
            duration = 0.0
    except (TypeError, ValueError, OverflowError):
        duration = 0.0
    result = SAFE_ERROR_RESULT_RU if failed else SAFE_PENDING_RESULT_RU
    summary = SAFE_ERROR_SUMMARY_RU if failed else SAFE_PENDING_SUMMARY_RU
    transcript = SAFE_ERROR_TRANSCRIPT_RU if failed else SAFE_PENDING_TRANSCRIPT_RU
    shortened = False
    try:
        dialogue = build_dialogue_input(record)
        if dialogue.turns:
            transcript, shortened = transcript_cell(dialogue)
    except (ValueError, TypeError, OverflowError):
        # Analysis failure must not hide a valid transcript, but a broken
        # transcript remains quarantined instead of being partially reconstructed.
        pass
    review = (
        "Звонок помещён в карантин: проверьте сохранённый анализ и расшифровку."
        if failed else "Смысловой анализ звонка ещё не завершён."
    )
    if shortened:
        review = f"{review} {TRANSCRIPT_OVERSIZE_REVIEW_RU}"
    values = [
        moscow_datetime(started).strftime("%Y-%m-%d %H:%M:%S"),
        manager_display(record.get("manager_name"), manager_map),
        {"outbound": "Исходящий", "inbound": "Входящий"}.get(
            str(record.get("direction") or "").lower(), "Не определено"
        ),
        format_duration(duration), "Не определено", str(record.get("phone") or ""),
        "Да", "—", summary, result, "—", "—", "—", "—", review, transcript,
    ]
    return projection_result(record, values, transcript, {}, analysis_done=False)


def call_projection(record: Mapping[str, Any], manager_map: Mapping[str, Any]) -> dict[str, Any]:
    if str(record.get("analysis_status") or "") != "done":
        return safe_call_projection(record, manager_map)
    started = parse_utc(record.get("started_at"))
    duration = float(record.get("duration_sec"))
    stored_analysis = required_json_object(record.get("analysis_json"), "analysis_json")
    analysis = guard_stored_analysis(record, stored_analysis)
    guarded_flags = json_object(analysis.get("quality_flags"))
    if guarded_flags.get("analysis_contract_invalid"):
        raise ValueError("analysis contract is stale")
    if (
        str(analysis.get("analysis_schema_version") or "").lower()
        != ANALYSIS_SCHEMA_VERSION_V3
        and not guarded_flags.get("role_attribution_untrusted")
    ):
        raise ValueError("analysis schema is stale")
    dialogue = build_dialogue_input(record)
    fields = json_object(analysis.get("display_fields"))
    interests = json_object(fields.get("interests"))
    next_step = json_object(fields.get("next_step"))
    flags = guarded_flags
    call_type = str(flags.get("call_type") or "").lower()
    products = interests.get("products")
    subjects = interests.get("subjects")
    topic = (
        (str(products[0]).strip() if isinstance(products, list) and products else "")
        or (str(subjects[0]).strip() if isinstance(subjects, list) and subjects else "")
        # Untrusted calls keep only the deterministic neutral topic list.
        or list_text(analysis.get("neutral_topics"))
        or "—"
    )
    summary = normalize_summary_time((
        str(analysis.get("manager_brief") or "").strip()
        or str(analysis.get("summary") or "").strip()
        or str(analysis.get("history_short") or "").strip()
        or str(analysis.get("history_summary") or "").strip()
        or "—"
    ), started)
    objections = list_text(fields.get("objections")) or list_text(analysis.get("objections")) or "—"
    action = str(next_step.get("action") or "").strip()
    if not action and isinstance(analysis.get("next_step"), str):
        action = str(analysis.get("next_step") or "").strip()
    due = str(next_step.get("due") or "").strip() or str(analysis.get("timeline") or "").strip()
    if not action:
        due = ""
    sentences = [
        item for item in [review_sentences_ru(analysis, flags)] if item
    ]
    needs_review = bool(
        analysis.get("needs_review")
        if analysis.get("needs_review") is not None
        else flags.get("needs_review")
    )
    if not dialogue.turns:
        raise ValueError("full transcript is empty")
    transcript, shortened = transcript_cell(dialogue)
    if not transcript:
        raise ValueError("full transcript is empty")
    if shortened:
        sentences.append(TRANSCRIPT_OVERSIZE_REVIEW_RU)
    review = "; ".join(dict.fromkeys(sentences)) or "—"
    # "Нужна проверка: Нет" next to a filled reason column is a contradiction
    # the reader resolves by trusting the flag and skipping the row.  There is
    # exactly one truth here: a stated reason *is* a request to check.
    needs_review = needs_review or review != "—"
    values = [
        moscow_datetime(started).strftime("%Y-%m-%d %H:%M:%S"),
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
        "Да" if needs_review else "Нет",
        topic,
        summary,
        result_text_ru(analysis),
        objections,
        action or "—",
        due or "—",
        manager_claim_evidence_ru(analysis) or "—",
        review,
        transcript,
    ]
    if any(len(str(value)) > MAX_CELL_CHARS for value in values):
        raise ValueError("cell exceeds 50000 characters")
    return projection_result(record, values, transcript, analysis, analysis_done=True)


def call_identity(record: Mapping[str, Any]) -> dict[str, Any]:
    call_key = validated_call_key(record.get("source_call_id"))
    analysis_status = str(record.get("analysis_status") or "")
    started_epoch = int(parse_utc(record.get("started_at")).timestamp())
    if analysis_status != "done":
        try:
            duration = float(record.get("duration_sec"))
            if not math.isfinite(duration) or duration < 0:
                duration = 0.0
        except (TypeError, ValueError, OverflowError):
            duration = 0.0
        try:
            transcript = render_transcript(record)
        except ValueError:
            transcript = SAFE_PENDING_TRANSCRIPT_RU
        legacy = ""
    else:
        duration = float(record.get("duration_sec"))
        legacy = render_legacy_identity_transcript(record)
        try:
            transcript = render_transcript(record)
        except ValueError:
            transcript = SAFE_ERROR_TRANSCRIPT_RU
    return {
        "id": int(record["id"]),
        "call_key": call_key,
        "started_epoch": started_epoch,
        "duration_sec": duration,
        "phone": str(record.get("phone") or ""),
        "transcript_sha": (
            hashlib.sha256(transcript.encode("utf-8")).hexdigest() if transcript else ""
        ),
        "legacy_transcript_sha": (
            hashlib.sha256(legacy.encode("utf-8")).hexdigest()
            if legacy and legacy != transcript
            else ""
        ),
        "safe_transcript_sha": hashlib.sha256(
            (SAFE_ERROR_TRANSCRIPT_RU if analysis_status == "done" else SAFE_PENDING_TRANSCRIPT_RU).encode(
                "utf-8"
            )
        ).hexdigest(),
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
    if len(row) < MANAGED_COLUMN_COUNT:
        raise ValueError("physical row has fewer than managed columns")
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
    transcript = str(row[TRANSCRIPT_COLUMN_INDEX] or "")
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
    return canonical_hash(
        list(row[:MANAGED_COLUMN_COUNT])
        + [""] * max(0, MANAGED_COLUMN_COUNT - len(row))
    )


def identity_matches(identity: Mapping[str, Any], call: Mapping[str, Any]) -> bool:
    precision = identity.get("time_precision")
    same_time = identity["utc"] == call["started_epoch"]
    if precision in {"minute", "second_or_minute"} and not same_time:
        same_time = identity["utc"] // 60 == call["started_epoch"] // 60
    if not same_time or identity["phone"] != normalized_phone(call["phone"]):
        return False
    if identity["transcript_sha"] not in {
        call["transcript_sha"],
        call.get("legacy_transcript_sha") or call["transcript_sha"],
        call.get("safe_transcript_sha") or call["transcript_sha"],
    }:
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
    if tuple(header[:MANAGED_COLUMN_COUNT]) != LIVE_HEADERS or any(
        not is_blank_sheet_value(value) for value in header[MANAGED_COLUMN_COUNT:25]
    ):
        raise ValueError("Google header contract mismatch")
    rows: list[list[Any]] = []
    blank_seen = False
    for raw in values[1:]:
        row = list(raw) + [""] * max(0, 25 - len(raw))
        filled = any(
            not is_blank_sheet_value(value) for value in row[:MANAGED_COLUMN_COUNT]
        )
        if not filled:
            blank_seen = True
            if any(not is_blank_sheet_value(value) for value in row[MANAGED_COLUMN_COUNT:25]):
                raise ValueError("reserved helper columns contain data")
            continue
        if blank_seen:
            raise ValueError("Google data rows are not contiguous")
        if any(not is_blank_sheet_value(value) for value in row[MANAGED_COLUMN_COUNT:25]):
            raise ValueError("reserved helper columns contain data")
        rows.append(row[:MANAGED_COLUMN_COUNT])
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
    columns = {
        str(item[1]) for item in connection.execute("PRAGMA table_info(call_records)")
    }
    recording_column = (
        "source_recording_id"
        if "source_recording_id" in columns
        else "NULL AS source_recording_id"
    )
    attempts_column = (
        "analysis_attempts_json"
        if "analysis_attempts_json" in columns
        else "NULL AS analysis_attempts_json"
    )
    rows = connection.execute(
        f"SELECT id,source_call_id,{recording_column},{attempts_column},started_at,phone,manager_name,direction,duration_sec,"
        "analysis_json,transcript_variants_json,transcript_text,analysis_status,sync_status "
        "FROM call_records ORDER BY id"
    ).fetchall()
    connection.close()
    calls: dict[str, Any] = {}
    identities: dict[str, Any] = {}
    errors: dict[str, Mapping[str, Any]] = {}
    recording_owners: dict[str, int] = {}
    for raw in rows:
        record = dict(raw)
        recording_id = str(record.get("source_recording_id") or "").strip()
        try:
            identity = call_identity(record)
        except (ValueError, TypeError, OverflowError) as exc:
            errors[incident_key(record)] = {
                "call_digest": "",
                "code": validation_error_code("identity", exc),
            }
            continue
        if identity["call_key"] in identities:
            raise IdentityCollisionError(
                "duplicate stable call_key",
                code="duplicate_stable_call_key",
                fingerprint=hashlib.sha256(
                    identity["call_key"].encode("utf-8")
                ).hexdigest(),
            )
        identities[identity["call_key"]] = identity
        if recording_id:
            if recording_id in recording_owners:
                raise IdentityCollisionError(
                    "duplicate normalized source_recording_id",
                    code="duplicate_source_recording_id",
                    fingerprint=hashlib.sha256(recording_id.encode("utf-8")).hexdigest(),
                )
            recording_owners[recording_id] = int(record["id"])
            record["source_recording_id"] = recording_id
        try:
            call = call_projection(record, manager_map)
        except (ValueError, TypeError, OverflowError) as exc:
            errors[incident_key(record)] = {
                "call_digest": call_key_digest(identity["call_key"]),
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
        "incidents": {},
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def call_key_digest(call_key: Any) -> str:
    """Irreversible handle of one call for the sidecar and its incidents.

    ``mango:mango_office:<source_call_id>`` embeds the provider's raw call id
    verbatim.  The sidecar is a durable owner-local journal that outlives the
    call, so it stores the digest instead: two runs still agree on "the same
    call", and nothing identifying can be read back out of the file.
    """
    text = str(call_key or "").strip()
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32] if text else ""


def incident_key(record: Mapping[str, Any]) -> str:
    """Stable and de-identified: never the raw call id, phone, file name or name.

    A call whose ``source_call_id`` is usable is keyed by the digest of its
    stable call key.  Without one there is no business identity at all, so the
    seed falls back to the technical coordinates of the stored row — local id,
    start, duration and the digest of the stored transcript container.  Together
    they separate two different broken rows that happen to share a start time;
    none of them is personal data, and the whole seed is hashed anyway.
    """
    source_call_id = str(record.get("source_call_id") or "").strip()
    if source_call_id:
        try:
            return "call:" + call_key_digest(validated_call_key(source_call_id))
        except ValueError:
            seed = f"source_call_id:{source_call_id}"
    else:
        variants_digest = hashlib.sha256(
            str(record.get("transcript_variants_json") or "").encode("utf-8")
        ).hexdigest()
        seed = (
            "no_source_call_id:"
            f"id={record.get('id')}|"
            f"started_at={record.get('started_at')}|"
            f"duration_sec={record.get('duration_sec')}|"
            f"variants_sha256={variants_digest}"
        )
    return "unresolved:" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:32]


def merge_incidents(
    previous: Mapping[str, Any],
    errors: Mapping[str, Mapping[str, Any]],
    *,
    now: str,
    incident_class: str = INCIDENT_CLASS_DATA,
    close_missing: bool = True,
) -> dict[str, dict[str, Any]]:
    """Keep one durable de-identified incident per still-failing call.

    ``first_seen_at`` survives a change of ``code``: the call has been broken
    since that moment, and only an actually fixed call closes the incident.

    A scan may close only what it can actually see.  The database scan produces
    ``data`` errors and knows nothing about a Google row that cannot be matched,
    so an empty data scan must not wipe an open ``reconcile`` incident — that is
    how a still-broken sheet used to come back green.  ``close_missing=False``
    is the stronger form of the same rule, used by a scan that *aborted*: it
    finished nothing, so it may add what it found and close nothing at all.
    """
    incidents = {
        str(key): dict(value)
        for key, value in previous.items()
        if isinstance(value, Mapping)
        and (
            not close_missing
            or str(value.get("class") or INCIDENT_CLASS_DATA) != incident_class
        )
    }
    for key, error in errors.items():
        prior = previous.get(key) if isinstance(previous.get(key), Mapping) else {}
        code = str(error.get("code") or "unknown")
        incident = {
            "class": incident_class,
            # Digest, never the raw provider call id: the sidecar is a durable
            # journal, and a call id is the identity of a real conversation.
            "call_digest": str(error.get("call_digest") or ""),
            "code": code,
            "first_seen_at": str(prior.get("first_seen_at") or "") or now,
            "last_seen_at": now,
        }
        previous_code = str(prior.get("code") or "")
        if previous_code and previous_code != code:
            incident["previous_code"] = previous_code
        if prior.get("first_seen_source"):
            incident["first_seen_source"] = str(prior["first_seen_source"])
        incidents[str(key)] = incident
    return incidents


def validated_incidents(raw: Any) -> dict[str, dict[str, Any]]:
    """A corrupt incident stops the run; it is never quietly filtered out."""
    if not isinstance(raw, Mapping):
        raise RuntimeError("publisher state incidents are invalid")
    incidents: dict[str, dict[str, Any]] = {}
    for key, value in raw.items():
        raw_key = str(key).strip()
        if not raw_key or not isinstance(value, Mapping):
            raise RuntimeError("publisher state incident entry is corrupt")
        if not INCIDENT_CODE_RE.fullmatch(str(value.get("code") or "")):
            raise RuntimeError("publisher state incident code is corrupt")
        if str(value.get("class") or INCIDENT_CLASS_DATA) not in INCIDENT_CLASSES:
            raise RuntimeError("publisher state incident class is corrupt")
        for field in ("first_seen_at", "last_seen_at"):
            try:
                parse_utc(value.get(field))
            except (TypeError, ValueError, OverflowError) as exc:
                raise RuntimeError(f"publisher state incident {field} is corrupt") from exc
        incident = dict(value)
        incident.setdefault("class", INCIDENT_CLASS_DATA)
        # A sidecar written before the de-identification still holds the raw
        # provider call key.  Reading it is where that stops: the value is
        # replaced by its digest instead of being carried forward.
        legacy_call_key = incident.pop("call_key", None)
        if legacy_call_key and not incident.get("call_digest"):
            incident["call_digest"] = call_key_digest(legacy_call_key)
        safe_key = (
            "call:" + call_key_digest(raw_key)
            if CALL_KEY_RE.fullmatch(raw_key)
            else raw_key
        )
        if safe_key != raw_key and not incident.get("call_digest"):
            incident["call_digest"] = call_key_digest(raw_key)
        digest = str(incident.get("call_digest") or "")
        if digest and not re.fullmatch(r"[0-9a-f]{32}", digest):
            raise RuntimeError("publisher state incident call digest is corrupt")
        if safe_key in incidents:
            raise RuntimeError("publisher state incident keys collide after migration")
        incidents[safe_key] = incident
    return incidents


def migrate_incidents(state: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Backward-compatible read of a ``state_v1`` sidecar.

    The old field was ``data_errors`` keyed by the local SQLite id and without
    timestamps.  Nothing is dropped: each old error becomes an incident whose
    ``first_seen_at`` is honestly marked as taken from the sidecar's own
    ``updated_at``.  Errors that carried a stable ``call_key`` keep their
    history across the migration; the rest are re-keyed on the next run.

    The raw ``call_key`` does not survive the migration: it is replaced by its
    digest, so an old sidecar stops carrying provider call ids the moment it is
    read.
    """
    if state.get("incidents") is not None or LEGACY_ERROR_FIELD not in state:
        return validated_incidents(state.get("incidents") or {})
    legacy = state.get(LEGACY_ERROR_FIELD)
    if not isinstance(legacy, Mapping):
        raise RuntimeError("publisher state data_errors are invalid")
    migrated_at = str(state.get("updated_at") or "") or datetime.now(timezone.utc).isoformat()
    incidents: dict[str, dict[str, Any]] = {}
    for key, value in legacy.items():
        if not isinstance(value, Mapping):
            raise RuntimeError("publisher state data_errors entry is corrupt")
        call_key = str(value.get("call_key") or "")
        digest = call_key_digest(call_key) if CALL_KEY_RE.fullmatch(call_key) else ""
        stable = (
            "call:" + digest
            if digest
            else "unresolved:"
            + hashlib.sha256(f"state_v1:{key}".encode("utf-8")).hexdigest()[:32]
        )
        incidents[stable] = {
            "class": INCIDENT_CLASS_DATA,
            "call_digest": digest,
            "code": str(value.get("code") or "unknown"),
            "first_seen_at": migrated_at,
            "last_seen_at": migrated_at,
            "first_seen_source": "migrated_state_v1_data_errors",
        }
    return validated_incidents(incidents)


def health_report(incidents: Mapping[str, Mapping[str, Any]], *, now: str) -> dict[str, Any]:
    breached = 0
    for incident in incidents.values():
        try:
            age_hours = (
                parse_utc(now) - parse_utc(incident.get("first_seen_at"))
            ).total_seconds() / 3600.0
        except (TypeError, ValueError, OverflowError):
            age_hours = float(INCIDENT_SLA_HOURS + 1)
        breached += int(age_hours > INCIDENT_SLA_HOURS)
    return {
        "status": "red" if breached else "amber" if incidents else "green",
        "open_incidents": len(incidents),
        "sla_breached": breached,
        "sla_hours": INCIDENT_SLA_HOURS,
    }


def apply_incidents(
    state: dict[str, Any],
    errors: Mapping[str, Mapping[str, Any]],
    *,
    incident_class: str = INCIDENT_CLASS_DATA,
    close_missing: bool = True,
) -> Mapping[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    state["incidents"] = merge_incidents(
        state.get("incidents") or {},
        errors,
        now=now,
        incident_class=incident_class,
        close_missing=close_missing,
    )
    state["health"] = health_report(state["incidents"], now=now)
    return state["health"]


def record_incidents(
    state: dict[str, Any],
    state_path: Path,
    errors: Mapping[str, Mapping[str, Any]],
    *,
    incident_class: str = INCIDENT_CLASS_DATA,
    close_missing: bool = True,
) -> Mapping[str, Any]:
    """Persist incidents to the owner sidecar even when the run then stops.

    Owner-sidecar policy: this local, 0600, owner-only JSON file is the only
    thing a dry run is ever allowed to write.  It is not an external system and
    carries no personal data — keys are hashes, values are codes and times —
    so recording an incident here does not make a dry run a publishing run.
    """
    health = apply_incidents(
        state, errors, incident_class=incident_class, close_missing=close_missing
    )
    atomic_owner_json(state_path, state)
    return health


def block_on_identity_errors(
    state: dict[str, Any], state_path: Path, errors: Mapping[str, Mapping[str, Any]]
) -> None:
    """An unreadable local row poisons matching, sorting and layout of the run.

    Projection errors stay isolated (one bad call, the neighbours publish), but
    an identity error means we cannot even say which Google row belongs to
    which call, so the whole run stops before any Google batch.
    """
    blocking = sorted(
        {
            str(error.get("code") or "")
            for error in errors.values()
            if str(error.get("code") or "").startswith("identity_")
        }
    )
    if not blocking:
        return
    record_incidents(state, state_path, errors)
    raise RuntimeError(
        "identity errors block the whole run before any Google write: "
        + ", ".join(blocking)
    )


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
    # A v1 sidecar is read, migrated in memory and only ever written back as v2:
    # the owner never has to hand-edit or delete their publication ledger.
    if (
        state.get("schema_version") not in {STATE_SCHEMA, STATE_SCHEMA_LEGACY}
        or state.get("destination_id") != destination_id
    ):
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
    state["incidents"] = migrate_incidents(state)
    state.pop(LEGACY_ERROR_FIELD, None)
    state["schema_version"] = STATE_SCHEMA
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
        # A row written by the older lenient projection is still identifiable.
        for field in ("transcript_sha", "legacy_transcript_sha", "safe_transcript_sha"):
            transcript_sha = str(identity.get(field) or "")
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
                # The exception message is built from the row, which is the
                # conversation: only the stage, the type and a digest survive.
                raise ReconcileError(
                    f"{safe_error_text('reconcile', exc)} "
                    f"at Google row {index + 2} (№ {display_number})",
                    code="reconcile_physical_row_invalid",
                    fingerprint=row_hash,
                ) from exc
        if len(candidates) != 1:
            display_number = row[0] if row else ""
            raise ReconcileError(
                "unidentified_or_ambiguous_physical_row "
                f"at Google row {index + 2} (№ {display_number})",
                code="reconcile_unidentified_or_ambiguous_row",
                fingerprint=row_hash,
            )
        call_key = next(iter(candidates))
        if call_key in call_to_row:
            first_index = call_to_row[call_key]
            display_number = row[0] if row else ""
            raise ReconcileError(
                "duplicate_physical_call at Google rows "
                f"{first_index + 2} and {index + 2} (№ {display_number})",
                code="reconcile_duplicate_physical_call",
                fingerprint=row_hash,
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
                        "endColumnIndex": MANAGED_COLUMN_COUNT,
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
                    "start": {
                        "sheetId": sheet_id,
                        "rowIndex": 1,
                        "columnIndex": SORT_KEY_COLUMN_INDEX,
                    },
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
                        "endColumnIndex": SORT_KEY_COLUMN_INDEX + 1,
                    },
                    "sortSpecs": [
                        {
                            "dimensionIndex": SORT_KEY_COLUMN_INDEX,
                            "sortOrder": "DESCENDING",
                        }
                    ],
                }
            },
            {
                "repeatCell": {
                    "range": {
                        "sheetId": sheet_id,
                        "startRowIndex": 1,
                        "endRowIndex": len(physical_keys) + 1,
                        "startColumnIndex": SORT_KEY_COLUMN_INDEX,
                        "endColumnIndex": SORT_KEY_COLUMN_INDEX + 1,
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
                    "range": {
                        **data_range,
                        "startColumnIndex": TRANSCRIPT_COLUMN_INDEX,
                        "endColumnIndex": TRANSCRIPT_COLUMN_INDEX + 1,
                    },
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
                "ranges": f"'{title}'!A2:Q{last_row}",
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
        if len(values) < MANAGED_COLUMN_COUNT:
            raise RuntimeError("managed-column format readback is incomplete")
        for column_index, cell in enumerate(values[:MANAGED_COLUMN_COUNT]):
            entered = cell.get("userEnteredValue") or {}
            if "formulaValue" in entered:
                column = chr(ord("A") + column_index)
                raise RuntimeError(
                    "Google formulas are forbidden in managed columns at "
                    f"{column}{index + 2} (№ {row[0]})"
                )
        j_format = values[9].get("userEnteredFormat") or {}
        transcript_format = values[TRANSCRIPT_COLUMN_INDEX].get("userEnteredFormat") or {}
        if j_format.get("wrapStrategy") != "WRAP" or j_format.get("verticalAlignment") != "TOP":
            raise RuntimeError(
                f"summary format readback mismatch at Google row {index + 2} (№ {row[0]})"
            )
        if transcript_format.get("wrapStrategy") != "CLIP" or transcript_format.get("verticalAlignment") != "TOP":
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
        columns = {
            str(item[1])
            for item in connection.execute("PRAGMA table_info(call_records)")
        }
        recording_column = (
            "source_recording_id"
            if "source_recording_id" in columns
            else "NULL AS source_recording_id"
        )
        attempts_column = (
            "analysis_attempts_json"
            if "analysis_attempts_json" in columns
            else "NULL AS analysis_attempts_json"
        )
        for call_key, proof in expected.items():
            row = connection.execute(
                f"SELECT id,source_call_id,{recording_column},{attempts_column},started_at,phone,manager_name,direction,duration_sec,"
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
        if current.get("analysis_done"):
            verified[call_key] = {
                "id": int(current["id"]),
                "source_fingerprint": current["source_fingerprint"],
            }
    state["updated_at"] = now
    atomic_owner_json(state_path, state)
    return verified


def publication_balance(
    identities: Mapping[str, Mapping[str, Any]],
    calls: Mapping[str, Mapping[str, Any]],
    errors: Mapping[str, Mapping[str, Any]],
    state: Mapping[str, Any],
    rows: Sequence[Sequence[Any]],
    call_to_row: Mapping[str, int],
) -> dict[str, Any]:
    """Account for every identifiable source call exactly once, from existing state.

    This is a report, not a new state machine: the categories are read off the
    ledger, the incidents and the current sheet that the publisher already
    maintains.  Its job is to make "we published everything" a checkable
    statement instead of an assumption — a call that belongs to no category, or
    to two, means the publisher has lost track of it.

    * ``failed_with_incident`` — the call cannot even be projected right now;
    * ``quarantined``          — projectable, but an open incident owns it;
    * ``reserved``             — planned or not yet planned: awaiting a write;
    * ``published``            — proven in the sheet, but the source has moved;
    * ``verified_current``     — proven in the sheet and still current.
    """
    entries = state.get("entries") if isinstance(state.get("entries"), Mapping) else {}
    incident_digests = {
        str(incident.get("call_digest") or "")
        for incident in (state.get("incidents") or {}).values()
        if isinstance(incident, Mapping) and str(incident.get("call_digest") or "")
    }
    failed_digests = {
        str(error.get("call_digest") or "")
        for error in errors.values()
        if str(error.get("call_digest") or "")
    }
    buckets: dict[str, set[str]] = {name: set() for name in BALANCE_CATEGORIES}
    source_calls = set(identities)
    source_digests = {call_key_digest(key) for key in source_calls}
    row_indexes = list(call_to_row.values())
    violations = {
        "unidentified_source_rows": sum(
            not str(error.get("call_digest") or "") for error in errors.values()
        ),
        "orphan_projected_calls": len(set(calls) - source_calls),
        "orphan_state_entries": len(set(entries) - source_calls),
        "orphan_sheet_mappings": len(set(call_to_row) - source_calls),
        "orphan_error_digests": len(failed_digests - source_digests),
        "orphan_incident_digests": len(incident_digests - source_digests),
        "invalid_sheet_indexes": sum(
            not isinstance(index, int) or index < 0 or index >= len(rows)
            for index in row_indexes
        ) + len(row_indexes) - len(set(row_indexes)),
    }
    unaccounted: set[str] = set()
    for call_key in sorted(source_calls):
        digest = call_key_digest(call_key)
        entry = entries.get(call_key) if isinstance(entries.get(call_key), Mapping) else {}
        call = calls.get(call_key)
        if digest in failed_digests:
            bucket = "quarantined" if call is not None else "failed_with_incident"
        elif call is None:
            unaccounted.add(call_key)
            continue
        elif digest in incident_digests:
            bucket = "quarantined"
        elif not entry or str(entry.get("status") or "") == "reserved":
            bucket = "reserved"
        elif str(entry.get("status") or "") != "verified":
            unaccounted.add(call_key)
            continue
        else:
            index = call_to_row.get(call_key)
            row = rows[index] if isinstance(index, int) and 0 <= index < len(rows) else None
            current = bool(
                row is not None
                and physical_row_hash(row)
                == physical_row_hash(desired_row(call, int(entry["display_number"])))
                and entry.get("source_fingerprint") == call["source_fingerprint"]
                and entry.get("projection_version") == PROJECTION_VERSION
            )
            bucket = "verified_current" if current else "published"
        buckets[bucket].add(call_key)
    counts = {name: len(items) for name, items in buckets.items()}
    assigned = [key for items in buckets.values() for key in items]
    violations["unaccounted_source_calls"] = len(unaccounted)
    return {
        "source_calls": len(source_calls),
        "analysis_done": sum(
            bool((calls.get(key) or {}).get("analysis_done")) for key in source_calls
        ),
        **counts,
        "integrity_violations": violations,
        "balanced": (
            not any(violations.values())
            and len(assigned) == len(set(assigned)) == len(source_calls)
        ),
    }


def require_closed_balance(balance: Mapping[str, Any]) -> None:
    """A publication that cannot account for its own calls does not write.

    The sum is re-derived here from the published counters instead of trusting
    the flag: the report is what a human reads, so the guard has to fail on the
    same numbers that a human would check, not on a private variable.
    """
    counted = sum(int(balance.get(name) or 0) for name in BALANCE_CATEGORIES)
    violations = balance.get("integrity_violations") or {}
    if (
        not balance.get("balanced")
        or counted != int(balance.get("source_calls") or 0)
        or not isinstance(violations, Mapping)
        or any(int(value or 0) for value in violations.values())
    ):
        raise RuntimeError(
            "publication balance does not close; external write is blocked: "
            f"categories={counted}, source_calls={balance.get('source_calls')}"
        )


def require_fresh_source_snapshot(
    *,
    selected: Sequence[str],
    original_calls: Mapping[str, Mapping[str, Any]],
    original_identities: Mapping[str, Mapping[str, Any]],
    fresh_calls: Mapping[str, Mapping[str, Any]],
    fresh_identities: Mapping[str, Mapping[str, Any]],
    fresh_errors: Mapping[str, Mapping[str, Any]],
    state: Mapping[str, Any],
    rows: Sequence[Sequence[Any]],
    call_to_row: Mapping[str, int],
) -> Mapping[str, Any]:
    """Last database check immediately before the sole external write."""
    if set(fresh_identities) != set(original_identities):
        raise RuntimeError("source population changed before external write")
    for call_key in selected:
        original = original_calls.get(call_key)
        fresh = fresh_calls.get(call_key)
        if (
            original is None
            or fresh is None
            or original.get("source_fingerprint") != fresh.get("source_fingerprint")
        ):
            raise RuntimeError("selected source changed before external write")
    balance = publication_balance(
        fresh_identities, fresh_calls, fresh_errors, state, rows, call_to_row
    )
    require_closed_balance(balance)
    return balance


def read_manager_map(path: Path) -> Mapping[str, Any]:
    payload = owner_json(path)
    mapping = payload.get("mapping")
    return dict(mapping) if isinstance(mapping, Mapping) else {}


@contextmanager
def sqlite_source_write_fence(path: Path):
    """Keep Analyze writes outside the final source-check/Google-write window."""
    connection = sqlite3.connect(path, timeout=30.0, isolation_level=None)
    try:
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute("BEGIN IMMEDIATE")
        yield
    finally:
        try:
            connection.rollback()
        finally:
            connection.close()


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
        if not call or not call.get("analysis_done") or not entry or call_key not in call_to_row:
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
        state = load_state(state_path, destination, required=args.execute)

        def scanned(current_manager_map: Mapping[str, Any]):
            """Read the database and leave a durable incident before dying."""
            try:
                return load_calls(db_path, current_manager_map)
            except DurableIncidentError as exc:
                # The scan aborted mid-way, so it may add what it saw and close
                # nothing: the calls it never reached are not proven healthy.
                record_incidents(
                    state,
                    state_path,
                    exc.incident_errors(),
                    incident_class=exc.incident_class,
                    close_missing=False,
                )
                raise

        calls, identities, data_errors = scanned(manager_map)
        block_on_identity_errors(state, state_path, data_errors)
        if args.execute:
            require_closed_analysis_cost(analysis_cost_summary(calls))

        def reconciled(
            current_rows: Sequence[Sequence[Any]],
        ) -> tuple[dict[str, int], dict[int, str]]:
            """Reconcile and leave a durable incident instead of only stderr."""
            try:
                matched = reconcile(current_rows, identities, state)
            except ReconcileError as exc:
                record_incidents(
                    state, state_path, exc.incident_errors(),
                    incident_class=exc.incident_class, close_missing=False,
                )
                raise
            # The sheet reconciles again: this is the only scan that can prove a
            # reconcile incident is over, so it is the only one that closes it.
            apply_incidents(state, {}, incident_class=INCIDENT_CLASS_RECONCILE)
            return matched

        gateway = LiveGoogleGateway(
            authorized_session(config["credentials_info"]), str(config["spreadsheet_id"])
        )
        validate_sheet_target(gateway.sheets(), title=sheet_title, sheet_id=sheet_id)
        _header, rows = normalize_values(gateway.values(sheet_title))
        call_to_row, row_to_call = reconciled(rows)
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
            health = apply_incidents(state, data_errors)
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
                "analysis_cost": analysis_cost_summary(calls),
                "health": health,
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
                "source_calls": len(identities),
                "analysis_done": sum(bool(call.get("analysis_done")) for call in calls.values()),
                "matched": len(call_to_row),
                "missing": sum(key not in call_to_row for key in calls),
                "stale": stale,
                "data_errors": len(data_errors),
                "data_error_codes": data_error_counts(data_errors),
                "analysis_cost": analysis_cost_summary(calls),
                "balance": publication_balance(
                    identities, calls, data_errors, audit_state, rows, call_to_row
                ),
                # A dry run persists its incidents to the owner sidecar: losing
                # the reason a call cannot be published — and the moment it
                # first broke — is worse than an owner-local file write.  Google
                # and the working SQLite are still untouched.
                "health": record_incidents(state, state_path, data_errors),
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
            fresh_call_to_row, row_to_call = reconciled(fresh_rows)
            verify_sheet_snapshot(
                rows=fresh_rows, call_to_row=fresh_call_to_row, identities=identities,
                state=state, exact_keys=recovered, recoverable_keys=calls,
            )
            rows, call_to_row = fresh_rows, fresh_call_to_row
            physical = {key: rows[index] for key, index in call_to_row.items()}
            manager_map = read_manager_map(manager_path)
            calls, identities, data_errors = scanned(manager_map)
            # A reload is a new read of the same database: it gets the same
            # identity gate as the first one, before any finalization or write.
            block_on_identity_errors(state, state_path, data_errors)
            require_closed_analysis_cost(analysis_cost_summary(calls))
            apply_incidents(state, data_errors)
            recovery_balance = publication_balance(
                identities, calls, data_errors, state, rows, call_to_row
            )
            require_closed_balance(recovery_balance)
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
        health = apply_incidents(state, data_errors)
        state, selected = reserve(state, calls, rows, call_to_row, limit=limit)
        if not selected:
            pending = [
                key for key, call in calls.items()
                if key in call_to_row
                and state["entries"].get(key, {}).get("status") == "verified"
                and call.get("analysis_done")
                and call["sync_status"] != "done"
            ]
            if pending:
                verify_layout(
                    gateway.layout(sheet_title, len(rows) + 1), rows,
                    int(config["summary_width_px"]),
                )
                _sync_header, sync_rows = normalize_values(gateway.values(sheet_title))
                sync_call_to_row, _sync_row_to_call = reconciled(sync_rows)
                verify_sheet_snapshot(
                    rows=sync_rows, call_to_row=sync_call_to_row,
                    identities=identities, state=state, recoverable_keys=calls,
                )
                rows, call_to_row = sync_rows, sync_call_to_row
            proofs = sync_proofs(calls, rows, call_to_row, state, pending)
            balance = publication_balance(
                identities, calls, data_errors, state, rows, call_to_row
            )
            require_closed_balance(balance)
            atomic_owner_json(state_path, state)
            write_sync_done(db_path, proofs, manager_map)
            return {
                "status": "no_change",
                "google_rows": len(rows),
                "data_errors": len(data_errors),
                "data_error_codes": data_error_counts(data_errors),
                "analysis_cost": analysis_cost_summary(calls),
                "balance": balance,
                "health": health,
                "external_write": False,
            }

        selected_set = set(selected)
        unchanged = {
            key: physical_row_hash(rows[index])
            for key, index in call_to_row.items()
            if key not in selected_set
        }
        # The last check before the only external write of the run: if the
        # publisher cannot place every finished analysis in exactly one
        # category, it does not know what it is about to publish.
        pre_write_balance = publication_balance(
            identities, calls, data_errors, state, rows, call_to_row
        )
        require_closed_balance(pre_write_balance)
        require_closed_analysis_cost(analysis_cost_summary(calls))
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
        with sqlite_source_write_fence(db_path):
            # Reserve/build can race with a new Analyze commit.  Re-read after both,
            # then compare the exact selected fingerprints before touching Google.
            prewrite_manager_map = read_manager_map(manager_path)
            prewrite_calls, prewrite_identities, prewrite_errors = scanned(
                prewrite_manager_map
            )
            block_on_identity_errors(state, state_path, prewrite_errors)
            require_closed_analysis_cost(analysis_cost_summary(prewrite_calls))
            require_fresh_source_snapshot(
                selected=selected,
                original_calls=calls,
                original_identities=identities,
                fresh_calls=prewrite_calls,
                fresh_identities=prewrite_identities,
                fresh_errors=prewrite_errors,
                state=state,
                rows=rows,
                call_to_row=call_to_row,
            )
            gateway.batch_sheet_requests(requests)
            request_count = len(requests)
            _post_header, post_rows = normalize_values(gateway.values(sheet_title))
            post_call_to_row, _post_row_to_call = reconciled(post_rows)
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
            final_call_to_row, _final_row_to_call = reconciled(final_rows)
            verify_sheet_snapshot(
                rows=final_rows, call_to_row=final_call_to_row, identities=identities,
                state=state, exact_keys=selected, unchanged_hashes=unchanged,
                recoverable_keys=calls,
            )
            if [physical_row_hash(row) for row in final_rows] != [
                physical_row_hash(row) for row in expected_rows
            ]:
                raise RuntimeError("Google final full readback mismatch")
            # The SQLite fence keeps Analyze out until this source is finalized.
            # The bounded comparison remains as defence against a changed manager
            # map or an unexpected writer that does not obey the working database.
            latest_manager_map: Mapping[str, Any] = prewrite_manager_map
            latest_calls = prewrite_calls
            latest_identities = prewrite_identities
            latest_errors = prewrite_errors
            # Three compensation writes are allowed; the fourth pass is a read-only
            # stability check so a third successful correction can actually settle.
            for _cas_attempt in range(4):
                latest_manager_map = read_manager_map(manager_path)
                latest_calls, latest_identities, latest_errors = scanned(latest_manager_map)
                block_on_identity_errors(state, state_path, latest_errors)
                require_closed_analysis_cost(analysis_cost_summary(latest_calls))
                changed = [
                    key
                    for key in selected
                    if key not in latest_calls
                    or latest_calls[key].get("source_fingerprint")
                    != state["entries"][key].get("source_fingerprint")
                ]
                if not changed:
                    break
                if _cas_attempt == 3:
                    raise RuntimeError("source kept changing during compensating publication")
                state, correction_selected = reserve(
                    state,
                    latest_calls,
                    final_rows,
                    final_call_to_row,
                    limit=len(changed),
                )
                if set(correction_selected) != set(changed):
                    raise RuntimeError("changed source cannot be compensated safely")
                atomic_owner_json(state_path, state)
                correction_requests, correction_expected = build_batch(
                    sheet_id=sheet_id,
                    rows=final_rows,
                    row_to_call=_final_row_to_call,
                    calls=latest_calls,
                    identities=latest_identities,
                    state=state,
                    selected=correction_selected,
                    summary_width_px=int(config["summary_width_px"]),
                )
                gateway.batch_sheet_requests(correction_requests)
                request_count += len(correction_requests)
                _final_header, final_rows = normalize_values(gateway.values(sheet_title))
                final_call_to_row, _final_row_to_call = reconciled(final_rows)
                verify_sheet_snapshot(
                    rows=final_rows,
                    call_to_row=final_call_to_row,
                    identities=latest_identities,
                    state=state,
                    exact_keys=correction_selected,
                    unchanged_hashes=unchanged,
                    recoverable_keys=latest_calls,
                )
                if [physical_row_hash(row) for row in final_rows] != [
                    physical_row_hash(row) for row in correction_expected
                ]:
                    raise RuntimeError("Google compensating readback mismatch")
            physical = {key: final_rows[index] for key, index in final_call_to_row.items()}
            health = apply_incidents(state, latest_errors)
            latest_cost = analysis_cost_summary(latest_calls)
            require_closed_analysis_cost(latest_cost)
            verified = finalize_verified(
                state, selected, latest_calls, physical, state_path
            )
        write_sync_done(db_path, verified, latest_manager_map)
        return {
            "status": "published",
            "published": len(selected),
            "google_rows": len(final_rows),
            "requests": request_count,
            "data_errors": len(latest_errors),
            "data_error_codes": data_error_counts(latest_errors),
            "analysis_cost": latest_cost,
            "balance": publication_balance(
                latest_identities, latest_calls, latest_errors, state,
                final_rows, final_call_to_row,
            ),
            "health": health,
            "external_write": True,
        }


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Exit 0 = green, 2 = open data incident, 1 = the run itself failed."""
    try:
        report = run(argv)
    except Exception as exc:  # noqa: BLE001 - structured fail-closed CLI
        # stdout of this script is read by an operator and by launchd logs.  An
        # exception message here can carry a transcript fragment, a phone number
        # or a file path, so none of it is printed: only the stage, the class
        # and an irreversible digest that two runs can still compare.
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": safe_error_text("publish", exc),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 1
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    balance = report.get("balance")
    if isinstance(balance, Mapping) and not bool(balance.get("balanced")):
        return 2
    cost = report.get("analysis_cost")
    if isinstance(cost, Mapping) and not bool(cost.get("balanced", 1)):
        return 2
    return 0 if str((report.get("health") or {}).get("status") or "") == "green" else 2


if __name__ == "__main__":
    raise SystemExit(main())
