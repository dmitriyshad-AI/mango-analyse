from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import subprocess
from collections import Counter
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence
from zoneinfo import ZoneInfo

from mango_mvp.productization.capture_staging import atomic_write_private_json
from mango_mvp.productization.mango_office_client import DEFAULT_STATS_FIELDS
from mango_mvp.productization.owner_only_io import (
    read_stable_regular_bytes,
    stable_regular_file_evidence,
)


MOSCOW = ZoneInfo("Europe/Moscow")
READY_MANIFEST_SCHEMA = "mango_calls_ready_v3"
DUAL_ENUMERATION_SCHEMA = "mango_exact_dual_enumeration_v2"
DUAL_ENUMERATION_NORMALIZATION = "mango_rows_call_day_v2"
MANGO_OFFICIAL_TOTAL_SCHEMA = "mango_extended_total_pages_v1"
MANGO_OFFICIAL_PAGE_LIMIT = 5000
CUTOVER_MANIFEST_SCHEMA = "mango_calls_cutover_v2"
PREVIOUS_HOST_SHUTDOWN_SNAPSHOT_SCHEMA = (
    "mango_calls_previous_host_shutdown_snapshot_v1"
)
EXTERNAL_WATCHDOG_SCHEMA = "m1_mango_calls_external_watchdog_observation_v1"
EXTERNAL_WATCHDOG_VERDICT_SCHEMA = "m1_mango_calls_external_watchdog_verdict_v1"
STAGE10_SCHEMA = "mango_calls_stage10_verdict_v2"
REQUIRED_CALLS_LAUNCHD_LABELS = (
    "com.mango.calls-two-processes",
    "com.mango.calls-process-a",
    "com.mango.calls-process-b",
    "com.mango.calls-capture",
    "com.mango.calls-pipeline",
    "com.mango.calls-watchdog",
    "com.mango.calls-publication-close-0600",
    "com.mango.calls-publication-close-0700",
    "com.mango.calls-publication-close-0800",
    "com.mango.calls-publication-alert-0830",
    "com.mango.calls-publication-status-0850",
)
REQUIRED_CALLS_LOCK_NAMES = ("process_a", "capture", "pipeline", "process_b")
CALLS_PROCESS_MATCHER_VERSION = "mango_calls_runtime_matchers_v1"
APPROVED_MODELS: Mapping[str, Mapping[str, Any]] = {
    "whisper": {
        "provider": "mlx",
        "model": "mlx-community/whisper-large-v3-mlx",
        "weights_revision": "49e6aa286ad60c14352c404340ded53710378a11",
        "library": "mlx-whisper",
        "library_version": "0.4.3",
        "language": "ru",
        "condition_on_previous_text": False,
        "word_timestamps": True,
        "split_stereo_channels": True,
        "cache_policy": "clear_free_cache_after_primary_file",
    },
    "gigaam": {
        "provider": "gigaam",
        "model": "v2_rnnt",
        "library": "gigaam",
        "library_version": "0.1.0",
        "device": "cpu",
        "starts_after_whisper_exit": True,
    },
    "resolve": {
        "provider": "codex_cli",
        "model": "gpt-5.4",
        "reasoning": "medium",
        "prompt_version": "resolve_dialogue_v1",
        "codex_cli_version": "0.142.3",
        "ephemeral": True,
        "external_processing": True,
    },
    "analyze": {
        "provider": "codex_cli",
        "model": "gpt-5.4-mini",
        "reasoning": "medium",
        "prompt_version": "analyze_compact_full_v1",
        "codex_cli_version": "0.142.3",
        "ephemeral": True,
        "external_processing": True,
    },
}
QUARANTINE_STATUSES = {
    "audio_integrity_quarantined",
    "recording_retry_expired",
    "multiple_recordings_needs_review",
    "duplicate_recording",
}
QUARANTINE_MANAGER_GUIDANCE: Mapping[str, tuple[str, str]] = {
    "quarantine_evidence_incomplete": (
        "Карантин зарегистрирован, но подтверждённая причина отсутствует.",
        "Проверить исходные данные и восстановить доказательство причины карантина.",
    ),
    "audio_integrity_quarantined": (
        "Контрольная сумма сохранённой аудиозаписи изменилась.",
        "Восстановить исходный файл по контрольной сумме или оставить звонок в карантине.",
    ),
    "recording_retry_expired": (
        "Аудиозапись не появилась в Mango в течение 72 часов.",
        "Проверить запись в Mango и повторить загрузку вручную, если файл появился.",
    ),
    "multiple_recordings_needs_review": (
        "Для звонка найдено несколько аудиозаписей.",
        "Выбрать соответствующую звонку запись вручную.",
    ),
    "duplicate_recording": (
        "Аудиозапись уже связана с другим событием звонка.",
        "Проверить связь звонка и записи, затем оставить только каноническое событие.",
    ),
    "dead_letter_transcribe": (
        "Распознавание не завершилось после допустимых попыток.",
        "Проверить аудиофайл и повторить распознавание вручную.",
    ),
    "dead_letter_resolve": (
        "Разделение ролей не завершилось после допустимых попыток.",
        "Проверить расшифровку и повторить разделение ролей вручную.",
    ),
    "dead_letter_analyze": (
        "Смысловой анализ не завершился после допустимых попыток.",
        "Проверить подготовленный диалог и повторить смысловой анализ вручную.",
    ),
}
SAFE_ALERT_KEYS = {
    "schema_version",
    "status",
    "stop_reason",
    "checked_at",
    "checked_through",
    "data_through",
    "mango_enumeration_complete",
    "consistency_ok",
    "closure_ok",
    "mango_unique",
    "ready_unique",
    "quarantine_unique",
    "pending_unique",
    "unexplained_missing",
    "pending_over_sla",
    "oldest_pending_age_minutes",
    "free_bytes",
    "required_free_bytes",
    "heartbeat_age_seconds",
    "foreign_host_count",
    "watchdog_alive",
}
SENSITIVE_ALERT_RE = re.compile(
    r"(?:\+7|\b8)[\s\-(]*\d{3}[\s\-)]*\d{3}[\s\-]*\d{2}[\s\-]*\d{2}"
    r"|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    r"|(?:token|secret|api[_-]?key|authorization)\s*[:=]",
    re.IGNORECASE,
)


def _canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _is_sha256(value: Any) -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{64}", str(value or "")))


def _official_total_proof_is_green(
    proof: Any,
    *,
    expected_count: int,
    expected_call_keys_sha256: str,
) -> bool:
    if not isinstance(proof, Mapping):
        return False
    proof_body = {key: value for key, value in proof.items() if key != "proof_sha256"}
    pages = proof.get("pages")
    total = proof.get("total_calls_count")
    if not (
        proof.get("schema_version") == MANGO_OFFICIAL_TOTAL_SCHEMA
        and type(proof.get("page_limit")) is int
        and proof.get("page_limit") == MANGO_OFFICIAL_PAGE_LIMIT
        and proof.get("proof_sha256") == _canonical_json_sha256(proof_body)
        and proof.get("complete") is True
        and type(total) is int
        and total == expected_count
        and proof.get("call_keys_sha256") == expected_call_keys_sha256
        and isinstance(pages, Sequence)
        and not isinstance(pages, (str, bytes))
        and type(proof.get("pages_count")) is int
        and proof.get("pages_count") == len(pages)
        and len(pages) >= 1
    ):
        return False
    offset = 0
    for page in pages:
        if not isinstance(page, Mapping):
            return False
        rows = page.get("rows")
        if not (
            type(page.get("offset")) is int
            and page.get("offset") == offset
            and type(rows) is int
            and 0 <= rows <= MANGO_OFFICIAL_PAGE_LIMIT
            and type(page.get("total_calls_count")) is int
            and page.get("total_calls_count") == total
            and page.get("status") == "complete"
            and _is_sha256(page.get("entry_ids_sha256"))
        ):
            return False
        offset += rows
    if offset != total:
        return False
    if total == 0:
        return len(pages) == 1 and pages[0].get("rows") == 0
    last_rows = pages[-1].get("rows")
    if last_rows == 0:
        return (
            len(pages) >= 2
            and pages[-1].get("offset") == total
            and pages[-2].get("rows") == MANGO_OFFICIAL_PAGE_LIMIT
        )
    return type(last_rows) is int and last_rows < MANGO_OFFICIAL_PAGE_LIMIT


def _ready_capture_proof_is_green(manifest: Mapping[str, Any]) -> bool:
    proof = manifest.get("capture_proof")
    if not isinstance(proof, Mapping):
        return False
    if manifest.get("capture_proof_sha256") != _canonical_json_sha256(proof):
        return False
    source = manifest.get("mango_enumeration_source")
    proof_source = proof.get("mango_enumeration_source")
    if not isinstance(source, Mapping) or not isinstance(proof_source, Mapping):
        return False
    raw_intervals = source.get("covered_intervals")
    if not isinstance(raw_intervals, Sequence) or isinstance(
        raw_intervals, (str, bytes)
    ):
        return False
    intervals = [item for item in raw_intervals if isinstance(item, Mapping)]
    if len(intervals) != len(raw_intervals):
        return False
    source_projection = {
        key: source.get(key)
        for key in (
            "mode",
            "since",
            "rolling_since",
            "until",
            "cursor",
            "pages",
            "pagination",
            "requests",
            "catch_up",
            "enumeration_consistency_ok",
            "dual_enumeration",
        )
    } | {
        "covered_intervals": [
            {
                key: interval.get(key)
                for key in (
                    "since",
                    "until",
                    "result_complete",
                    "rows",
                    "scope",
                    "authority_pass",
                )
            }
            for interval in intervals
        ]
    }
    if dict(proof_source) != source_projection:
        return False
    dual = source.get("dual_enumeration")
    passes = dual.get("passes") if isinstance(dual, Mapping) else None
    if not isinstance(passes, Sequence) or len(passes) != 2:
        return False
    primary = passes[0]
    if not isinstance(primary, Mapping):
        return False
    call_keys = proof.get("call_keys")
    calls_by_day = proof.get("calls_by_moscow_day")
    zero_by_day = proof.get("independent_zero_enumerations_by_day")
    if (
        not isinstance(call_keys, Sequence)
        or isinstance(call_keys, (str, bytes))
        or list(call_keys) != primary.get("call_keys")
        or not isinstance(calls_by_day, Mapping)
        or dict(calls_by_day) != primary.get("calls_by_moscow_day")
        or not isinstance(zero_by_day, Mapping)
    ):
        return False
    verdicts = manifest.get("daily_verdicts")
    if not isinstance(verdicts, Mapping):
        return False
    for day_key, zero_count in zero_by_day.items():
        verdict = verdicts.get(day_key)
        if (
            type(zero_count) is not int
            or not isinstance(verdict, Mapping)
            or verdict.get("independent_zero_enumerations") != zero_count
        ):
            return False
    authoritative_rows = sum(
        item.get("rows", 0)
        for item in intervals
        if item.get("scope") == "rolling_authority"
        and type(item.get("rows")) is int
    )
    auxiliary_rows = sum(
        item.get("rows", 0)
        for item in intervals
        if item.get("scope") == "recovery_auxiliary"
        and type(item.get("rows")) is int
    )
    return bool(
        proof.get("mango_enumeration_complete")
        == manifest.get("mango_enumeration_complete")
        and proof.get("api_requests") == source.get("requests") == len(intervals)
        and proof.get("api_authoritative_rows_total") == authoritative_rows
        and proof.get("api_auxiliary_rows_total") == auxiliary_rows
        and proof.get("api_rows_total") == authoritative_rows + auxiliary_rows
        and proof.get("api_events_total") == len(call_keys)
    )


def _dual_source_proof_is_green(source: Any) -> bool:
    if not isinstance(source, Mapping):
        return False
    proof = source.get("dual_enumeration")
    if not isinstance(proof, Mapping):
        return False
    proof_body = {key: value for key, value in proof.items() if key != "proof_sha256"}
    if proof.get("proof_sha256") != _canonical_json_sha256(proof_body):
        return False
    passes = proof.get("passes")
    comparison = proof.get("comparison")
    required_comparisons = {
        "normalized_unique_count_equal",
        "call_keys_equal",
        "call_keys_sha256_equal",
        "calls_by_moscow_day_equal",
        "calls_by_moscow_day_sha256_equal",
        "event_digest_sha256_equal",
        "primary_raw_balance_ok",
        "verification_raw_balance_ok",
        "partition_sha256_different",
        "official_total_equal",
    }
    if not (
        proof.get("schema_version") == DUAL_ENUMERATION_SCHEMA
        and proof.get("normalization_version")
        == DUAL_ENUMERATION_NORMALIZATION
        and type(proof.get("passes_required")) is int
        and proof.get("passes_required") == 2
        and type(proof.get("passes_completed")) is int
        and proof.get("passes_completed") == 2
        and isinstance(passes, Sequence)
        and not isinstance(passes, (str, bytes))
        and len(passes) == 2
        and [
            item.get("pass_id") if isinstance(item, Mapping) else None
            for item in passes
        ]
        == ["primary", "verification"]
        and isinstance(comparison, Mapping)
        and set(comparison) == required_comparisons
        and all(comparison.get(key) is True for key in required_comparisons)
        and proof.get("enumeration_consistency_ok") is True
        and source.get("enumeration_consistency_ok") is True
        and proof.get("mismatch_reason") == ""
        and str(proof.get("proof_run_id") or "").strip()
        and str(proof.get("tenant_id") or "").strip()
        and str(proof.get("base_url") or "").strip()
        and proof.get("fields_sha256")
        == _canonical_json_sha256(DEFAULT_STATS_FIELDS)
    ):
        return False
    try:
        _parse_strict_aware_datetime(proof.get("observed_at"))
        rolling_since = _parse_strict_aware_datetime(source.get("rolling_since"))
        until = _parse_strict_aware_datetime(source.get("until"))
        if (
            rolling_since >= until
            or _parse_strict_aware_datetime(proof.get("rolling_since"))
            != rolling_since
            or _parse_strict_aware_datetime(proof.get("until")) != until
        ):
            return False
    except (TypeError, ValueError):
        return False

    pass_facts: list[dict[str, Any]] = []
    pass_chunks: dict[int, list[Mapping[str, Any]]] = {}
    for pass_number, payload in enumerate(passes, start=1):
        if not isinstance(payload, Mapping):
            return False
        try:
            if (
                _parse_strict_aware_datetime(payload.get("rolling_since"))
                != rolling_since
                or _parse_strict_aware_datetime(payload.get("until")) != until
            ):
                return False
        except (TypeError, ValueError):
            return False
        requests = payload.get("requests")
        raw_rows = payload.get("raw_rows")
        chunks = payload.get("chunks")
        if (
            isinstance(requests, bool)
            or not isinstance(requests, int)
            or requests <= 0
            or isinstance(raw_rows, bool)
            or not isinstance(raw_rows, int)
            or raw_rows < 0
            or not isinstance(chunks, Sequence)
            or isinstance(chunks, (str, bytes))
            or len(chunks) != requests
        ):
            return False
        canonical_chunks: list[Mapping[str, Any]] = []
        chunk_cursor = rolling_since
        chunk_rows_total = 0
        for chunk in chunks:
            if not isinstance(chunk, Mapping):
                return False
            rows = chunk.get("rows")
            if (
                isinstance(rows, bool)
                or not isinstance(rows, int)
                or rows < 0
                or chunk.get("result_complete") is not True
            ):
                return False
            try:
                chunk_since = _parse_strict_aware_datetime(chunk.get("since"))
                chunk_until = _parse_strict_aware_datetime(chunk.get("until"))
            except (TypeError, ValueError):
                return False
            if (
                chunk_since != chunk_cursor
                or chunk_since >= chunk_until
                or chunk_since.microsecond
                or chunk_until.microsecond
            ):
                return False
            chunk_cursor = chunk_until
            chunk_rows_total += rows
            canonical_chunks.append(
                {
                    "since": chunk.get("since"),
                    "until": chunk.get("until"),
                    "result_complete": True,
                    "rows": rows,
                }
            )
        if chunk_cursor != until or chunk_rows_total != raw_rows:
            return False
        partition_sha256 = _canonical_json_sha256(
            [{"since": item["since"], "until": item["until"]} for item in canonical_chunks]
        )
        if payload.get("partition_sha256") != partition_sha256:
            return False

        multiset = payload.get("call_key_multiset")
        call_keys = payload.get("call_keys")
        calls_by_day = payload.get("calls_by_moscow_day")
        if (
            not isinstance(multiset, Sequence)
            or isinstance(multiset, (str, bytes))
            or not isinstance(call_keys, Sequence)
            or isinstance(call_keys, (str, bytes))
            or not isinstance(calls_by_day, Mapping)
        ):
            return False
        canonical_multiset = list(multiset)
        canonical_keys = list(call_keys)
        if (
            any(
                not isinstance(value, str)
                or not value
                or value != value.strip()
                for value in (*canonical_multiset, *canonical_keys)
            )
            or canonical_multiset != sorted(canonical_multiset)
            or len(canonical_multiset) != raw_rows
            or canonical_keys != sorted(set(canonical_keys))
            or sorted(set(canonical_multiset)) != canonical_keys
            or type(payload.get("normalized_unique_count")) is not int
            or payload.get("normalized_unique_count") != len(canonical_keys)
            or payload.get("call_key_multiset_sha256")
            != _canonical_json_sha256(canonical_multiset)
            or payload.get("call_keys_sha256")
            != _canonical_json_sha256(canonical_keys)
            or not _is_sha256(payload.get("raw_rows_sha256"))
            or not _is_sha256(payload.get("event_digest_sha256"))
        ):
            return False
        balance_fields = (
            "recordable_unique_rows",
            "without_recording_rows",
            "proven_duplicate_rows",
            "quarantined_rows",
            "error_rows",
            "unexplained_rows",
        )
        balance = [payload.get(field) for field in balance_fields]
        if (
            any(type(value) is not int or value < 0 for value in balance)
            or balance[0] + balance[1] != len(canonical_keys)
            or balance[2] != raw_rows - len(canonical_keys)
            or sum(balance) != raw_rows
            or any(balance[index] != 0 for index in (3, 4, 5))
            or payload.get("raw_balance_ok") is not True
        ):
            return False
        canonical_days: dict[str, list[str]] = {}
        for raw_day, raw_values in calls_by_day.items():
            try:
                day_key = date.fromisoformat(str(raw_day)).isoformat()
            except ValueError:
                return False
            if (
                raw_day != day_key
                or not isinstance(raw_values, Sequence)
                or isinstance(raw_values, (str, bytes))
            ):
                return False
            values = list(raw_values)
            if values != sorted(set(values)):
                return False
            canonical_days[day_key] = values
        flattened = [value for values in canonical_days.values() for value in values]
        if (
            sorted(flattened) != canonical_keys
            or len(flattened) != len(set(flattened))
            or payload.get("calls_by_moscow_day_sha256")
            != _canonical_json_sha256(
                {key: canonical_days[key] for key in sorted(canonical_days)}
            )
        ):
            return False
        facts = {
            key: payload.get(key)
            for key in (
                "normalized_unique_count",
                "call_keys",
                "call_keys_sha256",
                "calls_by_moscow_day",
                "calls_by_moscow_day_sha256",
                "event_digest_sha256",
            )
        }
        facts["chunks"] = canonical_chunks
        facts["partition_sha256"] = partition_sha256
        facts["raw_balance_ok"] = payload.get("raw_balance_ok")
        pass_facts.append(facts)
        pass_chunks[pass_number] = canonical_chunks

    computed_comparison = {
        f"{field}_equal": pass_facts[0][field] == pass_facts[1][field]
        for field in (
            "normalized_unique_count",
            "call_keys",
            "call_keys_sha256",
            "calls_by_moscow_day",
            "calls_by_moscow_day_sha256",
            "event_digest_sha256",
        )
    }
    computed_comparison["primary_raw_balance_ok"] = pass_facts[0][
        "raw_balance_ok"
    ] is True
    computed_comparison["verification_raw_balance_ok"] = pass_facts[1][
        "raw_balance_ok"
    ] is True
    computed_comparison["partition_sha256_different"] = (
        pass_facts[0]["partition_sha256"] != pass_facts[1]["partition_sha256"]
    )
    computed_comparison["official_total_equal"] = _official_total_proof_is_green(
        proof.get("official_total"),
        expected_count=pass_facts[0]["normalized_unique_count"],
        expected_call_keys_sha256=pass_facts[0]["call_keys_sha256"],
    )
    if dict(comparison) != computed_comparison or not all(
        computed_comparison.values()
    ):
        return False

    intervals = source.get("covered_intervals")
    requests = source.get("requests")
    if (
        not isinstance(intervals, Sequence)
        or isinstance(intervals, (str, bytes))
        or isinstance(requests, bool)
        or not isinstance(requests, int)
        or len(intervals) != requests
    ):
        return False
    observed: dict[int, list[Mapping[str, Any]]] = {1: [], 2: []}
    for interval in intervals:
        if not isinstance(interval, Mapping):
            return False
        scope = interval.get("scope")
        if scope == "rolling_authority":
            authority_pass = interval.get("authority_pass")
            if authority_pass not in {1, 2} or isinstance(authority_pass, bool):
                return False
            observed[authority_pass].append(
                {
                    "since": interval.get("since"),
                    "until": interval.get("until"),
                    "result_complete": interval.get("result_complete"),
                    "rows": interval.get("rows"),
                }
            )
        elif scope == "recovery_auxiliary":
            if "authority_pass" in interval:
                return False
            try:
                if (
                    _parse_strict_aware_datetime(interval.get("until"))
                    > rolling_since
                ):
                    return False
            except (TypeError, ValueError):
                return False
        else:
            return False
    return observed == pass_chunks


def parse_aware_datetime(value: Any) -> datetime:
    text_value = str(value or "").strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text_value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _parse_strict_aware_datetime(value: Any) -> datetime:
    text_value = str(value or "").strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text_value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def moscow_day_bounds_utc(day: date) -> tuple[datetime, datetime]:
    start = datetime.combine(day, time.min, MOSCOW)
    return start.astimezone(timezone.utc), (start + timedelta(days=1)).astimezone(timezone.utc)


def event_is_on_moscow_day(started_at: Any, day: date) -> bool:
    try:
        return parse_aware_datetime(started_at).astimezone(MOSCOW).date() == day
    except (TypeError, ValueError):
        return False


def validate_quarantine_items_payload(
    value: Any,
    *,
    day: date,
    expected_count: int,
    expected_without_reason: Optional[int] = None,
) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ["daily_verdict_quarantine_items_invalid"]
    keys: set[str] = set()
    normalized_order: list[tuple[datetime, str]] = []
    incomplete_count = 0
    for raw_item in value:
        if not isinstance(raw_item, Mapping) or set(raw_item) != {
            "call_key",
            "started_at",
            "code",
            "reason",
            "action",
        }:
            return ["daily_verdict_quarantine_items_invalid"]
        call_key = str(raw_item.get("call_key") or "").strip()
        code = str(raw_item.get("code") or "")
        guidance = QUARANTINE_MANAGER_GUIDANCE.get(code)
        if (
            not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}", call_key)
            or call_key in keys
            or guidance is None
            or raw_item.get("reason") != guidance[0]
            or raw_item.get("action") != guidance[1]
        ):
            return ["daily_verdict_quarantine_items_invalid"]
        try:
            started_at = parse_aware_datetime(raw_item.get("started_at"))
        except (TypeError, ValueError):
            return ["daily_verdict_quarantine_items_invalid"]
        if started_at.astimezone(MOSCOW).date() != day:
            return ["daily_verdict_quarantine_items_invalid"]
        keys.add(call_key)
        normalized_order.append((started_at, call_key))
        incomplete_count += code == "quarantine_evidence_incomplete"
    if (
        len(keys) != expected_count
        or normalized_order != sorted(normalized_order)
        or (
            expected_without_reason is not None
            and incomplete_count != expected_without_reason
        )
    ):
        return ["daily_verdict_quarantine_items_invalid"]
    return []


def sha256_file(path: Path) -> str:
    return str(stable_regular_file_evidence(path)["sha256"])


def _safe_git_environment() -> Mapping[str, str]:
    return {
        key: value
        for key, value in os.environ.items()
        if not key.upper().startswith("GIT_")
    }


def _safe_git_output(project_root: Path, *args: str) -> str:
    root = project_root.resolve(strict=True)
    return subprocess.check_output(
        [
            "/usr/bin/git",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.untrackedCache=false",
            "-C",
            str(root),
            *args,
        ],
        cwd=root,
        env=dict(_safe_git_environment()),
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()


def current_git_sha(project_root: Path) -> str:
    root = project_root.resolve(strict=True)
    top = Path(_safe_git_output(root, "rev-parse", "--show-toplevel")).resolve()
    if top != root:
        raise RuntimeError("git worktree root mismatch")
    return _safe_git_output(root, "rev-parse", "HEAD")


def git_worktree_is_clean(project_root: Path) -> bool:
    try:
        root = project_root.resolve(strict=True)
        top = Path(
            _safe_git_output(root, "rev-parse", "--show-toplevel")
        ).resolve()
        if top != root:
            return False
        tracked = _safe_git_output(root, "ls-files", "-v", "-z")
        if any(
            not entry.startswith("H ")
            for entry in tracked.split("\0")
            if entry
        ):
            return False
        status = _safe_git_output(
            root, "status", "--porcelain=v1", "--untracked-files=all"
        )
        command = [
            "/usr/bin/git",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.untrackedCache=false",
            "-C",
            str(root),
        ]
        environment = dict(_safe_git_environment())
        worktree_diff = subprocess.run(
            [*command, "diff-files", "--quiet", "--ignore-submodules=none", "--"],
            cwd=root,
            env=environment,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        index_diff = subprocess.run(
            [
                *command,
                "diff-index",
                "--cached",
                "--quiet",
                "--ignore-submodules=none",
                "HEAD",
                "--",
            ],
            cwd=root,
            env=environment,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
    except (OSError, subprocess.SubprocessError):
        return False
    return not status and worktree_diff == 0 and index_diff == 0


def approved_runtime_fingerprint() -> Mapping[str, Any]:
    return json.loads(json.dumps(APPROVED_MODELS, ensure_ascii=False))


def validate_runtime_fingerprint(value: Any) -> list[str]:
    if not isinstance(value, Mapping):
        return ["runtime_fingerprint_missing"]
    errors: list[str] = []
    for stage, expected in APPROVED_MODELS.items():
        actual = value.get(stage)
        if not isinstance(actual, Mapping):
            errors.append(f"{stage}_fingerprint_missing")
            continue
        for key, expected_value in expected.items():
            if actual.get(key) != expected_value:
                errors.append(f"{stage}_{key}_mismatch")
    return errors


def load_owner_only_json(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        raw = read_stable_regular_bytes(
            path, label=label, owner_only_mode=0o600
        )
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label}_invalid_json") from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"{label}_must_be_json_object")
    return payload


def read_host_id(path: Path) -> str:
    try:
        value = read_stable_regular_bytes(
            path, label="host_id", owner_only_mode=0o600
        ).decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise RuntimeError("host_id_invalid_encoding") from exc
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{2,127}", value):
        raise RuntimeError("host_id_invalid")
    return value


def verify_cutover_authority(
    *,
    cutover_manifest_path: Path,
    host_id_path: Path,
    previous_host_snapshot_path: Optional[Path] = None,
    expected_previous_host_id: Optional[str] = None,
    expected_code_sha: str,
    project_root: Path,
    expected_source_cursor_sha256: Optional[str] = None,
    now: Optional[datetime] = None,
    proof_max_age_minutes: int = 90,
    require_fresh_previous_host_proof: bool = False,
) -> Mapping[str, Any]:
    manifest = load_owner_only_json(cutover_manifest_path, label="cutover_manifest")
    host_id = read_host_id(host_id_path)
    current_sha = current_git_sha(project_root)
    errors: list[str] = []
    if manifest.get("schema_version") != CUTOVER_MANIFEST_SCHEMA:
        errors.append("cutover_schema_mismatch")
    if manifest.get("active_host_id") != host_id:
        errors.append("cutover_host_mismatch")
    if not re.fullmatch(r"[0-9a-f]{40}", expected_code_sha or ""):
        errors.append("expected_code_sha_invalid")
    if manifest.get("expected_code_sha") != expected_code_sha or current_sha != expected_code_sha:
        errors.append("cutover_code_sha_mismatch")
    if not git_worktree_is_clean(project_root):
        errors.append("cutover_worktree_dirty_or_unverifiable")
    cursor_sha = str(manifest.get("source_cursor_sha256") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", cursor_sha):
        errors.append("source_cursor_sha256_invalid")
    if expected_source_cursor_sha256 and cursor_sha != expected_source_cursor_sha256:
        errors.append("source_cursor_sha256_mismatch")
    snapshot_sha = str(manifest.get("previous_host_snapshot_sha256") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", snapshot_sha):
        errors.append("previous_host_snapshot_sha256_invalid")
    snapshot: Mapping[str, Any] = {}
    snapshot_loaded = False
    actual_snapshot_sha = ""
    if previous_host_snapshot_path is None:
        errors.append("previous_host_snapshot_missing_or_invalid")
    else:
        try:
            snapshot_raw = read_stable_regular_bytes(
                previous_host_snapshot_path,
                label="previous_host_snapshot",
                owner_only_mode=0o600,
            )
            decoded_snapshot = json.loads(snapshot_raw.decode("utf-8"))
            if not isinstance(decoded_snapshot, Mapping):
                raise ValueError("snapshot must be a JSON object")
            snapshot = decoded_snapshot
            snapshot_loaded = True
            actual_snapshot_sha = hashlib.sha256(snapshot_raw).hexdigest()
        except (OSError, RuntimeError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
            errors.append("previous_host_snapshot_missing_or_invalid")
    if actual_snapshot_sha and actual_snapshot_sha != snapshot_sha:
        errors.append("previous_host_snapshot_sha256_mismatch")
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    parsed_times: dict[str, datetime] = {}
    for field in ("previous_host_disabled_at", "previous_host_checked_at", "approved_at"):
        try:
            stamp = _parse_strict_aware_datetime(manifest.get(field))
        except (TypeError, ValueError):
            errors.append(f"{field}_invalid")
            continue
        parsed_times[field] = stamp
        if stamp > current:
            errors.append(f"{field}_in_future")
        if (
            require_fresh_previous_host_proof
            and field == "previous_host_checked_at"
            and current - stamp > timedelta(
            minutes=max(1, proof_max_age_minutes)
            )
        ):
            errors.append("previous_host_proof_stale")
    disabled = parsed_times.get("previous_host_disabled_at")
    checked = parsed_times.get("previous_host_checked_at")
    approved = parsed_times.get("approved_at")
    if disabled and checked and approved and not (disabled <= checked <= approved <= current):
        errors.append("cutover_chronology_invalid")
    if not str(manifest.get("approved_by") or "").strip():
        errors.append("approved_by_missing")

    manifest_previous_host_id = str(manifest.get("previous_host_id") or "")
    snapshot_previous_host_id = str(snapshot.get("previous_host_id") or "")
    if not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]{2,127}", expected_previous_host_id or ""
    ):
        errors.append("expected_previous_host_id_invalid")
    if manifest_previous_host_id != expected_previous_host_id:
        errors.append("cutover_previous_host_id_mismatch")
    if snapshot_loaded:
        if snapshot.get("schema_version") != PREVIOUS_HOST_SHUTDOWN_SNAPSHOT_SCHEMA:
            errors.append("previous_host_snapshot_schema_mismatch")
        if not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]{2,127}", snapshot_previous_host_id
        ):
            errors.append("previous_host_snapshot_host_id_invalid")
        elif snapshot_previous_host_id != expected_previous_host_id:
            errors.append("previous_host_snapshot_host_id_mismatch")
        elif snapshot_previous_host_id == host_id:
            errors.append("previous_host_snapshot_reuses_active_host")
        if snapshot.get("source_cursor_sha256") != cursor_sha:
            errors.append("previous_host_snapshot_cursor_sha256_mismatch")
        if snapshot.get("probe_ok") is not True:
            errors.append("previous_host_probe_unproven")
        if snapshot.get("launchd_scan_complete") is not True:
            errors.append("previous_host_launchd_scan_incomplete")
        if snapshot.get("process_scan_complete") is not True:
            errors.append("previous_host_process_scan_incomplete")
        if snapshot.get("process_matcher_version") != CALLS_PROCESS_MATCHER_VERSION:
            errors.append("previous_host_process_matcher_mismatch")
        if snapshot.get("plist_scan_complete") is not True:
            errors.append("previous_host_plist_scan_incomplete")
        if snapshot.get("cron_scan_complete") is not True:
            errors.append("previous_host_cron_scan_incomplete")
        if snapshot.get("lock_scan_complete") is not True:
            errors.append("previous_host_lock_scan_incomplete")

        checked_labels = snapshot.get("checked_launchd_labels")
        checked_label_set: set[str] = set()
        if (
            not isinstance(checked_labels, Sequence)
            or isinstance(checked_labels, (str, bytes))
            or any(
                not isinstance(label, str)
                or not re.fullmatch(r"com\.mango\.calls[-A-Za-z0-9.]*", label)
                for label in checked_labels
            )
            or len(set(checked_labels)) != len(checked_labels)
        ):
            errors.append("previous_host_checked_labels_invalid")
        else:
            checked_label_set = set(checked_labels)
        if not set(REQUIRED_CALLS_LAUNCHD_LABELS).issubset(checked_label_set):
            errors.append("previous_host_required_labels_unchecked")

        checked_locks = snapshot.get("checked_lock_names")
        if (
            not isinstance(checked_locks, Sequence)
            or isinstance(checked_locks, (str, bytes))
            or any(not isinstance(name, str) for name in checked_locks)
            or len(set(checked_locks)) != len(checked_locks)
            or not set(REQUIRED_CALLS_LOCK_NAMES).issubset(set(checked_locks))
        ):
            errors.append("previous_host_checked_locks_invalid")

        sequence_rules = (
            ("active_calls_labels", str),
            ("active_calls_pids", int),
            ("active_calls_commands", str),
            ("active_calls_plists", str),
            ("active_calls_cron_entries", str),
            ("held_lock_names", str),
        )
        active_evidence = False
        for field, expected_type in sequence_rules:
            values = snapshot.get(field)
            valid = isinstance(values, Sequence) and not isinstance(
                values, (str, bytes)
            )
            if valid and expected_type is int:
                valid = all(
                    not isinstance(value, bool)
                    and isinstance(value, int)
                    and value > 0
                    for value in values
                )
            elif valid:
                valid = all(
                    isinstance(value, str) and bool(value.strip())
                    for value in values
                )
            if not valid:
                errors.append(f"previous_host_{field}_invalid")
            elif values:
                active_evidence = True
        if active_evidence:
            errors.append("previous_host_calls_process_active")

        snapshot_times: dict[str, datetime] = {}
        for snapshot_field in ("captured_at_utc", "previous_host_disabled_at"):
            try:
                snapshot_times[snapshot_field] = _parse_strict_aware_datetime(
                    snapshot.get(snapshot_field)
                )
            except (TypeError, ValueError):
                errors.append(f"previous_host_snapshot_{snapshot_field}_invalid")
        if (
            snapshot_times.get("captured_at_utc")
            and checked
            and snapshot_times["captured_at_utc"] != checked
        ):
            errors.append("previous_host_snapshot_checked_at_mismatch")
        if (
            snapshot_times.get("previous_host_disabled_at")
            and disabled
            and snapshot_times["previous_host_disabled_at"] != disabled
        ):
            errors.append("previous_host_snapshot_disabled_at_mismatch")
    return {
        "ok": not errors,
        "errors": errors,
        "active_host_id": host_id,
        "current_code_sha": current_sha,
        "source_cursor_sha256": cursor_sha or None,
        "previous_host_snapshot_sha256": actual_snapshot_sha or None,
        "previous_host_disabled_at": (
            parsed_times.get("previous_host_disabled_at").isoformat()
            if parsed_times.get("previous_host_disabled_at")
            else None
        ),
        "approved_at": (
            parsed_times.get("approved_at").isoformat()
            if parsed_times.get("approved_at")
            else None
        ),
    }


def validate_external_watchdog_observation(
    observation: Any,
    *,
    expected_active_host_id: str,
    expected_previous_host_id: str,
    expected_code_sha: str,
    expected_cutover_manifest_sha256: str,
    expected_previous_host_snapshot_sha256: str,
    now: Optional[datetime] = None,
    max_observation_age_minutes: int = 20,
    max_heartbeat_age_minutes: int = 5,
) -> Mapping[str, Any]:
    """Validate a safe read-only observation produced outside both Macs."""
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    errors: list[str] = []
    active_old_process = False
    previous_labels_count = 0
    previous_pids_count = 0
    heartbeat_age_minutes: Optional[float] = None

    if not isinstance(observation, Mapping):
        observation = {}
        errors.append("observation_missing")
    if observation.get("schema_version") != EXTERNAL_WATCHDOG_SCHEMA:
        errors.append("observation_schema_mismatch")
    observer_id = str(observation.get("observer_id") or "")
    observer_id_valid = bool(
        re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{2,127}", observer_id)
    )
    if not observer_id_valid:
        errors.append("observer_id_invalid")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{2,127}", expected_active_host_id):
        errors.append("expected_active_host_id_invalid")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{2,127}", expected_previous_host_id):
        errors.append("expected_previous_host_id_invalid")
    if not re.fullmatch(r"[0-9a-f]{40}", expected_code_sha):
        errors.append("expected_code_sha_invalid")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_cutover_manifest_sha256):
        errors.append("expected_cutover_manifest_sha256_invalid")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_previous_host_snapshot_sha256):
        errors.append("expected_previous_host_snapshot_sha256_invalid")
    if observation.get("expected_code_sha") != expected_code_sha:
        errors.append("observation_code_sha_mismatch")
    if (
        observation.get("cutover_manifest_sha256")
        != expected_cutover_manifest_sha256
    ):
        errors.append("observation_cutover_sha_mismatch")
    try:
        observed_at = _parse_strict_aware_datetime(
            observation.get("observed_at_utc")
        )
        observation_age = (current - observed_at).total_seconds() / 60
        if observation_age < 0 or observation_age > max(
            1, max_observation_age_minutes
        ):
            errors.append("observation_stale_or_future")
    except (TypeError, ValueError):
        errors.append("observation_time_invalid")

    previous = observation.get("previous_host")
    if not isinstance(previous, Mapping):
        errors.append("previous_host_probe_unproven")
        previous = {}
    elif previous.get("probe_ok") is not True:
        errors.append("previous_host_probe_unproven")
    if previous.get("host_id") != expected_previous_host_id:
        errors.append("previous_host_id_mismatch")
    if (
        previous.get("shutdown_snapshot_sha256")
        != expected_previous_host_snapshot_sha256
    ):
        errors.append("previous_host_snapshot_sha256_mismatch")
    labels = previous.get("active_calls_labels")
    pids = previous.get("active_calls_pids")
    if (
        not isinstance(labels, Sequence)
        or isinstance(labels, (str, bytes))
        or any(not isinstance(value, str) or not value.strip() for value in labels)
    ):
        errors.append("previous_host_labels_invalid")
        labels = ()
    if (
        not isinstance(pids, Sequence)
        or isinstance(pids, (str, bytes))
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in pids
        )
    ):
        errors.append("previous_host_pids_invalid")
        pids = ()
    previous_labels_count = len(labels)
    previous_pids_count = len(pids)
    active_old_process = bool(previous_labels_count or previous_pids_count)
    if active_old_process:
        errors.append("previous_host_calls_process_active")

    m1 = observation.get("m1")
    if not isinstance(m1, Mapping):
        errors.append("m1_probe_unproven")
        m1 = {}
    elif m1.get("probe_ok") is not True:
        errors.append("m1_probe_unproven")
    if m1.get("host_id") != expected_active_host_id:
        errors.append("m1_host_id_mismatch")
    try:
        heartbeat = _parse_strict_aware_datetime(m1.get("heartbeat_at"))
        heartbeat_age_minutes = max(
            0.0, (current - heartbeat).total_seconds() / 60
        )
        if heartbeat > current or heartbeat_age_minutes > max(
            1, max_heartbeat_age_minutes
        ):
            errors.append("m1_heartbeat_stale_or_future")
    except (TypeError, ValueError):
        errors.append("m1_heartbeat_invalid")

    status = "p0" if active_old_process else "ok" if not errors else "alert"
    return {
        "schema_version": EXTERNAL_WATCHDOG_VERDICT_SCHEMA,
        "status": status,
        "ok": status == "ok",
        "errors": sorted(set(errors)),
        "observer_id_valid": observer_id_valid,
        "previous_host_active_labels_count": previous_labels_count,
        "previous_host_active_pids_count": previous_pids_count,
        "m1_heartbeat_age_minutes": (
            round(heartbeat_age_minutes, 3)
            if heartbeat_age_minutes is not None
            else None
        ),
        "safety": {
            "read_only_observation": True,
            "runs_asr": False,
            "runs_resolve_analyze": False,
            "writes_external_systems": False,
        },
    }


def _entry_dict(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    converter = getattr(value, "to_json_dict", None)
    converted = converter() if callable(converter) else {}
    return converted if isinstance(converted, Mapping) else {}


def _row_dict(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    try:
        return dict(value)
    except (TypeError, ValueError):
        return {}


def _latest_capture_by_call(
    entries: Iterable[Any], day: date
) -> Mapping[str, Mapping[str, Any]]:
    latest: dict[str, Mapping[str, Any]] = {}
    for raw in entries:
        entry = _entry_dict(raw)
        call_key = str(entry.get("provider_call_id") or "").strip()
        if not call_key or not event_is_on_moscow_day(entry.get("started_at"), day):
            continue
        prior = latest.get(call_key)
        if prior is None or str(entry.get("created_at") or "") >= str(
            prior.get("created_at") or ""
        ):
            latest[call_key] = entry
    return latest


def _ready_rows_by_call(
    rows: Iterable[Any], day: date
) -> tuple[Mapping[str, Mapping[str, Any]], int]:
    selected: dict[str, Mapping[str, Any]] = {}
    duplicates = 0
    for raw in rows:
        row = _row_dict(raw)
        call_key = str(row.get("source_call_id") or "").strip()
        if not call_key or not event_is_on_moscow_day(row.get("started_at"), day):
            continue
        if call_key in selected:
            duplicates += 1
        else:
            selected[call_key] = row
    return selected, duplicates


def _json_object(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    try:
        parsed = json.loads(str(value or "{}"))
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, Mapping) else {}


def has_dual_asr_or_exception(
    row: Mapping[str, Any],
    *,
    now: Optional[datetime] = None,
) -> bool:
    variants = _json_object(row.get("transcript_variants_json"))
    exception = variants.get("dual_asr_exception")
    if isinstance(exception, Mapping):
        approved_at_raw = exception.get("approved_at")
        if isinstance(approved_at_raw, str):
            try:
                approved_at = _parse_strict_aware_datetime(approved_at_raw)
            except (TypeError, ValueError):
                approved_at = None
        else:
            approved_at = None
        reason = exception.get("reason")
        approved_by = exception.get("approved_by")
        if (
            exception.get("approved") is True
            and isinstance(reason, str)
            and reason.strip()
            and isinstance(approved_by, str)
            and approved_by.strip()
            and approved_at is not None
            and approved_at <= (now or datetime.now(timezone.utc)).astimezone(
                timezone.utc
            )
        ):
            return True
    if not (
        variants.get("primary_provider") == "mlx"
        and variants.get("secondary_provider") == "gigaam"
    ):
        return False
    mode = str(variants.get("mode") or "").strip()
    if mode == "stereo":
        required_keys = ("manager", "client")
    elif mode == "mono_or_fallback":
        required_keys = ("full",)
    else:
        required_keys = None
    if required_keys is None:
        return False
    for key in required_keys:
        block = variants.get(key)
        if not isinstance(block, Mapping):
            return False
        if not (
            isinstance(block.get("variant_a"), str)
            and block["variant_a"].strip()
            and isinstance(block.get("variant_b"), str)
            and block["variant_b"].strip()
        ):
            return False
    return True


def ready_row_is_complete(
    row: Mapping[str, Any],
    *,
    now: Optional[datetime] = None,
) -> bool:
    return bool(
        str(row.get("transcription_status") or "") == "done"
        and has_dual_asr_or_exception(
            row,
            now=now,
        )
        and str(row.get("resolve_status") or "") in {"done", "skipped"}
        and str(row.get("analysis_status") or "") == "done"
        and _json_object(row.get("analysis_json"))
        and not str(row.get("dead_letter_stage") or "").strip()
        and not any(
            row.get(field)
            for field in (
                "pipeline_stage",
                "pipeline_worker_id",
                "pipeline_claimed_at",
                "analysis_worker_id",
                "analysis_claimed_at",
            )
        )
    )


def enumeration_source_covers_day(
    source: Any, day: date, *, require_full_day: bool = False
) -> bool:
    if not isinstance(source, Mapping):
        return False
    mode = str(source.get("mode") or "")
    if mode == "compatibility_not_for_service":
        return True
    if mode != "strict_service":
        return False
    if (
        not _dual_source_proof_is_green(source)
        or source.get("enumeration_consistency_ok") is not True
        or source.get("cursor") != "not_applicable_stats_request_result"
        or source.get("pagination") != "not_applicable_stats_request_result"
        or "pages" not in source
        or source.get("pages") is not None
        or isinstance(source.get("requests"), bool)
    ):
        return False
    try:
        request_count = int(source.get("requests"))
    except (TypeError, ValueError):
        return False
    if request_count <= 0:
        return False
    day_start, day_end = moscow_day_bounds_utc(day)
    try:
        source_since = parse_aware_datetime(source.get("since"))
        source_until = parse_aware_datetime(source.get("until"))
    except (TypeError, ValueError):
        return False
    if source_since > day_start or source_until <= day_start:
        return False
    if require_full_day and source_until < day_end:
        return False
    source_moscow_day = source_until.astimezone(MOSCOW).date()
    if day > source_moscow_day:
        return False
    coverage_end = (
        day_end
        if require_full_day
        else min(day_end, source_until) if day == source_moscow_day else day_end
    )
    intervals = source.get("covered_intervals")
    if not isinstance(intervals, Sequence) or isinstance(intervals, (str, bytes)):
        return False
    if len(intervals) != request_count:
        return False
    parsed: list[tuple[datetime, datetime]] = []
    try:
        for item in intervals:
            if not isinstance(item, Mapping) or item.get("result_complete") is not True:
                return False
            start = parse_aware_datetime(item.get("since"))
            end = parse_aware_datetime(item.get("until"))
            if start >= end:
                return False
            if (
                item.get("scope") == "rolling_authority"
                and end > day_start
                and start < coverage_end
            ):
                parsed.append((max(start, day_start), min(end, coverage_end)))
    except (TypeError, ValueError):
        return False
    cursor = day_start
    for start, end in sorted(parsed):
        if start > cursor:
            return False
        cursor = max(cursor, end)
    return cursor >= coverage_end


def validate_capture_enumeration_evidence(
    enumeration: Any,
    *,
    expected_source_mode: Optional[str] = None,
    expected_until: Any = None,
    expected_rolling_since: Any = None,
) -> list[str]:
    """Validate the loss-prevention evidence before it can move runtime state.

    Stage 10 deliberately returns a red verdict for incomplete business state,
    but malformed evidence is different: it must never be interpreted as an
    empty Mango day.  Keep this validator independent from the semantic digest
    so both cursor writes and ready-generation reuse fail closed.
    """

    if not isinstance(enumeration, Mapping):
        return ["enumeration_not_object"]
    errors: list[str] = []
    calls_by_day = enumeration.get("calls_by_moscow_day")
    zero_by_day = enumeration.get("independent_zero_enumerations_by_day")
    source = enumeration.get("mango_enumeration_source")
    source_mode = str(source.get("mode") or "") if isinstance(source, Mapping) else ""

    if expected_source_mode and source_mode != expected_source_mode:
        errors.append("enumeration_source_mode_mismatch")

    def canonical_day(raw_day: Any, label: str) -> Optional[str]:
        if not isinstance(raw_day, str):
            errors.append(f"{label}_day_key_not_string")
            return None
        try:
            canonical = date.fromisoformat(raw_day).isoformat()
        except ValueError:
            canonical = ""
        if raw_day != canonical:
            errors.append(f"{label}_day_key_not_canonical")
            return None
        return raw_day

    normalized_calls: dict[str, list[str]] = {}
    if calls_by_day is not None and not isinstance(calls_by_day, Mapping):
        errors.append("calls_by_moscow_day_not_object")
    elif isinstance(calls_by_day, Mapping):
        for raw_day, raw_calls in calls_by_day.items():
            day_key = canonical_day(raw_day, "calls_by_moscow_day")
            if not isinstance(raw_calls, Sequence) or isinstance(
                raw_calls, (str, bytes)
            ):
                errors.append("calls_by_moscow_day_value_not_array")
                continue
            values: list[str] = []
            for raw_call in raw_calls:
                if (
                    not isinstance(raw_call, str)
                    or not raw_call.strip()
                    or raw_call != raw_call.strip()
                ):
                    errors.append("calls_by_moscow_day_call_key_invalid")
                    continue
                values.append(raw_call)
            if day_key is not None:
                normalized_calls[day_key] = values

    normalized_zero: dict[str, int] = {}
    if zero_by_day is not None and not isinstance(zero_by_day, Mapping):
        errors.append("independent_zero_enumerations_by_day_not_object")
    elif isinstance(zero_by_day, Mapping):
        for raw_day, raw_count in zero_by_day.items():
            day_key = canonical_day(
                raw_day, "independent_zero_enumerations_by_day"
            )
            if (
                isinstance(raw_count, bool)
                or not isinstance(raw_count, int)
                or raw_count < 0
                or raw_count > 2
            ):
                errors.append("independent_zero_enumerations_count_invalid")
                continue
            if day_key is not None:
                normalized_zero[day_key] = raw_count

    if source_mode != "strict_service":
        return sorted(set(errors))

    if any(
        values != sorted(set(values)) for values in normalized_calls.values()
    ):
        errors.append("strict_calls_by_moscow_day_not_canonical")

    if enumeration.get("mango_enumeration_complete") is not True:
        errors.append("strict_enumeration_not_complete")
    if not isinstance(calls_by_day, Mapping):
        errors.append("strict_calls_by_moscow_day_missing")
    if not isinstance(zero_by_day, Mapping):
        errors.append("strict_zero_enumerations_by_day_missing")
    if not isinstance(source, Mapping):
        errors.append("strict_enumeration_source_missing")
        return sorted(set(errors))
    if (
        source.get("cursor") != "not_applicable_stats_request_result"
        or source.get("pagination") != "not_applicable_stats_request_result"
        or "pages" not in source
        or source.get("pages") is not None
    ):
        errors.append("strict_enumeration_source_contract_invalid")

    requests = source.get("requests")
    if isinstance(requests, bool) or not isinstance(requests, int) or requests <= 0:
        errors.append("strict_enumeration_requests_invalid")
        requests = None
    intervals = source.get("covered_intervals")
    if not isinstance(intervals, Sequence) or isinstance(intervals, (str, bytes)):
        errors.append("strict_enumeration_intervals_invalid")
        intervals = ()
    elif requests is not None and len(intervals) != requests:
        errors.append("strict_enumeration_request_count_mismatch")

    try:
        source_since = _parse_strict_aware_datetime(source.get("since"))
        source_until = _parse_strict_aware_datetime(source.get("until"))
        if source_since >= source_until:
            raise ValueError("empty source window")
    except (TypeError, ValueError):
        errors.append("strict_enumeration_window_invalid")
        source_since = source_until = None
    try:
        rolling_since = _parse_strict_aware_datetime(source.get("rolling_since"))
        if (
            source_since is None
            or source_until is None
            or rolling_since < source_since
            or rolling_since >= source_until
        ):
            raise ValueError("rolling window outside source window")
    except (TypeError, ValueError):
        errors.append("strict_enumeration_rolling_window_invalid")
        rolling_since = None
    if expected_rolling_since is not None:
        try:
            exact_rolling_since = _parse_strict_aware_datetime(
                expected_rolling_since
            )
            if rolling_since is None or rolling_since != exact_rolling_since:
                errors.append("strict_enumeration_rolling_since_mismatch")
        except (TypeError, ValueError):
            errors.append("strict_expected_rolling_since_invalid")
    if expected_until is not None:
        try:
            exact_until = _parse_strict_aware_datetime(expected_until)
            if source_until is None or source_until != exact_until:
                errors.append("strict_enumeration_until_mismatch")
        except (TypeError, ValueError):
            errors.append("strict_expected_until_invalid")

    dual_pass_chunks: dict[int, list[Mapping[str, Any]]] = {}
    dual_proof = source.get("dual_enumeration")
    if source.get("enumeration_consistency_ok") is not True:
        errors.append("strict_dual_enumeration_consistency_not_proven")
    if not isinstance(dual_proof, Mapping):
        errors.append("strict_dual_enumeration_missing")
    else:
        proof_body = {
            key: value for key, value in dual_proof.items() if key != "proof_sha256"
        }
        if dual_proof.get("proof_sha256") != _canonical_json_sha256(proof_body):
            errors.append("strict_dual_enumeration_proof_digest_invalid")
        if dual_proof.get("schema_version") != DUAL_ENUMERATION_SCHEMA:
            errors.append("strict_dual_enumeration_schema_invalid")
        if (
            dual_proof.get("normalization_version")
            != DUAL_ENUMERATION_NORMALIZATION
        ):
            errors.append("strict_dual_enumeration_normalization_invalid")
        if not str(dual_proof.get("tenant_id") or "").strip():
            errors.append("strict_dual_enumeration_tenant_missing")
        if not str(dual_proof.get("base_url") or "").strip():
            errors.append("strict_dual_enumeration_base_url_missing")
        if not str(dual_proof.get("proof_run_id") or "").strip():
            errors.append("strict_dual_enumeration_run_id_missing")
        try:
            _parse_strict_aware_datetime(dual_proof.get("observed_at"))
        except (TypeError, ValueError):
            errors.append("strict_dual_enumeration_observed_at_invalid")
        if dual_proof.get("fields_sha256") != _canonical_json_sha256(
            DEFAULT_STATS_FIELDS
        ):
            errors.append("strict_dual_enumeration_fields_digest_invalid")
        if (
            type(dual_proof.get("passes_required")) is not int
            or dual_proof.get("passes_required") != 2
            or type(dual_proof.get("passes_completed")) is not int
            or dual_proof.get("passes_completed") != 2
        ):
            errors.append("strict_dual_enumeration_pass_count_invalid")
        try:
            proof_since = _parse_strict_aware_datetime(
                dual_proof.get("rolling_since")
            )
            proof_until = _parse_strict_aware_datetime(dual_proof.get("until"))
            if proof_since != rolling_since or proof_until != source_until:
                errors.append("strict_dual_enumeration_window_mismatch")
        except (TypeError, ValueError):
            errors.append("strict_dual_enumeration_window_invalid")

        raw_passes = dual_proof.get("passes")
        passes = (
            list(raw_passes)
            if isinstance(raw_passes, Sequence)
            and not isinstance(raw_passes, (str, bytes))
            else []
        )
        if len(passes) != 2:
            errors.append("strict_dual_enumeration_passes_invalid")
        pass_facts: list[Mapping[str, Any]] = []
        for index, pass_payload in enumerate(passes[:2], start=1):
            if not isinstance(pass_payload, Mapping):
                errors.append("strict_dual_enumeration_pass_invalid")
                continue
            expected_pass_id = "primary" if index == 1 else "verification"
            if pass_payload.get("pass_id") != expected_pass_id:
                errors.append("strict_dual_enumeration_pass_identity_invalid")
            try:
                pass_since = _parse_strict_aware_datetime(
                    pass_payload.get("rolling_since")
                )
                pass_until = _parse_strict_aware_datetime(
                    pass_payload.get("until")
                )
                if pass_since != rolling_since or pass_until != source_until:
                    errors.append("strict_dual_enumeration_pass_window_mismatch")
            except (TypeError, ValueError):
                errors.append("strict_dual_enumeration_pass_window_invalid")

            raw_requests = pass_payload.get("requests")
            raw_rows = pass_payload.get("raw_rows")
            if (
                isinstance(raw_requests, bool)
                or not isinstance(raw_requests, int)
                or raw_requests <= 0
            ):
                errors.append("strict_dual_enumeration_requests_invalid")
                raw_requests = None
            if (
                isinstance(raw_rows, bool)
                or not isinstance(raw_rows, int)
                or raw_rows < 0
            ):
                errors.append("strict_dual_enumeration_rows_invalid")
                raw_rows = None

            raw_chunks = pass_payload.get("chunks")
            chunks = (
                list(raw_chunks)
                if isinstance(raw_chunks, Sequence)
                and not isinstance(raw_chunks, (str, bytes))
                else []
            )
            if not chunks or (
                raw_requests is not None and len(chunks) != raw_requests
            ):
                errors.append("strict_dual_enumeration_chunks_invalid")
            parsed_chunks: list[tuple[datetime, datetime, int]] = []
            canonical_chunks: list[Mapping[str, Any]] = []
            for chunk in chunks:
                if not isinstance(chunk, Mapping):
                    errors.append("strict_dual_enumeration_chunk_invalid")
                    continue
                chunk_rows = chunk.get("rows")
                if (
                    isinstance(chunk_rows, bool)
                    or not isinstance(chunk_rows, int)
                    or chunk_rows < 0
                    or chunk.get("result_complete") is not True
                ):
                    errors.append("strict_dual_enumeration_chunk_invalid")
                    continue
                try:
                    chunk_since = _parse_strict_aware_datetime(
                        chunk.get("since")
                    )
                    chunk_until = _parse_strict_aware_datetime(
                        chunk.get("until")
                    )
                    if (
                        chunk_since >= chunk_until
                        or chunk_since.microsecond
                        or chunk_until.microsecond
                    ):
                        raise ValueError
                except (TypeError, ValueError):
                    errors.append("strict_dual_enumeration_chunk_window_invalid")
                    continue
                parsed_chunks.append((chunk_since, chunk_until, chunk_rows))
                canonical_chunks.append(
                    {
                        "since": chunk.get("since"),
                        "until": chunk.get("until"),
                        "result_complete": True,
                        "rows": chunk_rows,
                    }
                )
            if rolling_since is not None and source_until is not None:
                chunk_cursor = rolling_since
                for chunk_since, chunk_until, _rows in parsed_chunks:
                    if chunk_since != chunk_cursor:
                        errors.append(
                            "strict_dual_enumeration_chunk_geometry_invalid"
                        )
                    chunk_cursor = chunk_until
                if chunk_cursor != source_until:
                    errors.append(
                        "strict_dual_enumeration_chunk_geometry_invalid"
                    )
            if raw_rows is not None and sum(
                item[2] for item in parsed_chunks
            ) != raw_rows:
                errors.append("strict_dual_enumeration_chunk_rows_mismatch")
            partition_sha256 = _canonical_json_sha256(
                [{"since": item["since"], "until": item["until"]} for item in canonical_chunks]
            )
            if pass_payload.get("partition_sha256") != partition_sha256:
                errors.append("strict_dual_enumeration_partition_digest_mismatch")
            dual_pass_chunks[index] = canonical_chunks

            raw_multiset = pass_payload.get("call_key_multiset")
            multiset = (
                list(raw_multiset)
                if isinstance(raw_multiset, Sequence)
                and not isinstance(raw_multiset, (str, bytes))
                else []
            )
            if (
                any(
                    not isinstance(value, str)
                    or not value
                    or value != value.strip()
                    for value in multiset
                )
                or multiset != sorted(multiset)
            ):
                errors.append("strict_dual_enumeration_multiset_invalid")
            if raw_rows is not None and len(multiset) != raw_rows:
                errors.append("strict_dual_enumeration_multiset_rows_mismatch")
            if pass_payload.get(
                "call_key_multiset_sha256"
            ) != _canonical_json_sha256(multiset):
                errors.append("strict_dual_enumeration_multiset_digest_mismatch")
            if not _is_sha256(pass_payload.get("raw_rows_sha256")):
                errors.append("strict_dual_enumeration_raw_rows_digest_invalid")

            raw_keys = pass_payload.get("call_keys")
            pass_keys = (
                list(raw_keys)
                if isinstance(raw_keys, Sequence)
                and not isinstance(raw_keys, (str, bytes))
                else []
            )
            if (
                any(
                    not isinstance(value, str)
                    or not value
                    or value != value.strip()
                    for value in pass_keys
                )
                or pass_keys != sorted(set(pass_keys))
            ):
                errors.append("strict_dual_enumeration_call_keys_invalid")
            if sorted(set(multiset)) != pass_keys:
                errors.append("strict_dual_enumeration_call_keys_mismatch")
            if (
                type(pass_payload.get("normalized_unique_count")) is not int
                or pass_payload.get("normalized_unique_count") != len(pass_keys)
            ):
                errors.append("strict_dual_enumeration_unique_count_mismatch")
            balance_fields = (
                "recordable_unique_rows",
                "without_recording_rows",
                "proven_duplicate_rows",
                "quarantined_rows",
                "error_rows",
                "unexplained_rows",
            )
            balance = [pass_payload.get(field) for field in balance_fields]
            if raw_rows is not None and (
                any(type(value) is not int or value < 0 for value in balance)
                or balance[0] + balance[1] != len(pass_keys)
                or balance[2] != raw_rows - len(pass_keys)
                or sum(balance) != raw_rows
                or any(balance[index] != 0 for index in (3, 4, 5))
                or pass_payload.get("raw_balance_ok") is not True
            ):
                errors.append("strict_dual_enumeration_raw_balance_invalid")
            if pass_payload.get("call_keys_sha256") != _canonical_json_sha256(
                pass_keys
            ):
                errors.append("strict_dual_enumeration_call_keys_digest_mismatch")

            raw_by_day = pass_payload.get("calls_by_moscow_day")
            pass_by_day: dict[str, list[str]] = {}
            if not isinstance(raw_by_day, Mapping):
                errors.append("strict_dual_enumeration_days_invalid")
            else:
                for raw_day, raw_day_keys in raw_by_day.items():
                    try:
                        day_key = date.fromisoformat(str(raw_day)).isoformat()
                    except ValueError:
                        day_key = ""
                    if raw_day != day_key or not isinstance(
                        raw_day_keys, Sequence
                    ) or isinstance(raw_day_keys, (str, bytes)):
                        errors.append("strict_dual_enumeration_days_invalid")
                        continue
                    day_keys = list(raw_day_keys)
                    if (
                        any(
                            not isinstance(value, str)
                            or not value
                            or value != value.strip()
                            for value in day_keys
                        )
                        or day_keys != sorted(set(day_keys))
                    ):
                        errors.append("strict_dual_enumeration_day_keys_invalid")
                    pass_by_day[day_key] = day_keys
            flattened_pass_days = [
                value for values in pass_by_day.values() for value in values
            ]
            if sorted(flattened_pass_days) != pass_keys or len(
                flattened_pass_days
            ) != len(set(flattened_pass_days)):
                errors.append("strict_dual_enumeration_days_mismatch")
            canonical_by_day = {
                key: pass_by_day[key] for key in sorted(pass_by_day)
            }
            if pass_payload.get(
                "calls_by_moscow_day_sha256"
            ) != _canonical_json_sha256(canonical_by_day):
                errors.append("strict_dual_enumeration_days_digest_mismatch")
            if not _is_sha256(pass_payload.get("event_digest_sha256")):
                errors.append("strict_dual_enumeration_event_digest_invalid")
            pass_facts.append(
                {
                    "raw_rows": raw_rows,
                    "call_key_multiset": multiset,
                    "call_key_multiset_sha256": pass_payload.get(
                        "call_key_multiset_sha256"
                    ),
                    "raw_rows_sha256": pass_payload.get("raw_rows_sha256"),
                    "normalized_unique_count": pass_payload.get(
                        "normalized_unique_count"
                    ),
                    "call_keys": pass_keys,
                    "call_keys_sha256": pass_payload.get("call_keys_sha256"),
                    "calls_by_moscow_day": canonical_by_day,
                    "calls_by_moscow_day_sha256": pass_payload.get(
                        "calls_by_moscow_day_sha256"
                    ),
                    "event_digest_sha256": pass_payload.get(
                        "event_digest_sha256"
                    ),
                    "partition_sha256": partition_sha256,
                    "raw_balance_ok": pass_payload.get("raw_balance_ok"),
                    "chunks": canonical_chunks,
                }
            )

        comparison_fields = (
            "normalized_unique_count",
            "call_keys",
            "call_keys_sha256",
            "calls_by_moscow_day",
            "calls_by_moscow_day_sha256",
            "event_digest_sha256",
        )
        computed_comparison: dict[str, bool] = {}
        if len(pass_facts) == 2:
            computed_comparison = {
                f"{field}_equal": pass_facts[0].get(field)
                == pass_facts[1].get(field)
                for field in comparison_fields
            }
            computed_comparison["primary_raw_balance_ok"] = pass_facts[0].get(
                "raw_balance_ok"
            ) is True
            computed_comparison["verification_raw_balance_ok"] = pass_facts[1].get(
                "raw_balance_ok"
            ) is True
            computed_comparison["partition_sha256_different"] = pass_facts[0].get(
                "partition_sha256"
            ) != pass_facts[1].get("partition_sha256")
            computed_comparison["official_total_equal"] = (
                _official_total_proof_is_green(
                    dual_proof.get("official_total"),
                    expected_count=pass_facts[0].get("normalized_unique_count"),
                    expected_call_keys_sha256=pass_facts[0].get(
                        "call_keys_sha256"
                    ),
                )
            )
            if not computed_comparison["official_total_equal"]:
                errors.append("strict_official_total_proof_invalid")
            if pass_facts[0].get("call_keys") != sorted(
                set(value for values in normalized_calls.values() for value in values)
            ):
                errors.append("strict_dual_enumeration_top_call_keys_mismatch")
            if pass_facts[0].get("calls_by_moscow_day") != {
                key: normalized_calls[key] for key in sorted(normalized_calls)
            }:
                errors.append("strict_dual_enumeration_top_days_mismatch")
        comparison = dual_proof.get("comparison")
        if not isinstance(comparison, Mapping) or dict(
            comparison
        ) != computed_comparison:
            errors.append("strict_dual_enumeration_comparison_invalid")
        comparison_green = bool(
            computed_comparison and all(computed_comparison.values())
        )
        if (
            dual_proof.get("enumeration_consistency_ok") is not comparison_green
            or source.get("enumeration_consistency_ok") is not comparison_green
        ):
            errors.append("strict_dual_enumeration_consistency_mismatch")
        expected_mismatch = "" if comparison_green else ",".join(
            sorted(
                key.removesuffix("_equal")
                for key, value in computed_comparison.items()
                if value is not True
            )
        )
        if dual_proof.get("mismatch_reason") != expected_mismatch:
            errors.append("strict_dual_enumeration_reason_mismatch")

    interval_rows_total = 0
    authoritative_rows_total = 0
    auxiliary_rows_total = 0
    interval_days: set[date] = set()
    parsed_intervals: list[tuple[datetime, datetime]] = []
    rolling_intervals: list[tuple[datetime, datetime]] = []
    authoritative_intervals_by_pass: dict[
        int, list[Mapping[str, Any]]
    ] = {1: [], 2: []}
    auxiliary_intervals: list[tuple[datetime, datetime]] = []
    for interval in intervals:
        if not isinstance(interval, Mapping):
            errors.append("strict_enumeration_interval_not_object")
            continue
        rows = interval.get("rows")
        if isinstance(rows, bool) or not isinstance(rows, int) or rows < 0:
            errors.append("strict_enumeration_interval_rows_invalid")
        else:
            interval_rows_total += rows
        scope = interval.get("scope")
        if scope not in {"rolling_authority", "recovery_auxiliary"}:
            errors.append("strict_enumeration_interval_scope_invalid")
        elif (
            scope == "rolling_authority"
            and isinstance(rows, int)
            and not isinstance(rows, bool)
        ):
            authoritative_rows_total += rows
            authority_pass = interval.get("authority_pass")
            if authority_pass not in {1, 2} or isinstance(
                authority_pass, bool
            ):
                errors.append("strict_enumeration_authority_pass_invalid")
            else:
                authoritative_intervals_by_pass[authority_pass].append(
                    {
                        "since": interval.get("since"),
                        "until": interval.get("until"),
                        "result_complete": interval.get("result_complete"),
                        "rows": rows,
                    }
                )
        elif (
            scope == "recovery_auxiliary"
            and isinstance(rows, int)
            and not isinstance(rows, bool)
        ):
            auxiliary_rows_total += rows
            if "authority_pass" in interval:
                errors.append("strict_enumeration_auxiliary_has_authority_pass")
        if interval.get("result_complete") is not True:
            errors.append("strict_enumeration_interval_incomplete")
        try:
            interval_since = _parse_strict_aware_datetime(interval.get("since"))
            interval_until = _parse_strict_aware_datetime(interval.get("until"))
            if interval_since >= interval_until:
                raise ValueError("empty interval")
            if source_since is not None and (
                interval_since < source_since or interval_until > source_until
            ):
                errors.append("strict_enumeration_interval_outside_window")
            parsed_intervals.append((interval_since, interval_until))
            if scope == "rolling_authority":
                rolling_intervals.append((interval_since, interval_until))
                if rolling_since is not None and (
                    interval_since < rolling_since
                    or source_until is None
                    or interval_until > source_until
                ):
                    errors.append(
                        "strict_enumeration_authority_interval_outside_rolling_window"
                    )
            elif scope == "recovery_auxiliary":
                auxiliary_intervals.append((interval_since, interval_until))
                if rolling_since is None or interval_until > rolling_since:
                    errors.append(
                        "strict_enumeration_auxiliary_overlaps_rolling_window"
                    )
            first_day = interval_since.astimezone(MOSCOW).date()
            last_day = (interval_until - timedelta(microseconds=1)).astimezone(
                MOSCOW
            ).date()
            if (last_day - first_day).days > 370:
                errors.append("strict_enumeration_interval_too_wide")
            else:
                while first_day <= last_day:
                    interval_days.add(first_day)
                    first_day += timedelta(days=1)
        except (TypeError, ValueError):
            errors.append("strict_enumeration_interval_window_invalid")

    if parsed_intervals and source_since is not None:
        if min(start for start, _end in parsed_intervals) != source_since:
            errors.append("strict_enumeration_source_start_mismatch")
    if rolling_since is not None and source_until is not None:
        for authority_pass in (1, 2):
            coverage_cursor = rolling_since
            for interval in authoritative_intervals_by_pass[authority_pass]:
                try:
                    interval_since = _parse_strict_aware_datetime(
                        interval.get("since")
                    )
                    interval_until = _parse_strict_aware_datetime(
                        interval.get("until")
                    )
                except (TypeError, ValueError):
                    continue
                if interval_since != coverage_cursor:
                    errors.append("strict_enumeration_rolling_geometry_invalid")
                coverage_cursor = interval_until
            if coverage_cursor != source_until:
                errors.append("strict_enumeration_rolling_coverage_incomplete")
            if authoritative_intervals_by_pass[authority_pass] != dual_pass_chunks.get(
                authority_pass, []
            ):
                errors.append("strict_enumeration_pass_chunks_mismatch")
    for index, (start, end) in enumerate(sorted(auxiliary_intervals)):
        if index and start < sorted(auxiliary_intervals)[index - 1][1]:
            errors.append("strict_enumeration_auxiliary_overlap")

    api_requests = enumeration.get("api_requests")
    api_rows_total = enumeration.get("api_rows_total")
    api_authoritative_rows_total = enumeration.get(
        "api_authoritative_rows_total"
    )
    api_auxiliary_rows_total = enumeration.get("api_auxiliary_rows_total")
    api_events_total = enumeration.get("api_events_total")
    for label, value in (
        ("api_requests", api_requests),
        ("api_rows_total", api_rows_total),
        ("api_authoritative_rows_total", api_authoritative_rows_total),
        ("api_auxiliary_rows_total", api_auxiliary_rows_total),
        ("api_events_total", api_events_total),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            errors.append(f"strict_{label}_invalid")
    if isinstance(api_requests, int) and not isinstance(api_requests, bool):
        if requests is not None and api_requests != requests:
            errors.append("strict_api_request_count_mismatch")
    if isinstance(api_rows_total, int) and not isinstance(api_rows_total, bool):
        if api_rows_total != interval_rows_total:
            errors.append("strict_api_row_count_mismatch")
    if isinstance(api_authoritative_rows_total, int) and not isinstance(
        api_authoritative_rows_total, bool
    ):
        if api_authoritative_rows_total != authoritative_rows_total:
            errors.append("strict_api_authoritative_row_count_mismatch")
    if isinstance(api_auxiliary_rows_total, int) and not isinstance(
        api_auxiliary_rows_total, bool
    ):
        if api_auxiliary_rows_total != auxiliary_rows_total:
            errors.append("strict_api_auxiliary_row_count_mismatch")

    flattened = [value for values in normalized_calls.values() for value in values]
    unique_calls = sorted(set(flattened))
    if len(flattened) != len(unique_calls):
        errors.append("strict_call_key_repeated_across_days")
    raw_call_keys = enumeration.get("call_keys")
    if not isinstance(raw_call_keys, Sequence) or isinstance(
        raw_call_keys, (str, bytes)
    ):
        errors.append("strict_call_keys_invalid")
    else:
        exact_call_keys = [
            value
            for value in raw_call_keys
            if isinstance(value, str) and value and value == value.strip()
        ]
        if len(exact_call_keys) != len(raw_call_keys):
            errors.append("strict_call_keys_invalid")
        elif exact_call_keys != sorted(set(exact_call_keys)):
            errors.append("strict_call_keys_not_canonical")
        elif exact_call_keys != unique_calls:
            errors.append("strict_call_keys_mismatch")
    if isinstance(api_events_total, int) and not isinstance(api_events_total, bool):
        if api_events_total != len(unique_calls):
            errors.append("strict_api_event_count_mismatch")
    if authoritative_rows_total > 0 and not unique_calls:
        errors.append("strict_api_rows_without_calls")

    evidence_days = set(normalized_calls) | set(normalized_zero)
    for day_key in evidence_days:
        day = date.fromisoformat(day_key)
        if not enumeration_source_covers_day(source, day):
            errors.append("strict_evidence_day_not_covered")
    fully_covered_days = {
        day
        for day in interval_days
        if enumeration_source_covers_day(source, day, require_full_day=True)
    }
    expected_zero_days = set(normalized_calls) | {
        day.isoformat() for day in fully_covered_days
    }
    if set(normalized_zero) != expected_zero_days:
        errors.append("strict_zero_enumeration_days_mismatch")
    for day_key in expected_zero_days:
        calls = normalized_calls.get(day_key, [])
        proof_count = normalized_zero.get(day_key)
        if calls and proof_count != 0:
            errors.append("strict_nonempty_day_has_zero_proof")
        elif not calls and day_key in {
            day.isoformat() for day in fully_covered_days
        } and proof_count != 2:
            errors.append("strict_full_empty_day_zero_proof_missing")

    return sorted(set(errors))


def build_stage10_verdict(
    *,
    day: date,
    enumeration: Mapping[str, Any],
    capture_entries: Sequence[Any],
    ready_rows: Sequence[Any],
    now: Optional[datetime] = None,
    pending_sla_hours: int = 72,
) -> Mapping[str, Any]:
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    evidence_schema_ok = not validate_capture_enumeration_evidence(enumeration)
    latest_capture = _latest_capture_by_call(capture_entries, day)
    ready, ready_duplicate_count = _ready_rows_by_call(ready_rows, day)
    calls_by_day = enumeration.get("calls_by_moscow_day")
    raw_call_keys = (
        calls_by_day.get(day.isoformat())
        if isinstance(calls_by_day, Mapping)
        else enumeration.get("call_keys")
    )
    enumerated = (
        [str(value).strip() for value in raw_call_keys if str(value or "").strip()]
        if isinstance(raw_call_keys, Sequence) and not isinstance(raw_call_keys, (str, bytes))
        else []
    )
    duplicate_call_keys = len(enumerated) - len(set(enumerated)) + ready_duplicate_count
    mango_keys = set(enumerated)
    source = enumeration.get("mango_enumeration_source")
    enumeration_complete = bool(
        enumeration.get("mango_enumeration_complete") is True
        and enumeration_source_covers_day(source, day)
    )
    zero_by_day = enumeration.get("independent_zero_enumerations_by_day")
    raw_zero_proofs = (
        zero_by_day.get(day.isoformat())
        if isinstance(zero_by_day, Mapping)
        else enumeration.get("independent_zero_enumerations")
    )
    zero_proofs = (
        raw_zero_proofs
        if isinstance(raw_zero_proofs, int)
        and not isinstance(raw_zero_proofs, bool)
        and raw_zero_proofs >= 0
        else 0
    )
    if not mango_keys and zero_proofs < 2:
        enumeration_complete = False

    db_quarantine_keys = {
        key
        for key, row in ready.items()
        if str(row.get("dead_letter_stage") or "")
        in {"transcribe", "resolve", "analyze"}
        and str(row.get("last_error") or "").strip()
    }
    active_db_keys = set(ready) - db_quarantine_keys
    # Presence in the sealed table is not enough: a stale/in-progress lease or
    # any unfinished stage must remain pending and can never prove closure.
    ready_keys = {
        key
        for key in active_db_keys
        if ready_row_is_complete(ready[key], now=current)
    }
    db_pending_keys = active_db_keys - ready_keys
    quarantine_keys = db_quarantine_keys | {
        key for key, entry in latest_capture.items()
        if str(entry.get("status") or "") in QUARANTINE_STATUSES
    }
    capture_pending_keys = set(latest_capture) - quarantine_keys
    pending_keys = (capture_pending_keys | db_pending_keys) - ready_keys - quarantine_keys
    enumerated_pending_keys = pending_keys & mango_keys
    state_overlap_count = sum(
        sum(key in state for state in (ready_keys, quarantine_keys, pending_keys)) > 1
        for key in mango_keys | ready_keys | quarantine_keys | pending_keys
    )
    unexplained_keys = mango_keys - ready_keys - quarantine_keys - pending_keys
    extra_state_keys = (ready_keys | quarantine_keys | pending_keys) - mango_keys

    pending_ages: list[float] = []
    for key in enumerated_pending_keys:
        entry = latest_capture.get(key) or {}
        try:
            age = (
                current
                - parse_aware_datetime(entry.get("started_at") or entry.get("created_at"))
            ).total_seconds() / 60
        except (TypeError, ValueError):
            age = float(pending_sla_hours * 60 + 1)
        pending_ages.append(max(0.0, age))
    pending_over_sla = sum(age > pending_sla_hours * 60 for age in pending_ages)
    known_event_keys = {
        str(_entry_dict(entry).get("event_key") or "").strip()
        for entry in capture_entries
        if str(_entry_dict(entry).get("event_key") or "").strip()
    }

    def quarantine_has_reason(key: str) -> bool:
        if key in db_quarantine_keys:
            row = ready[key]
            return bool(
                str(row.get("dead_letter_stage") or "")
                in {"transcribe", "resolve", "analyze"}
                and str(row.get("last_error") or "").strip()
            )
        entry = latest_capture[key]
        status = str(entry.get("status") or "")
        if status == "multiple_recordings_needs_review":
            return entry.get("remediation_code") == "manual_recording_selection"
        if status == "duplicate_recording":
            canonical = str(entry.get("canonical_event_key") or "").strip()
            return bool(canonical and canonical != entry.get("event_key") and canonical in known_event_keys)
        if status == "recording_retry_expired":
            try:
                age = current - parse_aware_datetime(
                    entry.get("started_at") or entry.get("created_at")
                )
            except (TypeError, ValueError):
                return False
            return bool(
                age >= timedelta(hours=pending_sla_hours)
                and str(entry.get("error") or "").strip()
                and entry.get("remediation_code")
                == "manual_review_or_retry_if_recording_appears"
            )
        if status == "audio_integrity_quarantined":
            return bool(
                entry.get("error") == "capture_target_integrity_mismatch"
                and entry.get("remediation_code")
                == "manual_restore_or_quarantine_corrupted_audio"
                and entry.get("recovery_state") == "immutable_audio_violation"
            )
        return bool(str(entry.get("error") or "").strip())

    quarantine_without_reason = sum(
        not quarantine_has_reason(key) for key in quarantine_keys & mango_keys
    )

    def quarantine_item(key: str) -> Mapping[str, str]:
        has_reason = quarantine_has_reason(key)
        if key in db_quarantine_keys:
            source_row = ready[key]
            code = f"dead_letter_{str(source_row.get('dead_letter_stage') or '')}"
        else:
            source_row = latest_capture[key]
            code = str(source_row.get("status") or "")
        if not has_reason:
            code = "quarantine_evidence_incomplete"
        reason, action = QUARANTINE_MANAGER_GUIDANCE[code]
        started_at = parse_aware_datetime(
            source_row.get("started_at") or source_row.get("created_at")
        ).isoformat()
        return {
            "call_key": key,
            "started_at": started_at,
            "code": code,
            "reason": reason,
            "action": action,
        }

    quarantine_items = sorted(
        (quarantine_item(key) for key in quarantine_keys & mango_keys),
        key=lambda item: (parse_aware_datetime(item["started_at"]), item["call_key"]),
    )
    active_ready_rows = [row for key, row in ready.items() if key in active_db_keys]
    ready_without_dual = sum(
        str(row.get("transcription_status") or "") != "done"
        or not has_dual_asr_or_exception(row, now=current)
        for row in active_ready_rows
    )
    ready_without_resolve = sum(
        str(row.get("resolve_status") or "") not in {"done", "skipped"}
        for row in active_ready_rows
    )
    ready_without_analyze = sum(
        str(row.get("analysis_status") or "") != "done"
        or not _json_object(row.get("analysis_json"))
        for row in active_ready_rows
    )
    if not isinstance(source, Mapping):
        source = {}
    consistency_ok = bool(
        evidence_schema_ok
        and enumeration_complete
        and not extra_state_keys
        and state_overlap_count == 0
        and not unexplained_keys
        and quarantine_without_reason == 0
        and duplicate_call_keys == 0
    )
    closure_ok = bool(
        consistency_ok
        and enumeration_source_covers_day(source, day, require_full_day=True)
        and not pending_keys
        and pending_over_sla == 0
        and ready_without_dual == 0
        and ready_without_resolve == 0
        and ready_without_analyze == 0
    )
    return {
        "schema_version": STAGE10_SCHEMA,
        "day": day.isoformat(),
        "generated_at": current.isoformat(),
        "mango_enumeration_complete": enumeration_complete,
        "mango_enumeration_source": dict(source),
        "mango_unique": len(mango_keys),
        "ready_unique": len(ready_keys & mango_keys),
        "ready_incomplete_unique": len(db_pending_keys & mango_keys),
        "quarantine_unique": len(quarantine_keys & mango_keys),
        "quarantine_items": quarantine_items,
        "pending_unique": len(enumerated_pending_keys),
        "unexplained_missing": len(unexplained_keys),
        "state_overlap_count": state_overlap_count,
        "pending_awaiting_recording": sum(
            str((latest_capture.get(key) or {}).get("status") or "")
            == "skipped_no_recording"
            for key in enumerated_pending_keys
        ),
        "pending_over_sla": pending_over_sla,
        "quarantine_without_reason": quarantine_without_reason,
        "ready_without_dual_asr_or_explicit_exception": ready_without_dual,
        "ready_without_resolve": ready_without_resolve,
        "ready_without_analyze": ready_without_analyze,
        "duplicate_call_keys": duplicate_call_keys,
        "oldest_pending_age_minutes": round(max(pending_ages), 3) if pending_ages else 0,
        "state_not_in_mango_enumeration": len(extra_state_keys),
        "independent_zero_enumerations": zero_proofs,
        "consistency_ok": consistency_ok,
        "closure_ok": closure_ok,
    }


def load_ready_rows(path: Path) -> list[Mapping[str, Any]]:
    if not path.is_file():
        return []
    with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True, timeout=30) as con:
        con.row_factory = sqlite3.Row
        tables = {str(row[0]) for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "call_records" not in tables:
            return []
        columns = {str(row[1]) for row in con.execute("PRAGMA table_info(call_records)")}
        wanted = (
            "source_call_id",
            "started_at",
            "transcription_status",
            "transcript_variants_json",
            "resolve_status",
            "analysis_status",
            "analysis_json",
            "pipeline_stage",
            "pipeline_worker_id",
            "pipeline_claimed_at",
            "analysis_worker_id",
            "analysis_claimed_at",
            "dead_letter_stage",
            "last_error",
        )
        selected = [name for name in wanted if name in columns]
        if not {"source_call_id", "started_at"}.issubset(selected):
            return []
        return [dict(row) for row in con.execute(f"SELECT {','.join(selected)} FROM call_records")]


def validate_ready_manifest_payload(
    manifest: Any,
    *,
    require_closure: bool = False,
    required_day: Optional[date] = None,
    require_consistency: bool = True,
    expected_code_sha: Optional[str] = None,
    expected_host_id: Optional[str] = None,
    allow_compatibility: bool = False,
) -> list[str]:
    if not isinstance(manifest, Mapping):
        return ["manifest_missing"]
    errors: list[str] = []
    if manifest.get("schema_version") != READY_MANIFEST_SCHEMA:
        errors.append("schema_version_mismatch")
    provenance_mode = str(manifest.get("provenance_mode") or "")
    compatibility = provenance_mode == "compatibility_not_for_service"
    if provenance_mode != "strict_service" and not (
        allow_compatibility and compatibility
    ):
        errors.append("provenance_mode_not_strict")
    if manifest.get("status") != "ready":
        errors.append("status_not_ready")
    if (
        require_consistency
        and required_day is None
        and manifest.get("consistency_ok") is not True
    ):
        errors.append("consistency_not_proven")
    if (
        required_day is None
        and require_closure
        and manifest.get("closure_ok") is not True
    ):
        errors.append("closure_not_proven")
    if not re.fullmatch(r"[0-9a-f]{40}", str(manifest.get("producer_git_sha") or "")):
        errors.append("producer_git_sha_invalid")
    if expected_code_sha and manifest.get("producer_git_sha") != expected_code_sha:
        errors.append("producer_git_sha_mismatch")
    host_id = str(manifest.get("host_id") or "")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{2,127}", host_id):
        errors.append("host_id_invalid")
    if expected_host_id and host_id != expected_host_id:
        errors.append("host_id_mismatch")
    if not str(manifest.get("run_id") or "").strip():
        errors.append("run_id_missing")
    window = manifest.get("mango_window")
    if not isinstance(window, Mapping) or not window.get("since") or not window.get("until"):
        errors.append("mango_window_missing")
    elif not compatibility:
        try:
            window_since = parse_aware_datetime(window.get("since"))
            window_until = parse_aware_datetime(window.get("until"))
            if window_since >= window_until:
                raise ValueError
        except (TypeError, ValueError):
            errors.append("mango_window_invalid")
        else:
            source = manifest.get("mango_enumeration_source")
            if not (
                isinstance(source, Mapping)
                and source.get("mode") == "strict_service"
            ):
                errors.append("mango_enumeration_source_not_strict")
            elif not _dual_source_proof_is_green(source):
                errors.append("mango_dual_enumeration_not_proven")
            intervals = source.get("covered_intervals") if isinstance(source, Mapping) else None
            if not isinstance(intervals, Sequence) or isinstance(intervals, (str, bytes)) or not intervals:
                errors.append("mango_covered_intervals_missing")
            else:
                parsed_intervals: list[tuple[datetime, datetime]] = []
                for item in intervals:
                    try:
                        if not isinstance(item, Mapping) or item.get("result_complete") is not True:
                            raise ValueError
                        start = parse_aware_datetime(item.get("since"))
                        end = parse_aware_datetime(item.get("until"))
                        if start >= end:
                            raise ValueError
                        parsed_intervals.append((start, end))
                    except (TypeError, ValueError):
                        errors.append("mango_covered_interval_invalid")
                        break
                if parsed_intervals:
                    relevant = sorted(
                        (max(start, window_since), min(end, window_until))
                        for start, end in parsed_intervals
                        if end > window_since and start < window_until
                    )
                    cursor = window_since
                    for start, end in relevant:
                        if start > cursor:
                            break
                        cursor = max(cursor, end)
                    if cursor < window_until:
                        errors.append("mango_window_not_fully_covered")
    if manifest.get("mango_enumeration_complete") is not True:
        errors.append("mango_enumeration_incomplete")
    if not compatibility:
        if not _is_sha256(manifest.get("capture_proof_sha256")):
            errors.append("capture_proof_sha256_invalid")
        if not _ready_capture_proof_is_green(manifest):
            errors.append("capture_proof_invalid")
        manifest_source = manifest.get("mango_enumeration_source")
        manifest_dual = (
            manifest_source.get("dual_enumeration")
            if isinstance(manifest_source, Mapping)
            else None
        )
        if (
            not isinstance(manifest_dual, Mapping)
            or not str(manifest.get("capture_proof_run_id") or "").strip()
            or manifest.get("capture_proof_run_id")
            != manifest_dual.get("proof_run_id")
        ):
            errors.append("capture_proof_run_id_invalid")
    daily_verdicts = manifest.get("daily_verdicts")
    moscow_dates = manifest.get("moscow_dates")
    if compatibility and daily_verdicts is None:
        daily_verdicts = {}
    if not isinstance(daily_verdicts, Mapping) or (
        not compatibility and not daily_verdicts
    ):
        errors.append("daily_verdicts_missing")
    elif daily_verdicts or not compatibility:
        verdict_dates = sorted(str(value) for value in daily_verdicts)
        if (
            not isinstance(moscow_dates, Sequence)
            or isinstance(moscow_dates, (str, bytes))
            or sorted(str(value) for value in moscow_dates) != verdict_dates
        ):
            errors.append("moscow_dates_mismatch")
        verdict_consistency: list[bool] = []
        verdict_closure: list[bool] = []
        count_fields = (
            "mango_unique",
            "ready_unique",
            "quarantine_unique",
            "pending_unique",
            "unexplained_missing",
            "state_overlap_count",
            "pending_awaiting_recording",
            "pending_over_sla",
            "quarantine_without_reason",
            "ready_without_dual_asr_or_explicit_exception",
            "ready_without_resolve",
            "ready_without_analyze",
            "duplicate_call_keys",
            "state_not_in_mango_enumeration",
            "independent_zero_enumerations",
        )
        for day_key, raw_verdict in daily_verdicts.items():
            try:
                date.fromisoformat(str(day_key))
            except ValueError:
                errors.append("daily_verdict_day_invalid")
                continue
            if not isinstance(raw_verdict, Mapping):
                errors.append("daily_verdict_invalid")
                continue
            if (
                raw_verdict.get("schema_version") != STAGE10_SCHEMA
                or raw_verdict.get("day") != str(day_key)
            ):
                errors.append("daily_verdict_identity_mismatch")
                continue
            try:
                parse_aware_datetime(raw_verdict.get("generated_at"))
                if any(field not in raw_verdict for field in count_fields) or (
                    "oldest_pending_age_minutes" not in raw_verdict
                ):
                    raise ValueError
                if any(isinstance(raw_verdict[field], bool) for field in count_fields):
                    raise ValueError
                counts = {
                    field: int(raw_verdict[field]) for field in count_fields
                }
                if any(value < 0 for value in counts.values()):
                    raise ValueError
                oldest_pending_age = float(
                    raw_verdict["oldest_pending_age_minutes"]
                )
                if oldest_pending_age < 0:
                    raise ValueError
            except (TypeError, ValueError):
                errors.append("daily_verdict_counts_invalid")
                continue
            if not enumeration_source_covers_day(
                raw_verdict.get("mango_enumeration_source"),
                date.fromisoformat(str(day_key)),
            ):
                errors.append("daily_verdict_enumeration_source_invalid")
            if (
                not compatibility
                and not (
                    isinstance(
                        raw_verdict.get("mango_enumeration_source"), Mapping
                    )
                    and str(
                        raw_verdict["mango_enumeration_source"].get("mode")
                    )
                    == "strict_service"
                )
            ):
                errors.append("daily_verdict_enumeration_source_not_strict")
            if counts["mango_unique"] != (
                counts["ready_unique"]
                + counts["quarantine_unique"]
                + counts["pending_unique"]
                + counts["unexplained_missing"]
            ):
                errors.append("daily_verdict_balance_mismatch")
            errors.extend(
                validate_quarantine_items_payload(
                    raw_verdict.get("quarantine_items"),
                    day=date.fromisoformat(str(day_key)),
                    expected_count=counts["quarantine_unique"],
                    expected_without_reason=counts["quarantine_without_reason"],
                )
            )
            if (
                counts["pending_awaiting_recording"] > counts["pending_unique"]
                or counts["pending_over_sla"] > counts["pending_unique"]
                or (
                    counts["pending_unique"] == 0
                    and oldest_pending_age != 0
                )
                or (
                    counts["mango_unique"] == 0
                    and counts["independent_zero_enumerations"] < 2
                    and raw_verdict.get("mango_enumeration_complete") is not False
                )
            ):
                errors.append("daily_verdict_pending_or_zero_proof_mismatch")
            expected_consistency = bool(
                raw_verdict.get("mango_enumeration_complete") is True
                and counts["state_overlap_count"] == 0
                and counts["unexplained_missing"] == 0
                and counts["quarantine_without_reason"] == 0
                and counts["duplicate_call_keys"] == 0
                and counts["state_not_in_mango_enumeration"] == 0
            )
            expected_closure = bool(
                expected_consistency
                and enumeration_source_covers_day(
                    raw_verdict.get("mango_enumeration_source"),
                    date.fromisoformat(str(day_key)),
                    require_full_day=True,
                )
                and counts["pending_unique"] == 0
                and counts["pending_over_sla"] == 0
                and counts["ready_without_dual_asr_or_explicit_exception"] == 0
                and counts["ready_without_resolve"] == 0
                and counts["ready_without_analyze"] == 0
            )
            if raw_verdict.get("consistency_ok") is not expected_consistency:
                errors.append("daily_verdict_consistency_mismatch")
            if raw_verdict.get("closure_ok") is not expected_closure:
                errors.append("daily_verdict_closure_mismatch")
            verdict_consistency.append(expected_consistency)
            verdict_closure.append(expected_closure)
        if verdict_consistency and manifest.get("consistency_ok") is not all(
            verdict_consistency
        ):
            errors.append("manifest_consistency_mismatch")
        if verdict_closure and manifest.get("closure_ok") is not all(verdict_closure):
            errors.append("manifest_closure_mismatch")
        if required_day is not None:
            required_verdict = daily_verdicts.get(required_day.isoformat())
            if not isinstance(required_verdict, Mapping):
                errors.append("required_day_verdict_missing")
            else:
                if (
                    require_consistency
                    and required_verdict.get("consistency_ok") is not True
                ):
                    errors.append("required_day_consistency_not_proven")
                if (
                    require_closure
                    and required_verdict.get("closure_ok") is not True
                ):
                    errors.append("required_day_closure_not_proven")
    if manifest.get("quick_check") != "ok" or manifest.get("integrity_check") != "ok":
        errors.append("sqlite_check_failed")
    snapshot = manifest.get("manifest_snapshot")
    if not compatibility and (
        not isinstance(snapshot, Mapping)
        or not isinstance(snapshot.get("end_offset"), int)
        or int(snapshot.get("end_offset")) < 0
        or not re.fullmatch(r"[0-9a-f]{64}", str(snapshot.get("sha256") or ""))
    ):
        errors.append("manifest_snapshot_invalid")
    for field in ("created_at_utc", "published_at"):
        if compatibility and not manifest.get(field):
            continue
        try:
            parse_aware_datetime(manifest.get(field))
        except (TypeError, ValueError):
            errors.append(f"{field}_invalid")
    errors.extend(validate_runtime_fingerprint(manifest.get("runtime_fingerprint")))
    return errors


def stage_capacity_report(
    *, benchmark: Mapping[str, Any], peak_snapshot: Mapping[str, Any], physical_memory_bytes: int
) -> Mapping[str, Any]:
    stages = ("whisper", "gigaam", "resolve", "analyze")
    missing = [stage for stage in stages if not isinstance(benchmark.get(stage), Mapping)]
    audio_hours = float(benchmark.get("audio_hours") or 0)
    total_wall = sum(float((benchmark.get(stage) or {}).get("wall_seconds") or 0) for stage in stages)
    peak_flow = float(peak_snapshot.get("peak_audio_hours_per_hour") or 0)
    capacity_per_hour = audio_hours * 3600 / total_wall if audio_hours > 0 and total_wall > 0 else 0
    headroom = capacity_per_hour / peak_flow if peak_flow > 0 else 0
    peak_memory = max(
        (int((benchmark.get(stage) or {}).get("peak_memory_bytes") or 0) for stage in stages),
        default=0,
    )
    memory_ratio = peak_memory / physical_memory_bytes if physical_memory_bytes > 0 else 1.0
    swap_peak = max(
        (int((benchmark.get(stage) or {}).get("swap_bytes") or 0) for stage in stages),
        default=0,
    )
    ok = not missing and headroom >= 2 and memory_ratio < 0.60
    return {
        "status": "ok" if ok else "blocked",
        "missing_stages": missing,
        "audio_hours": audio_hours,
        "total_stage_wall_seconds": round(total_wall, 3),
        "peak_audio_hours_per_hour": peak_flow,
        "capacity_audio_hours_per_hour": round(capacity_per_hour, 6),
        "headroom_ratio": round(headroom, 6),
        "peak_memory_bytes": peak_memory,
        "physical_memory_bytes": physical_memory_bytes,
        "peak_memory_ratio": round(memory_ratio, 6),
        "swap_peak_bytes": swap_peak,
        "capacity_ok": ok,
    }


def safe_alert_payload(value: Mapping[str, Any]) -> Mapping[str, Any]:
    projected = {
        str(key): item
        for key, item in value.items()
        if str(key) in SAFE_ALERT_KEYS and isinstance(item, (str, int, float, bool, type(None)))
    }
    text_value = json.dumps(projected, ensure_ascii=False, sort_keys=True)
    if SENSITIVE_ALERT_RE.search(text_value) or "/Users/" in text_value or ".sqlite" in text_value:
        raise RuntimeError("safe alert projection contains sensitive data")
    return projected


def foreign_host_ids(
    entries: Iterable[Any],
    *,
    active_host_id: str,
    foreign_after: Optional[datetime] = None,
) -> list[str]:
    found: set[str] = set()
    for raw in entries:
        entry = _entry_dict(raw)
        host_id = str(entry.get("host_id") or "").strip()
        if foreign_after is not None:
            try:
                created_at = parse_aware_datetime(entry.get("created_at"))
            except (TypeError, ValueError):
                found.add(host_id or "missing_host_id")
                continue
            if created_at <= foreign_after:
                continue
        if not host_id:
            found.add("missing_host_id")
        elif host_id != active_host_id:
            found.add(host_id)
    return sorted(found)


def status_counts(entries: Iterable[Any]) -> Mapping[str, int]:
    return dict(Counter(str(_entry_dict(entry).get("status") or "missing") for entry in entries))


def write_private_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_private_json(path, payload, indent=2)
