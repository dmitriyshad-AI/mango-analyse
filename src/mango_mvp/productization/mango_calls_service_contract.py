from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import stat
import subprocess
from collections import Counter
from contextlib import contextmanager
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence
from zoneinfo import ZoneInfo

from mango_mvp.productization.capture_staging import atomic_write_private_json


MOSCOW = ZoneInfo("Europe/Moscow")
READY_MANIFEST_SCHEMA = "mango_calls_ready_v2"
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


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
    )


@contextmanager
def _stable_regular_descriptor(
    path: Path,
    *,
    label: str,
    owner_only_mode: Optional[int] = None,
) -> Iterable[int]:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RuntimeError(f"{label}_unsafe_or_missing") from exc
    try:
        opened = os.fstat(descriptor)
        current = os.lstat(path)
        if (
            not stat.S_ISREG(opened.st_mode)
            or not stat.S_ISREG(current.st_mode)
            or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino)
        ):
            raise RuntimeError(f"{label}_must_be_regular_nofollow")
        if owner_only_mode is not None and (
            opened.st_uid != os.getuid()
            or stat.S_IMODE(opened.st_mode) != owner_only_mode
        ):
            raise RuntimeError(f"{label}_must_be_owner_only_{owner_only_mode:04o}")
        yield descriptor
        after = os.fstat(descriptor)
        current_after = os.lstat(path)
        if (
            _file_identity(opened) != _file_identity(after)
            or (after.st_dev, after.st_ino)
            != (current_after.st_dev, current_after.st_ino)
            or not stat.S_ISREG(current_after.st_mode)
        ):
            raise RuntimeError(f"{label}_changed_while_reading")
    finally:
        os.close(descriptor)


def read_stable_regular_bytes(
    path: Path,
    *,
    label: str,
    owner_only_mode: Optional[int] = None,
) -> bytes:
    chunks: list[bytes] = []
    with _stable_regular_descriptor(
        path, label=label, owner_only_mode=owner_only_mode
    ) as descriptor:
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
    return b"".join(chunks)


def sha256_file(path: Path) -> str:
    return str(stable_regular_file_evidence(path)["sha256"])


def stable_regular_file_evidence(
    path: Path, *, label: str = "sha256_source"
) -> Mapping[str, Any]:
    digest = hashlib.sha256()
    size_bytes = 0
    with _stable_regular_descriptor(path, label=label) as descriptor:
        opened = os.fstat(descriptor)
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
            size_bytes += len(chunk)
    return {
        "sha256": digest.hexdigest(),
        "size_bytes": size_bytes,
        "device": opened.st_dev,
        "inode": opened.st_ino,
        "mtime_ns": opened.st_mtime_ns,
    }


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


def _has_dual_asr_or_exception(
    row: Mapping[str, Any], *, now: Optional[datetime] = None
) -> bool:
    variants = _json_object(row.get("transcript_variants_json"))
    exception = variants.get("dual_asr_exception")
    if isinstance(exception, Mapping):
        try:
            approved_at = parse_aware_datetime(exception.get("approved_at"))
        except (TypeError, ValueError):
            approved_at = None
        if (
            exception.get("approved") is True
            and str(exception.get("reason") or "").strip()
            and str(exception.get("approved_by") or "").strip()
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
    blocks = [
        value
        for key in ("manager", "client", "full")
        if isinstance((value := variants.get(key)), Mapping)
    ]
    paired = 0
    for block in blocks:
        has_primary = bool(str(block.get("variant_a") or "").strip())
        has_secondary = bool(str(block.get("variant_b") or "").strip())
        # A variant from one channel must never prove the other channel/model.
        if has_primary != has_secondary:
            return False
        paired += int(has_primary and has_secondary)
    return paired > 0


def ready_row_is_complete(
    row: Mapping[str, Any], *, now: Optional[datetime] = None
) -> bool:
    return bool(
        str(row.get("transcription_status") or "") == "done"
        and _has_dual_asr_or_exception(row, now=now)
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
        source.get("cursor") != "not_applicable_stats_request_result"
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
            if end > day_start and start < coverage_end:
                parsed.append((max(start, day_start), min(end, coverage_end)))
    except (TypeError, ValueError):
        return False
    cursor = day_start
    for start, end in sorted(parsed):
        if start > cursor:
            return False
        cursor = max(cursor, end)
    return cursor >= coverage_end


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
    zero_proofs = int(
        (
            zero_by_day.get(day.isoformat())
            if isinstance(zero_by_day, Mapping)
            else enumeration.get("independent_zero_enumerations")
        )
        or 0
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
        or not _has_dual_asr_or_exception(row, now=current)
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
        enumeration_complete
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
