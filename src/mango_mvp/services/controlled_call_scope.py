from __future__ import annotations

import hashlib
import json
import fcntl
import os
import re
import signal
import stat
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Mapping, Optional
from zoneinfo import ZoneInfo

from sqlalchemy import text
from sqlalchemy.orm import Session

from mango_mvp.productization.mango_calls_service_contract import (
    current_git_sha,
    git_worktree_is_clean,
    read_host_id,
)
from mango_mvp.productization.owner_only_io import (
    atomic_replace_owner_only_bytes,
    inspect_stable_regular_file,
    path_has_cloud_marker,
    read_stable_regular_bytes,
    read_stable_regular_bytes_with_path,
    validate_owner_only_directory,
)

if TYPE_CHECKING:
    from mango_mvp.config import Settings


CONTROLLED_CALL_ALLOWLIST_SCHEMA = "mango_calls_controlled_allowlist_v2"
CONTROLLED_CAPTURE_REQUEST_SCHEMA = "mango_calls_controlled_capture_request_v1"
CONTROLLED_CALL_RUN_AUTHORITY_SCHEMA = "mango_calls_controlled_run_authority_v3"
CONTROLLED_CALL_LIFELINE_FD_ENV = "MANGO_CALLS_CONTROLLED_LIFELINE_FD"
CONTROLLED_CALL_ALLOWED_CLI_COMMANDS = frozenset(
    {
        "worker",
        "stats",
    }
)
CONTROLLED_CALL_ALLOWED_WORKER_STAGES = frozenset(
    {"transcribe", "backfill-second-asr", "resolve", "analyze"}
)


@dataclass(frozen=True)
class ControlledCallScope:
    source_call_id: str
    target_record_id: int
    source_audio_sha256: str
    source_audio_size_bytes: int
    tenant_id: str
    code_sha: str
    host_id: str
    allowlist_path: Path
    allowlist_sha256: str


@dataclass(frozen=True)
class ControlledCaptureRequest:
    source_call_id: str
    expected_count: int
    since: datetime
    until: datetime
    pipeline_root: Path
    tenant_id: str
    code_sha: str
    host_id: str
    request_path: Path
    request_sha256: str


def _setting_text(settings: "Settings", name: str) -> str:
    return str(getattr(settings, name, None) or "").strip()


def controlled_call_scope_configured(settings: "Settings") -> bool:
    mode = _setting_text(settings, "calls_processing_scope") or "service"
    return mode != "service" or any(
        _setting_text(settings, name)
        for name in (
            "controlled_call_allowlist_path",
            "controlled_call_allowlist_sha256",
            "controlled_call_tenant_id",
            "controlled_call_code_sha",
            "controlled_call_host_id",
            "controlled_call_host_id_path",
        )
    )


def _canonical_source_call_id(value: Any) -> str:
    if not isinstance(value, str) or value != value.strip():
        raise RuntimeError("controlled_call_source_call_id_invalid")
    if not value or len(value) > 256 or any(ord(char) < 32 for char in value):
        raise RuntimeError("controlled_call_source_call_id_invalid")
    return value


def _request_boundary(value: Any, *, label: str) -> datetime:
    if not isinstance(value, str) or value != value.strip():
        raise RuntimeError(f"controlled_capture_request_{label}_invalid")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RuntimeError(
            f"controlled_capture_request_{label}_invalid"
        ) from exc
    if (
        parsed.tzinfo is None
        or parsed.utcoffset() is None
        or parsed.microsecond
    ):
        raise RuntimeError(f"controlled_capture_request_{label}_invalid")
    return parsed.astimezone(timezone.utc)


def load_controlled_capture_request(
    *,
    path: Path,
    expected_sha256: str,
    expected_tenant_id: str,
    expected_code_sha: str,
    expected_host_id: str,
    host_id_path: Path,
    project_root: Path,
    expected_pipeline_root: Path,
    now: Optional[datetime] = None,
) -> ControlledCaptureRequest:
    """Load one immutable pre-download request for an isolated pilot call."""

    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise RuntimeError("controlled_capture_request_sha256_invalid")
    if not re.fullmatch(r"[0-9a-f]{40}", expected_code_sha):
        raise RuntimeError("controlled_capture_request_code_sha_invalid")
    if read_host_id(host_id_path) != expected_host_id:
        raise RuntimeError("controlled_capture_request_host_mismatch")
    raw, resolved = read_stable_regular_bytes_with_path(
        path,
        label="controlled_capture_request",
        owner_only_mode=0o600,
    )
    owner_local = (Path.home() / ".mango_local").resolve(strict=False)
    if path_has_cloud_marker(resolved) or owner_local not in resolved.parents:
        raise RuntimeError("controlled_capture_request_must_be_owner_local")
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if actual_sha256 != expected_sha256:
        raise RuntimeError("controlled_capture_request_sha256_mismatch")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("controlled_capture_request_invalid_json") from exc
    expected_keys = {
        "schema_version",
        "source_call_ids",
        "expected_count",
        "since",
        "until",
        "pipeline_root",
        "tenant_id",
        "code_sha",
        "host_id",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected_keys:
        raise RuntimeError("controlled_capture_request_schema_fields_mismatch")
    if payload.get("schema_version") != CONTROLLED_CAPTURE_REQUEST_SCHEMA:
        raise RuntimeError("controlled_capture_request_schema_mismatch")
    source_call_ids = payload.get("source_call_ids")
    if not isinstance(source_call_ids, list) or len(source_call_ids) != 1:
        raise RuntimeError("controlled_capture_request_requires_one_id")
    source_call_id = _canonical_source_call_id(source_call_ids[0])
    if type(payload.get("expected_count")) is not int or payload.get(
        "expected_count"
    ) != 1:
        raise RuntimeError("controlled_capture_request_expected_count_invalid")
    since = _request_boundary(payload.get("since"), label="since")
    until = _request_boundary(payload.get("until"), label="until")
    reference_now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    if (
        since >= until
        or until - since > timedelta(hours=1)
        or until > reference_now - timedelta(minutes=1)
    ):
        raise RuntimeError("controlled_capture_request_window_not_closed")
    moscow = ZoneInfo("Europe/Moscow")
    if since.astimezone(moscow).date() != (
        until - timedelta(seconds=1)
    ).astimezone(moscow).date():
        raise RuntimeError(
            "controlled_capture_request_crosses_moscow_day"
        )
    requested_pipeline = Path(str(payload.get("pipeline_root") or "")).expanduser()
    expected_pipeline = expected_pipeline_root.expanduser().resolve(strict=False)
    if (
        not requested_pipeline.is_absolute()
        or requested_pipeline.resolve(strict=False) != expected_pipeline
        or expected_pipeline == owner_local
        or owner_local not in expected_pipeline.parents
        or path_has_cloud_marker(expected_pipeline)
    ):
        raise RuntimeError("controlled_capture_request_pipeline_mismatch")
    if payload.get("tenant_id") != expected_tenant_id:
        raise RuntimeError("controlled_capture_request_tenant_mismatch")
    if payload.get("code_sha") != expected_code_sha:
        raise RuntimeError("controlled_capture_request_code_mismatch")
    if payload.get("host_id") != expected_host_id:
        raise RuntimeError("controlled_capture_request_host_mismatch")
    if (
        current_git_sha(project_root) != expected_code_sha
        or not git_worktree_is_clean(project_root)
    ):
        raise RuntimeError("controlled_capture_request_runtime_code_mismatch")
    return ControlledCaptureRequest(
        source_call_id=source_call_id,
        expected_count=1,
        since=since,
        until=until,
        pipeline_root=expected_pipeline,
        tenant_id=expected_tenant_id,
        code_sha=expected_code_sha,
        host_id=expected_host_id,
        request_path=resolved,
        request_sha256=actual_sha256,
    )


def load_controlled_call_allowlist(
    *,
    path: Path,
    expected_sha256: str,
    expected_tenant_id: str,
    expected_code_sha: str,
    expected_host_id: str,
    host_id_path: Path,
    project_root: Path,
) -> ControlledCallScope:
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise RuntimeError("controlled_call_allowlist_sha256_invalid")
    if not re.fullmatch(r"[0-9a-f]{40}", expected_code_sha):
        raise RuntimeError("controlled_call_code_sha_invalid")
    if not expected_tenant_id or len(expected_tenant_id) > 128:
        raise RuntimeError("controlled_call_tenant_id_invalid")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{2,127}", expected_host_id):
        raise RuntimeError("controlled_call_host_id_invalid")
    if read_host_id(host_id_path) != expected_host_id:
        raise RuntimeError("controlled_call_actual_host_mismatch")

    raw, resolved = read_stable_regular_bytes_with_path(
        path,
        label="controlled_call_allowlist",
        owner_only_mode=0o600,
    )
    owner_local = (Path.home() / ".mango_local").resolve(strict=False)
    if path_has_cloud_marker(resolved) or owner_local not in resolved.parents:
        raise RuntimeError("controlled_call_allowlist_must_be_owner_local")
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if actual_sha256 != expected_sha256:
        raise RuntimeError("controlled_call_allowlist_sha256_mismatch")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("controlled_call_allowlist_invalid_json") from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError("controlled_call_allowlist_must_be_object")
    expected_keys = {
        "schema_version",
        "source_call_ids",
        "target_record_id",
        "source_audio_sha256",
        "source_audio_size_bytes",
        "tenant_id",
        "code_sha",
        "host_id",
    }
    if set(payload) != expected_keys:
        raise RuntimeError("controlled_call_allowlist_schema_fields_mismatch")
    source_call_ids = payload.get("source_call_ids")
    if (
        not isinstance(source_call_ids, list)
        or len(source_call_ids) != 1
    ):
        raise RuntimeError("controlled_call_allowlist_must_contain_exactly_one_id")
    source_call_id = _canonical_source_call_id(source_call_ids[0])
    target_record_id = payload.get("target_record_id")
    source_audio_sha256 = payload.get("source_audio_sha256")
    source_audio_size_bytes = payload.get("source_audio_size_bytes")
    if not isinstance(target_record_id, int) or target_record_id < 1:
        raise RuntimeError("controlled_call_target_record_id_invalid")
    if not isinstance(source_audio_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", source_audio_sha256
    ):
        raise RuntimeError("controlled_call_source_audio_sha256_invalid")
    if not isinstance(source_audio_size_bytes, int) or source_audio_size_bytes < 1:
        raise RuntimeError("controlled_call_source_audio_size_invalid")
    if payload.get("schema_version") != CONTROLLED_CALL_ALLOWLIST_SCHEMA:
        raise RuntimeError("controlled_call_allowlist_schema_mismatch")
    if payload.get("tenant_id") != expected_tenant_id:
        raise RuntimeError("controlled_call_allowlist_tenant_mismatch")
    if payload.get("code_sha") != expected_code_sha:
        raise RuntimeError("controlled_call_allowlist_code_mismatch")
    if payload.get("host_id") != expected_host_id:
        raise RuntimeError("controlled_call_allowlist_host_mismatch")
    if (
        current_git_sha(project_root) != expected_code_sha
        or not git_worktree_is_clean(project_root)
    ):
        raise RuntimeError("controlled_call_runtime_code_mismatch")
    return ControlledCallScope(
        source_call_id=source_call_id,
        target_record_id=target_record_id,
        source_audio_sha256=source_audio_sha256,
        source_audio_size_bytes=source_audio_size_bytes,
        tenant_id=expected_tenant_id,
        code_sha=expected_code_sha,
        host_id=expected_host_id,
        allowlist_path=resolved,
        allowlist_sha256=actual_sha256,
    )


def load_controlled_call_scope(settings: "Settings") -> ControlledCallScope | None:
    mode = _setting_text(settings, "calls_processing_scope") or "service"
    names = {
        "path": _setting_text(settings, "controlled_call_allowlist_path"),
        "sha256": _setting_text(settings, "controlled_call_allowlist_sha256"),
        "tenant_id": _setting_text(settings, "controlled_call_tenant_id"),
        "code_sha": _setting_text(settings, "controlled_call_code_sha"),
        "host_id": _setting_text(settings, "controlled_call_host_id"),
        "host_id_path": _setting_text(
            settings, "controlled_call_host_id_path"
        ),
    }
    if mode == "service" and not any(names.values()):
        return None
    if mode != "controlled_1":
        raise RuntimeError("calls_processing_scope_invalid")
    if not all(names.values()):
        raise RuntimeError("controlled_call_scope_configuration_incomplete")
    return load_controlled_call_allowlist(
        path=Path(names["path"]),
        expected_sha256=names["sha256"],
        expected_tenant_id=names["tenant_id"],
        expected_code_sha=names["code_sha"],
        expected_host_id=names["host_id"],
        host_id_path=Path(names["host_id_path"]),
        project_root=Path(__file__).resolve().parents[3],
    )


def require_unique_controlled_call(
    session: Session,
    settings: "Settings",
) -> ControlledCallScope | None:
    scope = load_controlled_call_scope(settings)
    if scope is None:
        return None
    rows = session.execute(
        text(
            "SELECT id FROM call_records "
            "WHERE source_call_id = :controlled_source_call_id "
            "ORDER BY id ASC"
        ),
        {"controlled_source_call_id": scope.source_call_id},
    ).all()
    if len(rows) != 1 or int(rows[0][0]) != scope.target_record_id:
        raise RuntimeError("controlled_call_database_match_must_be_exactly_one")
    return scope


def controlled_audio_input_path(
    settings: "Settings",
    *,
    record_id: int,
    source_call_id: str | None,
    source_file: Path,
) -> Path:
    """Return the per-run verified audio copy for controlled ASR."""
    scope = load_controlled_call_scope(settings)
    if scope is None:
        return source_file
    if record_id != scope.target_record_id or source_call_id != scope.source_call_id:
        raise RuntimeError("controlled_call_audio_identity_mismatch")
    raw_path = _setting_text(settings, "controlled_call_audio_snapshot_path")
    expected_sha256 = _setting_text(
        settings, "controlled_call_audio_snapshot_sha256"
    )
    expected_size = getattr(
        settings, "controlled_call_audio_snapshot_size_bytes", None
    )
    if (
        not raw_path
        or expected_sha256 != scope.source_audio_sha256
        or expected_size != scope.source_audio_size_bytes
    ):
        raise RuntimeError("controlled_call_audio_snapshot_configuration_invalid")
    evidence = inspect_stable_regular_file(
        Path(raw_path),
        label="controlled_call_audio_snapshot",
        require_owner=True,
        require_single_link=True,
        owner_only_mode=0o600,
    )
    resolved = evidence.get("resolved_path")
    owner_local = (Path.home() / ".mango_local").resolve(strict=False)
    if not isinstance(resolved, Path) or (
        path_has_cloud_marker(resolved) or owner_local not in resolved.parents
    ):
        raise RuntimeError("controlled_call_audio_snapshot_must_be_owner_local")
    if (
        evidence.get("sha256") != expected_sha256
        or evidence.get("size_bytes") != expected_size
    ):
        raise RuntimeError("controlled_call_audio_snapshot_binding_mismatch")
    return resolved


def call_artifact_directory(
    settings: "Settings",
    *,
    export_dir: Path,
    source_file: Path,
    source_call_id: str | None,
) -> Path:
    scope = load_controlled_call_scope(settings)
    if scope is None:
        return export_dir / source_file.parent.name
    if source_call_id != scope.source_call_id:
        raise RuntimeError("controlled_call_artifact_identity_mismatch")
    owner_local = (Path.home() / ".mango_local").resolve(strict=False)
    target = export_dir / "controlled_1" / hashlib.sha256(
        scope.source_call_id.encode("utf-8")
    ).hexdigest()
    current = export_dir
    for index, directory in enumerate(
        (export_dir, export_dir / "controlled_1", target)
    ):
        if os.path.lexists(directory):
            opened = os.lstat(directory)
            if not stat.S_ISDIR(opened.st_mode):
                raise RuntimeError("controlled_call_artifact_directory_unsafe")
        else:
            if index == 0:
                directory.mkdir(parents=True, mode=0o700)
            else:
                directory.mkdir(mode=0o700)
        directory.chmod(0o700)
        current = validate_owner_only_directory(
            directory,
            label="controlled_call_artifact_directory",
            owner_only_mode=0o700,
        )
    if path_has_cloud_marker(current) or owner_local not in current.parents:
        raise RuntimeError("controlled_call_artifact_directory_must_be_owner_local")
    return current


def write_call_artifact_bytes(
    settings: "Settings",
    path: Path,
    payload: bytes,
) -> None:
    if load_controlled_call_scope(settings) is None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return
    validate_owner_only_directory(
        path.parent,
        label="controlled_call_artifact_directory",
        owner_only_mode=0o700,
    )
    atomic_replace_owner_only_bytes(
        path,
        payload,
        label="controlled_call_artifact",
    )


def read_call_artifact_text(
    settings: "Settings",
    path: Path,
    *,
    errors: str = "strict",
) -> str:
    if load_controlled_call_scope(settings) is None:
        return path.read_text(encoding="utf-8", errors=errors)
    raw = read_stable_regular_bytes(
        path,
        label="controlled_call_artifact",
        owner_only_mode=0o600,
    )
    return raw.decode("utf-8", errors=errors)


def enforce_controlled_cli_command(settings: "Settings", command: str) -> None:
    if not controlled_call_scope_configured(settings):
        return
    load_controlled_call_scope(settings)
    if command not in CONTROLLED_CALL_ALLOWED_CLI_COMMANDS:
        raise RuntimeError(f"controlled_call_scope_forbids_cli_command:{command}")


def _parse_authority_time(value: Any, *, label: str) -> datetime:
    if not isinstance(value, str):
        raise RuntimeError(f"controlled_call_run_authority_{label}_invalid")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RuntimeError(
            f"controlled_call_run_authority_{label}_invalid"
        ) from exc
    if parsed.tzinfo is None:
        raise RuntimeError(f"controlled_call_run_authority_{label}_invalid")
    return parsed.astimezone(timezone.utc)


def _validate_controlled_runtime_settings(
    settings: "Settings",
    *,
    stage: str,
) -> None:
    expected = {
        "transcribe_provider": "mlx",
        "gigaam_model": "v2_rnnt",
        "dual_merge_provider": "rule",
        "mono_role_assignment_mode": "rule",
        "resolve_llm_provider": "codex_cli",
        "resolve_dialogue_mode": "dialogue",
        "analyze_provider": "codex_cli",
    }
    for name, value in expected.items():
        if _setting_text(settings, name).lower() != value:
            raise RuntimeError(
                f"controlled_call_runtime_setting_mismatch:{name}"
            )
    if stage == "transcribe":
        if bool(settings.dual_transcribe_enabled) or _setting_text(
            settings, "secondary_transcribe_provider"
        ):
            raise RuntimeError("controlled_call_primary_asr_must_run_alone")
    elif not (
        bool(settings.dual_transcribe_enabled)
        and _setting_text(settings, "secondary_transcribe_provider").lower()
        == "gigaam"
    ):
        raise RuntimeError("controlled_call_secondary_asr_must_be_gigaam")
    rescue_provider = _setting_text(
        settings, "resolve_rescue_provider"
    ).lower()
    if rescue_provider not in {"", "none"} or bool(
        settings.resolve_rescue_dual_enabled
    ):
        raise RuntimeError("controlled_call_resolve_rescue_must_be_disabled")
    if not bool(settings.split_stereo_channels):
        raise RuntimeError("controlled_call_stereo_split_must_be_enabled")
    if os.getenv("MANGO_STRICT_ASR_RUNTIME", "").strip() != "1":
        raise RuntimeError("controlled_call_strict_asr_runtime_required")
    if os.getenv("MANGO_CODEX_SERVICE_TIER", "").strip().lower() != "flex":
        raise RuntimeError("controlled_call_codex_service_tier_must_be_flex")
    if bool(settings.llm_cache_enabled):
        raise RuntimeError("controlled_call_llm_cache_must_be_disabled")


def _validate_controlled_run_authority(
    settings: "Settings",
    *,
    scope: ControlledCallScope,
    stage: str,
) -> None:
    raw_path = _setting_text(settings, "controlled_call_run_authority_path")
    expected_sha256 = _setting_text(
        settings, "controlled_call_run_authority_sha256"
    )
    if not raw_path or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise RuntimeError("controlled_call_run_authority_missing")
    raw, resolved = read_stable_regular_bytes_with_path(
        Path(raw_path),
        label="controlled_call_run_authority",
        owner_only_mode=0o600,
    )
    owner_local = (Path.home() / ".mango_local").resolve(strict=False)
    if path_has_cloud_marker(resolved) or owner_local not in resolved.parents:
        raise RuntimeError("controlled_call_run_authority_must_be_owner_local")
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise RuntimeError("controlled_call_run_authority_sha256_mismatch")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("controlled_call_run_authority_invalid_json") from exc
    expected_keys = {
        "schema_version",
        "run_id",
        "stage",
        "issued_at",
        "expires_at",
        "allowlist_sha256",
        "code_sha",
        "host_id",
        "target_record_id",
        "source_audio_sha256",
        "source_audio_size_bytes",
        "authority_mode",
        "authority_evidence_path",
        "authority_evidence_sha256",
        "cutover_manifest_sha256",
        "pipeline_lock_path",
        "orchestrator_pid",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected_keys:
        raise RuntimeError("controlled_call_run_authority_schema_fields_mismatch")
    if payload.get("schema_version") != CONTROLLED_CALL_RUN_AUTHORITY_SCHEMA:
        raise RuntimeError("controlled_call_run_authority_schema_mismatch")
    if not re.fullmatch(r"[A-Za-z0-9_.-]{8,160}", str(payload.get("run_id") or "")):
        raise RuntimeError("controlled_call_run_authority_run_id_invalid")
    if payload.get("stage") != stage:
        raise RuntimeError("controlled_call_run_authority_stage_mismatch")
    if payload.get("allowlist_sha256") != scope.allowlist_sha256:
        raise RuntimeError("controlled_call_run_authority_allowlist_mismatch")
    if payload.get("code_sha") != scope.code_sha:
        raise RuntimeError("controlled_call_run_authority_code_mismatch")
    if payload.get("host_id") != scope.host_id:
        raise RuntimeError("controlled_call_run_authority_host_mismatch")
    if (
        payload.get("target_record_id") != scope.target_record_id
        or payload.get("source_audio_sha256") != scope.source_audio_sha256
        or payload.get("source_audio_size_bytes")
        != scope.source_audio_size_bytes
    ):
        raise RuntimeError("controlled_call_run_authority_target_mismatch")
    orchestrator_pid = payload.get("orchestrator_pid")
    if (
        not isinstance(orchestrator_pid, int)
        or orchestrator_pid <= 1
        or orchestrator_pid != os.getppid()
    ):
        raise RuntimeError("controlled_call_run_authority_parent_mismatch")
    authority_mode = payload.get("authority_mode")
    if authority_mode not in {
        "service_cutover_manifest",
        "isolated_controlled_request",
    }:
        raise RuntimeError("controlled_call_run_authority_mode_invalid")
    legacy_cutover_sha256 = payload.get("cutover_manifest_sha256")
    if (
        authority_mode == "service_cutover_manifest"
        and legacy_cutover_sha256 != payload.get("authority_evidence_sha256")
    ) or (
        authority_mode == "isolated_controlled_request"
        and legacy_cutover_sha256 != ""
    ):
        raise RuntimeError("controlled_call_run_authority_legacy_digest_invalid")
    authority_raw, authority_resolved = read_stable_regular_bytes_with_path(
        Path(str(payload.get("authority_evidence_path") or "")),
        label="controlled_call_authority_evidence",
        owner_only_mode=0o600,
    )
    owner_local = (Path.home() / ".mango_local").resolve(strict=False)
    if (
        path_has_cloud_marker(authority_resolved)
        or owner_local not in authority_resolved.parents
        or hashlib.sha256(authority_raw).hexdigest()
        != payload.get("authority_evidence_sha256")
    ):
        raise RuntimeError("controlled_call_run_authority_evidence_mismatch")
    lock_path = Path(str(payload.get("pipeline_lock_path") or ""))
    if path_has_cloud_marker(lock_path) or owner_local not in lock_path.resolve(
        strict=False
    ).parents:
        raise RuntimeError("controlled_call_run_authority_lock_mismatch")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        lock_fd = os.open(lock_path, flags)
    except OSError as exc:
        raise RuntimeError("controlled_call_run_authority_lock_mismatch") from exc
    try:
        lock_stat = os.fstat(lock_fd)
        if (
            not stat.S_ISREG(lock_stat.st_mode)
            or lock_stat.st_uid != os.getuid()
            or stat.S_IMODE(lock_stat.st_mode) != 0o600
        ):
            raise RuntimeError("controlled_call_run_authority_lock_mismatch")
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            raise RuntimeError("controlled_call_run_authority_lock_not_held")
    finally:
        os.close(lock_fd)
    issued_at = _parse_authority_time(payload.get("issued_at"), label="issued_at")
    expires_at = _parse_authority_time(payload.get("expires_at"), label="expires_at")
    now = datetime.now(timezone.utc)
    if (
        issued_at > now + timedelta(minutes=1)
        or expires_at <= now
        or expires_at <= issued_at
        or expires_at - issued_at > timedelta(hours=6)
    ):
        raise RuntimeError("controlled_call_run_authority_not_current")


def enforce_controlled_worker_stages(
    settings: "Settings",
    stages: list[str],
    *,
    stage_limit: int,
) -> None:
    scope = load_controlled_call_scope(settings)
    if scope is None:
        return
    forbidden = sorted(set(stages) - CONTROLLED_CALL_ALLOWED_WORKER_STAGES)
    if forbidden:
        raise RuntimeError(
            "controlled_call_scope_forbids_worker_stages:" + ",".join(forbidden)
        )
    if len(stages) != 1:
        raise RuntimeError("controlled_call_worker_requires_exactly_one_stage")
    if stage_limit != 1:
        raise RuntimeError("controlled_call_worker_requires_stage_limit_one")
    stage = stages[0]
    _validate_controlled_runtime_settings(settings, stage=stage)
    _validate_controlled_run_authority(
        settings,
        scope=scope,
        stage=stage,
    )


@contextmanager
def controlled_worker_parent_lifeline(
    settings: "Settings",
) -> Iterator[None]:
    """Kill the controlled worker group if its orchestrator disappears."""

    if not controlled_call_scope_configured(settings):
        yield
        return
    raw_fd = os.getenv(CONTROLLED_CALL_LIFELINE_FD_ENV, "")
    if not re.fullmatch(r"[0-9]{1,6}", raw_fd):
        raise RuntimeError("controlled_call_lifeline_missing")
    read_fd = int(raw_fd)
    try:
        descriptor = os.fstat(read_fd)
    except OSError as exc:
        raise RuntimeError("controlled_call_lifeline_invalid") from exc
    if not stat.S_ISFIFO(descriptor.st_mode):
        raise RuntimeError("controlled_call_lifeline_invalid")
    sentinel_pid = os.fork()
    if sentinel_pid == 0:
        try:
            signal.signal(signal.SIGTERM, lambda *_args: os._exit(0))
            while os.read(read_fd, 1):
                pass
            os.killpg(os.getpgrp(), signal.SIGKILL)
        except BaseException:
            os._exit(70)
        os._exit(0)
    os.close(read_fd)
    try:
        yield
    finally:
        try:
            os.kill(sentinel_pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            os.waitpid(sentinel_pid, 0)
        except ChildProcessError:
            pass
