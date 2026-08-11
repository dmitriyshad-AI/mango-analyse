from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import os
import re
import resource
import signal
import shutil
import socket
import sqlite3
import stat
import subprocess
import sys
import tempfile
import time
import uuid
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence
from zoneinfo import ZoneInfo

from mango_mvp.customer_timeline.import_cli import (
    TimelineImportCliConfig,
    run_timeline_import_cli,
)
from mango_mvp.customer_timeline.nightly_service import mango_processed_cursor
from mango_mvp.customer_timeline.safe_copy import file_sha256 as sha256_file
from mango_mvp.customer_timeline.store import CustomerTimelineSQLiteStore
from mango_mvp.customer_timeline.safety import (
    guard_customer_timeline_output_path,
    is_customer_timeline_prod_path,
    is_stable_runtime_path,
)
from mango_mvp.productization.capture_staging import (
    CaptureManifestStore,
    ManifestEntry,
    acknowledge_capture_recovery,
    capture_manifest_health,
    entry_from_json,
    entry_recording_ids,
    event_recording_ids,
    manifest_assets_exist,
    merge_recording_ids,
    atomic_write_private_json,
    stage_capture_events,
)
from mango_mvp.productization.mango_calls_service_contract import (
    READY_MANIFEST_SCHEMA,
    approved_runtime_fingerprint,
    build_stage10_verdict,
    current_git_sha,
    foreign_host_ids,
    load_ready_rows,
    moscow_day_bounds_utc,
    parse_aware_datetime,
    read_host_id,
    ready_row_is_complete,
    stage_capacity_report,
    validate_ready_manifest_payload,
    validate_runtime_fingerprint,
    verify_cutover_authority,
)
from mango_mvp.productization.mango_office import MangoOfficePayloadMapper
from mango_mvp.productization.mango_office_client import (
    DEFAULT_MANGO_BASE_URL,
    MangoOfficeClient,
    MangoOfficeCredentials,
)
from mango_mvp.productization.mango_recordings import MangoRecordingDownloader
from mango_mvp.productization.ready_publication import (
    commit_ready_generation,
    inspect_ready_publication,
    ready_publication_lock,
    recover_ready_generation,
)
from mango_mvp.productization.contracts import Direction, TelephonyCallEvent, TenantRef
from mango_mvp.services.transcribe import TranscribeService


SCHEMA_VERSION = "mango_calls_two_processes_v1"
SEQUENTIAL_PIPELINE_STAGES = (
    "transcribe",
    "backfill-second-asr",
    "resolve",
    "analyze",
)
# Backward-compatible import for callers/tests written before the name was
# corrected.  Execution remains strictly sequential.
PARALLEL_PIPELINE_STAGES = SEQUENTIAL_PIPELINE_STAGES
REQUIRED_PIPELINE_MODULES = ("sqlalchemy", "dotenv", "mlx_whisper", "gigaam", "mango_mvp.cli")
PHONE_RE = re.compile(r"(?:\+7|\b8)[\s\-(]*\d{3}[\s\-)]*\d{3}[\s\-]*\d{2}[\s\-]*\d{2}")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
SECRET_RE = re.compile(
    r"(?:token|api[_-]?key|secret|bearer|authorization)\s*[:=]\s*\S{12,}",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CallsTwoProcessesConfig:
    pipeline_root: Path
    timeline_db: Path
    timeline_allowed_root: Path
    python_executable: Path
    codex_binary: Path
    codex_home_root: Path
    foton_daily_dir: Optional[Path] = None
    tenant_id: str = "foton"
    base_url: str = DEFAULT_MANGO_BASE_URL
    bootstrap_since: Optional[str] = None
    first_lookback_hours: int = 24
    poll_overlap_minutes: int = 30
    pending_recording_retry_hours: int = 72
    recording_set_stabilization_minutes: int = 15
    api_window_hours: int = 12
    min_free_gib: float = 40.0
    stale_lock_seconds: int = 30 * 60
    stage_limit: int = 20
    poll_seconds: int = 10
    max_idle_cycles: int = 1
    freshness_max_age_minutes: int = 90
    manifest_recheck_sleep_sec: float = 2.0
    asr_mode: str = "mlx_dual"
    codex_resolve_model: str = "gpt-5.4"
    codex_analyze_model: str = "gpt-5.4-mini"
    codex_reasoning_effort: str = "medium"
    mlx_whisper_snapshot_path: Optional[Path] = None
    heavy_stage_timeout_seconds: int = 4 * 60 * 60
    expected_code_sha: Optional[str] = None
    host_id_path: Optional[Path] = None
    cutover_manifest_path: Optional[Path] = None
    cutover_proof_max_age_minutes: int = 90
    max_catch_up_days: int = 7
    require_cutover_authority: bool = False
    strict_ready_provenance: bool = False
    publication_root: Optional[Path] = None

    @classmethod
    def from_json(cls, path: Path) -> "CallsTwoProcessesConfig":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("config must be a JSON object")
        optional_daily = str(payload.get("foton_daily_dir") or "").strip()
        config = cls(
            pipeline_root=Path(str(payload["pipeline_root"])).expanduser(),
            timeline_db=Path(str(payload["timeline_db"])).expanduser(),
            timeline_allowed_root=Path(str(payload["timeline_allowed_root"])).expanduser(),
            python_executable=Path(str(payload.get("python_executable") or sys.executable)).expanduser(),
            codex_binary=Path(str(payload.get("codex_binary") or "codex")).expanduser(),
            codex_home_root=Path(str(payload["codex_home_root"])).expanduser(),
            foton_daily_dir=Path(optional_daily).expanduser() if optional_daily else None,
            tenant_id=str(payload.get("tenant_id") or "foton"),
            base_url=str(payload.get("base_url") or DEFAULT_MANGO_BASE_URL),
            bootstrap_since=optional_text(payload.get("bootstrap_since")),
            first_lookback_hours=int(payload.get("first_lookback_hours", 24)),
            poll_overlap_minutes=int(payload.get("poll_overlap_minutes", 30)),
            pending_recording_retry_hours=int(payload.get("pending_recording_retry_hours", 72)),
            recording_set_stabilization_minutes=int(
                payload.get("recording_set_stabilization_minutes", 15)
            ),
            api_window_hours=int(payload.get("api_window_hours", 12)),
            min_free_gib=float(payload.get("min_free_gib", 40.0)),
            stale_lock_seconds=int(payload.get("stale_lock_seconds", 30 * 60)),
            stage_limit=int(payload.get("stage_limit", 20)),
            poll_seconds=int(payload.get("poll_seconds", 10)),
            max_idle_cycles=int(payload.get("max_idle_cycles", 1)),
            freshness_max_age_minutes=int(payload.get("freshness_max_age_minutes", 90)),
            manifest_recheck_sleep_sec=float(payload.get("manifest_recheck_sleep_sec", 2.0)),
            asr_mode=str(payload.get("asr_mode") or "mlx_dual").strip().lower(),
            codex_resolve_model=str(payload.get("codex_resolve_model") or "gpt-5.4"),
            codex_analyze_model=str(payload.get("codex_analyze_model") or "gpt-5.4-mini"),
            codex_reasoning_effort=str(payload.get("codex_reasoning_effort") or "medium"),
            mlx_whisper_snapshot_path=(
                Path(str(payload["mlx_whisper_snapshot_path"])).expanduser()
                if payload.get("mlx_whisper_snapshot_path")
                else None
            ),
            heavy_stage_timeout_seconds=int(
                payload.get("heavy_stage_timeout_seconds", 4 * 60 * 60)
            ),
            expected_code_sha=optional_text(
                payload.get("expected_code_sha")
                or os.getenv("MANGO_CALLS_EXPECTED_CODE_SHA")
            ),
            host_id_path=(
                Path(str(payload["host_id_path"])).expanduser()
                if payload.get("host_id_path")
                else None
            ),
            cutover_manifest_path=(
                Path(str(payload["cutover_manifest_path"])).expanduser()
                if payload.get("cutover_manifest_path")
                else None
            ),
            cutover_proof_max_age_minutes=int(
                payload.get("cutover_proof_max_age_minutes", 90)
            ),
            max_catch_up_days=int(payload.get("max_catch_up_days", 7)),
            require_cutover_authority=bool(
                payload.get("require_cutover_authority", True)
            ),
            strict_ready_provenance=bool(
                payload.get("strict_ready_provenance", True)
            ),
            publication_root=(
                Path(str(payload["publication_root"])).expanduser()
                if payload.get("publication_root")
                else None
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        root = self.pipeline_root.resolve(strict=False)
        timeline_root = self.timeline_allowed_root.resolve(strict=False)
        timeline_db = self.timeline_db.resolve(strict=False)
        if is_stable_runtime_path(root) or is_stable_runtime_path(timeline_db):
            raise ValueError("pipeline and staging DB must stay outside stable_runtime")
        if is_customer_timeline_prod_path(timeline_db):
            raise ValueError("process B refuses customer_timeline prod paths")
        guard_customer_timeline_output_path(timeline_db, timeline_root)
        if self.stage_limit < 1 or self.poll_seconds < 1 or self.max_idle_cycles < 1:
            raise ValueError("worker drain settings must be positive")
        if self.freshness_max_age_minutes < 15:
            raise ValueError("freshness_max_age_minutes must be at least 15")
        if self.api_window_hours < 1 or self.api_window_hours > 24:
            raise ValueError("api_window_hours must be between 1 and 24")
        if self.pending_recording_retry_hours < 1:
            raise ValueError("pending_recording_retry_hours must be positive")
        if self.recording_set_stabilization_minutes < 0:
            raise ValueError("recording_set_stabilization_minutes must be non-negative")
        if self.asr_mode != "mlx_dual":
            raise ValueError("asr_mode must be mlx_dual; single-ASR fallback is disabled")
        if self.min_free_gib < 1:
            raise ValueError("min_free_gib must be at least 1")
        if self.heavy_stage_timeout_seconds < 60:
            raise ValueError("heavy_stage_timeout_seconds must be at least 60")
        if self.max_catch_up_days < 1:
            raise ValueError("max_catch_up_days must be positive")
        if self.require_cutover_authority != self.strict_ready_provenance:
            raise ValueError(
                "cutover authority and strict ready provenance must be enabled together"
            )
        if self.require_cutover_authority and not self.expected_code_sha:
            raise ValueError("expected_code_sha is required for cutover authority")
        if self.publication_root is not None:
            publication = self.publication_root.resolve(strict=False)
            owner_local = (Path.home() / ".mango_local").resolve(strict=False)
            try:
                publication.relative_to(owner_local)
            except ValueError:
                raise ValueError(
                    "publication_root must stay below $HOME/.mango_local"
                ) from None
            if publication == owner_local:
                raise ValueError("publication_root must be a dedicated directory")

    @property
    def capture_dir(self) -> Path:
        return self.pipeline_root / "capture"

    @property
    def recordings_dir(self) -> Path:
        return self.capture_dir / "recordings"

    @property
    def capture_manifest(self) -> Path:
        return self.capture_dir / "capture_manifest.jsonl"

    @property
    def working_dir(self) -> Path:
        return self.pipeline_root / "working"

    @property
    def working_audio_dir(self) -> Path:
        return self.working_dir / "audio"

    @property
    def metadata_csv(self) -> Path:
        return self.working_dir / "metadata.csv"

    @property
    def working_db(self) -> Path:
        return self.working_dir / "mango_calls_pipeline.sqlite"

    @property
    def transcripts_dir(self) -> Path:
        return self.working_dir / "transcripts"

    @property
    def ready_db(self) -> Path:
        return self.pipeline_root / "drop" / "mango_calls_ready.sqlite"

    @property
    def ready_manifest(self) -> Path:
        return self.ready_db.with_suffix(".manifest.json")

    @property
    def cursor_path(self) -> Path:
        return self.pipeline_root / "state" / "mango_api_freshness.json"

    @property
    def process_a_lock(self) -> Path:
        # Compatibility alias: direct process-a and the scheduled pipeline must
        # share one lock domain or either entry point could start a second ASR.
        return self.pipeline_lock

    @property
    def pipeline_lock(self) -> Path:
        return self.pipeline_root / "locks" / "pipeline.lock"

    @property
    def capture_lock(self) -> Path:
        return self.pipeline_root / "locks" / "capture.lock"

    @property
    def process_b_lock(self) -> Path:
        return self.pipeline_root / "locks" / "process_b.lock"

    @property
    def reports_dir(self) -> Path:
        return self.pipeline_root / "reports"

    @property
    def process_a_status_path(self) -> Path:
        return self.pipeline_root / "state" / "process_a_status.json"

    @property
    def process_a_heartbeat_path(self) -> Path:
        return self.pipeline_root / "state" / "process_a_heartbeat.json"

    @property
    def capture_status_path(self) -> Path:
        return self.pipeline_root / "state" / "capture_status.json"

    @property
    def host_id_file(self) -> Path:
        return self.host_id_path or self.pipeline_root / "state" / "host_id"

    @property
    def cutover_manifest_file(self) -> Path:
        return self.cutover_manifest_path or self.pipeline_root / "state" / "cutover_manifest.json"

    @property
    def cutover_cursor_lineage_path(self) -> Path:
        return self.pipeline_root / "state" / "cutover_cursor_lineage.json"

    @property
    def process_b_status_path(self) -> Path:
        return self.pipeline_root / "state" / "process_b_status.json"

    @property
    def process_b_cursor_path(self) -> Path:
        return self.pipeline_root / "state" / "process_b_cursor.json"

    @property
    def local_publication_root(self) -> Path:
        return self.publication_root or (
            Path.home() / ".mango_local" / "mango_calls_publication"
        )

    @property
    def ingest_dir(self) -> Path:
        return self.timeline_allowed_root / "mango_calls_two_processes"


CommandRunner = Callable[[Sequence[str], Mapping[str, str], Path], Mapping[str, Any]]
CaptureRunner = Callable[[CallsTwoProcessesConfig, datetime, datetime], Mapping[str, Any]]
ProducerRunner = Callable[[CallsTwoProcessesConfig, Path, Path, Optional[str]], Mapping[str, Any]]
ImportRunner = Callable[[TimelineImportCliConfig], Mapping[str, Any]]


def configured_host_id(config: CallsTwoProcessesConfig, *, required: bool) -> str:
    try:
        return read_host_id(config.host_id_file)
    except RuntimeError:
        if required:
            raise
        fallback = re.sub(r"[^A-Za-z0-9._-]", "-", socket.gethostname()).strip("-")
        return fallback or "legacy-local-host"


def cutover_authority_report(
    config: CallsTwoProcessesConfig, *, initialize_lineage: bool = False
) -> Mapping[str, Any]:
    if not config.require_cutover_authority:
        return {
            "ok": True,
            "mode": "compatibility_not_for_service",
            "active_host_id": configured_host_id(config, required=False),
        }
    assert config.expected_code_sha is not None
    report = dict(verify_cutover_authority(
        cutover_manifest_path=config.cutover_manifest_file,
        host_id_path=config.host_id_file,
        expected_code_sha=config.expected_code_sha,
        project_root=Path(__file__).resolve().parents[3],
        proof_max_age_minutes=config.cutover_proof_max_age_minutes,
        require_fresh_previous_host_proof=False,
    ))
    if report.get("ok") is not True:
        return report
    expected_cursor_sha = str(report.get("source_cursor_sha256") or "")
    cutover_sha = sha256_file(config.cutover_manifest_file)
    marker = read_json(config.cutover_cursor_lineage_path)
    marker_ok = bool(
        marker.get("schema_version") == "mango_calls_cutover_cursor_lineage_v1"
        and marker.get("source_cursor_sha256") == expected_cursor_sha
        and marker.get("cutover_manifest_sha256") == cutover_sha
        and marker.get("active_host_id") == report.get("active_host_id")
    )
    if not marker_ok and initialize_lineage:
        try:
            cursor_sha = sha256_file(config.cursor_path)
        except OSError:
            cursor_sha = ""
        if cursor_sha == expected_cursor_sha:
            write_json(
                config.cutover_cursor_lineage_path,
                {
                    "schema_version": "mango_calls_cutover_cursor_lineage_v1",
                    "source_cursor_sha256": expected_cursor_sha,
                    "cutover_manifest_sha256": cutover_sha,
                    "active_host_id": report.get("active_host_id"),
                    "verified_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            marker_ok = True
    if not marker_ok:
        report["ok"] = False
        report["errors"] = [*list(report.get("errors") or ()), "source_cursor_lineage_unproven"]
    report["source_cursor_lineage_ok"] = marker_ok
    return report


def run_capture(
    config: CallsTwoProcessesConfig,
    *,
    since: Optional[str] = None,
    until: Optional[str] = None,
    capture_runner: CaptureRunner = None,
) -> Mapping[str, Any]:
    config.validate()
    capture_runner = capture_runner or capture_mango_window
    started = datetime.now(timezone.utc)
    run_id = new_calls_run_id(started)
    try:
        with process_lease(
            config.capture_lock, stale_seconds=config.stale_lock_seconds
        ) as lock_info:
            authority = cutover_authority_report(config, initialize_lineage=True)
            if authority.get("ok") is not True:
                return finalize_report(
                    config,
                    run_id,
                    "capture",
                    "failed",
                    "cutover_authority_failed",
                    {"authority": authority, "lock": lock_info},
                )
            disk = disk_preflight(config)
            if not disk.get("ok"):
                return finalize_report(
                    config,
                    run_id,
                    "capture",
                    "failed",
                    "insufficient_disk_space",
                    {"authority": authority, "disk": disk, "lock": lock_info},
                )
            window_since, window_until = resolve_capture_window(
                config, since=since, until=until
            )
            capture = capture_runner(config, window_since, window_until)
            if capture.get("status") == "failed" or capture.get(
                "mango_enumeration_complete"
            ) is not True:
                return finalize_report(
                    config,
                    run_id,
                    "capture",
                    "failed",
                    "capture_or_enumeration_failed",
                    {
                        "authority": authority,
                        "disk": disk,
                        "capture": capture,
                        "lock": lock_info,
                    },
                )
            status = "partial" if capture.get("status") == "partial" else "ok"
            manifest_tail_incomplete = any(
                positive_int(capture.get(name)) > 0
                for name in (
                    "incomplete_trailing_manifest_records",
                    "recovered_trailing_manifest_records",
                )
            )
            report = finalize_report(
                config,
                run_id,
                "capture",
                status,
                (
                    "capture_manifest_tail_incomplete"
                    if manifest_tail_incomplete
                    else "capture_audio_incomplete"
                    if status == "partial"
                    else ""
                ),
                {
                    "authority": authority,
                    "disk": disk,
                    "capture": capture,
                    "window": {
                        "since": window_since.isoformat(),
                        "until": window_until.isoformat(),
                    },
                    "lock": lock_info,
                },
            )
            if not manifest_tail_incomplete:
                write_cursor(config.cursor_path, window_until, capture)
            recovered = positive_int(
                capture.get("recovered_trailing_manifest_records")
            )
            if recovered:
                incident_sha = str(capture.get("recovery_incident_sha256") or "")
                if not incident_sha:
                    raise RuntimeError("capture recovery incident identity is missing")
                acknowledge_capture_recovery(
                    config.capture_manifest,
                    expected_count=recovered,
                    expected_incident_sha256=incident_sha,
                )
            return report
    except LockBusy as exc:
        return finalize_report(
            config,
            run_id,
            "capture",
            "locked",
            "capture_locked",
            {"lock": exc.metadata},
        )
    except Exception as exc:
        return finalize_report(
            config,
            run_id,
            "capture",
            "failed",
            f"capture_exception:{type(exc).__name__}",
            {"diagnostic": safe_exception_diagnostic(exc)},
        )


def run_pipeline(
    config: CallsTwoProcessesConfig,
    *,
    command_runner: CommandRunner = None,
    process_b_runner: Callable[[CallsTwoProcessesConfig], Mapping[str, Any]] = None,
) -> Mapping[str, Any]:
    config.validate()
    try:
        with process_lease(
            config.pipeline_lock, stale_seconds=config.stale_lock_seconds
        ) as lock_info:
            return _run_pipeline_locked(
                config,
                command_runner=command_runner,
                process_b_runner=process_b_runner,
                lock_info=lock_info,
            )
    except LockBusy as exc:
        return {
            "schema_version": SCHEMA_VERSION,
            "process": "pipeline",
            "status": "locked",
            "stop_reason": "pipeline_locked",
            "new": 0,
            "reused": 0,
            "process_a": None,
            "process_b": None,
            "lock": exc.metadata,
        }


def _run_pipeline_locked(
    config: CallsTwoProcessesConfig,
    *,
    command_runner: CommandRunner,
    process_b_runner: Callable[[CallsTwoProcessesConfig], Mapping[str, Any]] | None,
    lock_info: Mapping[str, Any],
) -> Mapping[str, Any]:
    authority = cutover_authority_report(config)
    if authority.get("ok") is not True:
        return {
            "schema_version": SCHEMA_VERSION,
            "process": "pipeline",
            "status": "failed",
            "stop_reason": "cutover_authority_failed",
            "authority": authority,
        }
    cursor = read_json(config.cursor_path)
    try:
        manifest_end_offset = int(cursor.get("manifest_end_offset"))
    except (TypeError, ValueError):
        manifest_end_offset = -1
    expected_snapshot_sha = str(cursor.get("manifest_snapshot_sha256") or "")
    if (
        cursor.get("mango_enumeration_complete") is not True
        or manifest_end_offset < 0
        or not re.fullmatch(r"[0-9a-f]{64}", expected_snapshot_sha)
        or not config.capture_manifest.is_file()
        or config.capture_manifest.is_symlink()
    ):
        return {
            "schema_version": SCHEMA_VERSION,
            "process": "pipeline",
            "status": "failed",
            "stop_reason": "capture_snapshot_not_proven",
            "authority": authority,
        }
    snapshot = capture_manifest_snapshot(
        config.capture_manifest, end_offset=manifest_end_offset
    )
    if snapshot.get("sha256") != expected_snapshot_sha:
        return {
            "schema_version": SCHEMA_VERSION,
            "process": "pipeline",
            "status": "failed",
            "stop_reason": "capture_snapshot_sha256_mismatch",
            "authority": authority,
        }
    process_a = run_process_a(
        config,
        skip_capture=True,
        command_runner=command_runner,
        manifest_end_offset=manifest_end_offset,
        capture_evidence=cursor,
        pipeline_lock_info=lock_info,
    )
    counters = process_a.get("counters") if isinstance(
        process_a.get("counters"), Mapping
    ) else {}
    metadata = counters.get("metadata") if isinstance(
        counters.get("metadata"), Mapping
    ) else {}
    drop = counters.get("drop") if isinstance(counters.get("drop"), Mapping) else {}
    process_b: Optional[Mapping[str, Any]] = None
    if process_a.get("downstream_ready"):
        process_b = (process_b_runner or run_process_b)(config)
    status = str(process_a.get("status") or "failed")
    if process_b is not None and process_b.get("status") not in {"ok", "idle"}:
        status = "failed"
    no_change = bool(
        positive_int(metadata.get("audio_files")) == 0
        and not metadata.get("db_open_work")
        and drop.get("reused") is True
        and isinstance(process_b, Mapping)
        and process_b.get("status") == "idle"
        and process_b.get("stop_reason") == "drop_unchanged"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "process": "pipeline",
        "status": "idle" if no_change and status == "ok" else status,
        "stop_reason": (
            "unchanged_snapshot"
            if no_change
            else "process_b_not_ready"
            if process_b is not None and process_b.get("status") not in {"ok", "idle"}
            else str(process_a.get("stop_reason") or "")
        ),
        "new": positive_int(metadata.get("audio_files")),
        "reused": positive_int(metadata.get("skipped", {}).get("already_ingested"))
        if isinstance(metadata.get("skipped"), Mapping)
        else 0,
        "manifest_snapshot_end_offset": manifest_end_offset,
        "process_a": process_a,
        "process_b": process_b,
        "authority": authority,
        "lock": lock_info,
    }


def run_process_a(
    config: CallsTwoProcessesConfig,
    *,
    since: Optional[str] = None,
    until: Optional[str] = None,
    skip_capture: bool = False,
    skip_workers: bool = False,
    command_runner: CommandRunner = None,
    capture_runner: CaptureRunner = None,
    manifest_end_offset: Optional[int] = None,
    capture_evidence: Optional[Mapping[str, Any]] = None,
    pipeline_lock_info: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Any]:
    config.validate()
    command_runner = command_runner or run_command
    capture_runner = capture_runner or capture_mango_window
    started = datetime.now(timezone.utc)
    run_id = new_calls_run_id(started)
    try:
        lease = (
            nullcontext(pipeline_lock_info)
            if pipeline_lock_info is not None
            else process_lease(
                config.pipeline_lock, stale_seconds=config.stale_lock_seconds
            )
        )
        with lease as lock_info:
            authority = cutover_authority_report(
                config, initialize_lineage=not skip_capture
            )
            if authority.get("ok") is not True:
                return finalize_report(
                    config,
                    run_id,
                    "process_a",
                    "failed",
                    "cutover_authority_failed",
                    {"authority": authority, "lock": lock_info},
                )
            disk = disk_preflight(config)
            environment = environment_preflight(
                config,
                run_commands=not skip_workers,
                require_mango_credentials=not skip_capture,
            )
            if not bool(environment.get("ok")):
                return finalize_report(
                    config,
                    run_id,
                    "process_a",
                    "failed",
                    "environment_preflight_failed",
                    {"disk": disk, "environment": environment, "lock": lock_info},
                )
            if not bool(disk.get("ok")):
                return finalize_report(
                    config,
                    run_id,
                    "process_a",
                    "failed",
                    "insufficient_disk_space",
                    {"disk": disk, "environment": environment, "lock": lock_info},
                )
            window_since, window_until = resolve_capture_window(config, since=since, until=until)
            capture = (
                {
                    "status": "skipped",
                    "reason": "skip_capture",
                    **dict(capture_evidence or {}),
                }
                if skip_capture
                else capture_runner(config, window_since, window_until)
            )
            if capture.get("status") == "failed":
                return finalize_report(
                    config,
                    run_id,
                    "process_a",
                    "failed",
                    "capture_failed",
                    {"disk": disk, "environment": environment, "capture": capture, "lock": lock_info},
                )
            metadata = dict(
                prepare_ingest_inputs(config)
                if manifest_end_offset is None
                else prepare_ingest_inputs(
                    config,
                    manifest_end_offset=manifest_end_offset,
                    expected_manifest_sha256=optional_text(
                        (capture_evidence or {}).get("manifest_snapshot_sha256")
                    ),
                )
            )
            metadata["db_open_work"] = call_db_has_open_work(config.working_db)
            worker_reports: list[Mapping[str, Any]] = []
            if not skip_workers and (metadata["audio_files"] or metadata["db_open_work"]):
                heavy_cycle_deadline = (
                    time.monotonic() + config.heavy_stage_timeout_seconds
                )
                base_env = worker_environment(config)
                if metadata["audio_files"]:
                    prelude_commands = (
                        cli_command(config, "init-db"),
                        cli_command(
                            config,
                            "ingest",
                            "--recordings-dir",
                            str(config.working_audio_dir),
                            "--metadata-csv",
                            str(config.metadata_csv),
                        ),
                    )
                    for command in prelude_commands:
                        report = (
                            run_command(
                                command,
                                base_env,
                                config.working_dir,
                                deadline=heavy_cycle_deadline,
                            )
                            if command_runner is run_command
                            else command_runner(
                                command, base_env, config.working_dir
                            )
                        )
                        worker_reports.append(report)
                        if int(report.get("rc", 0)) != 0:
                            break
                if not any(
                    int(report.get("rc", 0)) != 0
                    for report in worker_reports
                ):
                    worker_reports.extend(
                        run_sequential_pipeline_workers(
                            config,
                            base_env,
                            command_runner,
                            include_llm=bool(environment.get("codex_network_ok")),
                            run_id=run_id,
                            cycle_deadline=heavy_cycle_deadline,
                        )
                    )
            failed_commands = [item for item in worker_reports if int(item.get("rc", 0)) != 0]
            if failed_commands:
                return finalize_report(
                    config,
                    run_id,
                    "process_a",
                    "failed",
                    "worker_command_failed",
                    {
                        "disk": disk,
                        "environment": environment,
                        "capture": capture,
                        "metadata": metadata,
                        "workers": compact_command_reports(worker_reports),
                        "lock": lock_info,
                    },
                )
            db_counts = call_db_counts(config.working_db) if config.working_db.exists() else empty_call_counts()
            if dead_letter_mass_failure(db_counts):
                return finalize_report(
                    config,
                    run_id,
                    "process_a",
                    "failed",
                    "dead_letter_mass_failure",
                    {
                        "disk": disk,
                        "environment": environment,
                        "capture": capture,
                        "metadata": metadata,
                        "workers": compact_command_reports(worker_reports),
                        "call_db": db_counts,
                        "dead_letter": db_counts.get("dead_letter_stage", {}),
                        "lock": lock_info,
                    },
                )
            manifest_tail_incomplete = any(
                positive_int(value) > 0
                for value in (
                    capture.get("incomplete_trailing_manifest_records"),
                    capture.get("recovered_trailing_manifest_records"),
                    metadata.get("incomplete_trailing_manifest_records"),
                    metadata.get("recovered_trailing_manifest_records"),
                )
            )
            asset_integrity_failed = positive_int(
                metadata.get("asset_integrity_failures")
            ) > 0
            if manifest_tail_incomplete or asset_integrity_failed:
                blocking_reason = (
                    "capture_manifest_tail_incomplete"
                    if manifest_tail_incomplete
                    else "capture_asset_integrity_failed"
                )
                drop = {
                    "status": "blocked",
                    "reason": blocking_reason,
                }
                report = finalize_report(
                    config,
                    run_id,
                    "process_a",
                    "partial",
                    blocking_reason,
                    {
                        "disk": disk,
                        "environment": environment,
                        "capture": capture,
                        "metadata": metadata,
                        "workers": compact_command_reports(worker_reports),
                        "call_db": db_counts,
                        "drop": drop,
                        "lock": lock_info,
                    },
                )
                recovered_tail_count = max(
                    positive_int(capture.get("recovered_trailing_manifest_records")),
                    positive_int(metadata.get("recovered_trailing_manifest_records")),
                )
                if recovered_tail_count:
                    incident_sha256_values = {
                        str(value)
                        for value in (
                            capture.get("recovery_incident_sha256"),
                            metadata.get("recovery_incident_sha256"),
                        )
                        if value
                    }
                    if len(incident_sha256_values) != 1:
                        raise RuntimeError("capture recovery incident identity is missing or inconsistent")
                    acknowledge_capture_recovery(
                        config.capture_manifest,
                        expected_count=recovered_tail_count,
                        expected_incident_sha256=incident_sha256_values.pop(),
                    )
                return report
            if not bool(environment.get("codex_network_ok")):
                if not skip_capture:
                    write_cursor(config.cursor_path, window_until, capture)
                return finalize_report(
                    config,
                    run_id,
                    "process_a",
                    "deferred",
                    "codex_network_unavailable",
                    {
                        "disk": disk,
                        "environment": environment,
                        "capture": capture,
                        "metadata": metadata,
                        "workers": compact_command_reports(worker_reports),
                        "call_db": db_counts,
                        "dead_letter": db_counts.get("dead_letter_stage", {}),
                        "window": {"since": window_since.isoformat(), "until": window_until.isoformat()},
                        "lock": lock_info,
                    },
                )
            drop = (
                publish_ready_db_if_changed(
                    config,
                    db_counts,
                    changed=bool(
                        metadata["audio_files"]
                        or (metadata["db_open_work"] and not skip_workers)
                    ),
                    run_id=run_id,
                    capture_evidence=capture,
                    manifest_end_offset=metadata.get(
                        "manifest_snapshot_end_offset"
                    ),
                    stage_reports=worker_reports,
                    runtime_fingerprint=environment.get("runtime_fingerprint"),
                )
                if config.working_db.exists()
                else {"status": "not_created"}
            )
            counters = {
                "disk": disk,
                "environment": environment,
                "capture": capture,
                "metadata": metadata,
                "workers": compact_command_reports(worker_reports),
                "call_db": db_counts,
                "dead_letter": db_counts.get("dead_letter_stage", {}),
                "drop": drop,
                "window": {"since": window_since.isoformat(), "until": window_until.isoformat()},
                "lock": lock_info,
            }
            balance_green = bool(
                isinstance(drop, Mapping)
                and drop.get("status") == "ready"
                and drop.get("consistency_ok") is True
            )
            capture_partial = capture.get("status") == "partial"
            report = finalize_report(
                config,
                run_id,
                "process_a",
                "ok" if balance_green and not capture_partial else "partial",
                (
                    ""
                    if balance_green and not capture_partial
                    else "capture_partial"
                    if capture_partial
                    else "stage10_consistency_not_proven"
                ),
                counters,
            )
            if not skip_capture:
                write_cursor(config.cursor_path, window_until, capture)
            return report
    except LockBusy as exc:
        return finalize_report(
            config,
            run_id,
            "process_a",
            "locked",
            "process_a_locked",
            {"lock": exc.metadata},
        )
    except Exception as exc:
        failure_run_id = run_id
        preserved_report = config.reports_dir / f"{run_id}_process_a.json"
        diagnostic = dict(safe_exception_diagnostic(exc))
        failure_counters: dict[str, Any] = {"diagnostic": diagnostic}
        if os.path.lexists(preserved_report):
            failure_run_id = new_calls_run_id(datetime.now(timezone.utc))
            diagnostic["preserved_report_run_id"] = run_id
            failure_counters["preserved_report"] = {"run_id": run_id}
        return finalize_report(
            config,
            failure_run_id,
            "process_a",
            "failed",
            f"process_a_exception:{type(exc).__name__}",
            failure_counters,
        )


def _run_process_b(
    config: CallsTwoProcessesConfig,
    *,
    producer_runner: ProducerRunner = None,
    import_runner: ImportRunner = run_timeline_import_cli,
) -> Mapping[str, Any]:
    started = datetime.now(timezone.utc)
    run_id = new_calls_run_id(started)
    producer_runner = producer_runner or run_increment_producer
    config.ingest_dir.mkdir(parents=True, exist_ok=True)
    if not config.ready_db.exists():
        return finalize_report(config, run_id, "process_b", "idle", "drop_missing", {"events": 0})
    drop_fingerprint = ready_drop_fingerprint(config)
    if not drop_fingerprint.get("manifest_valid"):
        return finalize_report(
            config,
            run_id,
            "process_b",
            "failed",
            "drop_manifest_mismatch" if drop_fingerprint.get("manifest_mismatch") else "drop_manifest_invalid",
            {"events": 0, "drop": drop_fingerprint},
        )
    if not config.timeline_db.is_file():
        # The target is reconstructible from the sealed full-drop scan.  Build
        # an empty staging schema before the read-only producer resolves
        # identities; otherwise target loss would make recovery impossible.
        with CustomerTimelineSQLiteStore(
            config.timeline_db,
            allowed_root=config.timeline_allowed_root,
        ):
            pass
    quick_before = sqlite_check(config.timeline_db, "quick_check")
    source_systems_before = call_event_source_systems(config.timeline_db)
    cursor = (
        mango_processed_cursor(config.timeline_db, tenant_id=config.tenant_id)
        if config.timeline_db.is_file()
        else {
            "source_system": "mango_processed_summary",
            "last_cursor_ts": None,
            "max_source_ts": None,
        }
    )
    increment_path = config.ingest_dir / "mango_processed_summary.jsonl"
    producer_report_path = config.ingest_dir / "mango_processed_summary_producer_report.json"
    # Always scan the sealed drop and let dedupe_key decide.  A source-only
    # cursor cannot prove that the target Timeline still exists or is complete,
    # so skipping an unchanged drop would make target loss permanent.
    producer = producer_runner(config, increment_path, producer_report_path, None)
    if str(producer.get("status") or "ok") not in {"ok", "ready"}:
        return finalize_report(
            config,
            run_id,
            "process_b",
            "failed",
            "producer_failed",
            {"producer": producer, "quick_check_before": quick_before},
        )
    if positive_int(producer.get("rows_selected")) != positive_int(producer.get("events_written")):
        return finalize_report(
            config,
            run_id,
            "process_b",
            "failed",
            "producer_event_count_mismatch",
            {"producer": producer, "quick_check_before": quick_before},
        )
    drop_after_producer = ready_drop_fingerprint(config)
    if (
        drop_after_producer.get("manifest_valid") is not True
        or drop_after_producer.get("sha256") != drop_fingerprint.get("sha256")
        or drop_after_producer.get("size_bytes") != drop_fingerprint.get("size_bytes")
    ):
        return finalize_report(
            config,
            run_id,
            "process_b",
            "failed",
            "drop_changed_during_producer",
            {
                "producer": producer,
                "quick_check_before": quick_before,
                "drop_before": drop_fingerprint,
                "drop_after": drop_after_producer,
            },
        )
    import_config = TimelineImportCliConfig(
        tenant_id=config.tenant_id,
        source_kind="mango_processed_summary",
        source_path=increment_path,
        allowed_root=config.timeline_allowed_root,
        timeline_db=config.timeline_db,
        source_ref=f"mango-calls-drop:{config.ready_db.name}",
        out_path=config.ingest_dir / "mango_processed_summary_import_report.json",
        apply=True,
        actor="mango_calls_process_b",
    )
    try:
        imported = import_runner(import_config)
    except RuntimeError as exc:
        if "writer lock" not in str(exc).casefold() and "lock" not in str(exc).casefold():
            raise
        return finalize_report(
            config,
            run_id,
            "process_b",
            "locked",
            "timeline_writer_locked",
            {"producer": producer, "quick_check_before": quick_before},
        )
    if import_config.out_path is not None:
        write_json(import_config.out_path, imported)
    quick_after = sqlite_check(config.timeline_db, "quick_check")
    integrity_after = sqlite_check(config.timeline_db, "integrity_check")
    source_systems_after = call_event_source_systems(config.timeline_db)
    unexpected = sorted(item for item in source_systems_after if item != "mango_processed_summary")
    import_valid = imported.get("validation_ok") is True
    status = "ok" if import_valid and quick_after == "ok" and integrity_after == "ok" and not unexpected else "failed"
    if not import_valid:
        stop_reason = "import_validation_failed"
    else:
        stop_reason = "" if status == "ok" else "source_system_or_integrity_failed"
    counters = {
        "producer": compact_producer_report(producer),
        "import": compact_import_report(imported),
        "quick_check_before": quick_before,
        "quick_check_after": quick_after,
        "integrity_check_after": integrity_after,
        "source_systems_before": source_systems_before,
        "source_systems_after": source_systems_after,
        "unexpected_source_systems": unexpected,
        "cursor_before": cursor,
        "producer_scan_mode": "full_drop_dedupe",
        "drop": drop_fingerprint,
    }
    if status == "ok":
        write_json(
            config.process_b_cursor_path,
            {
                "schema_version": "mango_calls_process_b_cursor_v1",
                "sha256": drop_fingerprint.get("sha256"),
                "size_bytes": drop_fingerprint.get("size_bytes"),
                "data_through": counters["producer"].get("max_event_at"),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )
    return finalize_report(config, run_id, "process_b", status, stop_reason, counters)


def run_process_b(
    config: CallsTwoProcessesConfig,
    *,
    producer_runner: ProducerRunner = None,
    import_runner: ImportRunner = run_timeline_import_cli,
) -> Mapping[str, Any]:
    started = datetime.now(timezone.utc)
    run_id = new_calls_run_id(started)
    try:
        config.validate()
    except Exception as exc:  # noqa: BLE001 - normalized fail-loud boundary
        return process_b_failure_report(
            run_id,
            f"process_b_config_exception:{type(exc).__name__}",
            exc,
        )
    try:
        with process_lease(config.process_b_lock, stale_seconds=config.stale_lock_seconds):
            authority = cutover_authority_report(config)
            if authority.get("ok") is not True:
                return finalize_report(
                    config,
                    run_id,
                    "process_b",
                    "failed",
                    "cutover_authority_failed",
                    {"authority": authority},
                )
            with ready_publication_lock(config.ready_db):
                recover_ready_generation(config.ready_db, lock_held=True)
                return _run_process_b(
                    config,
                    producer_runner=producer_runner,
                    import_runner=import_runner,
                )
    except LockBusy as exc:
        return finalize_report(
            config, run_id, "process_b", "locked", "process_b_locked", {"lock": exc.metadata},
        )
    except Exception as exc:  # noqa: BLE001 - Process B must report, never traceback
        stop_reason = f"process_b_exception:{type(exc).__name__}"
        try:
            return finalize_report(
                config,
                run_id,
                "process_b",
                "failed",
                stop_reason,
                {"diagnostic": safe_exception_diagnostic(exc)},
            )
        except Exception as finalize_exc:  # noqa: BLE001 - reports path may itself be broken
            return process_b_failure_report(
                run_id,
                f"process_b_finalize_exception:{type(finalize_exc).__name__}",
                finalize_exc,
                original_stop_reason=stop_reason,
            )


def process_b_failure_report(
    run_id: str,
    stop_reason: str,
    exc: Exception,
    *,
    original_stop_reason: str = "",
) -> Mapping[str, Any]:
    counters: dict[str, Any] = {"diagnostic": safe_exception_diagnostic(exc)}
    if original_stop_reason:
        counters["original_stop_reason"] = original_stop_reason
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "process": "process_b",
        "status": "failed",
        "stop_reason": stop_reason,
        "counters": counters,
        "safety": {
            "writes_timeline_staging": False,
            "writes_timeline_prod": False,
            "writes_stable_runtime": False,
            "writes_amo": False,
            "writes_crm": False,
            "writes_tallanto": False,
            "runs_asr": False,
            "runs_resolve_analyze": False,
            "runs_sync": False,
        },
    }


def run_cycle(config: CallsTwoProcessesConfig, **process_a_kwargs: Any) -> Mapping[str, Any]:
    config.validate()
    try:
        with process_lease(
            config.pipeline_lock, stale_seconds=config.stale_lock_seconds
        ) as lock_info:
            return _run_cycle_locked(config, lock_info, process_a_kwargs)
    except LockBusy as exc:
        return {
            "schema_version": SCHEMA_VERSION,
            "process": "cycle",
            "status": "locked",
            "stop_reason": "pipeline_locked",
            "process_a": None,
            "process_b": None,
            "lock": exc.metadata,
        }


def _run_cycle_locked(
    config: CallsTwoProcessesConfig,
    lock_info: Mapping[str, Any],
    process_a_kwargs: Mapping[str, Any],
) -> Mapping[str, Any]:
    first = run_process_a(
        config,
        **dict(process_a_kwargs),
        pipeline_lock_info=lock_info,
    )
    if not first.get("downstream_ready"):
        return {
            "schema_version": SCHEMA_VERSION,
            "process": "cycle",
            "status": str(first.get("status") or "failed"),
            "stop_reason": "process_a_not_ok",
            "process_a": first,
            "process_b": None,
            "lock": lock_info,
        }
    second = run_process_b(config)
    second_ok = second.get("status") in {"ok", "idle"}
    cycle_status = "failed" if not second_ok else ("partial" if first.get("status") == "partial" else "ok")
    return {
        "schema_version": SCHEMA_VERSION,
        "process": "cycle",
        "status": cycle_status,
        "stop_reason": "process_b_failed" if not second_ok else str(first.get("stop_reason") or ""),
        "process_a": first,
        "process_b": second,
        "lock": lock_info,
    }


class LockBusy(RuntimeError):
    def __init__(self, metadata: Mapping[str, Any]) -> None:
        super().__init__("process lock is busy")
        self.metadata = dict(metadata)


@contextmanager
def process_lease(path: Path, *, stale_seconds: int) -> Iterator[Mapping[str, Any]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    parent = os.lstat(path.parent)
    if (
        not stat.S_ISDIR(parent.st_mode)
        or stat.S_ISLNK(parent.st_mode)
        or parent.st_uid != os.getuid()
    ):
        raise RuntimeError("process lock directory is unsafe")
    path.parent.chmod(0o700)
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
        raise RuntimeError("process lock is unsafe") from exc
    handle = os.fdopen(descriptor, "a+", encoding="utf-8")
    opened = os.fstat(handle.fileno())
    current = os.lstat(path)
    if (
        not stat.S_ISREG(opened.st_mode)
        or opened.st_nlink != 1
        or opened.st_uid != os.getuid()
        or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino)
    ):
        handle.close()
        raise RuntimeError("process lock is unsafe")
    if stat.S_IMODE(opened.st_mode) != 0o600:
        os.fchmod(handle.fileno(), 0o600)
    handle.seek(0)
    previous = parse_json_object(handle.read())
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise LockBusy(previous) from exc
    locked_info = os.fstat(handle.fileno())
    current = os.lstat(path)
    if (locked_info.st_dev, locked_info.st_ino) != (current.st_dev, current.st_ino):
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()
        raise RuntimeError("process lock changed while acquiring")
    age_seconds = max(0.0, time.time() - locked_info.st_mtime)
    previous_pid = int(previous.get("pid") or 0)
    stale_recovered = bool(previous and (age_seconds > stale_seconds or not pid_exists(previous_pid)))
    metadata = {
        "pid": os.getpid(),
        "acquired_at": datetime.now(timezone.utc).isoformat(),
        "previous_pid": previous_pid or None,
        "previous_age_seconds": round(age_seconds, 3),
        "stale_recovered": stale_recovered,
    }
    handle.seek(0)
    handle.truncate()
    handle.write(json.dumps(metadata, ensure_ascii=False, sort_keys=True))
    handle.flush()
    os.fsync(handle.fileno())
    try:
        yield metadata
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def capture_mango_window(
    config: CallsTwoProcessesConfig,
    since: datetime,
    until: datetime,
) -> Mapping[str, Any]:
    api_key = os.getenv("MANGO_OFFICE_API_KEY", "").strip()
    api_salt = os.getenv("MANGO_OFFICE_API_SALT", "").strip()
    if not api_key or not api_salt:
        return {"status": "failed", "reason": "mango_credentials_missing"}
    credentials = MangoOfficeCredentials(api_key=api_key, api_salt=api_salt)
    client = MangoOfficeClient(credentials=credentials, base_url=config.base_url, timeout_sec=60)
    downloader = MangoRecordingDownloader(
        credentials=credentials,
        base_url=config.base_url,
        timeout_sec=60,
        link_retries=8,
        rate_limit_sleep_sec=30.0,
    )
    mapper = MangoOfficePayloadMapper()
    tenant = TenantRef(config.tenant_id)
    host_id = configured_host_id(
        config, required=config.require_cutover_authority
    )
    rows: list[Mapping[str, Any]] = []
    manifest_store = CaptureManifestStore(config.capture_manifest)
    if not os.path.lexists(config.capture_manifest) and capture_runtime_has_prior_state(config):
        raise RuntimeError("capture manifest is missing for an existing runtime")
    manifest_store.ensure_exists()
    manifest_store.recover_incomplete_tail()
    latest_manifest = manifest_store.latest_by_event_key()
    pending_entries = [
        entry
        for entry in latest_manifest.values()
        if entry.status in {"skipped_no_recording", "failed"}
    ]
    expired_retry_interval = timedelta(hours=24)
    due_expired_entries = [
        entry
        for entry in latest_manifest.values()
        if entry.status == "recording_retry_expired"
        and until.astimezone(timezone.utc)
        - parse_datetime(entry.created_at).astimezone(timezone.utc)
        >= expired_retry_interval
    ]
    due_expired_unknown = [
        entry for entry in due_expired_entries if not entry_recording_ids(entry)
    ]
    due_expired_unknown_by_key = {
        entry.event_key: entry for entry in due_expired_unknown
    }
    pending_keys = {
        entry.event_key for entry in (*pending_entries, *due_expired_unknown)
    }
    threshold = until - timedelta(hours=max(1, config.pending_recording_retry_hours))
    recent_entries = [entry for entry in latest_manifest.values() if parse_datetime(entry.started_at) >= threshold]
    expired_keys = {
        entry.event_key
        for entry in pending_entries
        if parse_datetime(entry.started_at) < threshold
    } | {entry.event_key for entry in due_expired_unknown}
    overlap = timedelta(minutes=config.poll_overlap_minutes)
    # The permanent service always proves a rolling TTL window.  Direct legacy
    # callers retain their explicit window; service JSON enables strict mode.
    rolling_day_start = (
        threshold.astimezone(ZoneInfo("Europe/Moscow"))
        .replace(hour=0, minute=0, second=0, microsecond=0)
        .astimezone(timezone.utc)
    )
    base_window_start = (
        min(since, rolling_day_start)
        if config.strict_ready_provenance
        else since
    )
    poll_windows = [(base_window_start, until)]
    for entry in {
        entry.event_key: entry for entry in (*pending_entries, *recent_entries)
    }.values():
        started = parse_datetime(entry.started_at)
        ended = parse_datetime(entry.ended_at) if entry.ended_at else started + timedelta(hours=1)
        poll_windows.append((started - overlap, ended + overlap))
    for entry in due_expired_unknown:
        poll_windows.append(
            moscow_day_bounds_utc(
                parse_datetime(entry.started_at)
                .astimezone(ZoneInfo("Europe/Moscow"))
                .date()
            )
        )
    merged_windows: list[tuple[datetime, datetime]] = []
    for start, end in sorted(poll_windows):
        if merged_windows and start <= merged_windows[-1][1]:
            merged_windows[-1] = (merged_windows[-1][0], max(end, merged_windows[-1][1]))
        else:
            merged_windows.append((start, end))
    api_requests = 0
    covered_intervals: list[Mapping[str, Any]] = []
    for window_start, window_end in merged_windows:
        chunk_start = window_start
        while chunk_start < window_end:
            chunk_end = min(window_end, chunk_start + timedelta(hours=config.api_window_hours))
            chunk_rows = client.poll_call_history(
                since=chunk_start, until=chunk_end
            )
            rows.extend(chunk_rows)
            api_requests += 1
            covered_intervals.append(
                {
                    "since": chunk_start.isoformat(),
                    "until": chunk_end.isoformat(),
                    "result_complete": True,
                    "rows": len(chunk_rows),
                }
            )
            chunk_start = chunk_end
    unique_events: dict[str, Any] = {}
    for row in rows:
        event = mapper.from_payload(tenant=tenant, payload=row)
        prior = unique_events.get(event.event_key)
        if prior is not None:
            refs = merge_recording_ids(event_recording_ids(prior), event_recording_ids(event))
            event = replace(event, recording_ref=refs[0] if refs else None, recording_refs=refs)
        unique_events[event.event_key] = event
    if config.strict_ready_provenance:
        # After cutover only the transferred SQLite generations are authority;
        # neighbouring archives must never silently suppress a new API event.
        known_recordings: set[str] = set()
        known_calls = (
            read_ingested_call_ids(config.working_db)
            | read_ingested_call_ids(config.ready_db)
        )
        fully_ready_calls = read_fully_ready_call_ids(config)
    else:
        known_recordings, known_calls = read_known_processed_ids(
            config.pipeline_root.parent
        )
        fully_ready_calls = set()
    recovery_events = [
        event
        for event in missing_capture_recovery_events(config, now=until)
        if not (
            config.strict_ready_provenance
            and event.provider_call_id in fully_ready_calls
        )
    ]
    recovery_keys = {event.event_key for event in recovery_events} | pending_keys
    for event in recovery_events:
        unique_events.setdefault(event.event_key, event)
    mapped_events = sorted(
        unique_events.values(),
        key=lambda event: (event.started_at, event.provider_call_id),
    )
    external_known_keys = {
        event.event_key
        for event in mapped_events
        if len(event_recording_ids(event)) < 2
        and event.event_key not in recovery_keys
        and (
            (
                config.strict_ready_provenance
                and event.provider_call_id in known_calls
            )
            or (
                (event.recording_ref and event.recording_ref in known_recordings)
                or event.provider_call_id in known_calls
            )
        )
    }
    events = [
        event
        for event in mapped_events
        if event.event_key not in external_known_keys
    ]
    summary = stage_capture_events(
        events=events,
        manifest_store=manifest_store,
        recordings_dir=config.recordings_dir,
        downloader=downloader,
        dry_run=False,
        sleep_sec=1.5,
        host_id=host_id,
        require_integrity_metadata=config.strict_ready_provenance,
    )
    latest = manifest_store.latest_by_event_key()
    pending_expired = 0
    expired_reenumerated = 0
    for event_key in expired_keys:
        entry = latest.get(event_key)
        original_due = due_expired_unknown_by_key.get(event_key)
        should_refresh_due = bool(
            entry is not None
            and original_due is not None
            and entry.status == "recording_retry_expired"
            and not entry_recording_ids(entry)
            and entry.created_at == original_due.created_at
        )
        should_expire_pending = bool(
            entry is not None
            and original_due is None
            and entry.status in {"skipped_no_recording", "failed"}
        )
        if entry is not None and (should_refresh_due or should_expire_pending):
            manifest_store.append(
                replace(
                    entry,
                    created_at=until.isoformat(),
                    status="recording_retry_expired",
                    error="recording_missing_after_retry_ttl",
                    remediation_code="manual_review_or_retry_if_recording_appears",
                    host_id=host_id,
                    recovery_state=(
                        "late_recording_reenumerated_still_missing"
                        if should_refresh_due
                        else entry.recovery_state
                    ),
                )
            )
            if should_refresh_due:
                expired_reenumerated += 1
            else:
                pending_expired += 1
    final_latest = manifest_store.latest_by_event_key()
    remaining_pending = {
        key
        for key, entry in final_latest.items()
        if entry.status in {"skipped_no_recording", "failed"}
    }
    open_multi_review = sum(entry.status == "multiple_recordings_needs_review" for entry in final_latest.values())
    open_integrity_quarantine = sum(
        entry.status == "audio_integrity_quarantined"
        for entry in final_latest.values()
    )
    incomplete_tail = manifest_store.incomplete_trailing_records
    recovered_tail = manifest_store.recovered_trailing_records
    recovery_incident_sha256 = manifest_store.recovery_incident_sha256
    # Pending/failed recordings and reasoned quarantine are queue state, not an
    # enumeration failure.  Only manifest lineage damage makes capture itself
    # partial; SLA/closure are evaluated separately by Stage 10/watchdog.
    status = (
        "ok"
        if summary.failed == 0 and incomplete_tail == 0 and recovered_tail == 0
        else "partial"
    )
    calls_by_moscow_day: dict[str, list[str]] = {}
    for event in mapped_events:
        day_key = event.started_at.astimezone(
            ZoneInfo("Europe/Moscow")
        ).date().isoformat()
        calls_by_moscow_day.setdefault(day_key, []).append(event.provider_call_id)
    for values in calls_by_moscow_day.values():
        values[:] = sorted(set(values))
    previous_cursor = read_json(config.cursor_path)
    previous_zero = previous_cursor.get("independent_zero_enumerations_by_day")
    if not isinstance(previous_zero, Mapping):
        previous_zero = {}
    zero_proofs: dict[str, int] = {
        key: 0 for key in calls_by_moscow_day
    }
    enumeration_start = min(start for start, _end in merged_windows)
    covered_days: set[date] = set()
    for start, end in merged_windows:
        day_cursor = start.astimezone(ZoneInfo("Europe/Moscow")).date()
        last_day = (end - timedelta(microseconds=1)).astimezone(
            ZoneInfo("Europe/Moscow")
        ).date()
        while day_cursor <= last_day:
            day_start, day_end = moscow_day_bounds_utc(day_cursor)
            if start <= day_start and end >= day_end:
                covered_days.add(day_cursor)
            day_cursor += timedelta(days=1)
    for covered_day in sorted(covered_days):
        key = covered_day.isoformat()
        zero_proofs[key] = (
            0
            if calls_by_moscow_day.get(key)
            else positive_int(previous_zero.get(key)) + 1
        )
    manifest_end_offset = config.capture_manifest.stat().st_size
    sealed_capture = capture_manifest_snapshot(
        config.capture_manifest, end_offset=manifest_end_offset
    )
    catch_up = (until - since) > timedelta(
        minutes=max(60, config.poll_overlap_minutes * 2)
    )
    return {
        "status": status,
        "host_id": host_id,
        "catch_up": catch_up,
        "sla_mode": "catch_up" if catch_up else "live",
        "mango_enumeration_complete": True,
        "mango_enumeration_source": {
            "mode": (
                "strict_service"
                if config.strict_ready_provenance
                else "compatibility_not_for_service"
            ),
            "since": enumeration_start.isoformat(),
            "rolling_since": base_window_start.isoformat(),
            "until": until.isoformat(),
            "cursor": "not_applicable_stats_request_result",
            "pages": None,
            "pagination": "not_applicable_stats_request_result",
            "requests": api_requests,
            "covered_intervals": covered_intervals,
            "catch_up": catch_up,
        },
        "call_keys": sorted(
            {event.provider_call_id for event in mapped_events}
        ),
        "calls_by_moscow_day": calls_by_moscow_day,
        "independent_zero_enumerations_by_day": zero_proofs,
        "manifest_end_offset": manifest_end_offset,
        "manifest_snapshot_sha256": sealed_capture["sha256"],
        "api_requests": api_requests,
        "api_rows_total": len(rows),
        "api_events_total": len(mapped_events),
        "api_events_already_known_external": len(external_known_keys),
        "pending_recording_retries": len(remaining_pending),
        "pending_recording_expired": pending_expired,
        "expired_recording_reenumerated_still_missing": expired_reenumerated,
        "open_multiple_recordings_needs_review": open_multi_review,
        "open_audio_integrity_quarantined": open_integrity_quarantine,
        "api_events_without_recording": sum(
            1 for event in mapped_events if not (event.recording_ref or event.recording_url)
        ),
        **summary.to_json_dict(),
        "incomplete_trailing_manifest_records": incomplete_tail,
        "recovered_trailing_manifest_records": recovered_tail,
        "recovery_incident_sha256": recovery_incident_sha256,
        "capture_assets_complete": bool(
            summary.failed == 0
            and not remaining_pending
            and pending_expired == 0
            and open_multi_review == 0
            and open_integrity_quarantine == 0
        ),
    }


def capture_runtime_has_prior_state(config: CallsTwoProcessesConfig) -> bool:
    if os.path.lexists(config.cursor_path):
        return True
    prior_status = read_regular_json_marker(config.process_a_status_path)
    if prior_status is None:
        return True
    if prior_status.get("checked_through") or prior_status.get("stop_reason") in {
        "capture_audio_incomplete",
        "capture_manifest_tail_incomplete",
    }:
        return True
    if not os.path.lexists(config.recordings_dir):
        return False
    if not config.recordings_dir.is_dir() or config.recordings_dir.is_symlink():
        return True
    try:
        return next(config.recordings_dir.iterdir(), None) is not None
    except OSError:
        return True


def read_regular_json_marker(path: Path) -> Optional[Mapping[str, Any]]:
    def stable_identity(descriptor: int) -> Optional[tuple[int, int, int, int, int]]:
        descriptor_stat = os.fstat(descriptor)
        path_stat = os.stat(path, follow_symlinks=False)
        if (
            not stat.S_ISREG(descriptor_stat.st_mode)
            or descriptor_stat.st_nlink < 1
            or not stat.S_ISREG(path_stat.st_mode)
            or path_stat.st_dev != descriptor_stat.st_dev
            or path_stat.st_ino != descriptor_stat.st_ino
        ):
            return None
        return (
            descriptor_stat.st_dev,
            descriptor_stat.st_ino,
            descriptor_stat.st_size,
            descriptor_stat.st_mtime_ns,
            descriptor_stat.st_ctime_ns,
        )

    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        return None if os.path.lexists(path) else {}
    except OSError:
        return None
    try:
        with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
            descriptor = -1
            before = stable_identity(handle.fileno())
            if before is None:
                return None
            text = handle.read()
            after = stable_identity(handle.fileno())
            if after != before:
                return None
    except (OSError, UnicodeDecodeError):
        return None
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if not text.strip():
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, Mapping) else None


def missing_capture_recovery_events(
    config: CallsTwoProcessesConfig,
    *,
    now: Optional[datetime] = None,
    expired_retry_interval: timedelta = timedelta(hours=24),
) -> tuple[TelephonyCallEvent, ...]:
    store = CaptureManifestStore(config.capture_manifest)
    recovered: list[TelephonyCallEvent] = []
    for entry in store.latest_by_event_key().values() if config.capture_manifest.exists() else ():
        if entry.status not in {
            "downloaded",
            "failed",
            "recording_retry_expired",
            "multiple_recordings_needs_review",
        } or not entry_recording_ids(entry):
            continue
        if entry.status == "recording_retry_expired":
            if now is None:
                continue
            attempted_at = parse_datetime(entry.created_at)
            if now.astimezone(timezone.utc) - attempted_at.astimezone(
                timezone.utc
            ) < expired_retry_interval:
                continue
        if entry.status not in {"failed", "recording_retry_expired"} and manifest_assets_exist(
            entry,
            config.recordings_dir,
            require_integrity_metadata=config.strict_ready_provenance,
        ):
            continue
        recovered.append(capture_event_from_manifest(entry))
    return tuple(recovered)


def capture_event_from_manifest(entry: ManifestEntry) -> TelephonyCallEvent:
    try:
        direction = Direction(entry.direction)
    except ValueError:
        direction = Direction.UNKNOWN
    return TelephonyCallEvent(
        tenant=TenantRef(entry.tenant_id),
        provider=entry.provider,
        provider_call_id=entry.provider_call_id,
        started_at=parse_datetime(entry.started_at),
        ended_at=parse_datetime(entry.ended_at) if entry.ended_at else None,
        direction=direction,
        client_phone=entry.client_phone,
        manager_ref=entry.manager_ref,
        recording_ref=entry.recording_id,
        recording_refs=entry.recording_ids,
        raw_payload={},
    )


def capture_manifest_snapshot(
    path: Path, *, end_offset: Optional[int] = None
) -> Mapping[str, Any]:
    if not path.is_file() or path.is_symlink():
        if end_offset is not None:
            raise RuntimeError("capture manifest snapshot source is missing")
        return {
            "entries": (),
            "end_offset": 0,
            "sha256": hashlib.sha256(b"").hexdigest(),
            "source_size_bytes": 0,
        }
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError("capture manifest snapshot source is not regular")
        limit = before.st_size if end_offset is None else int(end_offset)
        if limit < 0 or limit > before.st_size:
            raise RuntimeError("capture manifest snapshot offset is invalid")
        raw = handle.read(limit)
        after = os.fstat(handle.fileno())
        if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
            raise RuntimeError("capture manifest changed identity during snapshot")
    if len(raw) != limit:
        raise RuntimeError("capture manifest snapshot is incomplete")
    incomplete_tail = 0
    entries = []
    valid_size = 0
    cursor = 0
    for line in raw.splitlines(keepends=True):
        cursor += len(line)
        if not line.strip():
            valid_size = cursor
            continue
        try:
            entries.append(entry_from_json(json.loads(line.decode("utf-8"))))
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
            if end_offset is not None or line.endswith((b"\n", b"\r")):
                raise RuntimeError("capture manifest snapshot contains invalid JSONL")
            incomplete_tail = 1
            break
        valid_size = cursor
    if incomplete_tail:
        raw = raw[:valid_size]
        limit = valid_size
    return {
        "entries": tuple(entries),
        "end_offset": limit,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "source_size_bytes": before.st_size,
        "incomplete_trailing_records": incomplete_tail,
    }


def prepare_ingest_inputs(
    config: CallsTwoProcessesConfig,
    *,
    manifest_end_offset: Optional[int] = None,
    expected_manifest_sha256: Optional[str] = None,
) -> Mapping[str, Any]:
    config.working_audio_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    actions: dict[str, int] = {}
    skipped: dict[str, int] = {}
    stable_before = datetime.now(timezone.utc) - timedelta(
        minutes=max(0, config.recording_set_stabilization_minutes)
    )
    fully_ready_call_ids = read_fully_ready_call_ids(config)
    working_call_ids = read_ingested_call_ids(config.working_db)
    manifest_store = CaptureManifestStore(config.capture_manifest)
    # Load the recovery ledger as well as the valid append-only prefix.  This is
    # read-only: an incomplete tail is reported and never silently accepted.
    manifest_store.read_entries()
    snapshot = capture_manifest_snapshot(
        config.capture_manifest, end_offset=manifest_end_offset
    )
    if expected_manifest_sha256 and snapshot["sha256"] != expected_manifest_sha256:
        raise RuntimeError("capture manifest frozen prefix digest mismatch")
    latest: dict[str, ManifestEntry] = {}
    for entry in snapshot["entries"]:
        latest[entry.event_key] = entry
    for entry in sorted(latest.values(), key=lambda item: (item.started_at, item.event_key)):
        if entry.status != "downloaded" or not entry.local_audio_path:
            continue
        if parse_datetime(entry.started_at) > stable_before:
            skipped["recording_set_stabilizing"] = skipped.get("recording_set_stabilizing", 0) + 1
            continue
        source = Path(entry.local_audio_path)
        if entry.provider_call_id in fully_ready_call_ids:
            # The transferred DB is the durable completion authority.  Ready
            # historical audio is intentionally optional after cutover and
            # must not be copied or downloaded again.
            skipped["already_ingested"] = skipped.get("already_ingested", 0) + 1
            continue
        # A manifest row promised audio that is gone or empty: count it, so a
        # lost recording never reads as a clean run.
        if not manifest_assets_exist(
            entry,
            config.recordings_dir,
            require_integrity_metadata=config.strict_ready_provenance,
        ):
            source = Path(entry.local_audio_path)
            if source.is_file() and not source.is_symlink() and source.stat().st_size <= 0:
                reason = "audio_file_empty"
            elif source.is_file() and not source.is_symlink():
                reason = "audio_file_integrity_mismatch"
            else:
                reason = "audio_file_missing"
            skipped[reason] = skipped.get(reason, 0) + 1
            continue
        target = config.working_audio_dir / source.name
        action = hardlink_or_copy(source, target)
        actions[action] = actions.get(action, 0) + 1
        if entry.provider_call_id in working_call_ids:
            skipped["already_in_working"] = skipped.get("already_in_working", 0) + 1
            continue
        rows.append(
            {
                "filename": target.name,
                "call_id": entry.provider_call_id,
                "phone": entry.client_phone or "",
                "manager": entry.manager_ref or "",
                "started_at": entry.started_at,
                "direction": entry.direction,
                "source_event_key": entry.event_key,
            }
        )
    config.metadata_csv.parent.mkdir(parents=True, exist_ok=True)
    with config.metadata_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("filename", "call_id", "phone", "manager", "started_at", "direction", "source_event_key"),
        )
        writer.writeheader()
        writer.writerows(rows)
    return {
        "audio_files": len(rows),
        "link_actions": actions,
        "metadata_rows": len(rows),
        "skipped": skipped,
        "skipped_total": sum(skipped.values()),
        "asset_integrity_failures": sum(
            skipped.get(reason, 0)
            for reason in (
                "audio_file_missing",
                "audio_file_empty",
                "audio_file_integrity_mismatch",
            )
        ),
        "pending_stabilizing": skipped.get("recording_set_stabilizing", 0),
        "incomplete_total": (
            sum(
                skipped.get(reason, 0)
                for reason in (
                    "audio_file_missing",
                    "audio_file_empty",
                    "audio_file_integrity_mismatch",
                )
            )
            + int(snapshot.get("incomplete_trailing_records") or 0)
            + int(manifest_store.recovered_trailing_records or 0)
        ),
        "incomplete_trailing_manifest_records": int(
            snapshot.get("incomplete_trailing_records") or 0
        ),
        "recovered_trailing_manifest_records": int(
            manifest_store.recovered_trailing_records or 0
        ),
        "recovery_incident_sha256": manifest_store.recovery_incident_sha256,
        "manifest_snapshot_end_offset": snapshot["end_offset"],
        "manifest_snapshot_sha256": snapshot["sha256"],
        "manifest_source_size_bytes": snapshot["source_size_bytes"],
    }


def read_ingested_call_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30) as con:
            return {str(row[0]) for row in con.execute("SELECT source_call_id FROM call_records WHERE source_call_id IS NOT NULL")}
    except sqlite3.OperationalError as exc:
        if "no such table" in str(exc).casefold():
            return set()
        raise


def read_fully_ready_call_ids(config: CallsTwoProcessesConfig) -> set[str]:
    """Return calls durably complete in both working and sealed generations."""

    def complete(path: Path) -> set[str]:
        rows = load_ready_rows(path)
        counts: dict[str, int] = {}
        for row in rows:
            call_id = str(row.get("source_call_id") or "").strip()
            if call_id:
                counts[call_id] = counts.get(call_id, 0) + 1
        return {
            call_id
            for row in rows
            if (call_id := str(row.get("source_call_id") or "").strip())
            and counts[call_id] == 1
            and ready_row_is_complete(row)
        }

    if config.strict_ready_provenance:
        try:
            manifest = read_json(config.ready_manifest)
            if (
                validate_ready_manifest_payload(
                    manifest,
                    expected_code_sha=config.expected_code_sha,
                    expected_host_id=configured_host_id(config, required=True),
                )
                or manifest.get("sha256") != sha256_file(config.ready_db)
                or int(manifest.get("size_bytes") or -1)
                != config.ready_db.stat().st_size
            ):
                return set()
        except (OSError, TypeError, ValueError, RuntimeError):
            return set()

    return complete(config.working_db) & complete(config.ready_db)


def call_db_has_open_work(path: Path) -> bool:
    if not path.is_file():
        return False
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30) as con:
        con.row_factory = sqlite3.Row
        tables = {str(row[0]) for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "call_records" not in tables:
            return False
        now = datetime.now(timezone.utc)
        limits = {
            "transcribe": max(1, int(os.getenv("TRANSCRIBE_MAX_ATTEMPTS", "3"))),
            "resolve": max(1, int(os.getenv("RESOLVE_MAX_ATTEMPTS", "2"))),
            "analyze": max(1, int(os.getenv("ANALYZE_MAX_ATTEMPTS", "3"))),
        }

        def due(value: Any) -> bool:
            return not value or parse_datetime(str(value)) <= now

        def stale(value: Any, env_name: str) -> bool:
            seconds = max(60, int(os.getenv(env_name, "1800")))
            return not value or parse_datetime(str(value)) <= now - timedelta(seconds=seconds)

        for raw in con.execute("SELECT * FROM call_records WHERE dead_letter_stage IS NULL"):
            row = dict(raw)
            stage = row.get("pipeline_stage")
            if stage:
                if stale(row.get("pipeline_claimed_at"), "PIPELINE_LEASE_TIMEOUT_SEC"):
                    return True
                continue
            if row.get("analysis_status") == "in_progress":
                if stale(row.get("analysis_claimed_at"), "ANALYZE_LEASE_TIMEOUT_SEC"):
                    return True
                continue
            retry_due = due(row.get("next_retry_at"))
            transcription = row.get("transcription_status")
            if transcription in {"pending", "failed"} and retry_due and int(row.get("transcribe_attempts") or 0) < limits["transcribe"]:
                return True
            state = "not_needed"
            if transcription == "done" and row.get("transcript_variants_json"):
                try:
                    payload = json.loads(str(row["transcript_variants_json"]))
                except json.JSONDecodeError:
                    payload = {}
                if isinstance(payload, dict):
                    state = TranscribeService.secondary_backfill_state_from_payload(payload, secondary_provider="gigaam")
            if state in {"fresh", "retry"}:
                return True
            resolve = row.get("resolve_status")
            if transcription == "done" and resolve in {"pending", "failed"} and retry_due and int(row.get("resolve_attempts") or 0) < limits["resolve"]:
                return True
            if transcription == "done" and resolve in {None, "done", "skipped"} and row.get("analysis_status") in {"pending", "failed"} and retry_due and int(row.get("analyze_attempts") or 0) < limits["analyze"]:
                return True
        return False


def read_known_processed_ids(product_data_root: Path) -> tuple[set[str], set[str]]:
    recording_ids: set[str] = set()
    call_ids: set[str] = set()
    for manifest in sorted(product_data_root.glob("mango_update_after_*/recording_download_manifest.jsonl")):
        for line in manifest.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line.strip():
                continue
            row = parse_json_object(line)
            action = str(row.get("action") or row.get("status") or "").casefold()
            if action not in {
                "downloaded_recording",
                "skip_already_downloaded",
                "downloaded",
                "skipped_exists",
            }:
                continue
            recording_id = optional_text(row.get("recording_id") or row.get("recording_ref"))
            call_id = optional_text(row.get("provider_call_id") or row.get("call_id"))
            if recording_id:
                recording_ids.add(recording_id)
            if call_id:
                call_ids.add(call_id)
    for metadata in sorted(product_data_root.glob("mango_update_after_*/asr_ui_batch/metadata.csv")):
        try:
            with metadata.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    call_id = optional_text(
                        row.get("call_id") or row.get("provider_call_id") or row.get("source_call_id")
                    )
                    if call_id:
                        call_ids.add(call_id)
        except (OSError, csv.Error):
            continue
    return recording_ids, call_ids


def environment_preflight(
    config: CallsTwoProcessesConfig,
    *,
    run_commands: bool,
    require_mango_credentials: bool = True,
) -> Mapping[str, Any]:
    credentials_present = bool(
        os.getenv("MANGO_OFFICE_API_KEY", "").strip()
        and os.getenv("MANGO_OFFICE_API_SALT", "").strip()
    )
    python_ok = config.python_executable.is_file() and os.access(config.python_executable, os.X_OK)
    codex_ok = config.codex_binary.is_file() and os.access(config.codex_binary, os.X_OK)
    auth_ok = False
    modules_ok = False
    runtime_observation = observe_runtime_fingerprint(config)
    network_ok = codex_network_available() if run_commands else True
    if run_commands and python_ok:
        try:
            probe = subprocess.run(
                module_probe_command(config),
                cwd=Path(__file__).resolve().parents[3],
                env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[3] / "src")},
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=30,
            )
            modules_ok = probe.returncode == 0
        except subprocess.TimeoutExpired:
            modules_ok = False
    if run_commands and codex_ok:
        try:
            codex_home = prepare_codex_home(
                config.codex_home_root / "worker",
                strict=config.strict_ready_provenance,
            )
            auth = subprocess.run(
                [str(config.codex_binary), "login", "status"],
                env={
                    **os.environ,
                    "HOME": str(Path.home()),
                    "CODEX_HOME": str(codex_home),
                    "PATH": command_path(config),
                },
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=30,
            )
            auth_ok = auth.returncode == 0
        except subprocess.TimeoutExpired:
            auth_ok = False
    elif not run_commands:
        modules_ok = True
        auth_ok = True
    mango_ok = credentials_present or not require_mango_credentials
    checks = {
        "mango_credentials": mango_ok,
        "python_executable": python_ok,
        "asr_modules": modules_ok,
        "codex_binary": codex_ok,
        "codex_auth": auth_ok,
        "codex_network": network_ok,
        "runtime_fingerprint": runtime_observation.get("ok") is True,
    }
    required: tuple[str, ...] = (
        "mango_credentials",
        "python_executable",
        "asr_modules",
        "codex_binary",
        "codex_auth",
    )
    if config.strict_ready_provenance:
        required = (*required, "runtime_fingerprint")
    failed_checks = [name for name in required if run_commands and not checks[name]]
    return {
        "ok": not failed_checks,
        "checks": checks,
        "failed_checks": failed_checks,
        "degraded_checks": ["codex_network"] if run_commands and not network_ok else [],
        "credentials_present": credentials_present,
        "mango_credentials_required": require_mango_credentials,
        "python_ok": python_ok,
        "asr_modules_ok": modules_ok,
        "codex_binary_ok": codex_ok,
        "codex_authenticated": auth_ok,
        "codex_network_ok": network_ok,
        "asr_mode": config.asr_mode,
        "sequential_stages": list(
            pipeline_stages(config, include_llm=network_ok)
        ),
        "runtime_fingerprint": runtime_observation.get("fingerprint"),
        "runtime_fingerprint_errors": runtime_observation.get("errors"),
    }


def observe_runtime_fingerprint(config: CallsTwoProcessesConfig) -> Mapping[str, Any]:
    fingerprint = json.loads(
        json.dumps(approved_runtime_fingerprint(), ensure_ascii=False)
    )
    errors: list[str] = []
    versions: Mapping[str, Any] = {}
    if config.python_executable.is_file() and os.access(config.python_executable, os.X_OK):
        code = (
            "import importlib.metadata,json;"
            "print(json.dumps({n:importlib.metadata.version(n) for n in "
            "('mlx-whisper','gigaam')}))"
        )
        try:
            result = subprocess.run(
                [str(config.python_executable), "-c", code],
                text=True,
                capture_output=True,
                check=False,
                timeout=30,
            )
            if result.returncode == 0:
                parsed = json.loads(result.stdout)
                versions = parsed if isinstance(parsed, Mapping) else {}
            else:
                errors.append("asr_package_version_probe_failed")
        except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
            errors.append("asr_package_version_probe_failed")
    else:
        errors.append("python_executable_unavailable")
    fingerprint["whisper"]["library_version"] = str(
        versions.get("mlx-whisper") or ""
    )
    fingerprint["gigaam"]["library_version"] = str(versions.get("gigaam") or "")

    snapshot = config.mlx_whisper_snapshot_path
    if snapshot is None:
        fingerprint["whisper"]["weights_revision"] = ""
        errors.append("mlx_whisper_snapshot_path_missing")
    else:
        resolved = snapshot.expanduser().resolve(strict=False)
        if (
            not snapshot.exists()
            or not snapshot.is_dir()
            or resolved.name
            != str(approved_runtime_fingerprint()["whisper"]["weights_revision"])
        ):
            fingerprint["whisper"]["weights_revision"] = ""
            errors.append("mlx_whisper_snapshot_revision_unproven")
        else:
            fingerprint["whisper"]["weights_revision"] = resolved.name

    codex_version = ""
    if config.codex_binary.is_file() and os.access(config.codex_binary, os.X_OK):
        try:
            result = subprocess.run(
                [str(config.codex_binary), "--version"],
                text=True,
                capture_output=True,
                check=False,
                timeout=30,
            )
            match = re.search(r"\b(\d+\.\d+\.\d+)\b", result.stdout + result.stderr)
            codex_version = match.group(1) if result.returncode == 0 and match else ""
        except (OSError, subprocess.TimeoutExpired):
            pass
    for stage in ("resolve", "analyze"):
        fingerprint[stage]["codex_cli_version"] = codex_version
    fingerprint["resolve"]["model"] = config.codex_resolve_model
    fingerprint["analyze"]["model"] = config.codex_analyze_model
    fingerprint["resolve"]["reasoning"] = config.codex_reasoning_effort
    fingerprint["analyze"]["reasoning"] = config.codex_reasoning_effort
    errors.extend(validate_runtime_fingerprint(fingerprint))
    return {"ok": not errors, "errors": sorted(set(errors)), "fingerprint": fingerprint}


def module_probe_command(config: CallsTwoProcessesConfig) -> list[str]:
    modules = repr(REQUIRED_PIPELINE_MODULES)
    code = (
        "import importlib.util,sys; "
        f"mods={modules}; "
        "sys.exit(0 if all(importlib.util.find_spec(name) is not None for name in mods) else 1)"
    )
    return [str(config.python_executable), "-c", code]


def command_path(config: CallsTwoProcessesConfig) -> str:
    parts = [str(config.codex_binary.parent), os.environ.get("PATH", "")]
    return os.pathsep.join(part for part in parts if part)


def codex_network_available() -> bool:
    try:
        socket.getaddrinfo("chatgpt.com", 443, type=socket.SOCK_STREAM)
    except OSError:
        return False
    return True


def disk_preflight(config: CallsTwoProcessesConfig) -> Mapping[str, Any]:
    config.pipeline_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    root_info = os.lstat(config.pipeline_root)
    if (
        not stat.S_ISDIR(root_info.st_mode)
        or config.pipeline_root.is_symlink()
        or root_info.st_uid != os.getuid()
    ):
        raise RuntimeError("pipeline root is unsafe")
    config.pipeline_root.chmod(0o700)
    usage = shutil.disk_usage(config.pipeline_root)
    required = int(config.min_free_gib * 1024**3)
    return {"free_bytes": usage.free, "required_free_bytes": required, "ok": usage.free >= required}


def resolve_capture_window(
    config: CallsTwoProcessesConfig,
    *,
    since: Optional[str],
    until: Optional[str],
) -> tuple[datetime, datetime]:
    end = parse_datetime(until) if until else datetime.now(timezone.utc)
    if since:
        start = parse_datetime(since)
    else:
        cursor = read_json(config.cursor_path)
        raw = optional_text(cursor.get("until")) or config.bootstrap_since
        if raw:
            start = parse_datetime(raw) - timedelta(minutes=config.poll_overlap_minutes)
            if end - start > timedelta(days=config.max_catch_up_days):
                raise ValueError(
                    "capture gap exceeds max_catch_up_days; provide explicit --since after dry-run review"
                )
        else:
            start = end - timedelta(hours=config.first_lookback_hours)
    if end <= start:
        raise ValueError("capture until must be after since")
    return start, end


def worker_environment(config: CallsTwoProcessesConfig) -> Mapping[str, str]:
    project_root = Path(__file__).resolve().parents[3]
    isolated_codex = project_root / "scripts" / "run_codex_cli_isolated.sh"
    config.transcripts_dir.mkdir(parents=True, exist_ok=True)
    codex_home = prepare_codex_home(
        config.codex_home_root / "worker",
        strict=config.strict_ready_provenance,
    )
    return {
        **os.environ,
        "PATH": command_path(config),
        "DATABASE_URL": f"sqlite:///{config.working_db}",
        "TRANSCRIPT_EXPORT_DIR": str(config.transcripts_dir),
        "CODEX_HOME": str(codex_home),
        "PYTHONPATH": str(project_root / "src"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "SQLITE_BUSY_TIMEOUT_MS": "60000",
        "CODEX_CLI_COMMAND": str(isolated_codex),
        "MANGO_CODEX_REAL_BIN": str(config.codex_binary),
        "CODEX_CLI_TIMEOUT_SEC": "360",
        "CODEX_REASONING_EFFORT": config.codex_reasoning_effort,
        "CODEX_RESOLVE_MODEL": config.codex_resolve_model,
        "CODEX_ANALYZE_MODEL": config.codex_analyze_model,
        "RESOLVE_LLM_PROVIDER": "codex_cli",
        "RESOLVE_DIALOGUE_MODE": "dialogue",
        "RESOLVE_RESCUE_PROVIDER": "none",
        "RESOLVE_RESCUE_DUAL_ENABLED": "0",
        "ANALYZE_PROVIDER": "codex_cli",
        "MANGO_STRICT_ASR_RUNTIME": "1" if config.strict_ready_provenance else "0",
    }


def transcribe_environment(config: CallsTwoProcessesConfig, base: Mapping[str, str]) -> Mapping[str, str]:
    return {
        **base,
        "TRANSCRIBE_PROVIDER": "mlx",
        "DUAL_TRANSCRIBE_ENABLED": "1",
        "SECONDARY_TRANSCRIBE_PROVIDER": "gigaam",
        "DUAL_MERGE_PROVIDER": "rule",
        "MONO_ROLE_ASSIGNMENT_MODE": "rule",
        "TRANSCRIBE_LANGUAGE": "ru",
        "SPLIT_STEREO_CHANNELS": "1",
        "MLX_WHISPER_MODEL": str(
            config.mlx_whisper_snapshot_path
            or "mlx-community/whisper-large-v3-mlx"
        ),
        "MLX_CONDITION_ON_PREVIOUS_TEXT": "0",
        "MLX_WORD_TIMESTAMPS": "1",
        "GIGAAM_MODEL": "v2_rnnt",
        "GIGAAM_DEVICE": "cpu",
    }


def prepare_codex_home(target: Path, *, strict: bool = False) -> Path:
    source = Path.home() / ".codex"
    resolved_target = target.expanduser().resolve(strict=False)
    cloud_markers = {"yandex.disk", "icloud drive", "mobile documents", "dropbox", "onedrive"}
    if strict:
        if any(
            marker in part.casefold()
            for part in resolved_target.parts
            for marker in cloud_markers
        ):
            raise RuntimeError("isolated CODEX_HOME must stay outside cloud folders")
        project_root = Path(__file__).resolve().parents[3]
        if resolved_target == project_root or project_root in resolved_target.parents:
            raise RuntimeError("isolated CODEX_HOME must stay outside the Git worktree")
        owner_local = (Path.home() / ".mango_local").resolve(strict=False)
        if owner_local not in resolved_target.parents:
            raise RuntimeError("isolated CODEX_HOME must stay under ~/.mango_local")
    target.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(target, 0o700)
    allowed_existing = {
        "auth.json",
        "installation_id",
        "models_cache.json",
        "config.toml",
        "AGENTS.md",
    }
    unknown = sorted(entry.name for entry in target.iterdir() if entry.name not in allowed_existing)
    if strict and unknown:
        raise RuntimeError("isolated CODEX_HOME contains unknown persistent files")
    for name in ("auth.json", "installation_id", "models_cache.json"):
        src = source / name
        dst = target / name
        if src.is_file():
            shutil.copy2(src, dst)
            os.chmod(dst, 0o600)
    # Isolate batch classification from desktop plugins/MCP servers and account
    # personality. Resolve/Analyze receive all task context in their own prompt.
    for name, content in (
        ("config.toml", "# Isolated Mango batch runtime: no plugins or MCP servers.\n"),
        ("AGENTS.md", "Follow the supplied task prompt exactly and return only its requested result.\n"),
    ):
        path = target / name
        path.write_text(content, encoding="utf-8")
        os.chmod(path, 0o600)
    return target


def primary_transcribe_environment(
    config: CallsTwoProcessesConfig,
    base_env: Mapping[str, str],
) -> Mapping[str, str]:
    return {
        **transcribe_environment(config, base_env),
        "DUAL_TRANSCRIBE_ENABLED": "0",
        "SECONDARY_TRANSCRIBE_PROVIDER": "",
    }


def stage_worker_environment_for(
    config: CallsTwoProcessesConfig,
    base_env: Mapping[str, str],
    stage: str,
) -> Mapping[str, str]:
    if stage == "transcribe":
        return primary_transcribe_environment(config, base_env)
    if stage == "backfill-second-asr":
        return transcribe_environment(config, base_env)
    if stage in {"resolve", "analyze"}:
        return transcribe_environment(config, base_env)
    raise ValueError(f"unsupported parallel pipeline stage: {stage}")


def pipeline_stages(
    config: CallsTwoProcessesConfig,
    *,
    include_llm: bool = True,
) -> tuple[str, ...]:
    stages = SEQUENTIAL_PIPELINE_STAGES
    if include_llm:
        return stages
    return tuple(stage for stage in stages if stage not in {"resolve", "analyze"})


def run_sequential_pipeline_workers(
    config: CallsTwoProcessesConfig,
    base_env: Mapping[str, str],
    runner: CommandRunner,
    *,
    include_llm: bool = True,
    run_id: Optional[str] = None,
    cycle_deadline: Optional[float] = None,
) -> list[Mapping[str, Any]]:
    stages = pipeline_stages(config, include_llm=include_llm)
    if runner is not run_command:
        return [
            runner(
                worker_command(config, stage),
                stage_worker_environment_for(config, base_env, stage),
                config.working_dir,
            )
            for stage in stages
        ]
    logs_dir = config.working_dir / "logs"
    logs_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
    logs_dir.chmod(0o700)
    log_run_id = re.sub(
        r"[^A-Za-z0-9_.-]", "_", run_id or new_calls_run_id(datetime.now(timezone.utc))
    )
    reports: list[Mapping[str, Any]] = []
    heavy_cycle_deadline = cycle_deadline or (
        time.monotonic() + config.heavy_stage_timeout_seconds
    )
    for stage in stages:
        label = stage.replace("-", "_")
        log_path = logs_dir / f"stage_{label}_{log_run_id}.log"
        worker_env = {
            **stage_worker_environment_for(config, base_env, stage),
            "CODEX_HOME": str(
                prepare_codex_home(
                    config.codex_home_root / label,
                    strict=config.strict_ready_provenance,
                )
            ),
        }
        started_at = time.monotonic()
        if started_at >= heavy_cycle_deadline:
            log_path.write_text("heavy_cycle_timeout_before_stage\n", encoding="utf-8")
            log_path.chmod(0o600)
            reports.append(
                {
                    "rc": 124,
                    "command": f"worker:{stage}",
                    "log_path": str(log_path),
                    "log_size_bytes": log_path.stat().st_size,
                    "log_sha256": sha256_file(log_path),
                    "wall_seconds": 0.0,
                    "peak_rss_raw": None,
                    "swap_operations": 0,
                    "timed_out": True,
                    "timeout_scope": "heavy_cycle",
                }
            )
            break
        before_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        deadline = stage_timeout_deadline(
            started_at=started_at,
            timeout_seconds=config.heavy_stage_timeout_seconds,
            cycle_deadline=heavy_cycle_deadline,
        )
        heartbeat_path = config.process_a_heartbeat_path
        proc: subprocess.Popen[str] | None = None
        timed_out = False
        try:
            with log_path.open("x", encoding="utf-8") as log_handle:
                log_path.chmod(0o600)
                command = worker_command(config, stage)
                timed_command = (
                    ["/usr/bin/time", "-l", *command]
                    if Path("/usr/bin/time").is_file()
                    else command
                )
                proc = subprocess.Popen(
                    timed_command,
                    cwd=config.working_dir,
                    env=worker_env,
                    text=True,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                last_heartbeat = 0.0
                while proc.poll() is None:
                    current = time.monotonic()
                    if current - last_heartbeat >= 30:
                        write_json(
                            heartbeat_path,
                            {
                                "schema_version": "mango_calls_heavy_heartbeat_v1",
                                "stage": stage,
                                "pid": proc.pid,
                                "updated_at": datetime.now(timezone.utc).isoformat(),
                            },
                        )
                        last_heartbeat = current
                    if current >= deadline:
                        timed_out = True
                        terminate_process_group(proc)
                        break
                    time.sleep(1)
                rc = 124 if timed_out else int(proc.returncode or 0)
                if timed_out:
                    log_handle.write("stage_timeout\n")
        finally:
            if proc is not None and proc.poll() is None:
                terminate_process_group(proc)
            heartbeat_path.unlink(missing_ok=True)
        after_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        stage_metrics = parse_macos_time_metrics(log_path)
        reports.append(
            {
                "rc": rc,
                "command": f"worker:{stage}",
                "log_path": str(log_path),
                "log_size_bytes": log_path.stat().st_size,
                "log_sha256": sha256_file(log_path),
                "wall_seconds": round(time.monotonic() - started_at, 3),
                "peak_rss_raw": stage_metrics.get("peak_rss_bytes"),
                "swap_operations": stage_metrics.get(
                    "swap_operations",
                    max(0, int(after_usage.ru_nswap - before_usage.ru_nswap)),
                ),
                "timed_out": rc == 124,
            }
        )
        if rc != 0:
            break
    return reports


def stage_timeout_deadline(
    *,
    started_at: float,
    timeout_seconds: int,
    cycle_deadline: Optional[float] = None,
) -> float:
    """Apply both a per-command timeout and the shared heavy-cycle ceiling."""
    command_deadline = started_at + timeout_seconds
    return min(command_deadline, cycle_deadline) if cycle_deadline else command_deadline


def terminate_process_group(
    proc: subprocess.Popen[str],
    *,
    grace_seconds: float = 10.0,
) -> None:
    def group_exists() -> bool:
        try:
            os.killpg(proc.pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    if not group_exists():
        if proc.poll() is None:
            proc.wait(timeout=grace_seconds)
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        pass
    # The session leader may exit while a worker child ignores SIGTERM.  A
    # successful wait for the parent therefore does not prove that the process
    # group is gone.
    if group_exists():
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if proc.poll() is None:
        proc.wait(timeout=grace_seconds)
    deadline = time.monotonic() + grace_seconds
    while group_exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    if group_exists():
        raise RuntimeError("pipeline worker process group survived termination")


def parse_macos_time_metrics(path: Path) -> Mapping[str, int]:
    try:
        text_value = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return {}
    peak = re.findall(r"(?m)^\s*(\d+)\s+maximum resident set size\s*$", text_value)
    swaps = re.findall(r"(?m)^\s*(\d+)\s+swaps\s*$", text_value)
    result: dict[str, int] = {}
    if peak:
        result["peak_rss_bytes"] = int(peak[-1])
    if swaps:
        result["swap_operations"] = int(swaps[-1])
    return result


def run_parallel_pipeline_workers(
    config: CallsTwoProcessesConfig,
    base_env: Mapping[str, str],
    runner: CommandRunner,
    *,
    include_llm: bool = True,
    run_id: Optional[str] = None,
) -> list[Mapping[str, Any]]:
    """Compatibility alias; the implementation is intentionally sequential."""
    return run_sequential_pipeline_workers(
        config, base_env, runner, include_llm=include_llm, run_id=run_id
    )


def worker_command(config: CallsTwoProcessesConfig, stages: str) -> list[str]:
    return cli_command(
        config,
        "worker",
        "--stages",
        stages,
        "--stage-limit",
        str(config.stage_limit),
        "--poll-sec",
        str(config.poll_seconds),
        "--max-idle-cycles",
        str(config.max_idle_cycles),
    )


def cli_command(config: CallsTwoProcessesConfig, *args: str) -> list[str]:
    return [str(config.python_executable), "-m", "mango_mvp.cli", *args]


def run_command(
    command: Sequence[str],
    env: Mapping[str, str],
    cwd: Path,
    *,
    deadline: Optional[float] = None,
) -> Mapping[str, Any]:
    cwd.mkdir(parents=True, exist_ok=True)
    logs_dir = cwd / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    log_path = logs_dir / f"command_{stamp}.log"
    with log_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.Popen(
            list(command),
            cwd=cwd,
            env=dict(env),
            text=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        timed_out = False
        while proc.poll() is None:
            if deadline is not None and time.monotonic() >= deadline:
                timed_out = True
                terminate_process_group(proc)
                handle.write("heavy_cycle_timeout\n")
                break
            time.sleep(0.1)
        return_code = 124 if timed_out else int(proc.returncode or 0)
    report: dict[str, Any] = {
        "rc": return_code,
        "command": compact_command_name(command),
        "log_path": str(log_path),
        "timed_out": timed_out,
        "timeout_scope": "heavy_cycle" if timed_out else None,
    }
    if "ingest" in command:
        payload = parse_json_object(log_path.read_text(encoding="utf-8"))
        if payload:
            report["metrics"] = {
                key: payload.get(key)
                for key in ("processed", "inserted", "skipped", "failed", "failure_types")
                if key in payload
            }
    return report


def run_increment_producer(
    config: CallsTwoProcessesConfig,
    increment_path: Path,
    report_path: Path,
    since: Optional[str],
) -> Mapping[str, Any]:
    script = Path(__file__).resolve().parents[3] / "scripts" / "build_mango_call_timeline_increment.py"
    command = [
        str(config.python_executable),
        str(script),
        "--timeline-db",
        str(config.timeline_db),
        "--package-db",
        str(config.ready_db),
        "--out-jsonl",
        str(increment_path),
        "--report-out",
        str(report_path),
        "--tenant-id",
        config.tenant_id,
    ]
    if since:
        command.extend(["--since", since])
    proc = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[3],
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    report = read_json(report_path)
    return {"status": "ok" if proc.returncode == 0 else "failed", "rc": proc.returncode, **report}


def _ready_verdicts(
    config: CallsTwoProcessesConfig,
    *,
    ready_db: Optional[Path] = None,
    capture_evidence: Mapping[str, Any],
    manifest_end_offset: Optional[int],
) -> tuple[Mapping[str, Mapping[str, Any]], Mapping[str, Any], Mapping[str, Any]]:
    snapshot = capture_manifest_snapshot(
        config.capture_manifest, end_offset=manifest_end_offset
    )
    # Verdict and manifest must describe the sealed generation, never a live
    # working DB that may advance immediately after backup.
    rows = load_ready_rows(ready_db or config.ready_db)
    evidence: dict[str, Any] = dict(capture_evidence)
    calls_by_day = evidence.get("calls_by_moscow_day")
    if not isinstance(calls_by_day, Mapping) and not config.strict_ready_provenance:
        inferred: dict[str, list[str]] = {}
        for entry in snapshot["entries"]:
            call_key = str(entry.provider_call_id or "").strip()
            if not call_key or not entry.started_at:
                continue
            day_key = parse_aware_datetime(entry.started_at).astimezone(
                ZoneInfo("Europe/Moscow")
            ).date().isoformat()
            inferred.setdefault(day_key, []).append(call_key)
        for row in rows:
            call_key = str(row.get("source_call_id") or "").strip()
            raw_started = row.get("started_at")
            if not call_key or not raw_started:
                continue
            day_key = parse_aware_datetime(raw_started).astimezone(
                ZoneInfo("Europe/Moscow")
            ).date().isoformat()
            inferred.setdefault(day_key, []).append(call_key)
        for day_values in inferred.values():
            day_values[:] = sorted(set(day_values))
        evidence.update(
            mango_enumeration_complete=True,
            calls_by_moscow_day=inferred,
            independent_zero_enumerations_by_day={},
            mango_enumeration_source={
                "mode": "compatibility_not_for_service",
                "since": "not_proven",
                "until": "not_proven",
            },
        )
        calls_by_day = inferred
    zero_days = evidence.get("independent_zero_enumerations_by_day")
    day_keys = sorted(
        set(calls_by_day) | set(zero_days)
        if isinstance(calls_by_day, Mapping) and isinstance(zero_days, Mapping)
        else set(calls_by_day) if isinstance(calls_by_day, Mapping) else set()
    )
    verdicts = {
        day_key: build_stage10_verdict(
            day=datetime.fromisoformat(day_key).date(),
            enumeration=evidence,
            capture_entries=snapshot["entries"],
            ready_rows=rows,
        )
        for day_key in day_keys
    }
    return verdicts, snapshot, evidence


def publish_ready_db(
    config: CallsTwoProcessesConfig,
    counts: Mapping[str, Any],
    *,
    run_id: Optional[str] = None,
    capture_evidence: Optional[Mapping[str, Any]] = None,
    manifest_end_offset: Optional[int] = None,
    stage_reports: Sequence[Mapping[str, Any]] = (),
    runtime_fingerprint: Optional[Mapping[str, Any]] = None,
    publication_checkpoint: Optional[Callable[[str], None]] = None,
) -> Mapping[str, Any]:
    config.ready_db.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    config.ready_db.parent.chmod(0o700)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{config.ready_db.stem}.",
        suffix=".sqlite",
        dir=config.ready_db.parent,
    )
    os.fchmod(descriptor, 0o600)
    os.close(descriptor)
    temp = Path(temporary_name)
    source_before = sqlite_storage_signature(config.working_db)
    try:
        with sqlite3.connect(
            f"file:{config.working_db}?mode=ro", uri=True, timeout=60
        ) as source:
            source.execute("PRAGMA query_only=ON")
            with sqlite3.connect(temp) as target:
                source.backup(target)
                target.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                target.execute("PRAGMA journal_mode=DELETE")
                quick = str(target.execute("PRAGMA quick_check").fetchone()[0])
                integrity = str(target.execute("PRAGMA integrity_check").fetchone()[0])
        temp.chmod(0o600)
        source_after = sqlite_storage_signature(config.working_db)
        if source_before != source_after:
            raise RuntimeError("working DB changed while sealing ready generation")
        if quick != "ok" or integrity != "ok":
            raise RuntimeError("ready DB integrity check failed")
        if os.path.lexists(config.ready_db):
            existing = os.lstat(config.ready_db)
            if (
                not stat.S_ISREG(existing.st_mode)
                or config.ready_db.is_symlink()
                or existing.st_uid != os.getuid()
                or existing.st_nlink != 1
            ):
                raise RuntimeError("ready DB target is unsafe")
        sha = sha256_file(temp)
        temp_stat = temp.stat()
        evidence = dict(capture_evidence or read_json(config.cursor_path))
        verdicts, snapshot, evidence = _ready_verdicts(
            config,
            ready_db=temp,
            capture_evidence=evidence,
            manifest_end_offset=manifest_end_offset,
        )
        expected_capture_sha = optional_text(
            evidence.get("manifest_snapshot_sha256")
        )
        if (
            config.strict_ready_provenance
            and expected_capture_sha != snapshot["sha256"]
        ):
            raise RuntimeError("ready generation capture snapshot digest mismatch")
        consistency_ok = bool(verdicts) and all(
            verdict.get("consistency_ok") is True
            for verdict in verdicts.values()
        )
        closure_ok = bool(verdicts) and all(
            verdict.get("closure_ok") is True for verdict in verdicts.values()
        )
        if not config.strict_ready_provenance and not verdicts:
            consistency_ok = closure_ok = True
        project_root = Path(__file__).resolve().parents[3]
        producer_sha = config.expected_code_sha or current_git_sha(project_root)
        host_id = configured_host_id(
            config, required=config.require_cutover_authority
        )
        source = evidence.get("mango_enumeration_source")
        if not isinstance(source, Mapping):
            source = {}
        created = datetime.now(timezone.utc)
        observed_fingerprint = (
            dict(runtime_fingerprint)
            if isinstance(runtime_fingerprint, Mapping)
            else approved_runtime_fingerprint()
        )
        fingerprint_errors = validate_runtime_fingerprint(observed_fingerprint)
        if config.strict_ready_provenance and fingerprint_errors:
            raise RuntimeError(
                "runtime fingerprint is not proven: "
                + ",".join(fingerprint_errors)
            )
        manifest = {
            "schema_version": READY_MANIFEST_SCHEMA,
            "status": "ready",
            "published_at": created.isoformat(),
            "created_at_utc": created.isoformat(),
            "moscow_dates": sorted(verdicts),
            "producer_git_sha": producer_sha,
            "host_id": host_id,
            "run_id": run_id or new_calls_run_id(created),
            "ready_db": str(config.ready_db),
            "sha256": sha,
            "size_bytes": temp_stat.st_size,
            "ready_mtime_ns": temp_stat.st_mtime_ns,
            "quick_check": quick,
            "integrity_check": integrity,
            "provenance_mode": (
                "strict_service"
                if config.strict_ready_provenance
                else "compatibility_not_for_service"
            ),
            "mango_window": {
                "since": (
                    source.get("rolling_since")
                    or source.get("since")
                    or "not_proven"
                ),
                "until": source.get("until") or "not_proven",
            },
            "mango_enumeration_complete": (
                evidence.get("mango_enumeration_complete") is True
            ),
            "mango_enumeration_source": dict(source),
            "catch_up": bool(source.get("catch_up")),
            "sla_mode": "catch_up" if source.get("catch_up") else "live",
            "manifest_snapshot": {
                "end_offset": snapshot["end_offset"],
                "sha256": snapshot["sha256"],
            },
            "consistency_ok": consistency_ok,
            "closure_ok": closure_ok,
            "daily_verdicts": verdicts,
            "runtime_fingerprint": observed_fingerprint,
            "stage_metrics": compact_command_reports(stage_reports),
            "counts": counts,
            "source_storage": source_after,
        }
        validation_errors = validate_ready_manifest_payload(
            manifest,
            # A red Stage10 balance is a valid sealed generation and must be
            # published as evidence, while downstream consumers still reject
            # it through their default consistency gate.
            require_consistency=False,
            expected_code_sha=(
                config.expected_code_sha if config.strict_ready_provenance else None
            ),
            expected_host_id=(host_id if config.strict_ready_provenance else None),
            allow_compatibility=not config.strict_ready_provenance,
        )
        if validation_errors:
            raise RuntimeError(
                "staged ready manifest is invalid: "
                + ",".join(validation_errors)
            )
        commit_ready_generation(
            config.ready_db,
            temp,
            manifest,
            checkpoint=publication_checkpoint,
        )
        return manifest
    finally:
        temp.unlink(missing_ok=True)
        cleanup_sqlite_sidecars(temp)


def sqlite_storage_signature(path: Path) -> Mapping[str, Mapping[str, int]]:
    result: dict[str, Mapping[str, int]] = {}
    for label, candidate in (("db", path), ("wal", Path(str(path) + "-wal"))):
        if candidate.is_file():
            stat = candidate.stat()
            result[label] = {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return result


def publish_ready_db_if_changed(
    config: CallsTwoProcessesConfig,
    counts: Mapping[str, Any],
    *,
    changed: bool,
    run_id: Optional[str] = None,
    capture_evidence: Optional[Mapping[str, Any]] = None,
    manifest_end_offset: Optional[int] = None,
    stage_reports: Sequence[Mapping[str, Any]] = (),
    runtime_fingerprint: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Any]:
    recover_ready_generation(config.ready_db)
    manifest_path = config.ready_db.with_suffix(".manifest.json")
    if not changed and config.ready_db.is_file() and manifest_path.is_file():
        manifest = read_json(manifest_path)
        ready_stat = config.ready_db.stat()
        ready_unchanged = manifest.get("size_bytes") == ready_stat.st_size and manifest.get("ready_mtime_ns") == ready_stat.st_mtime_ns
        evidence = dict(capture_evidence or read_json(config.cursor_path))
        expected_offset = positive_int(
            manifest_end_offset
            if manifest_end_offset is not None
            else evidence.get("manifest_end_offset")
        )
        expected_snapshot_sha = optional_text(
            evidence.get("manifest_snapshot_sha256")
        )
        current_snapshot = capture_manifest_snapshot(
            config.capture_manifest,
            end_offset=expected_offset,
        ) if config.capture_manifest.is_file() else {
            "sha256": hashlib.sha256(b"").hexdigest(),
            "end_offset": 0,
        }
        if expected_snapshot_sha is None and not config.strict_ready_provenance:
            expected_snapshot_sha = str(current_snapshot["sha256"])
        prior_snapshot = manifest.get("manifest_snapshot")
        snapshot_same = bool(
            isinstance(prior_snapshot, Mapping)
            and positive_int(prior_snapshot.get("end_offset")) == expected_offset
            and prior_snapshot.get("sha256") == expected_snapshot_sha
            and current_snapshot.get("sha256") == expected_snapshot_sha
        )
        if (
            manifest.get("status") == "ready"
            and ready_unchanged
            and snapshot_same
            and manifest.get("source_storage")
            == sqlite_storage_signature(config.working_db)
            and not validate_ready_manifest_payload(
                manifest,
                expected_code_sha=(
                    config.expected_code_sha
                    if config.strict_ready_provenance
                    else None
                ),
                expected_host_id=(
                    configured_host_id(config, required=True)
                    if config.strict_ready_provenance
                    else None
                ),
                allow_compatibility=not config.strict_ready_provenance,
            )
        ):
            return {**manifest, "reused": True}
    return {
        **publish_ready_db(
            config,
            counts,
            run_id=run_id,
            capture_evidence=capture_evidence,
            manifest_end_offset=manifest_end_offset,
            stage_reports=stage_reports,
            runtime_fingerprint=runtime_fingerprint,
        ),
        "reused": False,
    }


def ready_drop_fingerprint(config: CallsTwoProcessesConfig) -> Mapping[str, Any]:
    manifest = read_json(config.ready_db.with_suffix(".manifest.json"))
    before = config.ready_db.stat()
    actual_sha = sha256_file(config.ready_db)
    manifest_sha = optional_text(manifest.get("sha256"))
    manifest_size = positive_int(manifest.get("size_bytes")) or None
    actual_size = config.ready_db.stat().st_size
    try:
        actual_quick_check = sqlite_check(config.ready_db, "quick_check")
        actual_integrity_check = sqlite_check(config.ready_db, "integrity_check")
    except (OSError, sqlite3.DatabaseError):
        actual_quick_check = "error"
        actual_integrity_check = "error"
    after = config.ready_db.stat()
    strict_manifest = bool(
        config.strict_ready_provenance
        or manifest.get("schema_version") == READY_MANIFEST_SCHEMA
    )
    if strict_manifest:
        errors = validate_ready_manifest_payload(
            manifest,
            expected_code_sha=(
                config.expected_code_sha if config.strict_ready_provenance else None
            ),
            expected_host_id=(
                configured_host_id(config, required=True)
                if config.strict_ready_provenance
                else None
            ),
            allow_compatibility=not config.strict_ready_provenance,
        )
    else:
        # Compatibility is intentionally available only to direct library
        # callers.  Service configs loaded from JSON always enable strict mode.
        errors = []
        if manifest.get("status") != "ready":
            errors.append("status_not_ready")
        if optional_text(manifest.get("quick_check")) != "ok":
            errors.append("quick_check_not_ok")
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        errors.append("ready_db_changed_while_reading")
    if optional_text(manifest.get("quick_check")) != "ok" or actual_quick_check != "ok":
        errors.append("quick_check_not_ok")
    if actual_integrity_check != "ok" or (
        strict_manifest and optional_text(manifest.get("integrity_check")) != "ok"
    ):
        errors.append("integrity_check_not_ok")
    if not manifest_sha:
        errors.append("sha256_missing")
    elif manifest_sha != actual_sha:
        errors.append("sha256_mismatch")
    if manifest_size is None:
        errors.append("size_missing")
    elif manifest_size != actual_size:
        errors.append("size_mismatch")
    return {
        "sha256": actual_sha,
        "size_bytes": actual_size,
        "manifest_sha256": manifest_sha,
        "manifest_size_bytes": manifest_size,
        "manifest_quick_check": optional_text(manifest.get("quick_check")),
        "actual_quick_check": actual_quick_check,
        "actual_integrity_check": actual_integrity_check,
        "consistency_ok": (
            manifest.get("consistency_ok") is True if strict_manifest else True
        ),
        "closure_ok": manifest.get("closure_ok") is True,
        "producer_git_sha": manifest.get("producer_git_sha"),
        "host_id": manifest.get("host_id"),
        "manifest_valid": not errors,
        "manifest_errors": errors,
        "manifest_mismatch": any(error.endswith("_mismatch") for error in errors),
    }


def cleanup_sqlite_sidecars(path: Path) -> None:
    for suffix in ("-wal", "-shm"):
        sidecar = Path(str(path) + suffix)
        try:
            sidecar.unlink(missing_ok=True)
        except FileNotFoundError:
            pass


def call_db_counts(path: Path) -> Mapping[str, Any]:
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30) as con:
        con.row_factory = sqlite3.Row
        tables = {str(row[0]) for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "call_records" not in tables:
            return empty_call_counts()
        result: dict[str, Any] = {"total": int(con.execute("SELECT COUNT(*) FROM call_records").fetchone()[0])}
        for column in ("transcription_status", "resolve_status", "analysis_status", "dead_letter_stage"):
            rows = con.execute(
                f"SELECT COALESCE({column}, '') AS value, COUNT(*) AS count FROM call_records GROUP BY COALESCE({column}, '')"
            ).fetchall()
            result[column] = {str(row["value"]): int(row["count"]) for row in rows}
        result["analysis_ready"] = int(
            con.execute(
                "SELECT COUNT(*) FROM call_records WHERE analysis_status='done' AND analysis_json IS NOT NULL AND analysis_json != ''"
            ).fetchone()[0]
        )
        result["max_analyzed_at"] = con.execute(
            "SELECT MAX(started_at) FROM call_records WHERE analysis_status='done' AND analysis_json IS NOT NULL AND analysis_json != ''"
        ).fetchone()[0]
        return result


def empty_call_counts() -> Mapping[str, Any]:
    return {
        "total": 0,
        "analysis_ready": 0,
        "transcription_status": {},
        "resolve_status": {},
        "analysis_status": {},
        "dead_letter_stage": {},
        "max_analyzed_at": None,
    }


def dead_letter_total(counts: Mapping[str, Any]) -> int:
    stages = counts.get("dead_letter_stage")
    if not isinstance(stages, Mapping):
        return 0
    return sum(positive_int(count) for stage, count in stages.items() if str(stage or "").strip())


def dead_letter_mass_failure(counts: Mapping[str, Any]) -> bool:
    total = positive_int(counts.get("total"))
    dead = dead_letter_total(counts)
    return total > 0 and dead > 3 and dead * 20 > total


def call_event_source_systems(path: Path) -> list[str]:
    if not path.exists():
        return []
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30) as con:
        con.execute("PRAGMA query_only=ON")
        tables = {str(row[0]) for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "timeline_events" not in tables:
            return []
        return [
            str(row[0])
            for row in con.execute(
                "SELECT DISTINCT source_system FROM timeline_events WHERE event_type='mango_call' ORDER BY source_system"
            )
        ]


def sqlite_check(path: Path, pragma: str) -> str:
    if not path.exists():
        return "missing"
    if pragma not in {"quick_check", "integrity_check"}:
        raise ValueError("unsupported sqlite check")
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=60) as con:
        con.execute("PRAGMA query_only=ON")
        return str(con.execute(f"PRAGMA {pragma}").fetchone()[0])


def finalize_report(
    config: CallsTwoProcessesConfig,
    run_id: str,
    process: str,
    status: str,
    stop_reason: str,
    counters: Mapping[str, Any],
) -> Mapping[str, Any]:
    workers = counters.get("workers") if isinstance(counters.get("workers"), Sequence) else ()
    worker_names = {
        str(item.get("command") or "")
        for item in workers
        if isinstance(item, Mapping)
    }
    imported = counters.get("import") if isinstance(counters.get("import"), Mapping) else {}
    report = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "process": process,
        "status": status,
        "stop_reason": stop_reason,
        "counters": counters,
        "safety": {
            "writes_timeline_staging": process == "process_b"
            and status == "ok"
            and positive_int(imported.get("writes_applied")) > 0,
            "writes_timeline_prod": False,
            "writes_stable_runtime": False,
            "writes_amo": False,
            "writes_crm": False,
            "writes_tallanto": False,
            "runs_asr": any("transcribe" in name for name in worker_names),
            "runs_resolve_analyze": any(
                name in {"worker:resolve", "worker:analyze", "worker:resolve,analyze"}
                for name in worker_names
            ),
            "runs_sync": False,
        },
    }
    report["downstream_ready"] = bool(
        process == "process_a"
        and status in {"ok", "partial"}
        and isinstance(counters.get("drop"), Mapping)
        and counters["drop"].get("status") == "ready"
        and counters["drop"].get("consistency_ok") is True
    )
    local_path = config.reports_dir / f"{run_id}_{process}.json"
    if status in {"failed", "partial"}:
        write_stage_status(config, report)
        config.reports_dir.mkdir(parents=True, exist_ok=True)
        write_json(local_path, report)
    else:
        config.reports_dir.mkdir(parents=True, exist_ok=True)
        write_json(local_path, report)
        write_stage_status(config, report)
    report["report_path"] = str(local_path)
    if config.foton_daily_dir is not None:
        daily_payload = safe_daily_payload(report)
        assert_no_pdn(daily_payload)
        config.foton_daily_dir.mkdir(parents=True, exist_ok=True)
        daily_path = config.foton_daily_dir / f"{run_id}_{process}_calls.json"
        write_json(daily_path, daily_payload)
        report["daily_report_path"] = str(daily_path)
    return report


def safe_daily_payload(report: Mapping[str, Any]) -> Mapping[str, Any]:
    counters = report.get("counters") if isinstance(report.get("counters"), Mapping) else {}
    safe = {
        "schema_version": report.get("schema_version"),
        "run_id": report.get("run_id"),
        "process": report.get("process"),
        "status": report.get("status"),
        "stop_reason": report.get("stop_reason"),
        "counters": scrub_daily_counters(counters),
        "safety": report.get("safety"),
    }
    return safe


def scrub_daily_counters(value: Any, key: str = "") -> Any:
    blocked_key_markers = (
        "path",
        "file",
        "phone",
        "email",
        "customer",
        "lead",
        "contact",
        "example",
        "command",
        "sha",
        "pid",
        "diagnostic",
    )
    if any(marker in key.casefold() for marker in blocked_key_markers):
        return None
    if isinstance(value, Mapping):
        return {
            str(item_key): scrub_daily_counters(item_value, str(item_key))
            for item_key, item_value in value.items()
            if not any(marker in str(item_key).casefold() for marker in blocked_key_markers)
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return len(value)
    if isinstance(value, str):
        return value if len(value) <= 120 else "[redacted_text]"
    return value


def assert_no_pdn(payload: Mapping[str, Any]) -> None:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    if PHONE_RE.search(text) or EMAIL_RE.search(text) or SECRET_RE.search(text):
        raise RuntimeError("foton-pdn-sweep blocked the daily report")


def safe_exception_diagnostic(exc: Exception) -> Mapping[str, str]:
    return {"type": type(exc).__name__}


def compact_import_report(report: Mapping[str, Any]) -> Mapping[str, Any]:
    summary = report.get("summary") if isinstance(report.get("summary"), Mapping) else {}
    writes = report.get("writes") if isinstance(report.get("writes"), Mapping) else {}
    return {
        "validation_ok": report.get("validation_ok"),
        "records_read": summary.get("records_read"),
        "records_accepted": summary.get("records_accepted"),
        "records_rejected": summary.get("records_rejected"),
        "writes_applied": summary.get("writes_applied"),
        "status_counts": writes.get("status_counts"),
        "source_system": report.get("source_system"),
    }


def compact_producer_report(report: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "status": report.get("status"),
        "rows_read": report.get("rows_read"),
        "rows_selected": report.get("rows_selected"),
        "events_written": report.get("events_written"),
        "identity_resolution_counts": report.get("identity_resolution_counts"),
        "call_type_counts": report.get("call_type_counts"),
        "brand_evidence_counts": report.get("brand_evidence_counts"),
        "brand_counts": report.get("brand_counts"),
        "max_event_at": report.get("max_event_at"),
    }


def write_stage_status(config: CallsTwoProcessesConfig, report: Mapping[str, Any]) -> None:
    process = str(report.get("process") or "")
    if process not in {"capture", "process_a", "process_b"}:
        return
    counters = report.get("counters") if isinstance(report.get("counters"), Mapping) else {}
    data_through: Any = None
    checked_through: Any = None
    if process == "capture":
        window = counters.get("window") if isinstance(
            counters.get("window"), Mapping
        ) else {}
        capture = counters.get("capture") if isinstance(
            counters.get("capture"), Mapping
        ) else {}
        data_through = window.get("until")
        checked_through = window.get("until") if capture.get(
            "mango_enumeration_complete"
        ) is True else None
        path = config.capture_status_path
    elif process == "process_a":
        call_db = counters.get("call_db") if isinstance(counters.get("call_db"), Mapping) else {}
        window = counters.get("window") if isinstance(counters.get("window"), Mapping) else {}
        data_through = call_db.get("max_analyzed_at")
        capture = counters.get("capture") if isinstance(counters.get("capture"), Mapping) else {}
        checked_through = window.get("until") if capture.get("status") != "skipped" else None
        path = config.process_a_status_path
    else:
        producer = counters.get("producer") if isinstance(counters.get("producer"), Mapping) else {}
        data_through = producer.get("max_event_at")
        checked_through = datetime.now(timezone.utc).isoformat()
        path = config.process_b_status_path
    previous = read_json(path)
    checked_at = datetime.now(timezone.utc).isoformat()
    if report.get("status") == "locked":
        data_through = data_through or previous.get("data_through")
        checked_through = previous.get("checked_through")
    elif report.get("status") == "idle":
        data_through = data_through or previous.get("data_through")
        checked_through = checked_at
    write_json(
        path,
        {
            "schema_version": "mango_calls_stage_status_v1",
            "process": process,
            "status": report.get("status"),
            "stop_reason": report.get("stop_reason"),
            "checked_at": checked_at,
            "checked_through": checked_through,
            "data_through": data_through,
        },
    )


def pipeline_freshness(
    config: CallsTwoProcessesConfig,
    *,
    now: Optional[datetime] = None,
) -> Mapping[str, Any]:
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    threshold = config.freshness_max_age_minutes * 60
    capture_health = capture_manifest_health(config.capture_manifest)
    tail_status = str(capture_health["tail_status"])
    recovery_status = str(capture_health["recovery_status"])
    recovery_unresolved_count = positive_int(capture_health["recovery_unresolved_count"])
    stages: dict[str, Any] = {}
    process_paths = (
        (
            ("capture", config.capture_status_path),
            ("process_a", config.process_a_status_path),
            ("process_b", config.process_b_status_path),
        )
        if config.strict_ready_provenance
        else (
            ("process_a", config.process_a_status_path),
            ("process_b", config.process_b_status_path),
        )
    )
    capture_process = "capture" if config.strict_ready_provenance else "process_a"
    for process, path in process_paths:
        state = read_json(path)
        raw_checked = optional_text(state.get("checked_through") or state.get("checked_at"))
        raw_data = optional_text(state.get("data_through"))
        checked = parse_datetime(raw_checked) if raw_checked else None
        data_at = parse_datetime(raw_data) if raw_data else None
        checked_age = max(0.0, (current - checked).total_seconds()) if checked else None
        data_age = max(0.0, (current - data_at).total_seconds()) if data_at else None
        status = (
            "missing"
            if checked is None
            else "stale"
            if checked_age is not None and checked_age > threshold
            else "fresh"
        )
        stage_stop_reason = state.get("stop_reason")
        if state.get("status") == "failed":
            status = "failed"
        elif process == capture_process and (tail_status == "invalid" or recovery_status == "invalid"):
            status = "failed"
            stage_stop_reason = (
                "capture_manifest_tail_invalid"
                if tail_status == "invalid"
                else "capture_recovery_ledger_invalid"
            )
        elif process == capture_process and (tail_status == "incomplete" or recovery_unresolved_count):
            status = "partial"
            stage_stop_reason = "capture_manifest_tail_incomplete"
        elif process == capture_process and tail_status == "missing":
            status = "missing"
            stage_stop_reason = "capture_manifest_missing"
        elif state.get("status") == "partial":
            status = "partial"
        elif state.get("stop_reason") == "drop_missing":
            status = "missing"
        stages[process] = {
            "status": status,
            "age_seconds": round(data_age, 3) if data_age is not None else None,
            "checked_age_seconds": round(checked_age, 3) if checked_age is not None else None,
            "checked_through": raw_checked,
            "data_through": raw_data,
            "last_run_status": state.get("status"),
            "stop_reason": stage_stop_reason,
        }
        if process == capture_process:
            stages[process]["capture_manifest_tail_status"] = tail_status
            stages[process]["capture_recovery_status"] = recovery_status
            stages[process]["capture_recovery_unresolved_count"] = recovery_unresolved_count
    heartbeat = read_json(config.process_a_heartbeat_path)
    heartbeat_state: Mapping[str, Any] = {}
    if heartbeat:
        raw_heartbeat = optional_text(heartbeat.get("updated_at"))
        try:
            heartbeat_at = parse_datetime(raw_heartbeat) if raw_heartbeat else None
            heartbeat_age = (
                max(0.0, (current - heartbeat_at).total_seconds())
                if heartbeat_at
                else None
            )
        except (TypeError, ValueError):
            heartbeat_age = None
        heartbeat_pid = positive_int(heartbeat.get("pid"))
        heartbeat_live = bool(
            heartbeat_age is not None
            and heartbeat_age <= 90
            and pid_exists(heartbeat_pid)
            and str(heartbeat.get("stage") or "") in SEQUENTIAL_PIPELINE_STAGES
        )
        heartbeat_state = {
            "status": "running" if heartbeat_live else "stale_or_dead",
            "age_seconds": round(heartbeat_age, 3) if heartbeat_age is not None else None,
            "stage": heartbeat.get("stage"),
        }
        if heartbeat_live and "process_a" in stages:
            stages["process_a"] = {
                **stages["process_a"],
                "status": "running",
                "stop_reason": "",
            }
    ok = all(item["status"] in {"fresh", "running"} for item in stages.values())
    return {
        "schema_version": "mango_calls_freshness_v1",
        "status": "fresh" if ok else "stale",
        "stages": stages,
        "heavy_heartbeat": heartbeat_state,
    }


def run_local_watchdog(
    config: CallsTwoProcessesConfig,
    *,
    now: Optional[datetime] = None,
) -> Mapping[str, Any]:
    """Read only local heartbeat/provenance state; never processes a call."""
    config.validate()
    freshness = pipeline_freshness(config, now=now)
    authority = cutover_authority_report(config)
    snapshot = capture_manifest_snapshot(config.capture_manifest)
    active_host = configured_host_id(
        config, required=config.require_cutover_authority
    )
    foreign_after: Optional[datetime] = None
    if authority.get("previous_host_disabled_at"):
        try:
            foreign_after = parse_aware_datetime(authority["previous_host_disabled_at"])
        except (TypeError, ValueError):
            foreign_after = None
    foreign = foreign_host_ids(
        snapshot["entries"],
        active_host_id=active_host,
        foreign_after=foreign_after,
    )
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    moscow_now = current.astimezone(ZoneInfo("Europe/Moscow"))
    yesterday = moscow_now.date() - timedelta(days=1)
    manifest = read_json(config.ready_manifest)
    manifest_errors = (
        validate_ready_manifest_payload(
            manifest,
            expected_code_sha=(
                config.expected_code_sha if config.strict_ready_provenance else None
            ),
            expected_host_id=(
                active_host if config.strict_ready_provenance else None
            ),
            allow_compatibility=not config.strict_ready_provenance,
        )
        if manifest
        else ["ready_manifest_missing"]
    )
    if inspect_ready_publication(config.ready_db).get("recovery_required"):
        manifest_errors.append("ready_publication_recovery_required")
    if manifest and config.strict_ready_provenance:
        try:
            ready_stat = os.lstat(config.ready_db)
            if (
                not stat.S_ISREG(ready_stat.st_mode)
                or config.ready_db.is_symlink()
                or sha256_file(config.ready_db) != manifest.get("sha256")
                or ready_stat.st_size != manifest.get("size_bytes")
            ):
                manifest_errors.append("ready_db_seal_mismatch")
        except OSError:
            manifest_errors.append("ready_db_seal_mismatch")
    verdicts = manifest.get("daily_verdicts") if isinstance(manifest, Mapping) else None
    yesterday_verdict = (
        verdicts.get(yesterday.isoformat())
        if isinstance(verdicts, Mapping)
        else None
    )
    yesterday_verdict = (
        yesterday_verdict if isinstance(yesterday_verdict, Mapping) else {}
    )
    today_verdict = (
        verdicts.get(moscow_now.date().isoformat())
        if isinstance(verdicts, Mapping)
        else None
    )
    today_verdict = today_verdict if isinstance(today_verdict, Mapping) else {}
    try:
        free_bytes = shutil.disk_usage(config.pipeline_root).free
    except OSError:
        free_bytes = 0
    required_free_bytes = int(config.min_free_gib * 1024**3)
    operational_reasons: list[str] = []
    if int(today_verdict.get("pending_over_sla") or 0) > 0:
        operational_reasons.append("pending_over_sla")
    if (
        int(today_verdict.get("mango_unique") or 0) > 0
        and int(today_verdict.get("pending_unique") or 0)
        == int(today_verdict.get("mango_unique") or 0)
        and float(today_verdict.get("oldest_pending_age_minutes") or 0) > 60
    ):
        operational_reasons.append("all_calls_pending_over_60_minutes")
    if free_bytes < required_free_bytes:
        operational_reasons.append("disk_below_threshold")
    if (moscow_now.hour, moscow_now.minute) >= (8, 30) and (
        yesterday_verdict.get("closure_ok") is not True
    ):
        operational_reasons.append("previous_day_not_closed")
    status_file = (
        config.local_publication_root
        / "status"
        / f"{yesterday.isoformat()}.json"
    )
    if (moscow_now.hour, moscow_now.minute) >= (9, 0) and not status_file.is_file():
        operational_reasons.append("daily_status_missing_after_0900")
    p0 = bool(foreign or authority.get("ok") is not True)
    status = (
        "p0"
        if p0
        else "alert"
        if manifest_errors or operational_reasons
        else "ok"
        if freshness.get("status") == "fresh"
        else "stale"
    )
    alert = {
        "schema_version": "mango_calls_watchdog_alert_v1",
        "status": status,
        "stop_reason": (
            "foreign_host_or_cutover_authority_failed"
            if p0
            else ",".join(
                [
                    *(("ready_manifest_invalid",) if manifest_errors else ()),
                    *operational_reasons,
                ]
            )
            if status == "alert"
            else "heartbeat_stale"
            if status == "stale"
            else ""
        ),
        "foreign_host_count": len(foreign),
        "pending_over_sla": int(today_verdict.get("pending_over_sla") or 0),
        "oldest_pending_age_minutes": float(
            today_verdict.get("oldest_pending_age_minutes") or 0
        ),
        "closure_ok": yesterday_verdict.get("closure_ok") is True,
        "free_bytes": free_bytes,
        "required_free_bytes": required_free_bytes,
        "watchdog_alive": True,
    }
    return {
        "schema_version": "mango_calls_local_watchdog_v1",
        "process": "watchdog",
        "status": status,
        "stop_reason": alert["stop_reason"],
        "freshness": freshness,
        "authority": authority,
        "foreign_host_ids": foreign,
        "daily": {
            "moscow_day": yesterday.isoformat(),
            "closure_ok": yesterday_verdict.get("closure_ok") is True,
            "today_pending_over_sla": int(
                today_verdict.get("pending_over_sla") or 0
            ),
            "status_present": status_file.is_file(),
            "manifest_errors": list(manifest_errors),
        },
        "safe_alert": safe_alert_payload(alert),
        "safety": {
            "read_only": True,
            "runs_asr": False,
            "runs_resolve_analyze": False,
            "writes_external_systems": False,
        },
    }


def compact_command_reports(reports: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    compacted: list[Mapping[str, Any]] = []
    for item in reports:
        row: dict[str, Any] = {
            "rc": item.get("rc"),
            "command": item.get("command"),
            "log_path": item.get("log_path"),
            "log_size_bytes": item.get("log_size_bytes"),
            "log_sha256": item.get("log_sha256"),
            "wall_seconds": item.get("wall_seconds"),
            "peak_rss_raw": item.get("peak_rss_raw"),
            "swap_operations": item.get("swap_operations"),
            "timed_out": item.get("timed_out"),
        }
        if isinstance(item.get("metrics"), Mapping):
            row["metrics"] = dict(item["metrics"])
        compacted.append(row)
    return compacted


def compact_command_name(command: Sequence[str]) -> str:
    if "worker" in command:
        try:
            return "worker:" + str(command[command.index("--stages") + 1])
        except (ValueError, IndexError):
            return "worker"
    if "ingest" in command:
        return "ingest"
    return str(command[-1]) if command else "unknown"


def _sha256_regular_nofollow(path: Path, *, label: str) -> str:
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0),
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
            raise RuntimeError(f"{label} is not a non-empty regular file: {path}")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
        path_stat = os.lstat(path)
        identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        if identity != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ) or (path_stat.st_dev, path_stat.st_ino) != (after.st_dev, after.st_ino):
            raise RuntimeError(f"{label} changed while hashing: {path}")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def hardlink_or_copy(source: Path, target: Path) -> str:
    source = source.absolute()
    source_stat = os.lstat(source)
    if not stat.S_ISREG(source_stat.st_mode) or source_stat.st_size <= 0:
        raise RuntimeError(f"source audio is not a non-empty regular file: {source}")
    source_digest = _sha256_regular_nofollow(source, label="source audio")
    if os.path.lexists(target):
        try:
            target_digest = _sha256_regular_nofollow(
                target, label="existing audio target"
            )
        except (OSError, RuntimeError) as exc:
            raise RuntimeError(f"existing audio target is unsafe: {target}") from exc
        if source_digest != target_digest:
            raise RuntimeError(f"existing audio differs: {target}")
        return "exists_same_hash"
    try:
        os.link(source, target, follow_symlinks=False)
        linked = os.lstat(target)
        current_source = os.lstat(source)
        if (
            not stat.S_ISREG(linked.st_mode)
            or (linked.st_dev, linked.st_ino)
            != (source_stat.st_dev, source_stat.st_ino)
            or (current_source.st_dev, current_source.st_ino)
            != (source_stat.st_dev, source_stat.st_ino)
        ):
            raise RuntimeError("source audio changed during hardlink")
        _fsync_directory(target.parent)
        return "hardlink"
    except FileExistsError:
        try:
            target_digest = _sha256_regular_nofollow(
                target, label="concurrent audio target"
            )
        except (OSError, RuntimeError) as exc:
            raise RuntimeError(f"concurrent audio target is unsafe: {target}") from exc
        if source_digest != target_digest:
            raise RuntimeError(f"concurrent audio target differs: {target}")
        return "exists_same_hash"
    except OSError:
        pass

    temporary = target.parent / (
        f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.copying"
    )
    descriptor = -1
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        source_descriptor = os.open(
            source,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        opened_source = os.fstat(source_descriptor)
        if (opened_source.st_dev, opened_source.st_ino) != (
            source_stat.st_dev,
            source_stat.st_ino,
        ):
            os.close(source_descriptor)
            raise RuntimeError("source audio changed before copy")
        with os.fdopen(source_descriptor, "rb") as source_handle, os.fdopen(
            descriptor, "wb", closefd=False
        ) as target_handle:
            shutil.copyfileobj(source_handle, target_handle, length=1024 * 1024)
            target_handle.flush()
            os.fsync(descriptor)
        if source_digest != _sha256_regular_nofollow(
            temporary, label="temporary audio copy"
        ):
            raise RuntimeError("audio changed or copy verification failed")
        try:
            os.link(temporary, target, follow_symlinks=False)
        except FileExistsError:
            try:
                target_digest = _sha256_regular_nofollow(
                    target, label="concurrent audio target"
                )
            except (OSError, RuntimeError) as exc:
                raise RuntimeError(
                    f"concurrent audio target is unsafe: {target}"
                ) from exc
            if source_digest != target_digest:
                raise RuntimeError(f"concurrent audio target differs: {target}")
            return "exists_same_hash"
        _fsync_directory(target.parent)
        return "copy"
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
        _fsync_directory(target.parent)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_cursor(path: Path, until: datetime, capture: Mapping[str, Any]) -> None:
    write_json(
        path,
        {
            "schema_version": "mango_api_freshness_v1",
            "until": until.isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "capture_status": capture.get("status"),
            "downloaded": capture.get("downloaded"),
            "failed": capture.get("failed"),
            "integrity_quarantined": capture.get("integrity_quarantined"),
            "host_id": capture.get("host_id"),
            "manifest_end_offset": capture.get("manifest_end_offset"),
            "manifest_snapshot_sha256": capture.get(
                "manifest_snapshot_sha256"
            ),
            "mango_enumeration_complete": capture.get(
                "mango_enumeration_complete"
            ),
            "mango_enumeration_source": capture.get(
                "mango_enumeration_source"
            ),
            "catch_up": bool(capture.get("catch_up")),
            "sla_mode": capture.get("sla_mode") or "live",
            "call_keys": capture.get("call_keys"),
            "calls_by_moscow_day": capture.get("calls_by_moscow_day"),
            "independent_zero_enumerations_by_day": capture.get(
                "independent_zero_enumerations_by_day"
            ),
        },
    )


def parse_datetime(value: str) -> datetime:
    text = value.strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def optional_text(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    return text or None


def positive_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def new_calls_run_id(started: datetime) -> str:
    return f"{started.strftime('%Y%m%dT%H%M%S%fZ')}-u{uuid.uuid4().hex[:12]}"


def parse_json_object(text: str) -> Mapping[str, Any]:
    try:
        value = json.loads(text) if text.strip() else {}
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, Mapping) else {}


def read_json(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        return {}
    return parse_json_object(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_private_json(path, payload, indent=2)
