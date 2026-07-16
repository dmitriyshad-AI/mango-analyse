from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import os
import re
import shutil
import socket
import sqlite3
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence

from mango_mvp.customer_timeline.import_cli import (
    TimelineImportCliConfig,
    run_timeline_import_cli,
)
from mango_mvp.customer_timeline.nightly_service import mango_processed_cursor
from mango_mvp.customer_timeline.safety import (
    guard_customer_timeline_output_path,
    is_customer_timeline_prod_path,
    is_stable_runtime_path,
)
from mango_mvp.productization.capture_staging import (
    CaptureManifestStore,
    stage_capture_events,
)
from mango_mvp.productization.mango_office import MangoOfficePayloadMapper
from mango_mvp.productization.mango_office_client import (
    DEFAULT_MANGO_BASE_URL,
    MangoOfficeClient,
    MangoOfficeCredentials,
)
from mango_mvp.productization.mango_recordings import MangoRecordingDownloader
from mango_mvp.productization.contracts import TenantRef


SCHEMA_VERSION = "mango_calls_two_processes_v1"
PARALLEL_PIPELINE_STAGES = (
    "transcribe",
    "backfill-second-asr",
    "resolve",
    "analyze",
)
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
    poll_overlap_minutes: int = 15
    api_window_hours: int = 12
    min_free_gib: float = 40.0
    stale_lock_seconds: int = 6 * 60 * 60
    stage_limit: int = 20
    poll_seconds: int = 10
    max_idle_cycles: int = 30
    freshness_max_age_minutes: int = 90
    asr_mode: str = "mlx_dual"
    codex_resolve_model: str = "gpt-5.4"
    codex_analyze_model: str = "gpt-5.4-mini"
    codex_reasoning_effort: str = "medium"

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
            poll_overlap_minutes=int(payload.get("poll_overlap_minutes", 15)),
            api_window_hours=int(payload.get("api_window_hours", 12)),
            min_free_gib=float(payload.get("min_free_gib", 40.0)),
            stale_lock_seconds=int(payload.get("stale_lock_seconds", 6 * 60 * 60)),
            stage_limit=int(payload.get("stage_limit", 20)),
            poll_seconds=int(payload.get("poll_seconds", 10)),
            max_idle_cycles=int(payload.get("max_idle_cycles", 30)),
            freshness_max_age_minutes=int(payload.get("freshness_max_age_minutes", 90)),
            asr_mode=str(payload.get("asr_mode") or "mlx_dual").strip().lower(),
            codex_resolve_model=str(payload.get("codex_resolve_model") or "gpt-5.4"),
            codex_analyze_model=str(payload.get("codex_analyze_model") or "gpt-5.4-mini"),
            codex_reasoning_effort=str(payload.get("codex_reasoning_effort") or "medium"),
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
        if self.asr_mode != "mlx_dual":
            raise ValueError("asr_mode must be mlx_dual; single-ASR fallback is disabled")
        if self.min_free_gib < 1:
            raise ValueError("min_free_gib must be at least 1")

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
    def cursor_path(self) -> Path:
        return self.pipeline_root / "state" / "mango_api_freshness.json"

    @property
    def process_a_lock(self) -> Path:
        return self.pipeline_root / "locks" / "process_a.lock"

    @property
    def reports_dir(self) -> Path:
        return self.pipeline_root / "reports"

    @property
    def process_a_status_path(self) -> Path:
        return self.pipeline_root / "state" / "process_a_status.json"

    @property
    def process_b_status_path(self) -> Path:
        return self.pipeline_root / "state" / "process_b_status.json"

    @property
    def process_b_cursor_path(self) -> Path:
        return self.pipeline_root / "state" / "process_b_cursor.json"

    @property
    def ingest_dir(self) -> Path:
        return self.timeline_allowed_root / "mango_calls_two_processes"


CommandRunner = Callable[[Sequence[str], Mapping[str, str], Path], Mapping[str, Any]]
CaptureRunner = Callable[[CallsTwoProcessesConfig, datetime, datetime], Mapping[str, Any]]
ProducerRunner = Callable[[CallsTwoProcessesConfig, Path, Path, Optional[str]], Mapping[str, Any]]
ImportRunner = Callable[[TimelineImportCliConfig], Mapping[str, Any]]


def run_process_a(
    config: CallsTwoProcessesConfig,
    *,
    since: Optional[str] = None,
    until: Optional[str] = None,
    skip_capture: bool = False,
    skip_workers: bool = False,
    command_runner: CommandRunner = None,
    capture_runner: CaptureRunner = None,
) -> Mapping[str, Any]:
    config.validate()
    command_runner = command_runner or run_command
    capture_runner = capture_runner or capture_mango_window
    started = datetime.now(timezone.utc)
    run_id = started.strftime("%Y%m%dT%H%M%SZ")
    config.reports_dir.mkdir(parents=True, exist_ok=True)
    try:
        with process_lease(config.process_a_lock, stale_seconds=config.stale_lock_seconds) as lock_info:
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
                {"status": "skipped", "reason": "skip_capture"}
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
            metadata = prepare_ingest_inputs(config)
            worker_reports: list[Mapping[str, Any]] = []
            if not skip_workers and metadata["audio_files"]:
                base_env = worker_environment(config)
                worker_reports.append(
                    command_runner(
                        cli_command(config, "init-db"),
                        base_env,
                        config.working_dir,
                    )
                )
                worker_reports.append(
                    command_runner(
                        cli_command(
                            config,
                            "ingest",
                            "--recordings-dir",
                            str(config.working_audio_dir),
                            "--metadata-csv",
                            str(config.metadata_csv),
                        ),
                        base_env,
                        config.working_dir,
                    )
                )
                worker_reports.extend(
                    run_parallel_pipeline_workers(
                        config,
                        base_env,
                        command_runner,
                        include_llm=bool(environment.get("codex_network_ok")),
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
            drop = publish_ready_db(config, db_counts) if config.working_db.exists() else {"status": "not_created"}
            if not skip_capture:
                write_cursor(config.cursor_path, window_until, capture)
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
            return finalize_report(config, run_id, "process_a", "ok", "", counters)
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
        return finalize_report(
            config,
            run_id,
            "process_a",
            "failed",
            f"process_a_exception:{type(exc).__name__}",
            {"diagnostic": safe_exception_diagnostic(exc)},
        )


def _run_process_b(
    config: CallsTwoProcessesConfig,
    *,
    producer_runner: ProducerRunner = None,
    import_runner: ImportRunner = run_timeline_import_cli,
) -> Mapping[str, Any]:
    started = datetime.now(timezone.utc)
    run_id = started.strftime("%Y%m%dT%H%M%SZ")
    producer_runner = producer_runner or run_increment_producer
    config.ingest_dir.mkdir(parents=True, exist_ok=True)
    if not config.ready_db.exists():
        return finalize_report(config, run_id, "process_b", "idle", "drop_missing", {"events": 0})
    drop_fingerprint = ready_drop_fingerprint(config)
    previous_cursor = read_json(config.process_b_cursor_path)
    if drop_fingerprint.get("sha256") and previous_cursor.get("sha256") == drop_fingerprint.get("sha256"):
        return finalize_report(
            config,
            run_id,
            "process_b",
            "idle",
            "drop_unchanged",
            {"events": 0, "drop": drop_fingerprint, "cursor_before": previous_cursor},
        )
    quick_before = sqlite_check(config.timeline_db, "quick_check")
    source_systems_before = call_event_source_systems(config.timeline_db)
    cursor = mango_processed_cursor(config.timeline_db, tenant_id=config.tenant_id)
    increment_path = config.ingest_dir / "mango_processed_summary.jsonl"
    producer_report_path = config.ingest_dir / "mango_processed_summary_producer_report.json"
    # A call may finish Analyze after newer calls were already imported. Scan the
    # sealed drop fully and let dedupe_key decide; a timestamp cursor can lose it.
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
    run_id = started.strftime("%Y%m%dT%H%M%SZ")
    try:
        config.validate()
    except Exception as exc:  # noqa: BLE001 - normalized fail-loud boundary
        return process_b_failure_report(
            run_id,
            f"process_b_config_exception:{type(exc).__name__}",
            exc,
        )
    try:
        return _run_process_b(
            config,
            producer_runner=producer_runner,
            import_runner=import_runner,
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
    first = run_process_a(config, **process_a_kwargs)
    if first.get("status") not in {"ok", "idle"}:
        return {
            "schema_version": SCHEMA_VERSION,
            "process": "cycle",
            "status": str(first.get("status") or "failed"),
            "stop_reason": "process_a_not_ok",
            "process_a": first,
            "process_b": None,
        }
    second = run_process_b(config)
    return {
        "schema_version": SCHEMA_VERSION,
        "process": "cycle",
        "status": "ok" if second.get("status") in {"ok", "idle", "locked"} else "failed",
        "stop_reason": "" if second.get("status") in {"ok", "idle", "locked"} else "process_b_failed",
        "process_a": first,
        "process_b": second,
    }


class LockBusy(RuntimeError):
    def __init__(self, metadata: Mapping[str, Any]) -> None:
        super().__init__("process lock is busy")
        self.metadata = dict(metadata)


@contextmanager
def process_lease(path: Path, *, stale_seconds: int) -> Iterator[Mapping[str, Any]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    handle.seek(0)
    previous = parse_json_object(handle.read())
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise LockBusy(previous) from exc
    age_seconds = max(0.0, time.time() - path.stat().st_mtime) if path.exists() else 0.0
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
    rows: list[Mapping[str, Any]] = []
    chunk_start = since
    api_requests = 0
    while chunk_start < until:
        chunk_end = min(until, chunk_start + timedelta(hours=config.api_window_hours))
        rows.extend(client.poll_call_history(since=chunk_start, until=chunk_end))
        api_requests += 1
        chunk_start = chunk_end
    unique_events: dict[str, Any] = {}
    for row in rows:
        event = mapper.from_payload(tenant=tenant, payload=row)
        unique_events[event.event_key] = event
    mapped_events = sorted(unique_events.values(), key=lambda event: (event.started_at, event.provider_call_id))
    known_recordings, known_calls = read_known_processed_ids(config.pipeline_root.parent)
    external_known_keys = {
        event.event_key
        for event in mapped_events
        if (event.recording_ref and event.recording_ref in known_recordings)
        or event.provider_call_id in known_calls
    }
    # Calls without a recording are deliberately not committed to the capture
    # manifest: Mango may attach the recording later, and a future overlap poll
    # must be able to pick it up.
    events = [
        event
        for event in mapped_events
        if (event.recording_ref or event.recording_url) and event.event_key not in external_known_keys
    ]
    summary = stage_capture_events(
        events=events,
        manifest_store=CaptureManifestStore(config.capture_manifest),
        recordings_dir=config.recordings_dir,
        downloader=downloader,
        dry_run=False,
        sleep_sec=1.5,
    )
    status = "ok" if summary.failed == 0 else "failed"
    return {
        "status": status,
        "api_requests": api_requests,
        "api_rows_total": len(rows),
        "api_events_total": len(mapped_events),
        "api_events_already_known_external": len(external_known_keys),
        "api_events_without_recording": sum(
            1 for event in mapped_events if not (event.recording_ref or event.recording_url)
        ),
        **summary.to_json_dict(),
    }


def prepare_ingest_inputs(config: CallsTwoProcessesConfig) -> Mapping[str, Any]:
    config.working_audio_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    actions: dict[str, int] = {}
    latest = CaptureManifestStore(config.capture_manifest).latest_by_event_key() if config.capture_manifest.exists() else {}
    for entry in sorted(latest.values(), key=lambda item: (item.started_at, item.event_key)):
        if entry.status != "downloaded" or not entry.local_audio_path:
            continue
        source = Path(entry.local_audio_path)
        if not source.is_file() or source.stat().st_size <= 0:
            continue
        target = config.working_audio_dir / source.name
        action = hardlink_or_copy(source, target)
        actions[action] = actions.get(action, 0) + 1
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
    return {"audio_files": len(rows), "link_actions": actions, "metadata_rows": len(rows)}


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
            codex_home = prepare_codex_home(config.codex_home_root / "worker")
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
    }
    required = ("mango_credentials", "python_executable", "asr_modules", "codex_binary", "codex_auth")
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
        "parallel_stages": list(pipeline_stages(config, include_llm=network_ok)),
    }


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
    config.pipeline_root.mkdir(parents=True, exist_ok=True)
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
        else:
            start = end - timedelta(hours=config.first_lookback_hours)
    if end <= start:
        raise ValueError("capture until must be after since")
    return start, end


def worker_environment(config: CallsTwoProcessesConfig) -> Mapping[str, str]:
    project_root = Path(__file__).resolve().parents[3]
    isolated_codex = project_root / "scripts" / "run_codex_cli_isolated.sh"
    config.transcripts_dir.mkdir(parents=True, exist_ok=True)
    codex_home = prepare_codex_home(config.codex_home_root / "worker")
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
        "GIGAAM_DEVICE": "cpu",
    }


def prepare_codex_home(target: Path) -> Path:
    source = Path.home() / ".codex"
    target.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(target, 0o700)
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
    stages = PARALLEL_PIPELINE_STAGES
    if include_llm:
        return stages
    return tuple(stage for stage in stages if stage not in {"resolve", "analyze"})


def run_parallel_pipeline_workers(
    config: CallsTwoProcessesConfig,
    base_env: Mapping[str, str],
    runner: CommandRunner,
    *,
    include_llm: bool = True,
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
    logs_dir.mkdir(parents=True, exist_ok=True)
    reports: list[Mapping[str, Any]] = []
    for stage in stages:
        label = stage.replace("-", "_")
        log_path = logs_dir / f"stage_{label}.log"
        worker_env = {
            **stage_worker_environment_for(config, base_env, stage),
            "CODEX_HOME": str(
                prepare_codex_home(config.codex_home_root / label)
            ),
        }
        with log_path.open("w", encoding="utf-8") as log_handle:
            proc = subprocess.run(
                worker_command(config, stage),
                cwd=config.working_dir,
                env=worker_env,
                text=True,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        rc = int(proc.returncode or 0)
        reports.append({"rc": rc, "command": f"worker:{stage}", "log_path": str(log_path)})
        if rc != 0:
            break
    return reports


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


def run_command(command: Sequence[str], env: Mapping[str, str], cwd: Path) -> Mapping[str, Any]:
    cwd.mkdir(parents=True, exist_ok=True)
    logs_dir = cwd / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    log_path = logs_dir / f"command_{stamp}.log"
    with log_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.run(
            list(command),
            cwd=cwd,
            env=dict(env),
            text=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    report: dict[str, Any] = {
        "rc": proc.returncode,
        "command": compact_command_name(command),
        "log_path": str(log_path),
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


def publish_ready_db(config: CallsTwoProcessesConfig, counts: Mapping[str, Any]) -> Mapping[str, Any]:
    config.ready_db.parent.mkdir(parents=True, exist_ok=True)
    temp = config.ready_db.with_suffix(".sqlite.tmp")
    cleanup_sqlite_sidecars(temp)
    with sqlite3.connect(f"file:{config.working_db}?mode=ro", uri=True, timeout=60) as source:
        source.execute("PRAGMA query_only=ON")
        with sqlite3.connect(temp) as target:
            source.backup(target)
            target.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            target.execute("PRAGMA journal_mode=DELETE")
            quick = str(target.execute("PRAGMA quick_check").fetchone()[0])
    if quick != "ok":
        raise RuntimeError("ready DB quick_check failed")
    temp.replace(config.ready_db)
    cleanup_sqlite_sidecars(temp)
    sha = sha256_file(config.ready_db)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "ready",
        "published_at": datetime.now(timezone.utc).isoformat(),
        "ready_db": str(config.ready_db),
        "sha256": sha,
        "size_bytes": config.ready_db.stat().st_size,
        "quick_check": quick,
        "counts": counts,
    }
    write_json(config.ready_db.with_suffix(".manifest.json"), manifest)
    return manifest


def ready_drop_fingerprint(config: CallsTwoProcessesConfig) -> Mapping[str, Any]:
    manifest = read_json(config.ready_db.with_suffix(".manifest.json"))
    actual_sha = sha256_file(config.ready_db)
    manifest_sha = optional_text(manifest.get("sha256"))
    manifest_size = positive_int(manifest.get("size_bytes")) or None
    actual_size = config.ready_db.stat().st_size
    return {
        "sha256": actual_sha,
        "size_bytes": actual_size,
        "manifest_sha256": manifest_sha,
        "manifest_size_bytes": manifest_size,
        "manifest_mismatch": bool(
            manifest_sha and (manifest_sha != actual_sha or manifest_size not in {None, actual_size})
        ),
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
    config.reports_dir.mkdir(parents=True, exist_ok=True)
    local_path = config.reports_dir / f"{run_id}_{process}.json"
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
    if process not in {"process_a", "process_b"}:
        return
    counters = report.get("counters") if isinstance(report.get("counters"), Mapping) else {}
    data_through: Any = None
    checked_through: Any = None
    if process == "process_a":
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
    if report.get("status") in {"locked", "idle"}:
        data_through = data_through or previous.get("data_through")
        checked_through = previous.get("checked_through")
    write_json(
        path,
        {
            "schema_version": "mango_calls_stage_status_v1",
            "process": process,
            "status": report.get("status"),
            "stop_reason": report.get("stop_reason"),
            "checked_at": datetime.now(timezone.utc).isoformat(),
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
    stages: dict[str, Any] = {}
    for process, path in (
        ("process_a", config.process_a_status_path),
        ("process_b", config.process_b_status_path),
    ):
        state = read_json(path)
        raw_checked = optional_text(state.get("checked_through") or state.get("checked_at"))
        raw_data = optional_text(state.get("data_through"))
        checked = parse_datetime(raw_checked) if raw_checked else None
        data_at = parse_datetime(raw_data) if raw_data else None
        checked_age = max(0.0, (current - checked).total_seconds()) if checked else None
        data_age = max(0.0, (current - data_at).total_seconds()) if data_at else None
        status = "missing" if data_at is None else "stale" if data_age > threshold else "fresh"
        if state.get("status") == "failed":
            status = "failed"
        elif state.get("stop_reason") == "drop_missing":
            status = "missing"
        stages[process] = {
            "status": status,
            "age_seconds": round(data_age, 3) if data_age is not None else None,
            "checked_age_seconds": round(checked_age, 3) if checked_age is not None else None,
            "checked_through": raw_checked,
            "data_through": raw_data,
            "last_run_status": state.get("status"),
            "stop_reason": state.get("stop_reason"),
        }
    ok = all(item["status"] == "fresh" for item in stages.values())
    return {"schema_version": "mango_calls_freshness_v1", "status": "fresh" if ok else "stale", "stages": stages}


def compact_command_reports(reports: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    compacted: list[Mapping[str, Any]] = []
    for item in reports:
        row: dict[str, Any] = {
            "rc": item.get("rc"),
            "command": item.get("command"),
            "log_path": item.get("log_path"),
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


def hardlink_or_copy(source: Path, target: Path) -> str:
    if target.exists():
        if sha256_file(source) != sha256_file(target):
            raise RuntimeError(f"existing audio differs: {target}")
        return "exists_same_hash"
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)
