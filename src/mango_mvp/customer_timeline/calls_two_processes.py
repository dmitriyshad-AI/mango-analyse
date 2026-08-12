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
from contextlib import closing, contextmanager, nullcontext
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta, timezone
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
    DUAL_ENUMERATION_NORMALIZATION,
    DUAL_ENUMERATION_SCHEMA,
    MANGO_OFFICIAL_LIST_SCHEMA,
    MANGO_OFFICIAL_PAGE_LIMIT,
    READY_MANIFEST_SCHEMA,
    _official_list_proof_is_green,
    approved_runtime_fingerprint,
    build_stage10_verdict,
    current_git_sha,
    enumeration_source_covers_day,
    foreign_host_ids,
    has_dual_asr_or_exception,
    load_ready_rows,
    moscow_day_bounds_utc,
    parse_aware_datetime,
    read_host_id,
    ready_row_is_complete,
    safe_alert_payload,
    stage_capacity_report,
    validate_capture_enumeration_evidence,
    validate_ready_manifest_payload,
    validate_runtime_fingerprint,
    verify_cutover_authority,
)
from mango_mvp.productization.mango_calls_config import (
    load_owner_only_runtime_config,
    strict_service_flags,
)
from mango_mvp.productization.owner_only_io import (
    atomic_replace_owner_only_bytes,
    copy_stable_regular_file_owner_only,
    path_has_cloud_marker,
    inspect_stable_regular_file,
    read_stable_regular_bytes,
    read_stable_regular_bytes_with_path,
    validate_owner_only_directory,
)
from mango_mvp.productization.mango_office import MangoOfficePayloadMapper
from mango_mvp.productization.mango_office_client import (
    DEFAULT_MANGO_BASE_URL,
    DEFAULT_STATS_FIELDS,
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
from mango_mvp.services.controlled_call_scope import (
    CONTROLLED_CALL_RUN_AUTHORITY_SCHEMA,
    ControlledCaptureRequest,
    ControlledCallScope,
    load_controlled_capture_request,
    load_controlled_call_allowlist,
)


SCHEMA_VERSION = "mango_calls_two_processes_v1"
LEGACY_CAPTURE_WINDOW_CERTIFICATE_SCHEMA = "mango_capture_window_certificate_v1"
CAPTURE_WINDOW_CERTIFICATE_SCHEMA = "mango_capture_window_certificate_v2"
SEQUENTIAL_PIPELINE_STAGES = (
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

REQUIRED_RUNTIME_CALL_RECORD_COLUMNS = frozenset(
    {
        "id",
        "source_file",
        "source_filename",
        "source_call_id",
        "audio_codec",
        "sample_rate",
        "channels",
        "duration_sec",
        "phone",
        "manager_name",
        "direction",
        "started_at",
        "transcription_status",
        "resolve_status",
        "analysis_status",
        "sync_status",
        "transcribe_attempts",
        "resolve_attempts",
        "analyze_attempts",
        "sync_attempts",
        "pipeline_stage",
        "pipeline_worker_id",
        "pipeline_claimed_at",
        "analysis_worker_id",
        "analysis_claimed_at",
        "next_retry_at",
        "dead_letter_stage",
        "transcript_manager",
        "transcript_client",
        "transcript_text",
        "transcript_variants_json",
        "resolve_json",
        "resolve_quality_score",
        "analysis_json",
        "amocrm_contact_id",
        "amocrm_lead_id",
        "last_error",
        "created_at",
        "updated_at",
    }
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
    codex_service_tier: str = "flex"
    processing_scope: str = "service"
    runtime_authority_mode: str = "service_cutover"
    controlled_capture_request_path: Optional[Path] = None
    controlled_capture_request_sha256: Optional[str] = None
    production_cursor_guard_path: Optional[Path] = None
    controlled_call_allowlist_path: Optional[Path] = None
    controlled_call_allowlist_sha256: Optional[str] = None
    mlx_whisper_snapshot_path: Optional[Path] = None
    heavy_stage_timeout_seconds: int = 4 * 60 * 60
    expected_code_sha: Optional[str] = None
    host_id_path: Optional[Path] = None
    cutover_manifest_path: Optional[Path] = None
    previous_host_snapshot_path: Optional[Path] = None
    expected_active_host_id: Optional[str] = None
    expected_previous_host_id: Optional[str] = None
    cutover_proof_max_age_minutes: int = 90
    max_catch_up_days: int = 7
    require_cutover_authority: bool = False
    strict_ready_provenance: bool = False
    publication_root: Optional[Path] = None
    # Internal, one-run bridge for a transfer cursor produced before strict
    # capture-window certificates existed.  Runtime JSON cannot enable it;
    # entrypoints set it only after cutover authority and the transferred
    # manifest prefix have both been verified.
    legacy_cursor_migration_mode: bool = False

    @classmethod
    def from_json(
        cls, path: Path, *, expected_sha256: str | None = None
    ) -> "CallsTwoProcessesConfig":
        payload = load_owner_only_runtime_config(
            path, expected_sha256=expected_sha256
        )
        require_cutover_authority, strict_ready_provenance = strict_service_flags(
            payload
        )
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
            codex_service_tier=str(
                payload.get("codex_service_tier") or "flex"
            ).strip(),
            processing_scope=str(
                payload.get("processing_scope") or "service"
            ).strip().lower(),
            runtime_authority_mode=str(
                payload.get("runtime_authority_mode") or "service_cutover"
            ).strip().lower(),
            controlled_capture_request_path=(
                Path(str(payload["controlled_capture_request_path"])).expanduser()
                if payload.get("controlled_capture_request_path")
                else None
            ),
            controlled_capture_request_sha256=optional_text(
                payload.get("controlled_capture_request_sha256")
            ),
            production_cursor_guard_path=(
                Path(str(payload["production_cursor_guard_path"])).expanduser()
                if payload.get("production_cursor_guard_path")
                else None
            ),
            controlled_call_allowlist_path=(
                Path(str(payload["controlled_call_allowlist_path"])).expanduser()
                if payload.get("controlled_call_allowlist_path")
                else None
            ),
            controlled_call_allowlist_sha256=optional_text(
                payload.get("controlled_call_allowlist_sha256")
            ),
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
            previous_host_snapshot_path=(
                Path(str(payload["previous_host_snapshot_path"])).expanduser()
                if payload.get("previous_host_snapshot_path")
                else None
            ),
            expected_active_host_id=optional_text(
                payload.get("expected_active_host_id")
            ),
            expected_previous_host_id=optional_text(
                payload.get("expected_previous_host_id")
            ),
            cutover_proof_max_age_minutes=int(
                payload.get("cutover_proof_max_age_minutes", 90)
            ),
            max_catch_up_days=int(payload.get("max_catch_up_days", 7)),
            require_cutover_authority=require_cutover_authority,
            strict_ready_provenance=strict_ready_provenance,
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
        if self.codex_service_tier != "flex":
            raise ValueError("codex_service_tier must be flex in strict M1 Calls runtime")
        if self.processing_scope not in {
            "service",
            "controlled_1_prepare",
            "controlled_1",
        }:
            raise ValueError("processing_scope is invalid")
        if self.runtime_authority_mode not in {
            "service_cutover",
            "isolated_controlled",
        }:
            raise ValueError("runtime_authority_mode is invalid")
        controlled_fields_present = bool(
            self.controlled_call_allowlist_path
            or self.controlled_call_allowlist_sha256
        )
        request_fields_present = bool(
            self.controlled_capture_request_path
            or self.controlled_capture_request_sha256
        )
        if self.processing_scope == "service" and (
            controlled_fields_present or request_fields_present
        ):
            raise ValueError("service scope must not configure controlled artifacts")
        if self.processing_scope in {"controlled_1_prepare", "controlled_1"}:
            if not self.strict_ready_provenance:
                raise ValueError("controlled runtime requires strict provenance")
            if self.stage_limit != 1:
                raise ValueError("controlled runtime requires stage_limit=1")
            owner_local = (Path.home() / ".mango_local").resolve(strict=False)
            resolved_pipeline = self.pipeline_root.resolve(strict=False)
            if owner_local not in resolved_pipeline.parents:
                raise ValueError(
                    "controlled pipeline_root must stay under $HOME/.mango_local"
                )
            if (
                self.runtime_authority_mode == "isolated_controlled"
                and not resolved_pipeline.name.startswith("controlled-")
            ):
                raise ValueError(
                    "isolated pipeline_root name must start with controlled-"
                )
            request_required = bool(
                self.processing_scope == "controlled_1_prepare"
                or self.runtime_authority_mode == "isolated_controlled"
            )
            if request_required:
                if (
                    self.controlled_capture_request_path is None
                    or not self.controlled_capture_request_path.is_absolute()
                    or not re.fullmatch(
                        r"[0-9a-f]{64}",
                        self.controlled_capture_request_sha256 or "",
                    )
                ):
                    raise ValueError("controlled capture request is required")
                resolved_request = self.controlled_capture_request_path.resolve(
                    strict=False
                )
                if (
                    path_has_cloud_marker(resolved_request)
                    or owner_local not in resolved_request.parents
                ):
                    raise ValueError("controlled request must stay under $HOME/.mango_local")
            elif request_fields_present:
                raise ValueError("service cutover controlled scope forbids capture request")
        if self.processing_scope == "controlled_1_prepare" and controlled_fields_present:
            raise ValueError("controlled preparation must not contain a post-capture allowlist")
        if (
            self.processing_scope == "controlled_1_prepare"
            and self.runtime_authority_mode != "isolated_controlled"
        ):
            raise ValueError("controlled preparation requires isolated authority")
        if self.processing_scope == "controlled_1":
            if self.controlled_call_allowlist_path is None:
                raise ValueError("controlled_1 allowlist path is required")
            if not re.fullmatch(
                r"[0-9a-f]{64}", self.controlled_call_allowlist_sha256 or ""
            ):
                raise ValueError("controlled_1 allowlist sha256 is required")
            allowlist = self.controlled_call_allowlist_path
            if not allowlist.is_absolute():
                raise ValueError("controlled_1 allowlist path must be absolute")
            owner_local = (Path.home() / ".mango_local").resolve(strict=False)
            resolved_allowlist = allowlist.resolve(strict=False)
            if (
                path_has_cloud_marker(resolved_allowlist)
                or owner_local not in resolved_allowlist.parents
            ):
                raise ValueError("controlled_1 allowlist must stay under $HOME/.mango_local")
        if self.min_free_gib < 1:
            raise ValueError("min_free_gib must be at least 1")
        if self.heavy_stage_timeout_seconds < 60:
            raise ValueError("heavy_stage_timeout_seconds must be at least 60")
        if self.max_catch_up_days < 1:
            raise ValueError("max_catch_up_days must be positive")
        if (
            self.runtime_authority_mode == "service_cutover"
            and self.require_cutover_authority != self.strict_ready_provenance
        ):
            raise ValueError(
                "cutover authority and strict ready provenance must be enabled together"
            )
        if self.runtime_authority_mode == "isolated_controlled" and (
            self.require_cutover_authority or not self.strict_ready_provenance
        ):
            raise ValueError("isolated controlled authority flags are invalid")
        if (
            self.runtime_authority_mode == "isolated_controlled"
            and self.processing_scope not in {"controlled_1_prepare", "controlled_1"}
        ):
            raise ValueError("isolated authority is only for controlled runtime")
        if self.runtime_authority_mode == "isolated_controlled":
            isolated_root = self.pipeline_root.resolve(strict=False)
            if self.production_cursor_guard_path is None:
                raise ValueError(
                    "isolated controlled production_cursor_guard_path is required"
                )
            production_cursor = self.production_cursor_guard_path.resolve(
                strict=False
            )
            if (
                not self.production_cursor_guard_path.is_absolute()
                or path_has_cloud_marker(production_cursor)
                or owner_local not in production_cursor.parents
                or production_cursor == self.cursor_path.resolve(strict=False)
                or isolated_root in production_cursor.parents
            ):
                raise ValueError(
                    "isolated production cursor guard must be owner-local and outside pipeline_root"
                )
            if self.publication_root is None:
                raise ValueError(
                    "isolated controlled publication_root is required"
                )
            for label, candidate in (
                ("timeline_allowed_root", self.timeline_allowed_root),
                ("timeline_db", self.timeline_db),
            ):
                resolved = candidate.resolve(strict=False)
                if isolated_root not in resolved.parents:
                    raise ValueError(
                        f"isolated controlled {label} must stay below pipeline_root"
                    )
            resolved_publication = self.publication_root.resolve(strict=False)
            if isolated_root not in resolved_publication.parents:
                raise ValueError(
                    "isolated controlled publication_root must stay below pipeline_root"
                )
        if self.legacy_cursor_migration_mode and not self.strict_ready_provenance:
            raise ValueError(
                "legacy cursor migration is available only in strict service mode"
            )
        if self.require_cutover_authority and not self.expected_code_sha:
            raise ValueError("expected_code_sha is required for cutover authority")
        if self.runtime_authority_mode == "isolated_controlled" and not re.fullmatch(
            r"[0-9a-f]{40}", self.expected_code_sha or ""
        ):
            raise ValueError("expected_code_sha is required for isolated authority")
        if self.runtime_authority_mode == "isolated_controlled" and not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]{2,127}",
            self.expected_active_host_id or "",
        ):
            raise ValueError("expected_active_host_id is required for isolated authority")
        if self.require_cutover_authority and not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]{2,127}",
            self.expected_active_host_id or "",
        ):
            raise ValueError(
                "expected_active_host_id is required for cutover authority"
            )
        if self.require_cutover_authority and not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]{2,127}",
            self.expected_previous_host_id or "",
        ):
            raise ValueError(
                "expected_previous_host_id is required for cutover authority"
            )
        if (
            self.require_cutover_authority
            and self.expected_active_host_id == self.expected_previous_host_id
        ):
            raise ValueError("active and previous host IDs must differ")
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
    def controlled_full_lock(self) -> Path:
        return self.pipeline_root / "locks" / "controlled_full.lock"

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
    def previous_host_snapshot_file(self) -> Path:
        return self.previous_host_snapshot_path or (
            self.pipeline_root / "state" / "previous_host_shutdown_snapshot.json"
        )

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


def controlled_call_scope_for_config(
    config: CallsTwoProcessesConfig,
) -> ControlledCallScope | None:
    if config.processing_scope != "controlled_1":
        return None
    if (
        config.controlled_call_allowlist_path is None
        or config.controlled_call_allowlist_sha256 is None
        or config.expected_code_sha is None
        or config.expected_active_host_id is None
    ):
        raise RuntimeError("controlled_1_configuration_incomplete")
    return load_controlled_call_allowlist(
        path=config.controlled_call_allowlist_path,
        expected_sha256=config.controlled_call_allowlist_sha256,
        expected_tenant_id=config.tenant_id,
        expected_code_sha=config.expected_code_sha,
        expected_host_id=config.expected_active_host_id,
        host_id_path=config.host_id_file,
        project_root=Path(__file__).resolve().parents[3],
    )


def controlled_capture_request_for_config(
    config: CallsTwoProcessesConfig,
) -> ControlledCaptureRequest | None:
    if config.runtime_authority_mode != "isolated_controlled":
        return None
    if (
        config.controlled_capture_request_path is None
        or config.controlled_capture_request_sha256 is None
        or config.expected_code_sha is None
        or config.expected_active_host_id is None
    ):
        raise RuntimeError("controlled_capture_request_configuration_incomplete")
    return load_controlled_capture_request(
        path=config.controlled_capture_request_path,
        expected_sha256=config.controlled_capture_request_sha256,
        expected_tenant_id=config.tenant_id,
        expected_code_sha=config.expected_code_sha,
        expected_host_id=config.expected_active_host_id,
        host_id_path=config.host_id_file,
        project_root=Path(__file__).resolve().parents[3],
        expected_pipeline_root=config.pipeline_root,
    )


@contextmanager
def controlled_worker_authority_environment(
    config: CallsTwoProcessesConfig,
    *,
    stage: str,
    run_id: str,
) -> Iterator[Mapping[str, str]]:
    if config.processing_scope != "controlled_1":
        yield {
            "MANGO_CALLS_CONTROLLED_RUN_AUTHORITY_PATH": "",
            "MANGO_CALLS_CONTROLLED_RUN_AUTHORITY_SHA256": "",
        }
        return
    scope = controlled_call_scope_for_config(config)
    assert scope is not None
    authority = controlled_read_only_cutover_authority_report(config)
    if authority.get("ok") is not True:
        raise RuntimeError("controlled_worker_cutover_authority_failed")
    legacy_cutover_sha256 = str(
        authority.get("verified_cutover_manifest_sha256") or ""
    )
    verified_authority_mode = str(
        authority.get("verified_authority_mode") or ""
    )
    verified_authority_path = str(
        authority.get("verified_authority_path") or ""
    )
    if (
        not verified_authority_mode
        and not verified_authority_path
        and re.fullmatch(r"[0-9a-f]{64}", legacy_cutover_sha256)
    ):
        verified_authority_mode = "service_cutover_manifest"
        verified_authority_path = str(config.cutover_manifest_file)
    verified_authority_sha256 = str(
        authority.get("verified_authority_sha256")
        or authority.get("verified_cutover_manifest_sha256")
        or ""
    )
    if (
        verified_authority_mode
        not in {"service_cutover_manifest", "isolated_controlled_request"}
        or not verified_authority_path
        or not re.fullmatch(r"[0-9a-f]{64}", verified_authority_sha256)
    ):
        raise RuntimeError("controlled_worker_authority_evidence_missing")
    controlled_call_bound_snapshot(config, scope)
    lock_metadata = read_json(config.pipeline_lock)
    if positive_int(lock_metadata.get("pid")) != os.getpid():
        raise RuntimeError("controlled_worker_requires_parent_pipeline_lock")
    authority_dir = config.pipeline_root / "state" / "controlled_worker_authority"
    authority_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    authority_dir.chmod(0o700)
    validate_owner_only_directory(
        authority_dir,
        label="controlled_worker_authority_directory",
        owner_only_mode=0o700,
    )
    now = datetime.now(timezone.utc)
    lifetime_seconds = min(
        max(60, config.heavy_stage_timeout_seconds + 300),
        6 * 60 * 60,
    )
    payload = {
        "schema_version": CONTROLLED_CALL_RUN_AUTHORITY_SCHEMA,
        "run_id": run_id,
        "stage": stage,
        "issued_at": now.isoformat(),
        "expires_at": (now + timedelta(seconds=lifetime_seconds)).isoformat(),
        "allowlist_sha256": scope.allowlist_sha256,
        "code_sha": scope.code_sha,
        "host_id": scope.host_id,
        "target_record_id": scope.target_record_id,
        "source_audio_sha256": scope.source_audio_sha256,
        "source_audio_size_bytes": scope.source_audio_size_bytes,
        "authority_mode": verified_authority_mode,
        "authority_evidence_path": verified_authority_path,
        "authority_evidence_sha256": verified_authority_sha256,
        "cutover_manifest_sha256": (
            verified_authority_sha256
            if verified_authority_mode == "service_cutover_manifest"
            else ""
        ),
        "pipeline_lock_path": str(config.pipeline_lock),
        "orchestrator_pid": os.getpid(),
    }
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    safe_run_id = re.sub(r"[^A-Za-z0-9_.-]", "_", run_id)
    safe_stage = re.sub(r"[^A-Za-z0-9_.-]", "_", stage)
    path = authority_dir / f"{safe_run_id}_{safe_stage}_{uuid.uuid4().hex}.json"
    atomic_replace_owner_only_bytes(
        path,
        raw,
        label="controlled_call_run_authority",
    )
    try:
        yield {
            "MANGO_CALLS_CONTROLLED_RUN_AUTHORITY_PATH": str(path),
            "MANGO_CALLS_CONTROLLED_RUN_AUTHORITY_SHA256": hashlib.sha256(
                raw
            ).hexdigest(),
        }
    finally:
        path.unlink(missing_ok=True)
        _fsync_directory(authority_dir)


def reject_controlled_call_broad_operation(
    config: CallsTwoProcessesConfig,
    operation: str,
) -> None:
    if config.processing_scope in {"controlled_1_prepare", "controlled_1"}:
        raise RuntimeError(
            f"{config.processing_scope}_forbids_broad_operation:{operation}"
        )


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
        previous_host_snapshot_path=config.previous_host_snapshot_file,
        expected_previous_host_id=config.expected_previous_host_id,
        expected_code_sha=config.expected_code_sha,
        project_root=Path(__file__).resolve().parents[3],
        proof_max_age_minutes=config.cutover_proof_max_age_minutes,
        require_fresh_previous_host_proof=False,
    ))
    if report.get("active_host_id") != config.expected_active_host_id:
        report["ok"] = False
        report["errors"] = [
            *list(report.get("errors") or ()),
            "expected_active_host_id_mismatch",
        ]
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
        fresh_report = dict(
            verify_cutover_authority(
                cutover_manifest_path=config.cutover_manifest_file,
                host_id_path=config.host_id_file,
                previous_host_snapshot_path=config.previous_host_snapshot_file,
                expected_previous_host_id=config.expected_previous_host_id,
                expected_code_sha=config.expected_code_sha,
                project_root=Path(__file__).resolve().parents[3],
                proof_max_age_minutes=config.cutover_proof_max_age_minutes,
                require_fresh_previous_host_proof=True,
            )
        )
        if fresh_report.get("ok") is not True:
            fresh_report["source_cursor_lineage_ok"] = False
            return fresh_report
        fresh_cutover_sha = sha256_file(config.cutover_manifest_file)
        if fresh_cutover_sha != cutover_sha:
            fresh_report["ok"] = False
            fresh_report["errors"] = [
                *list(fresh_report.get("errors") or ()),
                "cutover_manifest_changed_during_lineage_init",
            ]
            fresh_report["source_cursor_lineage_ok"] = False
            return fresh_report
        report = fresh_report
        expected_cursor_sha = str(report.get("source_cursor_sha256") or "")
        cutover_sha = fresh_cutover_sha
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
            marker_ok = sha256_file(config.cutover_manifest_file) == cutover_sha
    if not marker_ok:
        report["ok"] = False
        report["errors"] = [*list(report.get("errors") or ()), "source_cursor_lineage_unproven"]
    report["source_cursor_lineage_ok"] = marker_ok
    return report


def isolated_controlled_authority_report(
    config: CallsTwoProcessesConfig,
) -> Mapping[str, Any]:
    """Bind workers to one local request/cursor without claiming cutover."""

    try:
        request = controlled_capture_request_for_config(config)
        if request is None:
            raise RuntimeError("isolated controlled request is missing")
        cursor = read_json(config.cursor_path)
        verified_capture_window(config, cursor)
        selection = cursor.get("controlled_capture")
        if not isinstance(selection, Mapping):
            raise RuntimeError("controlled capture selection is missing")
        expected = {
            "request_sha256": request.request_sha256,
            "allowed_call_key": request.source_call_id,
            "expected_count": 1,
            "matched_count": 1,
            "attempted_other": 0,
            "since": request.since.isoformat(),
            "until": request.until.isoformat(),
        }
        if any(selection.get(key) != value for key, value in expected.items()):
            raise RuntimeError("controlled capture selection does not match request")
        if cursor.get("host_id") != request.host_id:
            raise RuntimeError("controlled capture host does not match request")
        evidence = read_stable_regular_bytes(
            request.request_path,
            label="controlled_capture_request",
            owner_only_mode=0o600,
        )
    except (OSError, RuntimeError, TypeError, ValueError):
        return {
            "ok": False,
            "errors": ["isolated_controlled_authority_unproven"],
            "source_cursor_lineage_ok": False,
            "controlled_cursor_binding_ok": False,
            "lineage_mode": "isolated_controlled",
            "shared_service_lineage_written": False,
        }
    digest = hashlib.sha256(evidence).hexdigest()
    ok = digest == request.request_sha256
    return {
        "ok": ok,
        "errors": [] if ok else ["controlled_capture_request_changed"],
        "active_host_id": request.host_id,
        "source_cursor_lineage_ok": False,
        "controlled_cursor_binding_ok": ok,
        "verified_authority_mode": "isolated_controlled_request",
        "verified_authority_path": str(request.request_path),
        "verified_authority_sha256": digest if ok else "",
        "lineage_mode": "isolated_controlled",
        "shared_service_lineage_written": False,
    }


def controlled_read_only_cutover_authority_report(
    config: CallsTwoProcessesConfig,
) -> Mapping[str, Any]:
    """Prove transferred cursor lineage without enabling service cutover."""
    if config.runtime_authority_mode == "isolated_controlled":
        return isolated_controlled_authority_report(config)
    if not config.require_cutover_authority or config.expected_code_sha is None:
        return {
            "ok": False,
            "errors": ["controlled_cutover_authority_required"],
            "source_cursor_lineage_ok": False,
            "controlled_cursor_binding_ok": False,
            "lineage_mode": "controlled_read_only",
            "shared_service_lineage_written": False,
        }
    try:
        cutover_before = read_stable_regular_bytes(
            config.cutover_manifest_file,
            label="controlled_cutover_manifest",
            owner_only_mode=0o600,
        )
    except RuntimeError:
        return {
            "ok": False,
            "errors": ["controlled_cutover_manifest_unreadable"],
            "source_cursor_lineage_ok": False,
            "controlled_cursor_binding_ok": False,
            "lineage_mode": "controlled_read_only",
            "shared_service_lineage_written": False,
        }
    report = dict(
        verify_cutover_authority(
            cutover_manifest_path=config.cutover_manifest_file,
            host_id_path=config.host_id_file,
            previous_host_snapshot_path=config.previous_host_snapshot_file,
            expected_previous_host_id=config.expected_previous_host_id,
            expected_code_sha=config.expected_code_sha,
            project_root=Path(__file__).resolve().parents[3],
            proof_max_age_minutes=config.cutover_proof_max_age_minutes,
            require_fresh_previous_host_proof=True,
        )
    )
    errors = list(report.get("errors") or ())
    if report.get("active_host_id") != config.expected_active_host_id:
        errors.append("expected_active_host_id_mismatch")
    expected_cursor_sha = str(report.get("source_cursor_sha256") or "")
    try:
        cursor = inspect_stable_regular_file(
            config.cursor_path,
            label="controlled_transferred_cursor",
            require_owner=True,
            require_single_link=True,
            owner_only_mode=0o600,
        )
    except RuntimeError:
        cursor = {}
        errors.append("controlled_transferred_cursor_unreadable")
    if (
        not re.fullmatch(r"[0-9a-f]{64}", expected_cursor_sha)
        or cursor.get("sha256") != expected_cursor_sha
    ):
        errors.append("controlled_source_cursor_mismatch")
    try:
        cutover_after = read_stable_regular_bytes(
            config.cutover_manifest_file,
            label="controlled_cutover_manifest",
            owner_only_mode=0o600,
        )
    except RuntimeError:
        cutover_after = b""
        errors.append("controlled_cutover_manifest_unreadable")
    if cutover_after != cutover_before:
        errors.append("controlled_cutover_manifest_changed_during_check")
    lineage_ok = bool(report.get("ok") is True and not errors)
    return {
        **report,
        "ok": lineage_ok,
        "errors": errors,
        "source_cursor_lineage_ok": lineage_ok,
        "controlled_cursor_binding_ok": lineage_ok,
        "verified_cutover_manifest_sha256": (
            hashlib.sha256(cutover_before).hexdigest() if lineage_ok else ""
        ),
        "verified_authority_mode": "service_cutover_manifest",
        "verified_authority_path": str(config.cutover_manifest_file),
        "verified_authority_sha256": (
            hashlib.sha256(cutover_before).hexdigest() if lineage_ok else ""
        ),
        "lineage_mode": "controlled_read_only",
        "shared_service_lineage_written": False,
    }


def working_db_is_authoritative(path: Path) -> bool:
    if not os.path.lexists(path):
        return False
    try:
        before = os.lstat(path)
        usable = bool(
            stat.S_ISREG(before.st_mode)
            and not path.is_symlink()
            and before.st_uid == os.getuid()
            and before.st_nlink == 1
            and before.st_size > 0
        )
        if usable:
            with closing(
                sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30)
            ) as connection:
                connection.execute("PRAGMA query_only=ON")
                table_info = list(
                    connection.execute("PRAGMA table_info(call_records)")
                )
                columns = {str(row[1]) for row in table_info}
                column_info = {str(row[1]): row for row in table_info}
                id_column = column_info.get("id")
                source_file_column = column_info.get("source_file")
                source_file_unique = False
                for index_row in connection.execute(
                    "PRAGMA index_list(call_records)"
                ):
                    if not bool(index_row[2]) or bool(index_row[4]):
                        continue
                    index_columns = [
                        str(row[0])
                        for row in connection.execute(
                            "SELECT name FROM pragma_index_info(?) "
                            "ORDER BY seqno",
                            (str(index_row[1]),),
                        )
                    ]
                    if index_columns == ["source_file"]:
                        source_file_unique = True
                        break
                critical_constraints_ok = bool(
                    id_column is not None
                    and str(id_column[2]).strip().upper() == "INTEGER"
                    and int(id_column[5]) == 1
                    and source_file_column is not None
                    and int(source_file_column[3]) == 1
                    and source_file_unique
                )
                usable = bool(
                    connection.execute("PRAGMA quick_check").fetchone()[0]
                    == "ok"
                    and connection.execute(
                        "SELECT COUNT(*) FROM sqlite_master "
                        "WHERE type='table' AND name='call_records'"
                    ).fetchone()[0]
                    == 1
                    and REQUIRED_RUNTIME_CALL_RECORD_COLUMNS.issubset(columns)
                    and critical_constraints_ok
                )
        after = os.lstat(path)
        return bool(
            usable
            and (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        )
    except (OSError, sqlite3.DatabaseError, TypeError):
        return False


def working_db_authority_issue(
    config: CallsTwoProcessesConfig,
) -> str | None:
    working_missing = not os.path.lexists(config.working_db)
    if not working_missing and working_db_is_authoritative(config.working_db):
        return None
    # A crash may leave the only durable ready generation in the publication
    # transaction before either canonical file exists.  Recover under the
    # publication lock before deciding whether this is a fresh runtime.  An
    # invalid existing working DB is never a valid bootstrap source, even when
    # no ready generation survives.
    with ready_publication_lock(config.ready_db):
        recover_ready_generation(config.ready_db, lock_held=True)
        ready_exists = bool(
            os.path.lexists(config.ready_db)
            or os.path.lexists(config.ready_manifest)
        )
    if working_missing and not ready_exists:
        return None
    if working_missing:
        return "working_db_missing_ready_generation_preserved"
    return (
        "working_db_invalid_ready_generation_preserved"
        if ready_exists
        else "working_db_invalid"
    )


def run_capture(
    config: CallsTwoProcessesConfig,
    *,
    since: Optional[str] = None,
    until: Optional[str] = None,
    capture_runner: CaptureRunner = None,
) -> Mapping[str, Any]:
    config.validate()
    reject_controlled_call_broad_operation(config, "capture")
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
            try:
                window_since, window_until = resolve_capture_window(
                    config, since=since, until=until
                )
                capture_config = config_for_capture_window(
                    config,
                    since=since,
                    window_since=window_since,
                    window_until=window_until,
                )
            except (RuntimeError, ValueError):
                return finalize_report(
                    config,
                    run_id,
                    "capture",
                    "failed",
                    "capture_enumeration_evidence_invalid",
                    {"authority": authority, "disk": disk, "lock": lock_info},
                )
            capture = capture_runner(
                capture_config,
                window_since,
                window_until,
            )
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
            try:
                enumeration_evidence_sha256 = capture_enumeration_exact_sha256(
                    capture,
                    expected_source_mode=(
                        "strict_service"
                        if config.strict_ready_provenance
                        else None
                    ),
                    expected_until=window_until,
                    expected_rolling_since=capture_rolling_window_start(
                        config,
                        since=window_since,
                        until=window_until,
                    ),
                )
                capture = certify_capture_window(
                    config,
                    capture,
                    requested_since=window_since,
                    requested_until=window_until,
                    enumeration_evidence_sha256=enumeration_evidence_sha256,
                )
            except (RuntimeError, ValueError):
                return finalize_report(
                    config,
                    run_id,
                    "capture",
                    "failed",
                    "capture_enumeration_evidence_invalid",
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
    reject_controlled_call_broad_operation(config, "pipeline")
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


def _sqlite_digest_value(value: Any) -> Mapping[str, Any]:
    if value is None:
        return {"type": "null", "value": None}
    if isinstance(value, bytes):
        return {
            "type": "blob",
            "size_bytes": len(value),
            "sha256": hashlib.sha256(value).hexdigest(),
        }
    if isinstance(value, bool):
        return {"type": "integer", "value": int(value)}
    if isinstance(value, int):
        return {"type": "integer", "value": value}
    if isinstance(value, float):
        return {"type": "real", "value": value}
    return {"type": "text", "value": str(value)}


def controlled_call_database_snapshot(
    path: Path,
    source_call_id: str,
    *,
    working_audio_dir: Path | None = None,
    require_source_audio: bool = False,
) -> Mapping[str, Any]:
    if not working_db_is_authoritative(path):
        raise RuntimeError("controlled_call_working_db_not_authoritative")
    with closing(
        sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30)
    ) as con:
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA query_only=ON")
        columns = [
            str(row[1])
            for row in con.execute("PRAGMA table_info(call_records)")
        ]
        if "id" not in columns or "source_call_id" not in columns:
            raise RuntimeError("controlled_call_working_db_schema_invalid")
        non_target_digest = hashlib.sha256()
        non_target_count = 0
        target_rows: list[dict[str, Any]] = []
        for raw_row in con.execute("SELECT * FROM call_records ORDER BY id ASC"):
            row = dict(raw_row)
            serialized = json.dumps(
                {
                    name: _sqlite_digest_value(row.get(name))
                    for name in columns
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            if str(row.get("source_call_id") or "") == source_call_id:
                target_rows.append(row)
            else:
                non_target_digest.update(len(serialized).to_bytes(8, "big"))
                non_target_digest.update(serialized)
                non_target_count += 1
        if len(target_rows) != 1:
            raise RuntimeError("controlled_call_database_match_must_be_exactly_one")
        target = target_rows[0]
        audio_evidence: Mapping[str, Any] = {
            "required": False,
            "ready": True,
        }
        if require_source_audio:
            if working_audio_dir is None:
                raise RuntimeError("controlled_call_working_audio_dir_required")
            raw_source_file = str(target.get("source_file") or "")
            source_file = Path(raw_source_file)
            if not source_file.is_absolute():
                raise RuntimeError("controlled_call_source_audio_must_be_absolute")
            root = working_audio_dir.resolve(strict=True)
            inspected = inspect_stable_regular_file(
                source_file,
                label="controlled_call_source_audio",
                require_owner=True,
            )
            resolved_source = inspected["resolved_path"]
            assert isinstance(resolved_source, Path)
            try:
                resolved_source.relative_to(root)
            except ValueError:
                raise RuntimeError(
                    "controlled_call_source_audio_outside_working_root"
                ) from None
            audio_evidence = {
                "required": True,
                "ready": True,
                "size_bytes": inspected["size_bytes"],
                "sha256": inspected["sha256"],
            }
        target_serialized = json.dumps(
            {
                name: _sqlite_digest_value(target.get(name))
                for name in columns
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        transcript = str(target.get("transcript_text") or "").encode("utf-8")
        analysis = str(target.get("analysis_json") or "").encode("utf-8")
        variants = str(target.get("transcript_variants_json") or "").encode(
            "utf-8"
        )
        return {
            "source_call_id": source_call_id,
            "target_row_sha256": hashlib.sha256(target_serialized).hexdigest(),
            "target": {
                "record_id": positive_int(target.get("id")),
                "channels": positive_int(target.get("channels")),
                "started_at": target.get("started_at"),
                "transcription_status": target.get("transcription_status"),
                "resolve_status": target.get("resolve_status"),
                "analysis_status": target.get("analysis_status"),
                "dead_letter_present": bool(
                    str(target.get("dead_letter_stage") or "").strip()
                ),
                "lease_present": any(
                    target.get(name)
                    for name in (
                        "pipeline_stage",
                        "pipeline_worker_id",
                        "pipeline_claimed_at",
                        "analysis_worker_id",
                        "analysis_claimed_at",
                    )
                ),
                "last_error_present": bool(
                    str(target.get("last_error") or "").strip()
                ),
                "transcribe_attempts": positive_int(
                    target.get("transcribe_attempts")
                ),
                "resolve_attempts": positive_int(target.get("resolve_attempts")),
                "analyze_attempts": positive_int(target.get("analyze_attempts")),
                "transcript_sha256": hashlib.sha256(transcript).hexdigest(),
                "transcript_variants_sha256": hashlib.sha256(variants).hexdigest(),
                "analysis_sha256": hashlib.sha256(analysis).hexdigest(),
                "ready_for_human_review": ready_row_is_complete(target),
                "source_audio": audio_evidence,
            },
            "non_target_row_count": non_target_count,
            "non_target_rows_sha256": non_target_digest.hexdigest(),
            "quick_check": str(con.execute("PRAGMA quick_check").fetchone()[0]),
        }


def controlled_call_bound_snapshot(
    config: CallsTwoProcessesConfig,
    scope: ControlledCallScope,
) -> Mapping[str, Any]:
    snapshot = controlled_call_database_snapshot(
        config.working_db,
        scope.source_call_id,
        working_audio_dir=config.working_audio_dir,
        require_source_audio=True,
    )
    target = snapshot.get("target")
    audio = target.get("source_audio") if isinstance(target, Mapping) else None
    if not (
        isinstance(target, Mapping)
        and isinstance(audio, Mapping)
        and target.get("record_id") == scope.target_record_id
        and audio.get("sha256") == scope.source_audio_sha256
        and audio.get("size_bytes") == scope.source_audio_size_bytes
    ):
        raise RuntimeError("controlled_one_allowlist_target_binding_mismatch")
    return snapshot


def create_isolated_controlled_allowlist(
    config: CallsTwoProcessesConfig,
    request: ControlledCaptureRequest,
) -> ControlledCallScope:
    """Promote a pre-download request after exactly one row was ingested."""

    if config.runtime_authority_mode != "isolated_controlled":
        raise RuntimeError("isolated controlled authority is required")
    snapshot = controlled_call_database_snapshot(
        config.working_db,
        request.source_call_id,
        working_audio_dir=config.working_audio_dir,
        require_source_audio=True,
    )
    target = snapshot.get("target")
    audio = target.get("source_audio") if isinstance(target, Mapping) else None
    if not isinstance(target, Mapping) or not isinstance(audio, Mapping):
        raise RuntimeError("controlled target/audio snapshot is incomplete")
    if positive_int(target.get("channels")) != 2:
        raise RuntimeError("controlled target audio must have exactly two channels")
    payload = {
        "schema_version": "mango_calls_controlled_allowlist_v2",
        "source_call_ids": [request.source_call_id],
        "target_record_id": target.get("record_id"),
        "source_audio_sha256": audio.get("sha256"),
        "source_audio_size_bytes": audio.get("size_bytes"),
        "tenant_id": request.tenant_id,
        "code_sha": request.code_sha,
        "host_id": request.host_id,
    }
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    parent = config.pipeline_root / "state" / "controlled"
    parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    parent.chmod(0o700)
    validate_owner_only_directory(
        parent,
        label="isolated_controlled_allowlist_parent",
        owner_only_mode=0o700,
    )
    path = parent / "allowlist.json"
    atomic_replace_owner_only_bytes(
        path,
        raw,
        label="isolated_controlled_allowlist",
    )
    digest = hashlib.sha256(raw).hexdigest()
    return load_controlled_call_allowlist(
        path=path,
        expected_sha256=digest,
        expected_tenant_id=request.tenant_id,
        expected_code_sha=request.code_sha,
        expected_host_id=request.host_id,
        host_id_path=config.host_id_file,
        project_root=Path(__file__).resolve().parents[3],
    )


@contextmanager
def controlled_audio_snapshot_environment(
    config: CallsTwoProcessesConfig,
    scope: ControlledCallScope,
    *,
    run_id: str,
) -> Iterator[
    tuple[Mapping[str, str], Mapping[str, Any], dict[str, Any]]
]:
    """Create one private, verified audio input shared by both ASR stages."""
    with closing(
        sqlite3.connect(
            f"file:{config.working_db}?mode=ro",
            uri=True,
            timeout=30,
        )
    ) as con:
        rows = con.execute(
            "SELECT id, source_file FROM call_records "
            "WHERE source_call_id=? ORDER BY id ASC",
            (scope.source_call_id,),
        ).fetchall()
    if (
        len(rows) != 1
        or positive_int(rows[0][0]) != scope.target_record_id
    ):
        raise RuntimeError("controlled_call_database_match_must_be_exactly_one")
    source = Path(str(rows[0][1] or ""))
    if not source.is_absolute():
        raise RuntimeError("controlled_call_source_audio_must_be_absolute")

    snapshots_root = config.pipeline_root / "state" / "controlled_runs"
    snapshots_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    snapshots_root.chmod(0o700)
    validate_owner_only_directory(
        snapshots_root,
        label="controlled_audio_snapshots_root",
        owner_only_mode=0o700,
    )
    if any(snapshots_root.iterdir()):
        raise RuntimeError(
            "controlled_call_audio_snapshot_stale_artifacts_present"
        )
    safe_run_id = re.sub(r"[^A-Za-z0-9_.-]", "_", run_id)
    run_root = snapshots_root / f"{safe_run_id}_{uuid.uuid4().hex}"
    run_root.mkdir(mode=0o700)
    validate_owner_only_directory(
        run_root,
        label="controlled_audio_snapshot_run_directory",
        owner_only_mode=0o700,
    )
    suffix = source.suffix if re.fullmatch(r"\.[A-Za-z0-9]{1,12}", source.suffix) else ".audio"
    snapshot_path = run_root / f"input{suffix}"
    cleanup: dict[str, Any] = {
        "ok": False,
        "snapshot_integrity_ok": False,
        "snapshot_removed": False,
        "run_directory_removed": False,
        "errors": [],
    }
    try:
        required_free = (
            int(config.min_free_gib * 1024**3)
            + scope.source_audio_size_bytes
        )
        if shutil.disk_usage(run_root).free < required_free:
            raise RuntimeError(
                "controlled_call_audio_snapshot_insufficient_disk_space"
            )
        evidence = copy_stable_regular_file_owner_only(
            source,
            snapshot_path,
            label="controlled_call_audio_snapshot",
            expected_sha256=scope.source_audio_sha256,
            expected_size_bytes=scope.source_audio_size_bytes,
        )
        yield (
            {
                "MANGO_CALLS_CONTROLLED_AUDIO_SNAPSHOT_PATH": str(snapshot_path),
                "MANGO_CALLS_CONTROLLED_AUDIO_SNAPSHOT_SHA256": scope.source_audio_sha256,
                "MANGO_CALLS_CONTROLLED_AUDIO_SNAPSHOT_SIZE_BYTES": str(
                    scope.source_audio_size_bytes
                ),
            },
            {
                "sha256": evidence.get("sha256"),
                "size_bytes": evidence.get("size_bytes"),
                "private_copy": True,
                "shared_by_asr_stages": True,
            },
            cleanup,
        )
    finally:
        errors: list[str] = []
        integrity_ok = False
        try:
            if os.path.lexists(snapshot_path):
                try:
                    final_evidence = inspect_stable_regular_file(
                        snapshot_path,
                        label="controlled_call_audio_snapshot",
                        require_owner=True,
                        require_single_link=True,
                        owner_only_mode=0o600,
                    )
                    integrity_ok = bool(
                        final_evidence.get("sha256")
                        == scope.source_audio_sha256
                        and final_evidence.get("size_bytes")
                        == scope.source_audio_size_bytes
                    )
                    if not integrity_ok:
                        errors.append(
                            "controlled_call_audio_snapshot_changed_during_run"
                        )
                except (RuntimeError, OSError):
                    errors.append(
                        "controlled_call_audio_snapshot_integrity_unproven"
                    )
            else:
                errors.append("controlled_call_audio_snapshot_missing_after_run")
        finally:
            try:
                snapshot_path.unlink(missing_ok=True)
            except OSError:
                errors.append("controlled_call_audio_snapshot_unlink_failed")
            snapshot_removed = not os.path.lexists(snapshot_path)
            if not snapshot_removed and (
                "controlled_call_audio_snapshot_unlink_failed" not in errors
            ):
                errors.append("controlled_call_audio_snapshot_unlink_failed")
            try:
                run_root.rmdir()
            except OSError:
                errors.append(
                    "controlled_call_audio_snapshot_cleanup_failed"
                )
            run_directory_removed = not os.path.lexists(run_root)
            cleanup.update(
                {
                    "ok": bool(
                        integrity_ok
                        and snapshot_removed
                        and run_directory_removed
                        and not errors
                    ),
                    "snapshot_integrity_ok": integrity_ok,
                    "snapshot_removed": snapshot_removed,
                    "run_directory_removed": run_directory_removed,
                    "errors": errors,
                }
            )


def _write_controlled_one_report(
    config: CallsTwoProcessesConfig,
    run_id: str,
    report: Mapping[str, Any],
) -> Mapping[str, Any]:
    config.reports_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    config.reports_dir.chmod(0o700)
    path = config.reports_dir / f"{run_id}_controlled_one.json"
    write_json(path, report)
    return {**dict(report), "report_path": str(path)}


def assert_no_live_controlled_heavy_worker(
    config: CallsTwoProcessesConfig,
) -> None:
    """Keep a parent crash from allowing a second heavy controlled run."""

    path = config.process_a_heartbeat_path
    if not os.path.lexists(path):
        return
    raw = read_stable_regular_bytes(
        path,
        label="controlled_heavy_heartbeat",
        owner_only_mode=0o600,
    )
    heartbeat = parse_json_object(raw.decode("utf-8"))
    pid = positive_int(heartbeat.get("pid"))
    stage = str(heartbeat.get("stage") or "")
    if stage in SEQUENTIAL_PIPELINE_STAGES and pid_exists(pid):
        raise RuntimeError("controlled_orphan_heavy_worker_live")


def run_controlled_one(
    config: CallsTwoProcessesConfig,
    *,
    command_runner: CommandRunner = None,
    capture_runner: Optional[Callable[..., Mapping[str, Any]]] = None,
    process_b_runner: Optional[
        Callable[[CallsTwoProcessesConfig], Mapping[str, Any]]
    ] = None,
    preview_runner: Optional[
        Callable[[Path, date], Mapping[str, Any]]
    ] = None,
    runtime_config_path: Optional[Path] = None,
    _pipeline_lock_info: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Any]:
    if config.processing_scope == "controlled_1_prepare":
        return run_controlled_one_from_request(
            config,
            command_runner=command_runner,
            capture_runner=capture_runner,
            process_b_runner=process_b_runner,
            preview_runner=preview_runner,
            runtime_config_path=runtime_config_path,
        )
    started = datetime.now(timezone.utc)
    run_id = new_calls_run_id(started)
    runner = command_runner or run_command
    config_valid = False
    try:
        config.validate()
        config_valid = True
        if config.processing_scope != "controlled_1":
            raise RuntimeError("controlled_one_requires_controlled_1_scope")
        assert_no_live_controlled_heavy_worker(config)
        with controlled_pipeline_lease(
            config,
            inherited_lock_info=_pipeline_lock_info,
        ) as lock_info:
            scope_before = controlled_call_scope_for_config(config)
            assert scope_before is not None
            authority = controlled_read_only_cutover_authority_report(config)
            if authority.get("ok") is not True:
                raise RuntimeError("controlled_one_cutover_authority_failed")
            before = controlled_call_bound_snapshot(config, scope_before)
            if before.get("quick_check") != "ok":
                raise RuntimeError("controlled_one_database_quick_check_failed")
            before_target = before.get("target")
            disk = disk_preflight(config)
            environment = environment_preflight(
                config,
                run_commands=runner is run_command,
                require_mango_credentials=False,
            )
            if disk.get("ok") is not True:
                raise RuntimeError("controlled_one_insufficient_disk_space")
            if environment.get("ok") is not True:
                raise RuntimeError("controlled_one_environment_preflight_failed")
            with controlled_audio_snapshot_environment(
                config,
                scope_before,
                run_id=run_id,
            ) as (
                audio_snapshot_env,
                audio_snapshot,
                audio_snapshot_cleanup,
            ):
                base_env = {
                    **worker_environment(config),
                    **audio_snapshot_env,
                }
                stage_reports = run_sequential_pipeline_workers(
                    config,
                    base_env,
                    runner,
                    include_llm=True,
                    run_id=run_id,
                    cycle_deadline=(
                        time.monotonic() + config.heavy_stage_timeout_seconds
                    ),
                )
            scope_after = controlled_call_scope_for_config(config)
            if scope_after != scope_before:
                raise RuntimeError("controlled_call_scope_changed_during_run")
            after = controlled_call_bound_snapshot(config, scope_before)
            if after.get("quick_check") != "ok":
                raise RuntimeError("controlled_one_database_quick_check_failed")
            non_target_unchanged = bool(
                before.get("non_target_row_count")
                == after.get("non_target_row_count")
                and before.get("non_target_rows_sha256")
                == after.get("non_target_rows_sha256")
            )
            if not non_target_unchanged:
                raise RuntimeError("controlled_one_non_target_rows_changed")
            after_target = after.get("target")
            source_audio_unchanged = bool(
                isinstance(before_target, Mapping)
                and isinstance(after_target, Mapping)
                and before_target.get("source_audio")
                == after_target.get("source_audio")
            )
            if not source_audio_unchanged:
                raise RuntimeError("controlled_one_source_audio_changed")
            compacted = compact_command_reports(stage_reports)
            failed = [
                item for item in compacted if int(item.get("rc") or 0) != 0
            ]
            target = after_target
            machine_ready = bool(
                isinstance(target, Mapping)
                and target.get("ready_for_human_review") is True
            )
            before_ready = bool(
                isinstance(before_target, Mapping)
                and before_target.get("ready_for_human_review") is True
            )
            processed_by_stage = {
                str(item.get("command") or ""): int(
                    (item.get("metrics") or {}).get("processed") or 0
                )
                for item in compacted
                if isinstance(item.get("metrics"), Mapping)
            }
            runtime_by_stage = {
                str(item.get("command") or ""): (
                    (item.get("metrics") or {}).get("runtime_receipt") or {}
                )
                for item in compacted
                if isinstance(item.get("metrics"), Mapping)
                and isinstance(
                    (item.get("metrics") or {}).get("runtime_receipt"),
                    Mapping,
                )
            }
            processed_total = sum(processed_by_stage.values())
            target_row_unchanged = bool(
                before.get("target_row_sha256")
                == after.get("target_row_sha256")
            )
            if processed_total == 0 and not target_row_unchanged:
                raise RuntimeError(
                    "controlled_one_zero_work_target_row_changed"
                )
            expected_stage_commands = {
                f"worker:{stage}" for stage in SEQUENTIAL_PIPELINE_STAGES
            }
            transcribe_receipt = runtime_by_stage.get("worker:transcribe") or {}
            backfill_receipt = runtime_by_stage.get(
                "worker:backfill-second-asr"
            ) or {}
            transcribe_providers = transcribe_receipt.get("provider_invocations")
            backfill_providers = backfill_receipt.get("provider_invocations")
            fresh_asr_sequence_proven = bool(
                isinstance(transcribe_providers, Mapping)
                and isinstance(backfill_providers, Mapping)
                and positive_int(transcribe_providers.get("mlx")) > 0
                and positive_int(backfill_providers.get("gigaam")) > 0
                and positive_int(
                    transcribe_receipt.get("mlx_cache_release_attempts")
                )
                > 0
                and positive_int(
                    transcribe_receipt.get("mlx_cache_release_successes")
                )
                == positive_int(
                    transcribe_receipt.get("mlx_cache_release_attempts")
                )
            )
            pilot_transition_proven = bool(
                not before_ready
                and machine_ready
                and not failed
                and source_audio_unchanged
                and fresh_asr_sequence_proven
                and set(processed_by_stage) == expected_stage_commands
                and processed_by_stage["worker:transcribe"] == 1
                and processed_by_stage["worker:backfill-second-asr"] == 1
                and processed_by_stage["worker:resolve"] == 1
                and processed_by_stage["worker:analyze"] == 1
                and all(
                    int((item.get("metrics") or {}).get("failed") or 0) == 0
                    and int((item.get("metrics") or {}).get("success") or 0)
                    == int((item.get("metrics") or {}).get("processed") or 0)
                    for item in compacted
                    if isinstance(item.get("metrics"), Mapping)
                )
            )
            execution_class = (
                "transitioned_to_ready"
                if (
                    not before_ready
                    and machine_ready
                    and processed_total > 0
                    and not failed
                )
                else "idempotent_noop"
                if (
                    before_ready
                    and machine_ready
                    and processed_total == 0
                    and target_row_unchanged
                )
                else "partial"
            )
            cleanup_ok = audio_snapshot_cleanup.get("ok") is True
            pilot_transition_proven = bool(
                pilot_transition_proven and cleanup_ok
            )
            status = (
                "failed"
                if failed or not cleanup_ok
                else "ok"
                if machine_ready
                else "partial"
            )
            stop_reason = (
                "worker_command_failed"
                if failed
                else "controlled_call_audio_snapshot_cleanup_failed"
                if not cleanup_ok
                else ""
                if machine_ready
                else "target_not_ready_for_human_review"
            )
            report = {
                "schema_version": "mango_calls_controlled_one_report_v1",
                "run_id": run_id,
                "process": "controlled_one",
                "status": status,
                "stop_reason": stop_reason,
                "processing_scope": config.processing_scope,
                "source_call_id": scope_before.source_call_id,
                "allowlist_sha256": scope_before.allowlist_sha256,
                "code_sha": scope_before.code_sha,
                "tenant_id": scope_before.tenant_id,
                "host_id": scope_before.host_id,
                "stage_order": list(SEQUENTIAL_PIPELINE_STAGES),
                "stages": compacted,
                "before": before,
                "after": after,
                "non_target_rows_unchanged": non_target_unchanged,
                "source_audio_unchanged": source_audio_unchanged,
                "asr_input_snapshot": audio_snapshot,
                "asr_input_snapshot_cleanup": audio_snapshot_cleanup,
                "target_row_unchanged": target_row_unchanged,
                "machine_result_ready_for_human_review": machine_ready,
                "execution_class": execution_class,
                "fresh_asr_sequence_proven": fresh_asr_sequence_proven,
                "pilot_transition_proven": pilot_transition_proven,
                "controlled_1_human_pass": False,
                "business_pass": False,
                "runtime_pass": False,
                "lock": lock_info,
                "safety": {
                    "captures_from_mango": False,
                    "runs_asr": any(
                        positive_int(count) > 0
                        for receipt in runtime_by_stage.values()
                        for providers in [receipt.get("provider_invocations")]
                        if isinstance(providers, Mapping)
                        for count in providers.values()
                    ),
                    "runs_resolve_analyze": any(
                        str(item.get("command") or "")
                        in {"worker:resolve", "worker:analyze"}
                        and int((item.get("metrics") or {}).get("processed") or 0)
                        > 0
                        for item in compacted
                        if isinstance(item.get("metrics"), Mapping)
                    ),
                    "writes_timeline_staging": False,
                    "writes_external_systems": False,
                    "writes_amo": False,
                    "publishes_google": False,
                    "publishes_yandex_disk": False,
                },
            }
            return _write_controlled_one_report(config, run_id, report)
    except LockBusy as exc:
        report = {
            "schema_version": "mango_calls_controlled_one_report_v1",
            "run_id": run_id,
            "process": "controlled_one",
            "status": "locked",
            "stop_reason": "pipeline_locked",
            "execution_class": "failed",
            "machine_result_ready_for_human_review": False,
            "fresh_asr_sequence_proven": False,
            "pilot_transition_proven": False,
            "controlled_1_human_pass": False,
            "business_pass": False,
            "runtime_pass": False,
            "lock": exc.metadata,
        }
    except Exception as exc:  # noqa: BLE001 - fail closed before the next stage.
        report = {
            "schema_version": "mango_calls_controlled_one_report_v1",
            "run_id": run_id,
            "process": "controlled_one",
            "status": "failed",
            "stop_reason": f"controlled_one_exception:{type(exc).__name__}",
            "execution_class": "failed",
            "machine_result_ready_for_human_review": False,
            "fresh_asr_sequence_proven": False,
            "pilot_transition_proven": False,
            "diagnostic": safe_exception_diagnostic(exc),
            "controlled_1_human_pass": False,
            "business_pass": False,
            "runtime_pass": False,
            "safety": {
                "writes_timeline_staging": False,
                "writes_external_systems": False,
                "writes_amo": False,
                "publishes_google": False,
                "publishes_yandex_disk": False,
            },
        }
    if not config_valid:
        return report
    try:
        return _write_controlled_one_report(config, run_id, report)
    except Exception:
        return report


def run_controlled_process_b(
    config: CallsTwoProcessesConfig,
) -> Mapping[str, Any]:
    """Import one sealed controlled drop only into its isolated Timeline DB."""

    config.validate()
    if (
        config.processing_scope != "controlled_1"
        or config.runtime_authority_mode != "isolated_controlled"
    ):
        raise RuntimeError("controlled process B requires isolated controlled scope")
    authority = isolated_controlled_authority_report(config)
    if authority.get("ok") is not True:
        raise RuntimeError("isolated controlled authority is not proven")
    try:
        with process_lease(
            config.process_b_lock,
            stale_seconds=config.stale_lock_seconds,
        ):
            with ready_publication_lock(config.ready_db):
                recover_ready_generation(config.ready_db, lock_held=True)
                return _run_process_b(config)
    except LockBusy as exc:
        return finalize_report(
            config,
            new_calls_run_id(datetime.now(timezone.utc)),
            "controlled_process_b",
            "locked",
            "process_b_locked",
            {"lock": exc.metadata},
        )


def controlled_timeline_effect_snapshot(
    config: CallsTwoProcessesConfig,
    source_call_id: str,
) -> Mapping[str, Any]:
    """Content-free durable receipt for one isolated Timeline target."""

    if not config.timeline_db.is_file() or config.timeline_db.is_symlink():
        return {
            "state": "absent",
            "total_rows": 0,
            "target_rows": 0,
            "mango_rows": 0,
        }
    timeline_source_id = f"provider:{source_call_id}"
    with closing(
        sqlite3.connect(
            f"file:{config.timeline_db}?mode=ro",
            uri=True,
            timeout=30,
        )
    ) as con:
        logical_digest = hashlib.sha256()
        for statement in con.iterdump():
            logical_digest.update(statement.encode("utf-8"))
            logical_digest.update(b"\n")
        total_rows = int(
            con.execute("SELECT COUNT(*) FROM timeline_events").fetchone()[0]
        )
        mango_rows = int(
            con.execute(
                "SELECT COUNT(*) FROM timeline_events "
                "WHERE source_system='mango_processed_summary' "
                "AND tenant_id=? AND event_type='mango_call'",
                (config.tenant_id,),
            ).fetchone()[0]
        )
        target_rows = int(
            con.execute(
                "SELECT COUNT(*) FROM timeline_events "
                "WHERE source_system='mango_processed_summary' "
                "AND tenant_id=? AND event_type='mango_call' AND source_id=?",
                (config.tenant_id, timeline_source_id),
            ).fetchone()[0]
        )
        quick_check = str(con.execute("PRAGMA quick_check").fetchone()[0])
    return {
        "state": "present",
        "total_rows": total_rows,
        "target_rows": target_rows,
        "mango_rows": mango_rows,
        "quick_check": quick_check,
        "logical_sha256": logical_digest.hexdigest(),
    }


def controlled_timeline_readback(
    config: CallsTwoProcessesConfig,
    source_call_id: str,
) -> Mapping[str, Any]:
    snapshot = controlled_timeline_effect_snapshot(config, source_call_id)
    if snapshot.get("state") != "present":
        return {"ok": False, **snapshot}
    timeline_source_id = f"provider:{source_call_id}"
    total_rows = positive_int(snapshot.get("total_rows"))
    mango_rows = positive_int(snapshot.get("mango_rows"))
    target_rows = positive_int(snapshot.get("target_rows"))
    quick_check = str(snapshot.get("quick_check") or "")
    return {
        "ok": (
            total_rows == 1
            and mango_rows == 1
            and target_rows == 1
            and quick_check == "ok"
        ),
        "total_rows": total_rows,
        "mango_rows": mango_rows,
        "target_rows": target_rows,
        "source_call_id_sha256": hashlib.sha256(
            source_call_id.encode("utf-8")
        ).hexdigest(),
        "timeline_source_id_sha256": hashlib.sha256(
            timeline_source_id.encode("utf-8")
        ).hexdigest(),
        "quick_check": quick_check,
    }


def run_controlled_local_previews(
    runtime_config_path: Path,
    day: date,
    expected_source_call_id: str | None = None,
) -> Mapping[str, Any]:
    """Build existing Google/Yandex local artifacts without external writes."""

    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        # The guarded worker executes ROOT/scripts/run_mango_calls_pipeline.py;
        # Python then exposes scripts/ but not its parent as an import root.
        # Add the verified repository root before importing the existing
        # coordinator namespace package.
        sys.path.insert(0, str(project_root))
    from scripts.run_mango_calls_publication_coordinator import run as run_preview

    google = run_preview(
        runtime_config_path,
        "current-plan",
        day=day,
        offline_only=True,
        controlled_preview=True,
    )
    yandex = run_preview(
        runtime_config_path,
        "daily-status",
        day=day,
        offline_only=True,
        controlled_preview=True,
    )
    google_plan = Path(str(google.get("plan") or ""))
    yandex_xlsx = Path(str(yandex.get("xlsx") or ""))
    yandex_manifest = Path(str(yandex.get("manifest") or ""))
    transcripts = yandex.get("transcripts")
    try:
        google_payload = json.loads(google_plan.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        google_payload = {}
    safe_rows = (
        google_payload.get("rows")
        if isinstance(google_payload, Mapping)
        else None
    )
    google_call_key_ok = bool(
        isinstance(safe_rows, list)
        and len(safe_rows) == 1
        and (
            expected_source_call_id is None
            or safe_rows[0].get("call_key") == expected_source_call_id
        )
    )
    artifacts_ok = bool(
        google.get("rows") == 1
        and google_call_key_ok
        and google_plan.is_file()
        and not google_plan.is_symlink()
        and yandex.get("rows") == 1
        and yandex_xlsx.is_file()
        and not yandex_xlsx.is_symlink()
        and yandex_manifest.is_file()
        and not yandex_manifest.is_symlink()
        and isinstance(transcripts, list)
        and len(transcripts) == 1
        and yandex.get("readback_ok") is True
    )
    return {
        "google": google,
        "yandex": yandex,
        "external_write": False,
        "tallanto_api_called": False,
        "artifacts_readback_ok": artifacts_ok,
        "ok": artifacts_ok and all(
            item.get("status") not in {"failed", "alert"}
            for item in (google, yandex)
        ),
    }


def controlled_production_cursor_snapshot(path: Path) -> Mapping[str, Any]:
    """Return content-free evidence for the production cursor sentinel."""

    path_hash = hashlib.sha256(str(path.resolve(strict=False)).encode()).hexdigest()
    if not os.path.lexists(path):
        return {"state": "absent", "path_sha256": path_hash}
    evidence = inspect_stable_regular_file(
        path,
        label="controlled_production_cursor_guard",
        require_owner=True,
        require_single_link=True,
        owner_only_mode=0o600,
    )
    current = os.lstat(path)
    return {
        "state": "present",
        "path_sha256": path_hash,
        "device": current.st_dev,
        "inode": current.st_ino,
        "size_bytes": evidence["size_bytes"],
        "mtime_ns": current.st_mtime_ns,
        "sha256": evidence["sha256"],
    }


def run_controlled_one_from_request(
    config: CallsTwoProcessesConfig,
    *,
    command_runner: CommandRunner = None,
    capture_runner: Optional[Callable[..., Mapping[str, Any]]] = None,
    process_b_runner: Optional[
        Callable[[CallsTwoProcessesConfig], Mapping[str, Any]]
    ] = None,
    preview_runner: Optional[
        Callable[[Path, date], Mapping[str, Any]]
    ] = None,
    runtime_config_path: Optional[Path] = None,
) -> Mapping[str, Any]:
    """Run one request from pre-download selection through local previews."""

    started = datetime.now(timezone.utc)
    run_id = new_calls_run_id(started)
    runner = command_runner or run_command
    capture_callable = capture_runner or capture_mango_window
    config_valid = False
    try:
        config.validate()
        config_valid = True
        if config.processing_scope != "controlled_1_prepare":
            raise RuntimeError("controlled preparation scope is required")
        request = controlled_capture_request_for_config(config)
        if request is None:
            raise RuntimeError("controlled capture request is missing")
        with process_lease(
            config.controlled_full_lock,
            stale_seconds=config.stale_lock_seconds,
        ) as full_lock_info:
            return _run_controlled_one_from_request_locked(
                config,
                request=request,
                run_id=run_id,
                runner=runner,
                capture_callable=capture_callable,
                process_b_runner=process_b_runner,
                preview_runner=preview_runner,
                runtime_config_path=runtime_config_path,
                full_lock_info=full_lock_info,
            )
    except LockBusy as exc:
        return {
            "schema_version": "mango_calls_controlled_one_full_report_v1",
            "run_id": run_id,
            "process": "controlled_one_full",
            "status": "locked",
            "stop_reason": "controlled_one_locked",
            "attempted": 0,
            "processed": 0,
            "attempted_other": 0,
            "lock": exc.metadata,
        }
    except Exception as exc:  # noqa: BLE001 - fail closed without call PII.
        report = {
            "schema_version": "mango_calls_controlled_one_full_report_v1",
            "run_id": run_id,
            "process": "controlled_one_full",
            "status": "failed",
            "stop_reason": f"controlled_one_full_exception:{type(exc).__name__}",
            "attempted": 0,
            "processed": 0,
            "attempted_other": 0,
            "diagnostic": safe_exception_diagnostic(exc),
            "controlled_1_human_pass": False,
            "business_pass": False,
            "runtime_pass": False,
            "safety": {
                "production_cursor_written": False,
                "writes_timeline_staging": False,
                "writes_external_systems": False,
                "writes_amo": False,
            },
        }
        if not config_valid:
            return report
        try:
            return _write_controlled_one_report(config, run_id, report)
        except Exception:
            return report


def _run_controlled_one_from_request_locked(
    config: CallsTwoProcessesConfig,
    *,
    request: ControlledCaptureRequest,
    run_id: str,
    runner: Any,
    capture_callable: Callable[..., Mapping[str, Any]],
    process_b_runner: Optional[
        Callable[[CallsTwoProcessesConfig], Mapping[str, Any]]
    ],
    preview_runner: Optional[Callable[[Path, date], Mapping[str, Any]]],
    runtime_config_path: Optional[Path],
    full_lock_info: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Execute one isolated request while the outer full-run lease is held."""
    effects: dict[str, Any] = {
        "attempted": 0,
        "attempted_other": 0,
        "downloaded": 0,
        "processed": 0,
        "writes_timeline_staging": False,
        "timeline_step_completed": False,
        "timeline_readback_ok": False,
        "local_previews_ok": False,
    }
    production_cursor_before: Mapping[str, Any] | None = None
    production_cursor_after: Mapping[str, Any] | None = None
    production_cursor_unchanged = False
    try:
        if config.production_cursor_guard_path is None:
            raise RuntimeError("controlled_production_cursor_guard_missing")
        assert_no_live_controlled_heavy_worker(config)
        production_cursor_before = controlled_production_cursor_snapshot(
            config.production_cursor_guard_path
        )
        disk = disk_preflight(config)
        environment = environment_preflight(
            config,
            run_commands=runner is run_command,
            require_mango_credentials=True,
        )
        if disk.get("ok") is not True:
            raise RuntimeError("controlled one has insufficient disk space")
        if environment.get("ok") is not True:
            raise RuntimeError("controlled one environment preflight failed")
        timeline_before = controlled_timeline_effect_snapshot(
            config,
            request.source_call_id,
        )
        with process_lease(
            config.pipeline_lock,
            stale_seconds=config.stale_lock_seconds,
        ) as lock_info:
            with process_lease(
                config.capture_lock,
                stale_seconds=config.stale_lock_seconds,
            ):
                capture = capture_callable(
                    config,
                    request.since,
                    request.until,
                    controlled_request=request,
                )
                selection = capture.get("controlled_capture")
                if isinstance(selection, Mapping):
                    for name in ("attempted", "attempted_other"):
                        value = selection.get(name)
                        if type(value) is int and value >= 0:
                            effects[name] = value
                downloaded = capture.get("downloaded")
                if type(downloaded) is int and downloaded >= 0:
                    effects["downloaded"] = downloaded
                if (
                    capture.get("status") not in {"ok", "partial"}
                    or capture.get("mango_enumeration_complete") is not True
                    or capture.get("enumeration_consistency_ok") is not True
                ):
                    raise RuntimeError("controlled capture failed")
                enumeration_sha256 = capture_enumeration_exact_sha256(
                    capture,
                    expected_source_mode="strict_service",
                    expected_until=request.until,
                    expected_rolling_since=request.since,
                )
                capture = certify_capture_window(
                    config,
                    capture,
                    requested_since=request.since,
                    requested_until=request.until,
                    enumeration_evidence_sha256=enumeration_sha256,
                )
                write_cursor(config.cursor_path, request.until, capture)
                metadata = dict(
                    prepare_ingest_inputs(
                        config,
                        manifest_end_offset=capture.get(
                            "manifest_end_offset"
                        ),
                        expected_manifest_sha256=str(
                            capture.get("manifest_snapshot_sha256") or ""
                        ),
                        controlled_request=request,
                    )
                )
                working_db_missing = not config.working_db.exists()
                prelude_env = dict(worker_environment(config))
                prelude_env.update(
                    {
                        "MANGO_CALLS_PROCESSING_SCOPE": "service",
                        "MANGO_CALLS_CONTROLLED_ALLOWLIST_PATH": "",
                        "MANGO_CALLS_CONTROLLED_ALLOWLIST_SHA256": "",
                        "MANGO_CALLS_CONTROLLED_TENANT_ID": "",
                        "MANGO_CALLS_CONTROLLED_CODE_SHA": "",
                        "MANGO_CALLS_CONTROLLED_HOST_ID": "",
                        "MANGO_CALLS_CONTROLLED_HOST_ID_PATH": "",
                    }
                )
                prelude_commands: list[Sequence[str]] = []
                if working_db_missing or metadata.get("audio_files"):
                    prelude_commands.append(cli_command(config, "init-db"))
                if metadata.get("audio_files"):
                    prelude_commands.append(
                        cli_command(
                            config,
                            "ingest",
                            "--recordings-dir",
                            str(config.working_audio_dir),
                            "--metadata-csv",
                            str(config.metadata_csv),
                        )
                    )
                prelude_reports: list[Mapping[str, Any]] = []
                deadline = time.monotonic() + config.heavy_stage_timeout_seconds
                for command in prelude_commands:
                    item = (
                        run_command(
                            command,
                            prelude_env,
                            config.working_dir,
                            deadline=deadline,
                            parent_lifeline=(
                                config.processing_scope
                                == "controlled_1_prepare"
                            ),
                        )
                        if runner is run_command
                        else runner(command, prelude_env, config.working_dir)
                    )
                    prelude_reports.append(item)
                    if int(item.get("rc") or 0) != 0:
                        raise RuntimeError("controlled ingest command failed")
                scope = create_isolated_controlled_allowlist(config, request)
            promoted = replace(
                config,
                processing_scope="controlled_1",
                controlled_call_allowlist_path=scope.allowlist_path,
                controlled_call_allowlist_sha256=scope.allowlist_sha256,
            )
            promoted.validate()
            heavy = run_controlled_one(
                promoted,
                command_runner=runner,
                _pipeline_lock_info=lock_info,
            )
        effects["processed"] = (
            1 if heavy.get("execution_class") == "transitioned_to_ready" else 0
        )
        if heavy.get("status") != "ok":
            raise RuntimeError("controlled heavy pipeline failed")
        cursor = read_json(promoted.cursor_path)
        manifest_end_offset = cursor.get("manifest_end_offset")
        if type(manifest_end_offset) is not int or manifest_end_offset < 0:
            raise RuntimeError("controlled capture manifest offset is invalid")
        counts = call_db_counts(promoted.working_db)
        drop = publish_ready_db_if_changed(
            promoted,
            counts,
            changed=bool(
                metadata.get("audio_files")
                or heavy.get("execution_class") == "transitioned_to_ready"
            ),
            run_id=run_id,
            capture_evidence=cursor,
            manifest_end_offset=manifest_end_offset,
            stage_reports=heavy.get("stages") or (),
            runtime_fingerprint=environment.get("runtime_fingerprint"),
        )
        if (
            drop.get("status") != "ready"
            or drop.get("consistency_ok") is not True
        ):
            raise RuntimeError("controlled ready drop is not green")
        effects["writes_timeline_staging"] = None
        try:
            timeline = (process_b_runner or run_controlled_process_b)(promoted)
        finally:
            try:
                timeline_after = controlled_timeline_effect_snapshot(
                    promoted,
                    request.source_call_id,
                )
                effects["writes_timeline_staging"] = bool(
                    timeline_after.get("state") != timeline_before.get("state")
                    or timeline_after.get("logical_sha256")
                    != timeline_before.get("logical_sha256")
                )
            except Exception:
                effects["writes_timeline_staging"] = None
                raise
        timeline_safety = timeline.get("safety")
        effects["timeline_step_completed"] = bool(
            timeline.get("status") in {"ok", "idle"}
        )
        reported_timeline_write = bool(
            isinstance(timeline_safety, Mapping)
            and timeline_safety.get("writes_timeline_staging") is True
        )
        if reported_timeline_write != effects["writes_timeline_staging"]:
            raise RuntimeError("controlled Timeline write receipt mismatch")
        if timeline.get("status") not in {"ok", "idle"}:
            raise RuntimeError("controlled Timeline staging import failed")
        timeline_readback = controlled_timeline_readback(
            promoted,
            request.source_call_id,
        )
        if timeline_readback.get("ok") is not True:
            raise RuntimeError("controlled Timeline target readback failed")
        effects["timeline_readback_ok"] = True
        heavy_after = heavy.get("after")
        heavy_target = (
            heavy_after.get("target")
            if isinstance(heavy_after, Mapping)
            else None
        )
        if not isinstance(heavy_target, Mapping) or not heavy_target.get(
            "started_at"
        ):
            raise RuntimeError("controlled target start time is missing")
        target_day = parse_datetime(
            str(heavy_target["started_at"])
        ).astimezone(ZoneInfo("Europe/Moscow")).date()
        previews: Mapping[str, Any]
        selected_preview_runner = preview_runner
        if selected_preview_runner is None and runtime_config_path is not None:
            selected_preview_runner = run_controlled_local_previews
        if selected_preview_runner is None or runtime_config_path is None:
            raise RuntimeError("controlled local preview runner is missing")
        if selected_preview_runner is run_controlled_local_previews:
            previews = selected_preview_runner(
                runtime_config_path,
                target_day,
                request.source_call_id,
            )
        else:
            previews = selected_preview_runner(runtime_config_path, target_day)
        if previews.get("ok") is not True:
            raise RuntimeError("controlled local previews failed")
        effects["local_previews_ok"] = True
        production_cursor_after = controlled_production_cursor_snapshot(
            config.production_cursor_guard_path
        )
        production_cursor_unchanged = bool(
            production_cursor_after == production_cursor_before
        )
        if not production_cursor_unchanged:
            raise RuntimeError("controlled_production_cursor_changed")
        selection = capture.get("controlled_capture")
        if not isinstance(selection, Mapping):
            raise RuntimeError("controlled capture counters are missing")
        report = {
            "schema_version": "mango_calls_controlled_one_full_report_v1",
            "run_id": run_id,
            "process": "controlled_one_full",
            "status": "ok",
            "stop_reason": "",
            "allowed_call_key": request.source_call_id,
            "expected_count": 1,
            "attempted": selection.get("attempted"),
            "processed": effects["processed"],
            "attempted_other": selection.get("attempted_other"),
            "downloaded": effects["downloaded"],
            "capture": capture,
            "ingest": {
                "metadata": metadata,
                "commands": compact_command_reports(prelude_reports),
            },
            "heavy": heavy,
            "ready": drop,
            "timeline_staging": timeline,
            "timeline_readback": timeline_readback,
            "local_previews": previews,
            "controlled_1_human_pass": False,
            "business_pass": False,
            "runtime_pass": False,
            "safety": {
                "old_mac_may_remain_active": True,
                "production_cursor_written": False,
                "production_cursor_unchanged": True,
                "production_cursor_guard": {
                    "before": production_cursor_before,
                    "after": production_cursor_after,
                },
                "writes_timeline_staging": effects["writes_timeline_staging"],
                "writes_external_systems": False,
                "writes_amo": False,
                "publishes_google": False,
                "publishes_yandex_disk": False,
            },
            "lock": lock_info,
            "full_lock": full_lock_info,
        }
        return _write_controlled_one_report(promoted, run_id, report)
    except Exception as exc:  # noqa: BLE001 - fail closed without call PII.
        if (
            production_cursor_before is not None
            and config.production_cursor_guard_path is not None
        ):
            try:
                production_cursor_after = controlled_production_cursor_snapshot(
                    config.production_cursor_guard_path
                )
                production_cursor_unchanged = bool(
                    production_cursor_after == production_cursor_before
                )
            except Exception:
                production_cursor_unchanged = False
        report = {
            "schema_version": "mango_calls_controlled_one_full_report_v1",
            "run_id": run_id,
            "process": "controlled_one_full",
            "status": "failed",
            "stop_reason": f"controlled_one_full_exception:{type(exc).__name__}",
            "attempted": effects["attempted"],
            "processed": effects["processed"],
            "attempted_other": effects["attempted_other"],
            "downloaded": effects["downloaded"],
            "effects": dict(effects),
            "diagnostic": safe_exception_diagnostic(exc),
            "controlled_1_human_pass": False,
            "business_pass": False,
            "runtime_pass": False,
            "safety": {
                "production_cursor_written": (
                    False if production_cursor_unchanged else None
                ),
                "production_cursor_unchanged": production_cursor_unchanged,
                "production_cursor_guard": {
                    "before": production_cursor_before,
                    "after": production_cursor_after,
                },
                "writes_timeline_staging": effects["writes_timeline_staging"],
                "writes_external_systems": False,
                "writes_amo": False,
            },
        }
        try:
            return _write_controlled_one_report(config, run_id, report)
        except Exception:
            return report


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
    raw_manifest_end_offset = cursor.get("manifest_end_offset")
    manifest_end_offset = (
        raw_manifest_end_offset
        if isinstance(raw_manifest_end_offset, int)
        and not isinstance(raw_manifest_end_offset, bool)
        else -1
    )
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


@contextmanager
def process_a_leases(
    config: CallsTwoProcessesConfig,
    *,
    pipeline_lock_info: Optional[Mapping[str, Any]],
    skip_capture: bool,
) -> Iterator[Mapping[str, Any]]:
    """Serialize process A capture behind pipeline, then capture, locks."""

    pipeline_lease = (
        nullcontext(pipeline_lock_info)
        if pipeline_lock_info is not None
        else process_lease(
            config.pipeline_lock,
            stale_seconds=config.stale_lock_seconds,
        )
    )
    with pipeline_lease as pipeline_info:
        base_info = dict(pipeline_info or {})
        if skip_capture:
            yield base_info
            return
        with process_lease(
            config.capture_lock,
            stale_seconds=config.stale_lock_seconds,
        ) as capture_info:
            yield {**base_info, "capture_lease": dict(capture_info)}


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
    reject_controlled_call_broad_operation(config, "process_a")
    command_runner = command_runner or run_command
    capture_runner = capture_runner or capture_mango_window
    started = datetime.now(timezone.utc)
    run_id = new_calls_run_id(started)
    try:
        lease = process_a_leases(
            config,
            pipeline_lock_info=pipeline_lock_info,
            skip_capture=skip_capture,
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
            if working_issue := working_db_authority_issue(config):
                return finalize_report(
                    config,
                    run_id,
                    "process_a",
                    "failed",
                    working_issue,
                    {
                        "authority": authority,
                        "drop": {
                            "status": "preserved",
                            "reason": working_issue,
                        },
                        "lock": lock_info,
                    },
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
            try:
                window_since, window_until = resolve_capture_window(
                    config, since=since, until=until
                )
                capture_config = (
                    config
                    if skip_capture
                    else config_for_capture_window(
                        config,
                        since=since,
                        window_since=window_since,
                        window_until=window_until,
                    )
                )
            except (RuntimeError, ValueError):
                return finalize_report(
                    config,
                    run_id,
                    "process_a",
                    "failed",
                    "capture_enumeration_evidence_invalid",
                    {"disk": disk, "environment": environment, "lock": lock_info},
                )
            effective_capture_evidence = capture_evidence
            if skip_capture and effective_capture_evidence is None:
                effective_capture_evidence = read_json(config.cursor_path)
            capture = (
                {
                    "status": "skipped",
                    "reason": "skip_capture",
                    **dict(effective_capture_evidence or {}),
                }
                if skip_capture
                else capture_runner(
                    capture_config,
                    window_since,
                    window_until,
                )
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
            try:
                source_evidence = capture.get("mango_enumeration_source")
                captured_until = window_until
                expected_rolling_since: Any = capture_rolling_window_start(
                    config,
                    since=window_since,
                    until=window_until,
                )
                if skip_capture:
                    if config.strict_ready_provenance:
                        (
                            _certified_since,
                            captured_until,
                            expected_rolling_since,
                        ) = verified_capture_window(config, capture)
                    else:
                        captured_until = capture.get("until") or (
                            source_evidence.get("until")
                            if isinstance(source_evidence, Mapping)
                            else None
                        )
                        expected_rolling_since = (
                            source_evidence.get("rolling_since")
                            if isinstance(source_evidence, Mapping)
                            else None
                        )
                enumeration_evidence_sha256 = capture_enumeration_exact_sha256(
                    capture,
                    expected_source_mode=(
                        "strict_service"
                        if config.strict_ready_provenance
                        else None
                    ),
                    expected_until=captured_until,
                    expected_rolling_since=expected_rolling_since,
                )
                if not skip_capture:
                    capture = certify_capture_window(
                        config,
                        capture,
                        requested_since=window_since,
                        requested_until=window_until,
                        enumeration_evidence_sha256=(
                            enumeration_evidence_sha256
                        ),
                    )
            except (RuntimeError, TypeError, ValueError):
                return finalize_report(
                    config,
                    run_id,
                    "process_a",
                    "failed",
                    "capture_enumeration_evidence_invalid",
                    {
                        "disk": disk,
                        "environment": environment,
                        "capture": capture,
                        "lock": lock_info,
                    },
                )
            effective_manifest_end_offset = manifest_end_offset
            if (
                effective_manifest_end_offset is None
                and skip_capture
                and config.strict_ready_provenance
            ):
                raw_manifest_end_offset = capture.get("manifest_end_offset")
                if (
                    isinstance(raw_manifest_end_offset, bool)
                    or not isinstance(raw_manifest_end_offset, int)
                    or raw_manifest_end_offset < 0
                ):
                    return finalize_report(
                        config,
                        run_id,
                        "process_a",
                        "failed",
                        "capture_enumeration_evidence_invalid",
                        {
                            "disk": disk,
                            "environment": environment,
                            "capture": capture,
                            "lock": lock_info,
                        },
                    )
                effective_manifest_end_offset = raw_manifest_end_offset
            metadata = dict(
                prepare_ingest_inputs(config)
                if effective_manifest_end_offset is None
                else prepare_ingest_inputs(
                    config,
                    manifest_end_offset=effective_manifest_end_offset,
                    expected_manifest_sha256=optional_text(
                        capture.get("manifest_snapshot_sha256")
                    ),
                )
            )
            metadata["db_open_work"] = call_db_has_open_work(config.working_db)
            if positive_int(metadata.get("legacy_topology_blocked")):
                return finalize_report(
                    config,
                    run_id,
                    "process_a",
                    "partial",
                    "legacy_asr_topology_blocked",
                    {
                        "disk": disk,
                        "environment": environment,
                        "capture": capture,
                        "metadata": metadata,
                        "workers": (),
                        "lock": lock_info,
                    },
                )
            worker_reports: list[Mapping[str, Any]] = []
            working_db_missing = not config.working_db.exists()
            if not skip_workers and (
                working_db_missing
                or metadata["audio_files"]
                or metadata["db_open_work"]
            ):
                heavy_cycle_deadline = (
                    time.monotonic() + config.heavy_stage_timeout_seconds
                )
                base_env = worker_environment(config)
                prelude_commands: list[Sequence[str]] = []
                if working_db_missing or metadata["audio_files"]:
                    prelude_commands.append(cli_command(config, "init-db"))
                if metadata["audio_files"]:
                    prelude_commands.append(
                        cli_command(
                            config,
                            "ingest",
                            "--recordings-dir",
                            str(config.working_audio_dir),
                            "--metadata-csv",
                            str(config.metadata_csv),
                        )
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
                if (
                    (metadata["audio_files"] or metadata["db_open_work"])
                    and not any(
                        int(report.get("rc", 0)) != 0
                        for report in worker_reports
                    )
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
            # A quiet database needs no network-dependent Resolve/Analyze
            # work.  Still seal its enumeration evidence so a zero-call day
            # can advance from the first proof to an honest closed verdict.
            remaining_open_work = call_db_has_open_work(config.working_db)
            if (
                not bool(environment.get("codex_network_ok"))
                and remaining_open_work
            ):
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
                        or positive_int(
                            metadata.get("legacy_topology_normalized")
                        )
                        or positive_int(
                            metadata.get("legacy_state_normalized")
                        )
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
    if config.runtime_authority_mode == "isolated_controlled":
        prior = read_json(config.process_b_cursor_path)
        controlled_request = controlled_capture_request_for_config(config)
        controlled_snapshot = (
            controlled_timeline_effect_snapshot(
                config,
                controlled_request.source_call_id,
            )
            if controlled_request is not None
            else {}
        )
        if (
            prior.get("schema_version")
            == "mango_calls_process_b_cursor_v1"
            and prior.get("sha256") == drop_fingerprint.get("sha256")
            and prior.get("size_bytes") == drop_fingerprint.get("size_bytes")
            and controlled_snapshot.get("state") == "present"
            and positive_int(controlled_snapshot.get("total_rows")) == 1
            and positive_int(controlled_snapshot.get("target_rows")) == 1
            and positive_int(controlled_snapshot.get("mango_rows")) == 1
            and controlled_snapshot.get("quick_check") == "ok"
        ):
            return finalize_report(
                config,
                run_id,
                "process_b",
                "idle",
                "controlled_drop_already_imported",
                {
                    "events": 0,
                    "drop": drop_fingerprint,
                    "producer_scan_mode": "controlled_exact_drop_reuse",
                    "import": {
                        "validation_ok": True,
                        "records_read": 0,
                        "records_accepted": 0,
                        "records_rejected": 0,
                        "writes_applied": 0,
                        "status_counts": {},
                        "source_system": "mango_processed_summary",
                    },
                },
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
        reject_controlled_call_broad_operation(config, "process_b")
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
    reject_controlled_call_broad_operation(config, "cycle")
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
def controlled_pipeline_lease(
    config: CallsTwoProcessesConfig,
    *,
    inherited_lock_info: Optional[Mapping[str, Any]] = None,
) -> Iterator[Mapping[str, Any]]:
    """Reuse only a currently held, same-process controlled pipeline lock."""

    if inherited_lock_info is None:
        with process_lease(
            config.pipeline_lock,
            stale_seconds=config.stale_lock_seconds,
        ) as acquired:
            yield acquired
        return
    expected = dict(inherited_lock_info)
    raw = read_stable_regular_bytes(
        config.pipeline_lock,
        label="controlled_inherited_pipeline_lock",
        owner_only_mode=0o600,
    )
    current = parse_json_object(raw.decode("utf-8"))
    if (
        current != expected
        or positive_int(current.get("pid")) != os.getpid()
    ):
        raise RuntimeError("controlled_inherited_pipeline_lock_mismatch")
    descriptor = os.open(
        config.pipeline_lock,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0),
    )
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield expected
        else:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            raise RuntimeError("controlled_inherited_pipeline_lock_not_held")
    finally:
        os.close(descriptor)


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


def _canonical_json_sha256(value: Any) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _mango_event_scalar_identity(event: TelephonyCallEvent) -> Mapping[str, Any]:
    recording_fields = {
        "recording_ref",
        "recording_id",
        "record_id",
        "records",
        "recording_url",
        "record_url",
        "recording_link",
    }
    return {
        "event_key": event.event_key,
        "provider_call_id": event.provider_call_id,
        "started_at": event.started_at.astimezone(timezone.utc).isoformat(),
        "ended_at": (
            event.ended_at.astimezone(timezone.utc).isoformat()
            if event.ended_at is not None
            else None
        ),
        "direction": event.direction.value,
        "client_phone": event.client_phone,
        "manager_ref": event.manager_ref,
        "raw_non_recording_payload": {
            str(key): value
            for key, value in event.raw_payload.items()
            if str(key) not in recording_fields
        },
    }


def normalize_mango_enumeration_rows(
    mapper: MangoOfficePayloadMapper,
    tenant: TenantRef,
    rows: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, TelephonyCallEvent], tuple[str, ...], str]:
    """Normalize one API pass deterministically and reject scalar conflicts."""

    groups: dict[str, list[TelephonyCallEvent]] = {}
    raw_call_keys: list[str] = []
    canonical_rows: list[str] = []
    for row in rows:
        event = mapper.from_payload(tenant=tenant, payload=row)
        groups.setdefault(event.event_key, []).append(event)
        raw_call_keys.append(event.provider_call_id)
        canonical_rows.append(
            json.dumps(
                dict(row),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
        )
    normalized: dict[str, TelephonyCallEvent] = {}
    for event_key, group in groups.items():
        identities = {
            json.dumps(
                _mango_event_scalar_identity(event),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
            for event in group
        }
        if len(identities) != 1:
            raise RuntimeError(
                "Mango enumeration contains conflicting duplicate call rows"
            )
        refs = sorted(
            {
                recording_id
                for event in group
                for recording_id in event_recording_ids(event)
            }
        )
        urls = sorted(
            {
                str(event.recording_url).strip()
                for event in group
                if str(event.recording_url or "").strip()
            }
        )
        selected = min(
            group,
            key=lambda event: json.dumps(
                dict(event.raw_payload),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ),
        )
        normalized[event_key] = replace(
            selected,
            recording_ref=refs[0] if refs else None,
            recording_refs=tuple(refs),
            recording_url=urls[0] if urls else None,
        )
    return (
        normalized,
        tuple(sorted(raw_call_keys)),
        _canonical_json_sha256(sorted(canonical_rows)),
    )


def mango_enumeration_pass_proof(
    *,
    pass_id: str,
    rows: Sequence[Mapping[str, Any]],
    intervals: Sequence[Mapping[str, Any]],
    events_by_key: Mapping[str, TelephonyCallEvent],
    call_key_multiset: Sequence[str],
    raw_rows_sha256: str,
    rolling_since: datetime,
    until: datetime,
) -> Mapping[str, Any]:
    call_keys = sorted(
        event.provider_call_id for event in events_by_key.values()
    )
    calls_by_day: dict[str, list[str]] = {}
    event_projection: list[Mapping[str, Any]] = []
    for event in events_by_key.values():
        day_key = event.started_at.astimezone(
            ZoneInfo("Europe/Moscow")
        ).date().isoformat()
        calls_by_day.setdefault(day_key, []).append(event.provider_call_id)
        event_projection.append(
            {
                **dict(_mango_event_scalar_identity(event)),
                "recording_ids": sorted(event_recording_ids(event)),
                "recording_url": event.recording_url,
            }
        )
    for values in calls_by_day.values():
        values[:] = sorted(values)
    chunks = [
        {
            "since": interval.get("since"),
            "until": interval.get("until"),
            "result_complete": interval.get("result_complete"),
            "rows": interval.get("rows"),
        }
        for interval in intervals
    ]
    recordable_unique_rows = sum(
        bool(event_recording_ids(event)) for event in events_by_key.values()
    )
    without_recording_rows = len(call_keys) - recordable_unique_rows
    proven_duplicate_rows = len(rows) - len(call_keys)
    raw_balance_ok = len(rows) == sum(
        (
            recordable_unique_rows,
            without_recording_rows,
            proven_duplicate_rows,
            0,  # quarantined rows fail the pass before proof creation
            0,  # malformed rows fail the pass before proof creation
            0,  # no unexplained row is permitted
        )
    )
    return {
        "pass_id": pass_id,
        "rolling_since": rolling_since.astimezone(timezone.utc).isoformat(),
        "until": until.astimezone(timezone.utc).isoformat(),
        "requests": len(chunks),
        "raw_rows": len(rows),
        "chunks": chunks,
        "partition_sha256": _canonical_json_sha256(
            [{"since": item["since"], "until": item["until"]} for item in chunks]
        ),
        "recordable_unique_rows": recordable_unique_rows,
        "without_recording_rows": without_recording_rows,
        "proven_duplicate_rows": proven_duplicate_rows,
        "quarantined_rows": 0,
        "error_rows": 0,
        "unexplained_rows": 0,
        "raw_balance_ok": raw_balance_ok,
        "call_key_multiset": list(call_key_multiset),
        "call_key_multiset_sha256": _canonical_json_sha256(
            list(call_key_multiset)
        ),
        "raw_rows_sha256": raw_rows_sha256,
        "call_keys": call_keys,
        "normalized_unique_count": len(call_keys),
        "call_keys_sha256": _canonical_json_sha256(call_keys),
        "calls_by_moscow_day": {
            key: calls_by_day[key] for key in sorted(calls_by_day)
        },
        "calls_by_moscow_day_sha256": _canonical_json_sha256(
            {key: calls_by_day[key] for key in sorted(calls_by_day)}
        ),
        "event_digest_sha256": _canonical_json_sha256(
            sorted(
                event_projection,
                key=lambda item: str(item.get("event_key") or ""),
            )
        ),
    }


def build_dual_enumeration_proof(
    config: CallsTwoProcessesConfig,
    *,
    rolling_since: datetime,
    until: datetime,
    primary: Mapping[str, Any],
    verification: Mapping[str, Any],
    official_list: Mapping[str, Any],
    proof_run_id: str,
    observed_at: datetime,
) -> Mapping[str, Any]:
    comparison_fields = (
        "normalized_unique_count",
        "call_keys",
        "call_keys_sha256",
        "calls_by_moscow_day",
        "calls_by_moscow_day_sha256",
        "event_digest_sha256",
    )
    comparison = {
        f"{field}_equal": primary.get(field) == verification.get(field)
        for field in comparison_fields
    }
    comparison["primary_raw_balance_ok"] = primary.get("raw_balance_ok") is True
    comparison["verification_raw_balance_ok"] = (
        verification.get("raw_balance_ok") is True
    )
    comparison["partition_sha256_different"] = primary.get(
        "partition_sha256"
    ) != verification.get("partition_sha256")
    comparison["official_list_equal"] = _official_list_proof_is_green(
        official_list,
        expected_call_keys=primary.get("call_keys") or (),
        expected_since=rolling_since,
        expected_until=until,
    )
    matched = all(comparison.values())
    mismatches = sorted(
        key.removesuffix("_equal")
        for key, value in comparison.items()
        if value is not True
    )
    proof = {
        "schema_version": DUAL_ENUMERATION_SCHEMA,
        "normalization_version": DUAL_ENUMERATION_NORMALIZATION,
        "tenant_id": config.tenant_id,
        "base_url": config.base_url,
        "fields_sha256": _canonical_json_sha256(DEFAULT_STATS_FIELDS),
        "rolling_since": rolling_since.astimezone(timezone.utc).isoformat(),
        "until": until.astimezone(timezone.utc).isoformat(),
        "proof_run_id": proof_run_id,
        "observed_at": observed_at.astimezone(timezone.utc).isoformat(),
        "passes_required": 2,
        "passes_completed": 2,
        "passes": [dict(primary), dict(verification)],
        "official_list": dict(official_list),
        "comparison": comparison,
        "enumeration_consistency_ok": matched,
        "mismatch_reason": "" if matched else ",".join(mismatches),
    }
    return {**proof, "proof_sha256": _canonical_json_sha256(proof)}


def _mango_wire_datetime(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Mango window boundary must be timezone-aware")
    return datetime.fromtimestamp(int(value.timestamp()), tz=timezone.utc)


def _mango_extended_wire_datetime(value: datetime) -> str:
    """Serialize an extended-statistics boundary in the PBX account timezone."""

    return _mango_wire_datetime(value).astimezone(
        ZoneInfo("Europe/Moscow")
    ).strftime("%d.%m.%Y %H:%M:%S")


def build_mango_official_list_proof(
    *,
    call_ids: Sequence[str],
    page_receipts: Sequence[Mapping[str, Any]],
    since: datetime,
    until: datetime,
) -> Mapping[str, Any]:
    canonical_ids = sorted(call_ids)
    proof = {
        "schema_version": MANGO_OFFICIAL_LIST_SCHEMA,
        "page_limit": MANGO_OFFICIAL_PAGE_LIMIT,
        "request": {
            "since_utc": _mango_wire_datetime(since).isoformat(),
            "until_utc": _mango_wire_datetime(until).isoformat(),
            "start_date": _mango_extended_wire_datetime(since),
            "end_date": _mango_extended_wire_datetime(until),
            "timezone": "Europe/Moscow",
            "datetime_format": "dd.mm.YYYY HH:MM:SS",
        },
        "pages": [dict(item) for item in page_receipts],
        "pages_count": len(page_receipts),
        "observed_count": len(canonical_ids),
        "call_keys": canonical_ids,
        "call_keys_sha256": _canonical_json_sha256(canonical_ids),
        "terminal_empty_page": bool(
            page_receipts and page_receipts[-1].get("rows") == 0
        ),
        "complete": bool(
            page_receipts and page_receipts[-1].get("rows") == 0
        ),
    }
    return {**proof, "proof_sha256": _canonical_json_sha256(proof)}


def poll_mango_official_list_pages(
    client: MangoOfficeClient,
    *,
    since: datetime,
    until: datetime,
    expected_call_ids: Sequence[str],
) -> Mapping[str, Any]:
    """Cross-check basic rows against the independent extended-statistics list."""

    offset = 0
    expected_ids = sorted(str(value) for value in expected_call_ids)
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("expected Mango call IDs must be unique")
    expected_id_set = set(expected_ids)
    observed_ids: list[str] = []
    page_receipts: list[Mapping[str, Any]] = []
    max_pages = max(
        1,
        (len(expected_ids) + MANGO_OFFICIAL_PAGE_LIMIT - 1)
        // MANGO_OFFICIAL_PAGE_LIMIT
        + 1,
    )
    for _page_number in range(max_pages):
        request_token = client.post_command(
            "/vpbx/stats/calls/request",
            {
                "start_date": _mango_extended_wire_datetime(since),
                "end_date": _mango_extended_wire_datetime(until),
                "limit": MANGO_OFFICIAL_PAGE_LIMIT,
                "offset": offset,
            },
        )
        if not isinstance(request_token, Mapping) or not str(
            request_token.get("key") or ""
        ).strip():
            raise RuntimeError("Mango extended stats request returned no key")
        result: Any = None
        for attempt in range(client.stats_result_poll_attempts):
            result = client.post_command(
                "/vpbx/stats/calls/result", request_token
            )
            if isinstance(result, Mapping) and result.get("status") in {
                "request",
                "work",
            }:
                if attempt + 1 >= client.stats_result_poll_attempts:
                    raise RuntimeError("Mango extended stats polling deadline exhausted")
                client.sleeper(client.stats_result_poll_interval_sec)
                continue
            break
        if (
            not isinstance(result, Mapping)
            or result.get("status") != "complete"
            or type(result.get("result")) is not int
            or result.get("result") != 1000
        ):
            raise RuntimeError("Mango extended stats result is not complete")
        raw_buckets = result.get("data")
        if not isinstance(raw_buckets, Sequence) or isinstance(
            raw_buckets, (str, bytes)
        ):
            raise RuntimeError("Mango extended stats data is invalid")
        page_ids: list[str] = []
        bucket_evidence: list[Mapping[str, Any]] = []
        for bucket in raw_buckets:
            if not isinstance(bucket, Mapping):
                raise RuntimeError("Mango extended stats bucket is invalid")
            bucket_total = bucket.get("total_calls_count")
            rows = bucket.get("list")
            if (
                not isinstance(rows, Sequence)
                or isinstance(rows, (str, bytes))
            ):
                raise RuntimeError("Mango extended stats total/page is invalid")
            if bucket_total is not None and (
                type(bucket_total) is not int or bucket_total < 0
            ):
                raise RuntimeError("Mango extended stats bucket total is invalid")
            period_raw = bucket.get("period")
            if period_raw is not None and (
                not isinstance(period_raw, str) or not period_raw.strip()
            ):
                raise RuntimeError("Mango extended stats bucket period is invalid")
            period = period_raw.strip() if period_raw is not None else ""
            bucket_ids: list[str] = []
            for row in rows:
                if not isinstance(row, Mapping):
                    raise RuntimeError("Mango extended stats row is invalid")
                call_id = str(row.get("entry_id") or "").strip()
                if not call_id:
                    raise RuntimeError("Mango extended stats row has no entry_id")
                page_ids.append(call_id)
                bucket_ids.append(call_id)
            bucket_evidence.append(
                {
                    "period": period or None,
                    "declared_total_calls_count": bucket_total,
                    "rows": len(bucket_ids),
                    "entry_ids": list(bucket_ids),
                    "entry_ids_sha256": _canonical_json_sha256(
                        sorted(bucket_ids)
                    ),
                }
            )
        if len(page_ids) > MANGO_OFFICIAL_PAGE_LIMIT:
            raise RuntimeError("Mango extended stats page exceeded requested limit")
        if len(page_ids) != len(set(page_ids)):
            raise RuntimeError("Mango extended stats page contains duplicate IDs")
        if set(page_ids).intersection(observed_ids):
            raise RuntimeError("Mango extended stats pages overlap")
        unexpected_ids = set(page_ids) - expected_id_set
        if unexpected_ids:
            raise RuntimeError("Mango extended stats returned an unexpected call")
        observed_ids.extend(page_ids)
        page_receipts.append(
            {
                "offset": offset,
                "rows": len(page_ids),
                "entry_ids": list(page_ids),
                "entry_ids_sha256": _canonical_json_sha256(sorted(page_ids)),
                "buckets": bucket_evidence,
                "status": "complete",
            }
        )
        if not page_ids:
            break
        if len(observed_ids) > len(expected_ids):
            raise RuntimeError("Mango extended stats returned too many calls")
        offset = len(observed_ids)
    else:
        raise RuntimeError("Mango extended stats terminal page is missing")
    if sorted(observed_ids) != expected_ids:
        raise RuntimeError("Mango basic and extended call sets differ")
    return build_mango_official_list_proof(
        call_ids=observed_ids,
        page_receipts=page_receipts,
        since=since,
        until=until,
    )


def capture_mango_window(
    config: CallsTwoProcessesConfig,
    since: datetime,
    until: datetime,
    *,
    controlled_request: ControlledCaptureRequest | None = None,
) -> Mapping[str, Any]:
    since = _mango_wire_datetime(since)
    until = _mango_wire_datetime(until)
    if controlled_request is not None:
        expected_request = controlled_capture_request_for_config(config)
        if (
            expected_request != controlled_request
            or since != controlled_request.since
            or until != controlled_request.until
            or controlled_request.pipeline_root
            != config.pipeline_root.resolve(strict=False)
        ):
            raise RuntimeError("controlled capture request/config mismatch")
    proof_observed_at = datetime.now(timezone.utc)
    proof_run_id = new_calls_run_id(proof_observed_at)
    api_key = os.getenv("MANGO_OFFICE_API_KEY", "").strip()
    api_salt = os.getenv("MANGO_OFFICE_API_SALT", "").strip()
    if not api_key or not api_salt:
        return {"status": "failed", "reason": "mango_credentials_missing"}
    if config.strict_ready_provenance:
        previous_cursor, previous_cursor_sha256 = read_capture_cursor_snapshot(
            config.cursor_path
        )
        if config.legacy_cursor_migration_mode:
            if not previous_cursor or not legacy_transfer_cursor_can_be_replaced(
                config,
                previous_cursor,
            ):
                raise RuntimeError(
                    "legacy capture cursor changed before migration"
                )
            # The transfer cursor proves only its frozen manifest prefix.  It
            # is not current API evidence and contributes no zero-day proof.
            zero_evidence_cursor: Mapping[str, Any] = {}
        else:
            if previous_cursor:
                verified_capture_window(
                    config,
                    previous_cursor,
                    allow_pre_dual_anchor=True,
                )
            zero_evidence_cursor = previous_cursor
    else:
        previous_cursor = read_json(config.cursor_path)
        previous_cursor_sha256 = None
        zero_evidence_cursor = previous_cursor
    credentials = MangoOfficeCredentials(api_key=api_key, api_salt=api_salt)
    primary_client = MangoOfficeClient(
        credentials=credentials,
        base_url=config.base_url,
        timeout_sec=60,
    )
    verification_client = (
        MangoOfficeClient(
            credentials=credentials,
            base_url=config.base_url,
            timeout_sec=60,
        )
        if config.strict_ready_provenance
        else primary_client
    )
    mapper = MangoOfficePayloadMapper()
    tenant = TenantRef(config.tenant_id)
    host_id = configured_host_id(
        config, required=config.require_cutover_authority
    )
    scoped_rows: list[tuple[Mapping[str, Any], str]] = []
    manifest_store = CaptureManifestStore(config.capture_manifest)
    if not os.path.lexists(config.capture_manifest) and capture_runtime_has_prior_state(config):
        raise RuntimeError("capture manifest is missing for an existing runtime")
    if not config.strict_ready_provenance:
        manifest_store.ensure_exists()
        manifest_store.recover_incomplete_tail()
    latest_manifest = manifest_store.latest_by_event_key()
    if controlled_request is not None:
        unexpected_manifest_calls = sorted(
            {
                entry.provider_call_id
                for entry in latest_manifest.values()
                if entry.provider_call_id != controlled_request.source_call_id
            }
        )
        unexpected_db_calls = sorted(
            read_ingested_call_ids(config.working_db)
            - {controlled_request.source_call_id}
        )
        if unexpected_manifest_calls or unexpected_db_calls:
            raise RuntimeError(
                "controlled isolated root contains another call"
            )
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
    base_window_start = capture_rolling_window_start(
        config,
        since=since,
        until=until,
    )
    retry_windows: list[tuple[datetime, datetime]] = []
    for entry in {
        entry.event_key: entry for entry in (*pending_entries, *recent_entries)
    }.values():
        started = parse_datetime(entry.started_at)
        ended = parse_datetime(entry.ended_at) if entry.ended_at else started + timedelta(hours=1)
        retry_start = started - overlap
        retry_end = min(ended + overlap, until)
        if retry_start < retry_end:
            retry_windows.append((retry_start, retry_end))
    for entry in due_expired_unknown:
        retry_start, retry_end = moscow_day_bounds_utc(
            parse_datetime(entry.started_at)
            .astimezone(ZoneInfo("Europe/Moscow"))
            .date()
        )
        retry_end = min(retry_end, until)
        if retry_start < retry_end:
            retry_windows.append((retry_start, retry_end))
    if controlled_request is not None:
        retry_windows = []

    def merge_windows(
        windows: Sequence[tuple[datetime, datetime]],
    ) -> list[tuple[datetime, datetime]]:
        merged: list[tuple[datetime, datetime]] = []
        for start, end in sorted(windows):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
            else:
                merged.append((start, end))
        return merged

    def poll_exact_windows(
        client: MangoOfficeClient,
        windows: Sequence[tuple[datetime, datetime]],
        *,
        scope: str,
        authority_pass: Optional[int] = None,
    ) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
        rows: list[Mapping[str, Any]] = []
        intervals: list[Mapping[str, Any]] = []
        for window_start, window_end in windows:
            chunk_start = window_start
            while chunk_start < window_end:
                chunk_end = min(
                    window_end,
                    chunk_start + timedelta(hours=config.api_window_hours),
                )
                chunk_rows = client.poll_call_history(
                    since=chunk_start,
                    until=chunk_end,
                )
                for row in chunk_rows if config.strict_ready_provenance else ():
                    started_raw = next(
                        (
                            row.get(name)
                            for name in (
                                "started_at",
                                "start_time",
                                "timestamp",
                                "date",
                                "start",
                            )
                            if row.get(name) not in (None, "")
                        ),
                        None,
                    )
                    if started_raw is None:
                        raise RuntimeError("Mango stats row has no start time")
                    row_started_at = parse_datetime(str(started_raw))
                    if not chunk_start <= row_started_at <= chunk_end:
                        raise RuntimeError(
                            "Mango stats row is outside its requested wire window"
                        )
                rows.extend(chunk_rows)
                interval: dict[str, Any] = {
                    "since": chunk_start.isoformat(),
                    "until": chunk_end.isoformat(),
                    "result_complete": True,
                    "rows": len(chunk_rows),
                    "scope": scope,
                }
                if authority_pass is not None:
                    interval["authority_pass"] = authority_pass
                intervals.append(interval)
                chunk_start = chunk_end
        return rows, intervals

    dual_enumeration: Optional[Mapping[str, Any]] = None
    covered_intervals: list[Mapping[str, Any]] = []
    auxiliary_rows: list[Mapping[str, Any]] = []
    enumeration_start = base_window_start
    if config.strict_ready_provenance:
        # The two authoritative passes are intentionally independent API
        # requests over one immutable rolling window.  Recovery is clipped
        # before that window, polled only once, and can never affect the
        # completeness proof.
        rolling_windows = [(base_window_start, until)]
        auxiliary_windows = merge_windows(
            [
                (start, min(end, base_window_start))
                for start, end in retry_windows
                if start < base_window_start
                and start < min(end, base_window_start)
            ]
        )
        if auxiliary_windows:
            enumeration_start = min(
                base_window_start,
                min(start for start, _end in auxiliary_windows),
            )
        try:
            primary_rows, primary_intervals = poll_exact_windows(
                primary_client,
                rolling_windows,
                scope="rolling_authority",
                authority_pass=1,
            )
        except (OSError, RuntimeError, TimeoutError, ValueError):
            return {
                "status": "failed",
                "reason": "primary_mango_enumeration_failed",
                "mango_enumeration_complete": False,
                "enumeration_consistency_ok": False,
            }
        primary_boundaries = {
            parse_datetime(str(interval[key]))
            for interval in primary_intervals
            for key in ("since", "until")
        }
        verification_split = _mango_wire_datetime(
            base_window_start + (until - base_window_start) / 3
        )
        if verification_split in primary_boundaries:
            verification_split += timedelta(seconds=1)
        if not base_window_start < verification_split < until:
            raise RuntimeError("independent Mango partition is unavailable")
        verification_windows = [
            (base_window_start, verification_split),
            (verification_split, until),
        ]
        try:
            (
                primary_events,
                primary_call_key_multiset,
                primary_raw_rows_sha256,
            ) = normalize_mango_enumeration_rows(
                mapper,
                tenant,
                primary_rows,
            )
        except (KeyError, RuntimeError, TypeError, ValueError):
            return {
                "status": "failed",
                "reason": "primary_mango_enumeration_invalid",
                "mango_enumeration_complete": False,
                "enumeration_consistency_ok": False,
            }
        try:
            verification_rows, verification_intervals = poll_exact_windows(
                verification_client,
                verification_windows,
                scope="rolling_authority",
                authority_pass=2,
            )
        except (OSError, RuntimeError, TimeoutError, ValueError):
            return {
                "status": "failed",
                "reason": "verification_mango_enumeration_failed",
                "mango_enumeration_complete": False,
                "enumeration_consistency_ok": False,
                "api_requests": len(primary_intervals),
                "api_rows_total": len(primary_rows),
            }
        try:
            (
                verification_events,
                verification_call_key_multiset,
                verification_raw_rows_sha256,
            ) = normalize_mango_enumeration_rows(
                mapper,
                tenant,
                verification_rows,
            )
        except (KeyError, RuntimeError, TypeError, ValueError):
            return {
                "status": "failed",
                "reason": "verification_mango_enumeration_invalid",
                "mango_enumeration_complete": False,
                "enumeration_consistency_ok": False,
            }
        primary_proof = mango_enumeration_pass_proof(
            pass_id="primary",
            rows=primary_rows,
            intervals=primary_intervals,
            events_by_key=primary_events,
            call_key_multiset=primary_call_key_multiset,
            raw_rows_sha256=primary_raw_rows_sha256,
            rolling_since=base_window_start,
            until=until,
        )
        verification_proof = mango_enumeration_pass_proof(
            pass_id="verification",
            rows=verification_rows,
            intervals=verification_intervals,
            events_by_key=verification_events,
            call_key_multiset=verification_call_key_multiset,
            raw_rows_sha256=verification_raw_rows_sha256,
            rolling_since=base_window_start,
            until=until,
        )
        try:
            official_list = poll_mango_official_list_pages(
                primary_client,
                since=base_window_start,
                until=until,
                expected_call_ids=[
                    event.provider_call_id for event in primary_events.values()
                ],
            )
        except (OSError, RuntimeError, TimeoutError, TypeError, ValueError):
            return {
                "status": "failed",
                "reason": "official_mango_list_verification_failed",
                "mango_enumeration_complete": False,
                "enumeration_consistency_ok": False,
                "api_requests": len(primary_intervals)
                + len(verification_intervals),
            }
        dual_enumeration = build_dual_enumeration_proof(
            config,
            rolling_since=base_window_start,
            until=until,
            primary=primary_proof,
            verification=verification_proof,
            official_list=official_list,
            proof_run_id=proof_run_id,
            observed_at=proof_observed_at,
        )
        if dual_enumeration.get("enumeration_consistency_ok") is not True:
            return {
                "status": "failed",
                "reason": "independent_mango_enumeration_mismatch",
                "mango_enumeration_complete": False,
                "enumeration_consistency_ok": False,
                "dual_enumeration": dual_enumeration,
                "api_requests": len(primary_intervals)
                + len(verification_intervals),
                "api_rows_total": len(primary_rows)
                + len(verification_rows),
            }
        try:
            auxiliary_rows, auxiliary_intervals = poll_exact_windows(
                primary_client,
                auxiliary_windows,
                scope="recovery_auxiliary",
            )
        except (OSError, RuntimeError, TimeoutError, ValueError):
            return {
                "status": "failed",
                "reason": "auxiliary_mango_enumeration_failed",
                "mango_enumeration_complete": False,
                "enumeration_consistency_ok": False,
                "dual_enumeration": dual_enumeration,
            }
        try:
            auxiliary_events, _aux_multiset, _aux_rows_sha = (
                normalize_mango_enumeration_rows(
                    mapper,
                    tenant,
                    auxiliary_rows,
                )
            )
        except (KeyError, RuntimeError, TypeError, ValueError):
            return {
                "status": "failed",
                "reason": "auxiliary_mango_enumeration_invalid",
                "mango_enumeration_complete": False,
                "enumeration_consistency_ok": False,
                "dual_enumeration": dual_enumeration,
            }
        unique_events = dict(primary_events)
        for event_key, event in auxiliary_events.items():
            unique_events.setdefault(event_key, event)
        authoritative_event_keys = set(primary_events)
        scoped_rows.extend((row, "rolling_authority") for row in primary_rows)
        scoped_rows.extend((row, "rolling_authority") for row in verification_rows)
        scoped_rows.extend((row, "recovery_auxiliary") for row in auxiliary_rows)
        covered_intervals = [
            *primary_intervals,
            *verification_intervals,
            *auxiliary_intervals,
        ]
    else:
        merged_windows = merge_windows(
            [(base_window_start, until), *retry_windows]
        )
        enumeration_start = min(start for start, _end in merged_windows)
        for window_start, window_end in merged_windows:
            chunk_start = window_start
            while chunk_start < window_end:
                chunk_end = min(
                    window_end,
                    chunk_start + timedelta(hours=config.api_window_hours),
                )
                if chunk_start < base_window_start < chunk_end:
                    chunk_end = base_window_start
                scope = (
                    "rolling_authority"
                    if chunk_start >= base_window_start
                    else "recovery_auxiliary"
                )
                rows, intervals = poll_exact_windows(
                    primary_client,
                    [(chunk_start, chunk_end)],
                    scope=scope,
                )
                scoped_rows.extend((row, scope) for row in rows)
                covered_intervals.extend(intervals)
                chunk_start = chunk_end
        unique_events = {}
        authoritative_event_keys: set[str] = set()
        for row, scope in scoped_rows:
            event = mapper.from_payload(tenant=tenant, payload=row)
            prior = unique_events.get(event.event_key)
            if prior is not None:
                refs = merge_recording_ids(
                    event_recording_ids(prior),
                    event_recording_ids(event),
                )
                event = replace(
                    event,
                    recording_ref=refs[0] if refs else None,
                    recording_refs=refs,
                )
            unique_events[event.event_key] = event
            if scope == "rolling_authority":
                authoritative_event_keys.add(event.event_key)
    api_requests = len(covered_intervals)
    if config.strict_ready_provenance:
        _current_cursor, current_cursor_sha256 = read_capture_cursor_snapshot(
            config.cursor_path
        )
        if current_cursor_sha256 != previous_cursor_sha256:
            raise RuntimeError("capture cursor changed during Mango enumeration")
        manifest_store.ensure_exists()
        manifest_store.recover_incomplete_tail()
    # Only events observed in the current Mango responses are authoritative
    # enumeration evidence.  Local recovery entries are still staged below,
    # but must not be allowed to expand or fabricate the API balance.
    enumerated_events = sorted(
        (unique_events[event_key] for event_key in authoritative_event_keys),
        key=lambda event: (event.started_at, event.provider_call_id),
    )
    if config.strict_ready_provenance:
        # After cutover only the transferred SQLite generations are authority;
        # neighbouring archives must never silently suppress a new API event.
        # A damaged processing DB must not stop the lightweight capture loop,
        # but it is never trusted for deduplication either.
        working_db_trusted = working_db_is_authoritative(config.working_db)
        trusted_ready_calls = trusted_ready_call_ids_for_capture(config)
        known_recordings: set[str] = set()
        known_calls = (
            (
                read_ingested_call_ids(config.working_db)
                if working_db_trusted
                else set()
            )
            | trusted_ready_calls
        )
        fully_ready_calls = trusted_ready_calls
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
    controlled_capture: Mapping[str, Any] | None = None
    if controlled_request is not None:
        matched = [
            event
            for event in enumerated_events
            if event.provider_call_id == controlled_request.source_call_id
        ]
        if len(enumerated_events) != 1 or len(matched) != 1:
            return {
                "status": "failed",
                "reason": "controlled_capture_window_not_exactly_one",
                "mango_enumeration_complete": True,
                "enumeration_consistency_ok": True,
                "attempted": 0,
                "attempted_other": 0,
            }
        if not event_recording_ids(matched[0]) and not matched[0].recording_url:
            return {
                "status": "failed",
                "reason": "controlled_capture_target_has_no_recording",
                "mango_enumeration_complete": True,
                "enumeration_consistency_ok": True,
                "attempted": 0,
                "attempted_other": 0,
            }
        events = [
            event
            for event in events
            if event.provider_call_id == controlled_request.source_call_id
        ]
        controlled_capture = {
            "request_sha256": controlled_request.request_sha256,
            "allowed_call_key": controlled_request.source_call_id,
            "expected_count": 1,
            "matched_count": 1,
            "enumerated_count": 1,
            "enumerated_other_count": 0,
            "attempted": len(events),
            "attempted_other": 0,
            "since": controlled_request.since.isoformat(),
            "until": controlled_request.until.isoformat(),
        }
    downloader = MangoRecordingDownloader(
        credentials=credentials,
        base_url=config.base_url,
        timeout_sec=60,
        link_retries=8,
        rate_limit_sleep_sec=30.0,
    )
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
    for event in enumerated_events:
        day_key = event.started_at.astimezone(
            ZoneInfo("Europe/Moscow")
        ).date().isoformat()
        calls_by_moscow_day.setdefault(day_key, []).append(event.provider_call_id)
    for values in calls_by_moscow_day.values():
        values[:] = sorted(set(values))
    previous_zero = zero_evidence_cursor.get(
        "independent_zero_enumerations_by_day"
    )
    if not isinstance(previous_zero, Mapping):
        previous_zero = {}
    zero_proofs: dict[str, int] = {
        key: 0 for key in calls_by_moscow_day
    }
    covered_days: set[date] = set()
    for start, end in ((base_window_start, until),):
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
            else 2
            if config.strict_ready_provenance
            else min(2, positive_int(previous_zero.get(key)) + 1)
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
        "controlled_capture": controlled_capture,
        "enumeration_consistency_ok": (
            dual_enumeration.get("enumeration_consistency_ok")
            if isinstance(dual_enumeration, Mapping)
            else None
        ),
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
            "enumeration_consistency_ok": (
                dual_enumeration.get("enumeration_consistency_ok")
                if isinstance(dual_enumeration, Mapping)
                else None
            ),
            **(
                {"dual_enumeration": dict(dual_enumeration)}
                if isinstance(dual_enumeration, Mapping)
                else {}
            ),
        },
        "call_keys": sorted(
            {event.provider_call_id for event in enumerated_events}
        ),
        "calls_by_moscow_day": calls_by_moscow_day,
        "independent_zero_enumerations_by_day": zero_proofs,
        "manifest_end_offset": manifest_end_offset,
        "manifest_snapshot_sha256": sealed_capture["sha256"],
        "api_requests": api_requests,
        "api_rows_total": len(scoped_rows),
        "api_authoritative_rows_total": sum(
            scope == "rolling_authority" for _row, scope in scoped_rows
        ),
        "api_auxiliary_rows_total": sum(
            scope == "recovery_auxiliary" for _row, scope in scoped_rows
        ),
        "api_events_total": len(enumerated_events),
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
    controlled_request: ControlledCaptureRequest | None = None,
) -> Mapping[str, Any]:
    if controlled_request is None:
        reject_controlled_call_broad_operation(config, "prepare_ingest_inputs")
    elif (
        config.processing_scope != "controlled_1_prepare"
        or config.runtime_authority_mode != "isolated_controlled"
        or controlled_capture_request_for_config(config) != controlled_request
    ):
        raise RuntimeError("controlled ingest request/config mismatch")
    if controlled_request is not None:
        early_working_call_ids = read_ingested_call_ids(config.working_db)
        if early_working_call_ids - {controlled_request.source_call_id}:
            raise RuntimeError("controlled working DB contains another call")
        early_snapshot = capture_manifest_snapshot(
            config.capture_manifest,
            end_offset=manifest_end_offset,
        )
        if (
            expected_manifest_sha256
            and early_snapshot["sha256"] != expected_manifest_sha256
        ):
            raise RuntimeError("capture manifest frozen prefix digest mismatch")
        early_latest: dict[str, ManifestEntry] = {}
        for entry in early_snapshot["entries"]:
            early_latest[entry.event_key] = entry
        if any(
            entry.provider_call_id != controlled_request.source_call_id
            for entry in early_latest.values()
        ):
            raise RuntimeError("controlled capture manifest contains another call")
    config.working_audio_dir.mkdir(parents=True, exist_ok=True)
    legacy_topology = normalize_recoverable_legacy_call_states(
        config.working_db
    )
    rows: list[dict[str, str]] = []
    actions: dict[str, int] = {}
    skipped: dict[str, int] = {}
    stable_before = datetime.now(timezone.utc) - timedelta(
        minutes=max(0, config.recording_set_stabilization_minutes)
    )
    fully_ready_call_ids = read_fully_ready_call_ids(config)
    working_call_ids = read_ingested_call_ids(config.working_db)
    if controlled_request is not None and (
        working_call_ids - {controlled_request.source_call_id}
    ):
        raise RuntimeError("controlled working DB contains another call")
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
    if controlled_request is not None and any(
        entry.provider_call_id != controlled_request.source_call_id
        for entry in latest.values()
    ):
        raise RuntimeError("controlled capture manifest contains another call")
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
        "legacy_topology_normalized": legacy_topology["normalized"],
        "legacy_downstream_invalidated": legacy_topology[
            "downstream_invalidated"
        ],
        "legacy_state_normalized": legacy_topology["state_normalized"],
        "legacy_dead_letter_state_normalized": legacy_topology[
            "dead_letter_state_normalized"
        ],
        "legacy_resolve_state_normalized": legacy_topology[
            "resolve_state_normalized"
        ],
        "legacy_topology_blocked": legacy_topology["blocked"],
        "legacy_topology_blocked_reasons": legacy_topology["blocked_reasons"],
    }


def normalize_recoverable_legacy_call_states(path: Path) -> Mapping[str, Any]:
    """Repair only legacy states whose stored evidence proves one exact outcome.

    The two supported repairs are a missing ASR mode on an otherwise terminal
    row and a missing Resolve state on a row with proven dual ASR.  Expired
    leases may be reclaimed; live or ambiguous states remain unchanged and
    become an explicit Process A blocker instead of being sent downstream.
    """

    result: dict[str, Any] = {
        "normalized": 0,
        "downstream_invalidated": 0,
        "state_normalized": 0,
        "dead_letter_state_normalized": 0,
        "resolve_state_normalized": 0,
        "blocked": 0,
        "blocked_reasons": {},
    }
    if not path.is_file():
        return result

    required_columns = {
        "id",
        "source_call_id",
        "transcription_status",
        "transcript_variants_json",
        "resolve_status",
        "analysis_status",
        "analysis_json",
        "dead_letter_stage",
        "pipeline_stage",
        "pipeline_worker_id",
        "pipeline_claimed_at",
        "analysis_worker_id",
        "analysis_claimed_at",
        "resolve_attempts",
        "analyze_attempts",
        "next_retry_at",
        "resolve_json",
        "resolve_quality_score",
        "last_error",
    }

    def complete_block(payload: Mapping[str, Any], key: str) -> bool:
        block = payload.get(key)
        return bool(
            isinstance(block, Mapping)
            and isinstance(block.get("variant_a"), str)
            and block["variant_a"].strip()
            and isinstance(block.get("variant_b"), str)
            and block["variant_b"].strip()
        )

    def populated_block(payload: Mapping[str, Any], key: str) -> bool:
        if key not in payload or payload.get(key) is None:
            return False
        block = payload.get(key)
        return bool(block) if isinstance(block, Mapping) else True

    def downstream_is_recoverable(row: Mapping[str, Any]) -> bool:
        if row.get("pipeline_stage"):
            return True
        if row.get("analysis_status") == "in_progress":
            return True
        resolve_status = row.get("resolve_status")
        if (
            resolve_status in {"pending", "failed"}
            and int(row.get("resolve_attempts") or 0)
            < max(1, int(os.getenv("RESOLVE_MAX_ATTEMPTS", "2")))
        ):
            return True
        return bool(
            resolve_status in {"done", "skipped"}
            and row.get("analysis_status") in {"pending", "failed"}
            and int(row.get("analyze_attempts") or 0)
            < max(1, int(os.getenv("ANALYZE_MAX_ATTEMPTS", "3")))
        )

    def add_block(reason: str) -> None:
        result["blocked"] += 1
        reasons = result["blocked_reasons"]
        reasons[reason] = reasons.get(reason, 0) + 1

    def lease_is_expired(value: Any, env_name: str) -> bool:
        if not value:
            return True
        try:
            claimed_at = parse_datetime(str(value))
        except (TypeError, ValueError):
            return False
        timeout = max(60, int(os.getenv(env_name, "1800")))
        return claimed_at <= datetime.now(timezone.utc) - timedelta(seconds=timeout)

    with sqlite3.connect(path, timeout=30) as con:
        con.row_factory = sqlite3.Row
        tables = {
            str(row[0])
            for row in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if "call_records" not in tables:
            return result
        columns = {
            str(row[1])
            for row in con.execute("PRAGMA table_info(call_records)")
        }
        if not required_columns.issubset(columns):
            return result

        con.execute("BEGIN IMMEDIATE")
        canonicalized = con.execute(
            """
            UPDATE call_records
               SET dead_letter_stage=NULL
             WHERE dead_letter_stage IS NOT NULL
               AND TRIM(dead_letter_stage)=''
            """
        )
        result["dead_letter_state_normalized"] = int(canonicalized.rowcount or 0)
        result["state_normalized"] = result["dead_letter_state_normalized"]
        call_id_counts: dict[str, int] = {}
        for count_row in con.execute("SELECT source_call_id FROM call_records"):
            normalized_call_id = str(count_row[0] or "").strip()
            if normalized_call_id:
                call_id_counts[normalized_call_id] = (
                    call_id_counts.get(normalized_call_id, 0) + 1
                )
        rows = con.execute(
            """
            SELECT id, source_call_id, transcription_status,
                   transcript_variants_json, resolve_status, analysis_status,
                   analysis_json, dead_letter_stage, pipeline_stage,
                   pipeline_worker_id, pipeline_claimed_at,
                   analysis_worker_id, analysis_claimed_at,
                   resolve_attempts, analyze_attempts, next_retry_at,
                   resolve_json, resolve_quality_score, last_error
              FROM call_records
             WHERE transcription_status='done'
               AND dead_letter_stage IS NULL
             ORDER BY id
            """
        )
        for raw_row in rows:
            row = dict(raw_row)
            raw = row.get("transcript_variants_json")
            try:
                payload = json.loads(str(raw or ""))
            except json.JSONDecodeError:
                payload = None
            if ready_row_is_complete(row):
                continue
            try:
                analysis = json.loads(str(row.get("analysis_json") or ""))
            except json.JSONDecodeError:
                analysis = None
            lease_fields = (
                "pipeline_stage",
                "pipeline_worker_id",
                "pipeline_claimed_at",
                "analysis_worker_id",
                "analysis_claimed_at",
            )
            downstream_terminal = bool(
                row.get("resolve_status") in {"done", "skipped"}
                and row.get("analysis_status") == "done"
                and not any(row.get(field) for field in lease_fields)
            )
            if downstream_terminal and not (
                isinstance(analysis, Mapping) and bool(analysis)
            ):
                add_block("terminal_payload_invalid")
                continue
            if isinstance(payload, Mapping):
                mode = str(payload.get("mode") or "").strip()
                if has_dual_asr_or_exception(row):
                    raw_resolve_status = row.get("resolve_status")
                    resolve_status = str(raw_resolve_status or "").strip()
                    if not resolve_status:
                        call_id = str(row.get("source_call_id") or "").strip()
                        if not call_id or call_id_counts.get(call_id) != 1:
                            add_block("non_unique_source_call_id")
                            continue
                        analysis_status = row.get("analysis_status")
                        pipeline_leased = any(
                            row.get(field)
                            for field in (
                                "pipeline_stage",
                                "pipeline_worker_id",
                                "pipeline_claimed_at",
                            )
                        )
                        analysis_leased = bool(
                            analysis_status == "in_progress"
                            or row.get("analysis_worker_id")
                            or row.get("analysis_claimed_at")
                        )
                        pipeline_lease_expired = bool(
                            pipeline_leased
                            and lease_is_expired(
                                row.get("pipeline_claimed_at"),
                                "PIPELINE_LEASE_TIMEOUT_SEC",
                            )
                        )
                        analysis_lease_expired = bool(
                            analysis_leased
                            and lease_is_expired(
                                row.get("analysis_claimed_at"),
                                "ANALYZE_LEASE_TIMEOUT_SEC",
                            )
                        )
                        if analysis_status not in {
                            "pending",
                            "failed",
                            "in_progress",
                        } or (pipeline_leased and not pipeline_lease_expired) or (
                            analysis_leased and not analysis_lease_expired
                        ):
                            add_block("resolve_state_missing_or_leased")
                            continue
                        candidate = {
                            **row,
                            "resolve_status": "pending",
                            "analysis_status": "pending",
                            "resolve_attempts": 0,
                            "analyze_attempts": 0,
                            "pipeline_stage": None,
                            "pipeline_worker_id": None,
                            "pipeline_claimed_at": None,
                            "analysis_worker_id": None,
                            "analysis_claimed_at": None,
                            "next_retry_at": None,
                            "resolve_json": None,
                            "resolve_quality_score": None,
                            "analysis_json": None,
                            "last_error": None,
                        }
                        if not (
                            has_dual_asr_or_exception(candidate)
                            and downstream_is_recoverable(candidate)
                            and not ready_row_is_complete(candidate)
                        ):
                            add_block("resolve_state_normalization_failed")
                            continue
                        updated = con.execute(
                            """
                            UPDATE call_records
                               SET resolve_status='pending',
                                   analysis_status='pending',
                                   resolve_attempts=0,
                                   analyze_attempts=0,
                                   pipeline_stage=NULL,
                                   pipeline_worker_id=NULL,
                                   pipeline_claimed_at=NULL,
                                   analysis_worker_id=NULL,
                                   analysis_claimed_at=NULL,
                                   next_retry_at=NULL,
                                   resolve_json=NULL,
                                   resolve_quality_score=NULL,
                                   analysis_json=NULL,
                                   last_error=NULL
                             WHERE id=?
                               AND (
                                    (resolve_status IS NULL AND ? IS NULL)
                                    OR resolve_status=?
                               )
                               AND analysis_status IN ('pending', 'failed', 'in_progress')
                            """,
                            (row["id"], raw_resolve_status, raw_resolve_status),
                        )
                        if int(updated.rowcount or 0) != 1:
                            add_block("resolve_state_changed_concurrently")
                            continue
                        result["state_normalized"] += 1
                        result["resolve_state_normalized"] += 1
                        if row.get("resolve_quality_score") is not None or any(
                            row.get(field)
                            for field in (
                                "resolve_json",
                                "analysis_json",
                                "last_error",
                            )
                        ) or int(row.get("resolve_attempts") or 0) or int(
                            row.get("analyze_attempts") or 0
                        ):
                            result["downstream_invalidated"] += 1
                        continue
                    if downstream_is_recoverable(row):
                        continue
                    add_block("terminal_payload_invalid")
                    continue
                if mode in {"stereo", "mono_or_fallback"}:
                    if payload.get("primary_provider") != "mlx":
                        add_block("primary_provider_mismatch")
                        continue
                    backfill_state = (
                        TranscribeService.secondary_backfill_state_from_payload(
                            dict(payload),
                            secondary_provider="gigaam",
                        )
                    )
                    if backfill_state in {"fresh", "retry"}:
                        if downstream_terminal:
                            add_block(
                                "secondary_asr_after_downstream_terminal"
                            )
                            continue
                        if downstream_is_recoverable(row):
                            continue
                        add_block("terminal_payload_invalid")
                        continue
                    add_block("strict_asr_topology_invalid")
                    continue
            else:
                mode = ""

            if mode:
                add_block("unknown_mode")
                continue
            if not isinstance(payload, Mapping):
                add_block("invalid_variants_json")
                continue
            if payload.get("legacy_topology_normalization") is not None:
                add_block("normalization_audit_conflict")
                continue
            if (
                payload.get("primary_provider") != "mlx"
                or payload.get("secondary_provider") != "gigaam"
            ):
                add_block("provider_mismatch")
                continue

            call_id = str(row.get("source_call_id") or "").strip()
            if not call_id or call_id_counts.get(call_id) != 1:
                add_block("non_unique_source_call_id")
                continue
            terminal = bool(
                str(row.get("resolve_status") or "") in {"done", "skipped"}
                and str(row.get("analysis_status") or "") == "done"
                and isinstance(analysis, Mapping)
                and bool(analysis)
                and not any(
                    row.get(field)
                    for field in lease_fields
                )
            )
            if not terminal:
                add_block("non_terminal_or_leased")
                continue

            stereo_complete = bool(
                complete_block(payload, "manager")
                and complete_block(payload, "client")
                and not populated_block(payload, "full")
            )
            mono_complete = bool(
                complete_block(payload, "full")
                and not populated_block(payload, "manager")
                and not populated_block(payload, "client")
            )
            if stereo_complete == mono_complete:
                add_block("ambiguous_or_incomplete_topology")
                continue

            normalized = dict(payload)
            normalized["mode"] = (
                "stereo" if stereo_complete else "mono_or_fallback"
            )
            normalized["legacy_topology_normalization"] = {
                "method": "complete_shape_xor_reset_downstream_v1",
                "source_json_sha256": hashlib.sha256(
                    str(raw).encode("utf-8")
                ).hexdigest(),
            }
            serialized = json.dumps(normalized, ensure_ascii=False)
            candidate = {
                **row,
                "transcript_variants_json": serialized,
                "resolve_status": "pending",
                "analysis_status": "pending",
                "resolve_attempts": 0,
                "analyze_attempts": 0,
                "pipeline_stage": None,
                "pipeline_worker_id": None,
                "pipeline_claimed_at": None,
                "analysis_worker_id": None,
                "analysis_claimed_at": None,
                "next_retry_at": None,
                "resolve_json": None,
                "resolve_quality_score": None,
                "analysis_json": None,
                "last_error": None,
            }
            if not (
                has_dual_asr_or_exception(candidate)
                and downstream_is_recoverable(candidate)
                and not ready_row_is_complete(candidate)
            ):
                add_block("normalization_postcondition_failed")
                continue
            updated = con.execute(
                """
                UPDATE call_records
                   SET transcript_variants_json=?,
                       resolve_status='pending',
                       analysis_status='pending',
                       resolve_attempts=0,
                       analyze_attempts=0,
                       pipeline_stage=NULL,
                       pipeline_worker_id=NULL,
                       pipeline_claimed_at=NULL,
                       analysis_worker_id=NULL,
                       analysis_claimed_at=NULL,
                       next_retry_at=NULL,
                       resolve_json=NULL,
                       resolve_quality_score=NULL,
                       analysis_json=NULL,
                       last_error=NULL
                 WHERE id=?
                   AND transcript_variants_json=?
                   AND transcription_status='done'
                   AND dead_letter_stage IS NULL
                   AND resolve_status IN ('done', 'skipped')
                   AND analysis_status='done'
                   AND COALESCE(pipeline_stage, '')=''
                   AND COALESCE(pipeline_worker_id, '')=''
                   AND COALESCE(pipeline_claimed_at, '')=''
                   AND COALESCE(analysis_worker_id, '')=''
                   AND COALESCE(analysis_claimed_at, '')=''
                """,
                (
                    serialized,
                    row["id"],
                    raw,
                ),
            )
            if updated.rowcount != 1:
                raise RuntimeError(
                    "legacy ASR topology changed during normalization"
                )
            result["normalized"] += 1
            result["downstream_invalidated"] += 1
        con.commit()
    return result


# Kept for callers outside this module while the more accurate name rolls out.
normalize_unambiguous_legacy_asr_topologies = normalize_recoverable_legacy_call_states


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


def trusted_ready_call_ids_for_capture(
    config: CallsTwoProcessesConfig,
) -> set[str]:
    """Return only unique complete calls from a sealed ready generation.

    Capture must fail open for deduplication: an absent, stale, malformed or
    unsealed downstream generation can cause extra local work, but can never
    suppress a fresh Mango recording.
    """

    try:
        manifest = read_json(config.ready_manifest)
        before = os.lstat(config.ready_db)
        if (
            not stat.S_ISREG(before.st_mode)
            or config.ready_db.is_symlink()
            or before.st_uid != os.getuid()
            or before.st_nlink != 1
            or manifest.get("ready_db") != str(config.ready_db)
            or manifest.get("sha256") != sha256_file(config.ready_db)
            or manifest.get("size_bytes") != before.st_size
            or manifest.get("ready_mtime_ns") != before.st_mtime_ns
            or validate_ready_manifest_payload(
                manifest,
                require_consistency=False,
                expected_code_sha=config.expected_code_sha,
                expected_host_id=configured_host_id(config, required=True),
            )
        ):
            return set()
        rows = load_ready_rows(config.ready_db)
        after = os.lstat(config.ready_db)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            return set()
    except (OSError, sqlite3.DatabaseError, TypeError, ValueError, RuntimeError):
        return set()
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

        for raw in con.execute(
            "SELECT * FROM call_records WHERE COALESCE(dead_letter_stage, '')=''"
        ):
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
                downstream_terminal = bool(
                    row.get("resolve_status") in {"done", "skipped"}
                    and row.get("analysis_status") == "done"
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
                if not downstream_terminal:
                    return True
            resolve = row.get("resolve_status")
            if transcription == "done" and resolve in {"pending", "failed"} and retry_due and int(row.get("resolve_attempts") or 0) < limits["resolve"]:
                return True
            if transcription == "done" and resolve in {"done", "skipped"} and row.get("analysis_status") in {"pending", "failed"} and retry_due and int(row.get("analyze_attempts") or 0) < limits["analyze"]:
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


def ensure_codex_runtime_anchor(config: CallsTwoProcessesConfig) -> Path:
    """Create and validate the private container for ephemeral Codex homes."""

    anchor = config.codex_home_root.expanduser()
    anchor.mkdir(parents=True, exist_ok=True, mode=0o700)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(anchor, flags)
    except OSError as exc:
        raise RuntimeError("codex_runtime_anchor_unsafe") from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISDIR(opened.st_mode) or opened.st_uid != os.getuid():
            raise RuntimeError("codex_runtime_anchor_wrong_owner_or_type")
        os.fchmod(descriptor, 0o700)
    finally:
        os.close(descriptor)
    resolved = validate_owner_only_directory(
        anchor,
        label="codex_runtime_anchor",
        owner_only_mode=0o700,
    )
    _validate_codex_location(
        resolved,
        require_owner_local=config.strict_ready_provenance,
    )
    return resolved


@contextmanager
def temporary_codex_runtime(
    config: CallsTwoProcessesConfig,
    *,
    label: str,
) -> Iterator[Mapping[str, str]]:
    """Build a clean Codex home/process home for exactly one subprocess."""

    safe_label = re.sub(r"[^A-Za-z0-9_.-]", "_", label).strip("._-")
    if not safe_label:
        raise RuntimeError("temporary Codex runtime label is invalid")
    parent = ensure_codex_runtime_anchor(config)
    with tempfile.TemporaryDirectory(
        prefix=f".mango-codex-{safe_label}-",
        dir=parent,
    ) as raw_runtime:
        runtime = Path(raw_runtime)
        runtime.chmod(0o700)
        process_home = runtime / "home"
        process_tmp = runtime / "tmp"
        process_home.mkdir(mode=0o700)
        process_tmp.mkdir(mode=0o700)
        codex_home = prepare_codex_home(
            runtime / "codex-home",
            strict=config.strict_ready_provenance,
        )
        yield {
            "CODEX_HOME": str(codex_home),
            "MANGO_CODEX_PROCESS_HOME": str(process_home),
            "MANGO_CODEX_PROCESS_TMPDIR": str(process_tmp),
            "TMPDIR": str(process_tmp),
        }


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
            with temporary_codex_runtime(
                config,
                label="preflight",
            ) as codex_runtime:
                auth = subprocess.run(
                    [str(config.codex_binary), "login", "status"],
                    env={
                        "HOME": codex_runtime["MANGO_CODEX_PROCESS_HOME"],
                        "CODEX_HOME": codex_runtime["CODEX_HOME"],
                        "TMPDIR": codex_runtime[
                            "MANGO_CODEX_PROCESS_TMPDIR"
                        ],
                        "PATH": command_path(config),
                        "LANG": "en_US.UTF-8",
                        "LC_ALL": "en_US.UTF-8",
                        "NO_COLOR": "1",
                    },
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                    timeout=30,
                )
                auth_ok = auth.returncode == 0
        except (OSError, RuntimeError, subprocess.TimeoutExpired):
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
    current = _mango_wire_datetime(datetime.now(timezone.utc))
    if until:
        end = _mango_wire_datetime(parse_datetime(until))
        if config.strict_ready_provenance and end > current:
            raise ValueError("strict capture until cannot be in the future")
    else:
        end = current
        if config.strict_ready_provenance:
            # A scheduled dual pass must query a settled boundary.  Otherwise
            # a call or recording that appears between the two full API passes
            # causes perpetual safe-but-unproductive mismatches under load.
            end -= timedelta(
                minutes=config.recording_set_stabilization_minutes
            )
    if since:
        start = _mango_wire_datetime(parse_datetime(since))
    else:
        cursor = read_json(config.cursor_path)
        if config.strict_ready_provenance and cursor:
            verified_capture_window(
                config,
                cursor,
                allow_pre_dual_anchor=True,
            )
        raw = optional_text(cursor.get("until")) or config.bootstrap_since
        if raw:
            start = _mango_wire_datetime(parse_datetime(raw)) - timedelta(
                minutes=config.poll_overlap_minutes
            )
            if end - start > timedelta(days=config.max_catch_up_days):
                raise ValueError(
                    "capture gap exceeds max_catch_up_days; provide explicit --since after dry-run review"
                )
        else:
            start = end - timedelta(hours=config.first_lookback_hours)
    start = _mango_wire_datetime(start)
    end = _mango_wire_datetime(end)
    if end <= start:
        raise ValueError("capture until must be after since")
    return start, end


def legacy_transfer_cursor_can_be_replaced(
    config: CallsTwoProcessesConfig,
    cursor: Mapping[str, Any],
) -> bool:
    """Recognize only the exact pre-certificate transfer-cursor shape.

    The handoff release wrote none of the fields below.  Their presence means
    that a current cursor lost or had its certificate changed, which must not
    be mistaken for a legitimate legacy rollout.
    """

    if any(
        field in cursor
        for field in (
            "capture_window_certificate",
            "api_requests",
            "api_rows_total",
            "api_authoritative_rows_total",
            "api_events_total",
        )
    ):
        return False
    if (
        cursor.get("schema_version") != "mango_api_freshness_v1"
        or cursor.get("mango_enumeration_complete") is not True
    ):
        return False
    try:
        parsed_until = datetime.fromisoformat(
            str(cursor.get("until") or "").replace("Z", "+00:00")
        )
        if parsed_until.tzinfo is None or parsed_until.utcoffset() is None:
            return False
        end_offset = cursor.get("manifest_end_offset")
        if isinstance(end_offset, bool) or not isinstance(end_offset, int):
            return False
        if end_offset <= 0:
            return False
        expected_sha256 = cursor.get("manifest_snapshot_sha256")
        if not isinstance(expected_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", expected_sha256
        ):
            return False
        snapshot = capture_manifest_snapshot(
            config.capture_manifest,
            end_offset=end_offset,
        )
    except (OSError, RuntimeError, TypeError, ValueError):
        return False
    return (
        snapshot.get("end_offset") == end_offset
        and snapshot.get("sha256") == expected_sha256
    )


def config_for_capture_window(
    config: CallsTwoProcessesConfig,
    *,
    since: Optional[str],
    window_since: datetime,
    window_until: datetime,
) -> CallsTwoProcessesConfig:
    """Require a continuous strict window and authorize one legacy bridge.

    Every existing certified cursor is checked against the proposed window.
    A current-but-invalid cursor is a hard stop.  Only the exact old handoff
    shape, with its manifest prefix still intact and an operator-supplied start
    boundary, gets the transient migration flag.
    """

    if not config.strict_ready_provenance:
        return config
    cursor, _cursor_sha256 = read_capture_cursor_snapshot(config.cursor_path)
    if not cursor:
        return config
    if "capture_window_certificate" in cursor:
        _prior_since, prior_until, _prior_rolling_since = (
            verified_capture_window(
                config,
                cursor,
                allow_pre_dual_anchor=True,
            )
        )
        if (
            window_until < prior_until
            or capture_rolling_window_start(
                config,
                since=window_since,
                until=window_until,
            )
            > prior_until
        ):
            raise RuntimeError(
                "certified capture replacement window is not continuous"
            )
        return config
    if since is None:
        raise RuntimeError(
            "legacy capture cursor replacement requires explicit since"
        )
    if not legacy_transfer_cursor_can_be_replaced(config, cursor):
        raise RuntimeError("legacy capture cursor is not eligible for migration")
    try:
        legacy_until = datetime.fromisoformat(
            str(cursor.get("until") or "").replace("Z", "+00:00")
        ).astimezone(timezone.utc)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("legacy capture cursor boundary is invalid") from exc
    if (
        window_until < legacy_until
        or capture_rolling_window_start(
            config,
            since=window_since,
            until=window_until,
        )
        > legacy_until
    ):
        raise RuntimeError("legacy capture replacement window is not continuous")
    return replace(config, legacy_cursor_migration_mode=True)


def capture_rolling_window_start(
    config: CallsTwoProcessesConfig,
    *,
    since: datetime,
    until: datetime,
) -> datetime:
    """Return the exact rolling boundary that strict evidence must prove."""

    if (
        not config.strict_ready_provenance
        or config.runtime_authority_mode == "isolated_controlled"
    ):
        return since
    threshold = until - timedelta(
        hours=max(1, config.pending_recording_retry_hours)
    )
    rolling_day_start = (
        threshold.astimezone(ZoneInfo("Europe/Moscow"))
        .replace(hour=0, minute=0, second=0, microsecond=0)
        .astimezone(timezone.utc)
    )
    earliest = min(since, rolling_day_start)
    return (
        earliest.astimezone(ZoneInfo("Europe/Moscow"))
        .replace(hour=0, minute=0, second=0, microsecond=0)
        .astimezone(timezone.utc)
    )


def worker_environment(config: CallsTwoProcessesConfig) -> Mapping[str, str]:
    project_root = Path(__file__).resolve().parents[3]
    isolated_codex = project_root / "scripts" / "run_codex_cli_isolated.sh"
    config.transcripts_dir.mkdir(parents=True, exist_ok=True)
    return {
        **os.environ,
        "PATH": command_path(config),
        "DATABASE_URL": f"sqlite:///{config.working_db}",
        "TRANSCRIPT_EXPORT_DIR": str(config.transcripts_dir),
        # Each heavy stage replaces these empty sentinels with a fresh,
        # one-subprocess runtime. Prelude init/ingest never invokes Codex.
        "CODEX_HOME": "",
        "MANGO_CODEX_PROCESS_HOME": "",
        "MANGO_CODEX_PROCESS_TMPDIR": "",
        "PYTHONPATH": str(project_root / "src"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "SQLITE_BUSY_TIMEOUT_MS": "60000",
        "CODEX_CLI_COMMAND": str(isolated_codex),
        "MANGO_CODEX_REAL_BIN": str(config.codex_binary),
        "CODEX_CLI_TIMEOUT_SEC": "360",
        "CODEX_REASONING_EFFORT": config.codex_reasoning_effort,
        "MANGO_CODEX_SERVICE_TIER": config.codex_service_tier,
        "CODEX_RESOLVE_MODEL": config.codex_resolve_model,
        "CODEX_ANALYZE_MODEL": config.codex_analyze_model,
        "RESOLVE_LLM_PROVIDER": "codex_cli",
        "RESOLVE_DIALOGUE_MODE": "dialogue",
        "RESOLVE_RESCUE_PROVIDER": "none",
        "RESOLVE_RESCUE_DUAL_ENABLED": "0",
        "ANALYZE_PROVIDER": "codex_cli",
        "MANGO_STRICT_ASR_RUNTIME": "1" if config.strict_ready_provenance else "0",
        # Always overwrite these values so a parent shell cannot accidentally
        # leak a stale pilot scope into the ordinary service or vice versa.
        "MANGO_CALLS_PROCESSING_SCOPE": config.processing_scope,
        "MANGO_CALLS_CONTROLLED_ALLOWLIST_PATH": (
            str(config.controlled_call_allowlist_path)
            if config.controlled_call_allowlist_path
            else ""
        ),
        "MANGO_CALLS_CONTROLLED_ALLOWLIST_SHA256": (
            config.controlled_call_allowlist_sha256 or ""
        ),
        "MANGO_CALLS_CONTROLLED_TENANT_ID": (
            config.tenant_id if config.processing_scope == "controlled_1" else ""
        ),
        "MANGO_CALLS_CONTROLLED_CODE_SHA": (
            config.expected_code_sha or ""
            if config.processing_scope == "controlled_1"
            else ""
        ),
        "MANGO_CALLS_CONTROLLED_HOST_ID": (
            config.expected_active_host_id or ""
            if config.processing_scope == "controlled_1"
            else ""
        ),
        "MANGO_CALLS_CONTROLLED_HOST_ID_PATH": (
            str(config.host_id_file)
            if config.processing_scope == "controlled_1"
            else ""
        ),
        "MANGO_CALLS_CONTROLLED_RUN_AUTHORITY_PATH": "",
        "MANGO_CALLS_CONTROLLED_RUN_AUTHORITY_SHA256": "",
        "MANGO_CALLS_CONTROLLED_LIFELINE_FD": "",
        "MANGO_CALLS_CONTROLLED_AUDIO_SNAPSHOT_PATH": "",
        "MANGO_CALLS_CONTROLLED_AUDIO_SNAPSHOT_SHA256": "",
        "MANGO_CALLS_CONTROLLED_AUDIO_SNAPSHOT_SIZE_BYTES": "",
        **(
            {
                "LLM_CACHE_ENABLED": "0",
                "LLM_CACHE_DIR": str(
                    config.pipeline_root
                    / "state"
                    / "controlled_llm_cache_disabled"
                ),
            }
            if config.processing_scope == "controlled_1"
            else {}
        ),
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


def _validate_codex_location(
    resolved_target: Path, *, require_owner_local: bool
) -> None:
    if path_has_cloud_marker(resolved_target):
        raise RuntimeError("isolated CODEX_HOME must stay outside cloud folders")
    project_root = Path(__file__).resolve().parents[3]
    if resolved_target == project_root or project_root in resolved_target.parents:
        raise RuntimeError("isolated CODEX_HOME must stay outside the Git worktree")
    owner_local = (Path.home() / ".mango_local").resolve(strict=False)
    if require_owner_local and owner_local not in resolved_target.parents:
        raise RuntimeError("isolated CODEX_HOME must stay under ~/.mango_local")


def _cleanup_isolated_codex_atomic_temps(
    target: Path,
    allowed_names: set[str],
) -> None:
    removed = False
    for entry in target.iterdir():
        if not any(
            re.fullmatch(
                rf"\.{re.escape(name)}\.[A-Za-z0-9_-]+\.tmp",
                entry.name,
            )
            for name in allowed_names
        ):
            continue
        read_stable_regular_bytes(
            entry,
            label="isolated_codex_atomic_temp",
            owner_only_mode=0o600,
        )
        entry.unlink()
        removed = True
    if removed:
        _fsync_directory(target)


def prepare_codex_home(target: Path, *, strict: bool = False) -> Path:
    source = Path.home() / ".codex"
    resolved_target = target.expanduser().resolve(strict=False)
    _validate_codex_location(resolved_target, require_owner_local=strict)
    target.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(target, 0o700)
    resolved_target = validate_owner_only_directory(
        target,
        label="isolated_codex_home",
        owner_only_mode=0o700,
    )
    _validate_codex_location(resolved_target, require_owner_local=strict)
    allowed_existing = {
        "auth.json",
        "installation_id",
        "models_cache.json",
        "config.toml",
        "AGENTS.md",
    }
    _cleanup_isolated_codex_atomic_temps(target, allowed_existing)
    unknown = sorted(entry.name for entry in target.iterdir() if entry.name not in allowed_existing)
    if strict and unknown:
        raise RuntimeError("isolated CODEX_HOME contains unknown persistent files")
    for name in ("auth.json", "installation_id", "models_cache.json"):
        src = source / name
        dst = target / name
        if src.is_file():
            source_mode = 0o600 if strict and name == "auth.json" else None
            label = f"codex_source_{name.replace('.', '_')}"
            if strict and name == "auth.json":
                payload, source_path = read_stable_regular_bytes_with_path(
                    src,
                    label=label,
                    owner_only_mode=source_mode,
                )
                _validate_codex_location(
                    source_path,
                    require_owner_local=False,
                )
            else:
                payload = read_stable_regular_bytes(
                    src,
                    label=label,
                    owner_only_mode=source_mode,
                )
            atomic_replace_owner_only_bytes(
                dst,
                payload,
                label=f"isolated_codex_{name.replace('.', '_')}",
            )
        elif name == "auth.json" and os.path.lexists(dst):
            if stat.S_ISDIR(os.lstat(dst).st_mode):
                raise RuntimeError("stale isolated Codex auth target is a directory")
            dst.unlink()
            _fsync_directory(target)
    # Isolate batch classification from desktop plugins/MCP servers and account
    # personality. Resolve/Analyze receive all task context in their own prompt.
    for name, content in (
        ("config.toml", "# Isolated Mango batch runtime: no plugins or MCP servers.\n"),
        ("AGENTS.md", "Follow the supplied task prompt exactly and return only its requested result.\n"),
    ):
        path = target / name
        atomic_replace_owner_only_bytes(
            path,
            content.encode("utf-8"),
            label=f"isolated_codex_{name.replace('.', '_')}",
        )
    for name in allowed_existing:
        path = target / name
        if os.path.lexists(path):
            read_stable_regular_bytes(
                path,
                label=f"isolated_codex_{name.replace('.', '_')}",
                owner_only_mode=0o600,
            )
    return resolved_target


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
    raise ValueError(f"unsupported sequential pipeline stage: {stage}")


def pipeline_stages(
    config: CallsTwoProcessesConfig,
    *,
    include_llm: bool = True,
) -> tuple[str, ...]:
    stages = SEQUENTIAL_PIPELINE_STAGES
    if include_llm:
        return stages
    return tuple(stage for stage in stages if stage not in {"resolve", "analyze"})


def controlled_stage_report(
    config: CallsTwoProcessesConfig,
    report: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Turn a zero-exit worker with failed/incomplete metrics into a STOP."""
    if config.processing_scope != "controlled_1":
        return report
    metrics = report.get("metrics")
    worker_rc = report.get("rc")
    valid = bool(
        isinstance(worker_rc, int)
        and not isinstance(worker_rc, bool)
        and worker_rc == 0
        and isinstance(metrics, Mapping)
        and all(
            isinstance(metrics.get(name), int)
            and not isinstance(metrics.get(name), bool)
            and int(metrics[name]) >= 0
            for name in ("processed", "success", "failed")
        )
        and int(metrics["failed"]) == 0
        and int(metrics["success"]) == int(metrics["processed"])
        and int(metrics["processed"]) <= 1
        and int(metrics["success"]) <= 1
    )
    if valid:
        return {**dict(report), "controlled_stage_contract_ok": True}
    return {
        **dict(report),
        "worker_rc": report.get("rc"),
        "rc": 65,
        "controlled_stage_contract_ok": False,
        "orchestrator_stop_reason": "controlled_stage_metrics_failed",
    }


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
        reports: list[Mapping[str, Any]] = []
        for stage in stages:
            controlled_call_scope_for_config(config)
            with controlled_worker_authority_environment(
                config,
                stage=stage,
                run_id=run_id or new_calls_run_id(datetime.now(timezone.utc)),
            ) as authority_env:
                with temporary_codex_runtime(
                    config,
                    label=stage.replace("-", "_"),
                ) as codex_runtime:
                    stage_report = controlled_stage_report(
                        config,
                        runner(
                            worker_command(config, stage),
                            {
                                **stage_worker_environment_for(
                                    config,
                                    base_env,
                                    stage,
                                ),
                                **authority_env,
                                **codex_runtime,
                            },
                            config.working_dir,
                        ),
                    )
                    reports.append(stage_report)
                    if int(stage_report.get("rc") or 0) != 0:
                        break
        return reports
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
        controlled_call_scope_for_config(config)
        label = stage.replace("-", "_")
        log_path = logs_dir / f"stage_{label}_{log_run_id}.log"
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
        lifeline_read_fd = -1
        lifeline_write_fd = -1
        timed_out = False
        authority_scope = controlled_worker_authority_environment(
            config,
            stage=stage,
            run_id=log_run_id,
        )
        authority_env = authority_scope.__enter__()
        codex_runtime_scope = temporary_codex_runtime(
            config,
            label=label,
        )
        codex_runtime: Mapping[str, str] = {}
        codex_runtime_entered = False
        try:
            codex_runtime = codex_runtime_scope.__enter__()
            codex_runtime_entered = True
            worker_env = {
                **stage_worker_environment_for(config, base_env, stage),
                **authority_env,
                **codex_runtime,
            }
            pass_fds: tuple[int, ...] = ()
            if config.processing_scope == "controlled_1":
                lifeline_read_fd, lifeline_write_fd = os.pipe()
                worker_env["MANGO_CALLS_CONTROLLED_LIFELINE_FD"] = str(
                    lifeline_read_fd
                )
                pass_fds = (lifeline_read_fd,)
            with log_path.open("x", encoding="utf-8") as log_handle:
                log_path.chmod(0o600)
                command = worker_command(config, stage)
                timed_command = stage_subprocess_command(config, command)
                proc = subprocess.Popen(
                    timed_command,
                    cwd=config.working_dir,
                    env=worker_env,
                    text=True,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    pass_fds=pass_fds,
                )
                if lifeline_read_fd >= 0:
                    os.close(lifeline_read_fd)
                    lifeline_read_fd = -1
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
            try:
                if proc is not None:
                    terminate_process_group(proc)
            finally:
                try:
                    if lifeline_read_fd >= 0:
                        os.close(lifeline_read_fd)
                    if lifeline_write_fd >= 0:
                        os.close(lifeline_write_fd)
                finally:
                    try:
                        heartbeat_path.unlink(missing_ok=True)
                    finally:
                        try:
                            if codex_runtime_entered:
                                codex_runtime_scope.__exit__(None, None, None)
                        finally:
                            authority_scope.__exit__(None, None, None)
        after_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        stage_metrics = parse_macos_time_metrics(log_path)
        worker_metrics = parse_worker_stage_metrics(log_path, stage)
        stage_report = controlled_stage_report(
            config,
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
                "metrics": worker_metrics,
            },
        )
        reports.append(stage_report)
        if int(stage_report.get("rc") or 0) != 0:
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


def parse_worker_stage_metrics(path: Path, stage: str) -> Mapping[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return {}
    decoder = json.JSONDecoder()
    objects: list[Mapping[str, Any]] = []
    cursor = 0
    while cursor < len(raw):
        start = raw.find("{", cursor)
        if start < 0:
            break
        try:
            value, end = decoder.raw_decode(raw, start)
        except json.JSONDecodeError:
            cursor = start + 1
            continue
        if isinstance(value, Mapping):
            objects.append(value)
        cursor = end
    for payload in reversed(objects):
        totals = payload.get("totals")
        if not isinstance(totals, Mapping):
            continue
        stage_totals = totals.get(stage)
        if not isinstance(stage_totals, Mapping):
            continue
        strict_totals: dict[str, int] = {}
        for name in ("processed", "success", "failed"):
            value = stage_totals.get(name)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
            ):
                return {}
            strict_totals[name] = value
        strict_run_counts: dict[str, int] = {}
        for name in ("cycles", "idle_cycles"):
            value = payload.get(name)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
                or (name == "cycles" and value < 1)
            ):
                return {}
            strict_run_counts[name] = value
        if strict_run_counts["idle_cycles"] > strict_run_counts["cycles"]:
            return {}
        receipts = payload.get("runtime_receipts")
        if not isinstance(receipts, Mapping):
            return {}
        stage_receipt = receipts.get(stage)
        if not isinstance(stage_receipt, Mapping):
            return {}
        expected_receipt_fields = {
            "provider_invocations",
            "mlx_cache_release_attempts",
            "mlx_cache_release_successes",
        }
        if set(stage_receipt) != expected_receipt_fields:
            return {}
        providers = stage_receipt.get("provider_invocations")
        if not isinstance(providers, Mapping):
            return {}
        strict_providers: dict[str, int] = {}
        for provider, count in providers.items():
            if (
                not isinstance(provider, str)
                or not provider
                or not isinstance(count, int)
                or isinstance(count, bool)
                or count < 0
            ):
                return {}
            strict_providers[provider] = count
        strict_receipt_counts: dict[str, int] = {}
        for name in (
            "mlx_cache_release_attempts",
            "mlx_cache_release_successes",
        ):
            value = stage_receipt.get(name)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
            ):
                return {}
            strict_receipt_counts[name] = value
        runtime_receipt: Mapping[str, Any] = {
            "provider_invocations": strict_providers,
            **strict_receipt_counts,
        }
        return {
            **strict_totals,
            **strict_run_counts,
            "stop_reason": optional_text(payload.get("stop_reason")),
            "runtime_receipt": runtime_receipt,
        }
    return {}


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


def stage_subprocess_command(
    config: CallsTwoProcessesConfig,
    command: Sequence[str],
) -> list[str]:
    """Keep the controlled worker directly parented by its orchestrator."""
    if config.processing_scope == "controlled_1":
        return list(command)
    if Path("/usr/bin/time").is_file():
        return ["/usr/bin/time", "-l", *command]
    return list(command)


def cli_command(config: CallsTwoProcessesConfig, *args: str) -> list[str]:
    return [str(config.python_executable), "-m", "mango_mvp.cli", *args]


def run_command(
    command: Sequence[str],
    env: Mapping[str, str],
    cwd: Path,
    *,
    deadline: Optional[float] = None,
    parent_lifeline: bool = False,
) -> Mapping[str, Any]:
    cwd.mkdir(parents=True, exist_ok=True)
    logs_dir = cwd / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    log_path = logs_dir / f"command_{stamp}.log"
    with log_path.open("w", encoding="utf-8") as handle:
        lifeline_read_fd = -1
        lifeline_write_fd = -1
        child_env = dict(env)
        pass_fds: tuple[int, ...] = ()
        if parent_lifeline:
            lifeline_read_fd, lifeline_write_fd = os.pipe()
            child_env["MANGO_CALLS_CONTROLLED_LIFELINE_FD"] = str(
                lifeline_read_fd
            )
            pass_fds = (lifeline_read_fd,)
        proc = subprocess.Popen(
            (
                parent_lifeline_subprocess_command(command)
                if parent_lifeline
                else list(command)
            ),
            cwd=cwd,
            env=child_env,
            text=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            pass_fds=pass_fds,
        )
        if lifeline_read_fd >= 0:
            os.close(lifeline_read_fd)
        timed_out = False
        try:
            while proc.poll() is None:
                if deadline is not None and time.monotonic() >= deadline:
                    timed_out = True
                    terminate_process_group(proc)
                    handle.write("heavy_cycle_timeout\n")
                    break
                time.sleep(0.1)
        finally:
            if proc.poll() is None:
                terminate_process_group(proc)
            if lifeline_write_fd >= 0:
                os.close(lifeline_write_fd)
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
        "--strict-service-ready",
    ]
    if since:
        command.extend(["--since", since])
    lifeline_read_fd = -1
    lifeline_write_fd = -1
    producer_env = dict(os.environ)
    pass_fds: tuple[int, ...] = ()
    timed_command = command
    if config.runtime_authority_mode == "isolated_controlled":
        lifeline_read_fd, lifeline_write_fd = os.pipe()
        producer_env["MANGO_CALLS_CONTROLLED_LIFELINE_FD"] = str(
            lifeline_read_fd
        )
        pass_fds = (lifeline_read_fd,)
        timed_command = parent_lifeline_subprocess_command(command)
    try:
        proc = subprocess.run(
            timed_command,
            cwd=Path(__file__).resolve().parents[3],
            env=producer_env,
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            start_new_session=(
                config.runtime_authority_mode == "isolated_controlled"
            ),
            pass_fds=pass_fds,
        )
    finally:
        if lifeline_read_fd >= 0:
            os.close(lifeline_read_fd)
        if lifeline_write_fd >= 0:
            os.close(lifeline_write_fd)
    report = read_json(report_path)
    return {"status": "ok" if proc.returncode == 0 else "failed", "rc": proc.returncode, **report}


def parent_lifeline_subprocess_command(
    command: Sequence[str],
) -> list[str]:
    helper = Path(__file__).resolve().parents[3] / "scripts" / "run_parent_lifeline.py"
    return [sys.executable, str(helper), *command]


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
        enumeration_evidence_sha256 = capture_enumeration_evidence_sha256(
            evidence,
            expected_source_mode=(
                "strict_service" if config.strict_ready_provenance else None
            ),
        )
        capture_proof = (
            capture_enumeration_exact_projection(
                evidence,
                expected_source_mode="strict_service",
            )
            if config.strict_ready_provenance
            else None
        )
        capture_proof_sha256 = (
            _canonical_json_sha256(capture_proof)
            if capture_proof is not None
            else ready_capture_proof_sha256(evidence)
        )
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
        source_dual = source.get("dual_enumeration")
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
            "controlled_capture": evidence.get("controlled_capture"),
            "enumeration_evidence_sha256": enumeration_evidence_sha256,
            "capture_proof_sha256": capture_proof_sha256,
            "capture_proof": capture_proof,
            "capture_proof_run_id": (
                source_dual.get("proof_run_id")
                if isinstance(source_dual, Mapping)
                else None
            ),
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


def capture_enumeration_evidence_sha256(
    evidence: Mapping[str, Any],
    *,
    expected_source_mode: Optional[str] = None,
    expected_until: Any = None,
    expected_rolling_since: Any = None,
) -> str:
    """Digest semantic day evidence for deterministic ready-generation reuse.

    This intentionally ignores polling geometry and operational telemetry.
    Cursor certificates must use :func:`capture_enumeration_exact_sha256`.
    """

    validation_errors = validate_capture_enumeration_evidence(
        evidence,
        expected_source_mode=expected_source_mode,
        expected_until=expected_until,
        expected_rolling_since=expected_rolling_since,
    )
    if validation_errors:
        raise RuntimeError(
            "invalid capture enumeration evidence: "
            + ",".join(validation_errors)
        )
    calls_by_day = evidence.get("calls_by_moscow_day")
    zero_by_day = evidence.get("independent_zero_enumerations_by_day")
    source = evidence.get("mango_enumeration_source")
    source = source if isinstance(source, Mapping) else {}

    def canonical_day_keys(mapping: Any, *, label: str) -> set[str]:
        if not isinstance(mapping, Mapping):
            return set()
        result: set[str] = set()
        for raw_day in mapping:
            if not isinstance(raw_day, str):
                raise RuntimeError(f"{label} contains a non-string day key")
            try:
                canonical = datetime.fromisoformat(raw_day).date().isoformat()
            except ValueError:
                canonical = ""
            if raw_day != canonical:
                raise RuntimeError(f"{label} contains a non-canonical day key")
            result.add(raw_day)
        return result

    day_keys = sorted(
        canonical_day_keys(calls_by_day, label="calls_by_moscow_day")
        | canonical_day_keys(
            zero_by_day,
            label="independent_zero_enumerations_by_day",
        )
    )
    days: dict[str, Mapping[str, Any]] = {}
    for day_key in day_keys:
        raw_calls = (
            calls_by_day.get(day_key)
            if isinstance(calls_by_day, Mapping)
            else None
        )
        normalized_calls = (
            sorted(
                str(value).strip()
                for value in raw_calls
                if str(value or "").strip()
            )
            if isinstance(raw_calls, Sequence)
            and not isinstance(raw_calls, (str, bytes))
            else None
        )
        try:
            parsed_day = datetime.fromisoformat(day_key).date()
        except ValueError:
            covered = full_day_covered = False
        else:
            covered = enumeration_source_covers_day(source, parsed_day)
            full_day_covered = enumeration_source_covers_day(
                source, parsed_day, require_full_day=True
            )
        days[day_key] = {
            "call_keys": normalized_calls,
            "zero_proofs": min(
                2,
                positive_int(
                    zero_by_day.get(day_key)
                    if isinstance(zero_by_day, Mapping)
                    else 0
                ),
            ),
            "covered": covered,
            "full_day_covered": full_day_covered,
        }
    projected = {
        "mango_enumeration_complete": evidence.get(
            "mango_enumeration_complete"
        ),
        "source_mode": source.get("mode"),
        "catch_up": source.get("catch_up"),
        "dual_enumeration_contract": (
            {
                "schema_version": source["dual_enumeration"].get(
                    "schema_version"
                ),
                "normalization_version": source["dual_enumeration"].get(
                    "normalization_version"
                ),
                "passes_required": source["dual_enumeration"].get(
                    "passes_required"
                ),
                "enumeration_consistency_ok": source[
                    "dual_enumeration"
                ].get("enumeration_consistency_ok"),
            }
            if isinstance(source.get("dual_enumeration"), Mapping)
            else None
        ),
        "days": days,
        "call_keys_fallback": (
            None
            if isinstance(calls_by_day, Mapping)
            else sorted(
                str(value).strip()
                for value in evidence.get("call_keys", ())
                if str(value or "").strip()
            )
            if isinstance(evidence.get("call_keys"), Sequence)
            and not isinstance(evidence.get("call_keys"), (str, bytes))
            else None
        ),
        "independent_zero_enumerations_fallback": (
            None
            if isinstance(zero_by_day, Mapping)
            else min(
                2,
                positive_int(evidence.get("independent_zero_enumerations")),
            )
        ),
    }
    serialized = json.dumps(
        projected,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def ready_capture_proof_sha256(evidence: Mapping[str, Any]) -> str:
    source = evidence.get("mango_enumeration_source")
    if isinstance(source, Mapping) and source.get("mode") == "strict_service":
        return capture_enumeration_exact_sha256(
            evidence,
            expected_source_mode="strict_service",
        )
    return capture_enumeration_evidence_sha256(evidence)


def capture_enumeration_legacy_exact_sha256(
    evidence: Mapping[str, Any],
) -> str:
    """Reproduce the v1 certificate projection for one migration capture.

    This is never accepted by workers or ready publication.  It exists only
    so an already certified single-pass cursor can anchor the next dual API
    capture without silently discarding its continuity boundary.
    """

    source = evidence.get("mango_enumeration_source")
    source = source if isinstance(source, Mapping) else {}
    raw_intervals = source.get("covered_intervals")
    intervals = raw_intervals if isinstance(raw_intervals, Sequence) else ()
    calls_by_day = evidence.get("calls_by_moscow_day")
    zero_by_day = evidence.get("independent_zero_enumerations_by_day")
    projected = {
        "mango_enumeration_complete": evidence.get(
            "mango_enumeration_complete"
        ),
        "mango_enumeration_source": {
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
            )
        }
        | {
            "covered_intervals": [
                {
                    key: interval.get(key)
                    for key in (
                        "since",
                        "until",
                        "result_complete",
                        "rows",
                        "scope",
                    )
                }
                for interval in intervals
                if isinstance(interval, Mapping)
            ]
        },
        "call_keys": list(evidence.get("call_keys") or ()),
        "calls_by_moscow_day": {
            key: list(calls_by_day[key]) for key in sorted(calls_by_day)
        }
        if isinstance(calls_by_day, Mapping)
        else None,
        "independent_zero_enumerations_by_day": {
            key: zero_by_day[key] for key in sorted(zero_by_day)
        }
        if isinstance(zero_by_day, Mapping)
        else None,
        "api_requests": evidence.get("api_requests"),
        "api_rows_total": evidence.get("api_rows_total"),
        "api_authoritative_rows_total": evidence.get(
            "api_authoritative_rows_total"
        ),
        "api_events_total": evidence.get("api_events_total"),
    }
    return _canonical_json_sha256(projected)


def capture_enumeration_exact_projection(
    evidence: Mapping[str, Any],
    *,
    expected_source_mode: Optional[str] = None,
    expected_until: Any = None,
    expected_rolling_since: Any = None,
) -> Mapping[str, Any]:
    """Build the exact strict API proof projection.

    Ready publication intentionally ignores harmless polling telemetry.  A
    cursor certificate has a different job: every executed interval, row
    count and canonical call/zero collection must remain byte-semantically
    identical after it is signed.
    """

    validation_errors = validate_capture_enumeration_evidence(
        evidence,
        expected_source_mode=expected_source_mode,
        expected_until=expected_until,
        expected_rolling_since=expected_rolling_since,
    )
    if validation_errors:
        raise RuntimeError(
            "invalid capture enumeration evidence: "
            + ",".join(validation_errors)
        )
    source = evidence.get("mango_enumeration_source")
    source = source if isinstance(source, Mapping) else {}
    raw_intervals = source.get("covered_intervals")
    intervals = raw_intervals if isinstance(raw_intervals, Sequence) else ()
    calls_by_day = evidence.get("calls_by_moscow_day")
    zero_by_day = evidence.get("independent_zero_enumerations_by_day")
    projected = {
        "mango_enumeration_complete": evidence.get(
            "mango_enumeration_complete"
        ),
        "mango_enumeration_source": {
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
        }
        | {
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
                if isinstance(interval, Mapping)
            ]
        },
        "call_keys": list(evidence.get("call_keys") or ()),
        "calls_by_moscow_day": {
            key: list(calls_by_day[key])
            for key in sorted(calls_by_day)
        }
        if isinstance(calls_by_day, Mapping)
        else None,
        "independent_zero_enumerations_by_day": {
            key: zero_by_day[key] for key in sorted(zero_by_day)
        }
        if isinstance(zero_by_day, Mapping)
        else None,
        "api_requests": evidence.get("api_requests"),
        "api_rows_total": evidence.get("api_rows_total"),
        "api_authoritative_rows_total": evidence.get(
            "api_authoritative_rows_total"
        ),
        "api_auxiliary_rows_total": evidence.get(
            "api_auxiliary_rows_total"
        ),
        "api_events_total": evidence.get("api_events_total"),
    }
    if isinstance(evidence.get("controlled_capture"), Mapping):
        projected["controlled_capture"] = evidence["controlled_capture"]
    return projected


def capture_enumeration_exact_sha256(
    evidence: Mapping[str, Any],
    *,
    expected_source_mode: Optional[str] = None,
    expected_until: Any = None,
    expected_rolling_since: Any = None,
) -> str:
    projected = capture_enumeration_exact_projection(
        evidence,
        expected_source_mode=expected_source_mode,
        expected_until=expected_until,
        expected_rolling_since=expected_rolling_since,
    )
    return _canonical_json_sha256(projected)


def certify_capture_window(
    config: CallsTwoProcessesConfig,
    capture: Mapping[str, Any],
    *,
    requested_since: datetime,
    requested_until: datetime,
    enumeration_evidence_sha256: str,
) -> Mapping[str, Any]:
    """Bind a validated strict capture to the caller's exact request window."""

    if not config.strict_ready_provenance:
        return dict(capture)
    source = capture.get("mango_enumeration_source")
    dual_proof = (
        source.get("dual_enumeration")
        if isinstance(source, Mapping)
        else None
    )
    if not (
        isinstance(dual_proof, Mapping)
        and dual_proof.get("tenant_id") == config.tenant_id
        and dual_proof.get("base_url") == config.base_url
        and dual_proof.get("fields_sha256")
        == _canonical_json_sha256(DEFAULT_STATS_FIELDS)
    ):
        raise RuntimeError("capture dual enumeration identity is invalid")
    expected_rolling_since = capture_rolling_window_start(
        config,
        since=requested_since,
        until=requested_until,
    )
    body: dict[str, Any] = {
        "schema_version": CAPTURE_WINDOW_CERTIFICATE_SCHEMA,
        "requested_since": requested_since.astimezone(timezone.utc).isoformat(),
        "requested_until": requested_until.astimezone(timezone.utc).isoformat(),
        "expected_rolling_since": expected_rolling_since.astimezone(
            timezone.utc
        ).isoformat(),
        "pending_recording_retry_hours": config.pending_recording_retry_hours,
        "tenant_id": config.tenant_id,
        "base_url": config.base_url,
        "enumeration_evidence_sha256": enumeration_evidence_sha256,
        "manifest_end_offset": capture.get("manifest_end_offset"),
        "manifest_snapshot_sha256": capture.get("manifest_snapshot_sha256"),
        "expected_code_sha": config.expected_code_sha,
        "host_id": capture.get("host_id"),
    }
    serialized = json.dumps(
        body,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    body["certificate_sha256"] = hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()
    return {**dict(capture), "capture_window_certificate": body}


def verified_capture_window(
    config: CallsTwoProcessesConfig,
    capture: Mapping[str, Any],
    *,
    allow_pre_dual_anchor: bool = False,
) -> tuple[datetime, datetime, datetime]:
    """Verify and return the request window sealed by ``run_capture``."""

    certificate = capture.get("capture_window_certificate")
    if not isinstance(certificate, Mapping):
        raise RuntimeError("capture window certificate is missing")
    body = {
        key: value
        for key, value in certificate.items()
        if key != "certificate_sha256"
    }
    serialized = json.dumps(
        body,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    expected_certificate_sha256 = hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()
    certificate_schema = certificate.get("schema_version")
    legacy_anchor = bool(
        certificate_schema == LEGACY_CAPTURE_WINDOW_CERTIFICATE_SCHEMA
        and allow_pre_dual_anchor
    )
    if (
        certificate_schema not in {
            CAPTURE_WINDOW_CERTIFICATE_SCHEMA,
            *(
                (LEGACY_CAPTURE_WINDOW_CERTIFICATE_SCHEMA,)
                if allow_pre_dual_anchor
                else ()
            ),
        }
        or certificate.get("certificate_sha256")
        != expected_certificate_sha256
        or certificate.get("pending_recording_retry_hours")
        != config.pending_recording_retry_hours
        or certificate.get("tenant_id") != config.tenant_id
        or certificate.get("base_url") != config.base_url
        or not re.fullmatch(
            r"[0-9a-f]{40}",
            str(certificate.get("expected_code_sha") or ""),
        )
        or certificate.get("host_id") != capture.get("host_id")
        or certificate.get("manifest_end_offset")
        != capture.get("manifest_end_offset")
        or certificate.get("manifest_snapshot_sha256")
        != capture.get("manifest_snapshot_sha256")
    ):
        raise RuntimeError("capture window certificate is invalid")
    manifest_end_offset = certificate.get("manifest_end_offset")
    manifest_snapshot_sha256 = certificate.get("manifest_snapshot_sha256")
    if (
        isinstance(manifest_end_offset, bool)
        or not isinstance(manifest_end_offset, int)
        or manifest_end_offset < 0
        or not re.fullmatch(
            r"[0-9a-f]{64}", str(manifest_snapshot_sha256 or "")
        )
    ):
        raise RuntimeError("capture window certificate manifest is invalid")
    try:
        manifest_snapshot = capture_manifest_snapshot(
            config.capture_manifest,
            end_offset=manifest_end_offset,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "capture window certificate manifest prefix is unavailable"
        ) from exc
    if (
        manifest_snapshot.get("end_offset") != manifest_end_offset
        or manifest_snapshot.get("sha256") != manifest_snapshot_sha256
    ):
        raise RuntimeError(
            "capture window certificate manifest prefix changed"
        )
    try:
        requested_since = parse_datetime(
            str(certificate.get("requested_since") or "")
        )
        requested_until = parse_datetime(
            str(certificate.get("requested_until") or "")
        )
        expected_rolling_since = parse_datetime(
            str(certificate.get("expected_rolling_since") or "")
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError("capture window certificate timestamps are invalid") from exc
    if requested_since >= requested_until or expected_rolling_since != (
        capture_rolling_window_start(
            config,
            since=requested_since,
            until=requested_until,
        )
    ):
        raise RuntimeError("capture window certificate boundaries are invalid")
    cursor_until = capture.get("until")
    if cursor_until is None or parse_datetime(str(cursor_until)) != requested_until:
        raise RuntimeError("capture cursor until differs from its certificate")
    if legacy_anchor:
        source = capture.get("mango_enumeration_source")
        intervals = (
            source.get("covered_intervals")
            if isinstance(source, Mapping)
            else None
        )
        if (
            capture.get("mango_enumeration_complete") is not True
            or not isinstance(source, Mapping)
            or source.get("mode") != "strict_service"
            or parse_datetime(str(source.get("rolling_since") or ""))
            != expected_rolling_since
            or parse_datetime(str(source.get("until") or ""))
            != requested_until
            or not isinstance(intervals, Sequence)
            or isinstance(intervals, (str, bytes))
            or not intervals
        ):
            raise RuntimeError("legacy capture anchor evidence is invalid")
        coverage_cursor = expected_rolling_since
        try:
            for interval in sorted(
                (
                    item
                    for item in intervals
                    if isinstance(item, Mapping)
                    and item.get("scope") == "rolling_authority"
                    and item.get("result_complete") is True
                ),
                key=lambda item: parse_datetime(str(item.get("since") or "")),
            ):
                interval_since = parse_datetime(
                    str(interval.get("since") or "")
                )
                interval_until = parse_datetime(
                    str(interval.get("until") or "")
                )
                if interval_since > coverage_cursor:
                    raise RuntimeError
                coverage_cursor = max(coverage_cursor, interval_until)
        except (RuntimeError, TypeError, ValueError) as exc:
            raise RuntimeError("legacy capture anchor coverage is invalid") from exc
        if coverage_cursor < requested_until:
            raise RuntimeError("legacy capture anchor coverage is incomplete")
        evidence_sha256 = capture_enumeration_legacy_exact_sha256(capture)
    else:
        source = capture.get("mango_enumeration_source")
        dual_proof = (
            source.get("dual_enumeration")
            if isinstance(source, Mapping)
            else None
        )
        if not (
            isinstance(dual_proof, Mapping)
            and dual_proof.get("tenant_id") == certificate.get("tenant_id")
            and dual_proof.get("base_url") == certificate.get("base_url")
            and dual_proof.get("fields_sha256")
            == _canonical_json_sha256(DEFAULT_STATS_FIELDS)
        ):
            raise RuntimeError("capture window certificate API identity changed")
        evidence_sha256 = capture_enumeration_exact_sha256(
            capture,
            expected_source_mode="strict_service",
            expected_until=requested_until,
            expected_rolling_since=expected_rolling_since,
        )
    if certificate.get("enumeration_evidence_sha256") != evidence_sha256:
        raise RuntimeError("capture window certificate evidence changed")
    return requested_since, requested_until, expected_rolling_since


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
        evidence_same = manifest.get(
            "enumeration_evidence_sha256"
        ) == capture_enumeration_evidence_sha256(
            evidence,
            expected_source_mode=(
                "strict_service" if config.strict_ready_provenance else None
            ),
        )
        capture_proof_same = manifest.get(
            "capture_proof_sha256"
        ) == ready_capture_proof_sha256(evidence)
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
        verdicts = manifest.get("daily_verdicts")
        time_sensitive_pending = bool(
            isinstance(verdicts, Mapping)
            and any(
                isinstance(verdict, Mapping)
                and positive_int(verdict.get("pending_unique")) > 0
                for verdict in verdicts.values()
            )
        )
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
            and evidence_same
            and capture_proof_same
            and not time_sensitive_pending
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
    diagnostic = {"type": type(exc).__name__}
    code = str(exc).strip()
    if code.startswith("controlled_") and re.fullmatch(
        r"[a-z0-9_.:-]{1,160}", code
    ):
        diagnostic["code"] = code
    return diagnostic


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
        checked_age = (current - checked).total_seconds() if checked else None
        data_age = (current - data_at).total_seconds() if data_at else None
        status = (
            "missing"
            if checked is None
            else "future"
            if (checked_age is not None and checked_age < 0)
            or (data_age is not None and data_age < 0)
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
        elif state.get("status") not in {"ok", "idle"}:
            raw_status = str(state.get("status") or "")
            status = (
                raw_status
                if raw_status in {"blocked", "deferred", "locked"}
                else "invalid"
            )
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
                (current - heartbeat_at).total_seconds() if heartbeat_at else None
            )
        except (TypeError, ValueError):
            heartbeat_age = None
        heartbeat_pid = positive_int(heartbeat.get("pid"))
        heartbeat_live = bool(
            heartbeat_age is not None
            and 0 <= heartbeat_age <= 90
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
            "controlled_stage_contract_ok": item.get(
                "controlled_stage_contract_ok"
            ),
            "orchestrator_stop_reason": item.get(
                "orchestrator_stop_reason"
            ),
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
            "capture_window_certificate": capture.get(
                "capture_window_certificate"
            ),
            "mango_enumeration_complete": capture.get(
                "mango_enumeration_complete"
            ),
            "enumeration_consistency_ok": capture.get(
                "enumeration_consistency_ok"
            ),
            "mango_enumeration_source": capture.get(
                "mango_enumeration_source"
            ),
            "catch_up": bool(capture.get("catch_up")),
            "sla_mode": capture.get("sla_mode") or "live",
            "call_keys": capture.get("call_keys"),
            "calls_by_moscow_day": capture.get("calls_by_moscow_day"),
            "api_requests": capture.get("api_requests"),
            "api_rows_total": capture.get("api_rows_total"),
            "api_authoritative_rows_total": capture.get(
                "api_authoritative_rows_total"
            ),
            "api_auxiliary_rows_total": capture.get(
                "api_auxiliary_rows_total"
            ),
            "api_events_total": capture.get("api_events_total"),
            "independent_zero_enumerations_by_day": capture.get(
                "independent_zero_enumerations_by_day"
            ),
            "controlled_capture": capture.get("controlled_capture"),
        },
    )


def parse_datetime(value: str) -> datetime:
    text = value.strip().replace("Z", "+00:00")
    if text.isdigit():
        return datetime.fromtimestamp(int(text), tz=timezone.utc)
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


def read_capture_cursor_snapshot(
    path: Path,
) -> tuple[Mapping[str, Any], Optional[str]]:
    if not os.path.lexists(path):
        return {}, None
    raw = read_stable_regular_bytes(
        path,
        label="mango_capture_cursor",
        owner_only_mode=0o600,
    )
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("existing capture cursor is malformed") from exc
    if not isinstance(payload, Mapping) or not payload:
        raise RuntimeError("existing capture cursor is malformed")
    return payload, hashlib.sha256(raw).hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_private_json(path, payload, indent=2)
