from __future__ import annotations

import fcntl
import json
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Sequence

from mango_mvp.customer_profile.builder import CustomerProfileBuilder, CustomerProfileBuildOptions
from mango_mvp.customer_timeline.bot_safe_summary import BotSafeSummaryBuildConfig, build_bot_safe_summaries
from mango_mvp.customer_timeline.contracts import (
    BotContextChunk,
    IdentityMatchClass,
    TimelineDirection,
    TimelineEvent,
    TimelineEventType,
)
from mango_mvp.customer_timeline.ids import normalize_key, require_text, stable_digest
from mango_mvp.customer_timeline.ingestion import (
    AmoSnapshotNormalizer,
    MangoCallSummaryNormalizer,
    TallantoSnapshotNormalizer,
    TimelineImportService,
    TimelineNormalizedBatch,
    TimelineNormalizer,
    TimelineSourceRecord,
)
from mango_mvp.customer_timeline.store import CustomerTimelineSQLiteStore


NIGHTLY_INCREMENTAL_SCHEMA_VERSION = "customer_timeline_nightly_incremental_v1"
DEFAULT_SAFETY_MARGIN_SECONDS = 300
DEFAULT_LOCK_POLL_SECONDS = 0.2


@dataclass(frozen=True)
class IncrementalSourceConfig:
    name: str
    source_system: str
    path: Path
    tenant_id: str = "foton"
    source_ref: Optional[str] = None
    normalizer: str = "jsonl"
    required: bool = True
    ignore_cursor: bool = False
    preserve_cursor: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", normalize_key(self.name, "source name"))
        object.__setattr__(self, "source_system", normalize_key(self.source_system, "source_system"))
        object.__setattr__(self, "tenant_id", normalize_key(self.tenant_id, "tenant_id"))
        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(self, "normalizer", normalize_key(self.normalizer, "normalizer"))
        if self.source_ref is not None:
            object.__setattr__(self, "source_ref", require_text(self.source_ref, "source_ref"))

    @property
    def effective_source_ref(self) -> str:
        return self.source_ref or f"{self.source_system}:{self.path.name}"


@dataclass(frozen=True)
class ProfileRebuildConfig:
    profiles_db: Path
    master_calls_db: Optional[Path] = None
    build_id: Optional[str] = None


@dataclass(frozen=True)
class BotSafeRebuildConfig:
    allowed_root: Path
    apply: bool = False
    limit: Optional[int] = None


@dataclass(frozen=True)
class NightlyIncrementalConfig:
    timeline_db: Path
    allowed_root: Path
    sources: Sequence[IncrementalSourceConfig]
    journal_path: Path
    tenant_id: str = "foton"
    safety_margin_seconds: int = DEFAULT_SAFETY_MARGIN_SECONDS
    lock_timeout_seconds: float = 30.0
    actor: str = "customer_timeline_nightly_incremental"
    profile_rebuild: Optional[ProfileRebuildConfig] = None
    bot_safe_rebuild: Optional[BotSafeRebuildConfig] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "timeline_db", Path(self.timeline_db))
        object.__setattr__(self, "allowed_root", Path(self.allowed_root))
        object.__setattr__(self, "journal_path", Path(self.journal_path))
        object.__setattr__(self, "tenant_id", normalize_key(self.tenant_id, "tenant_id"))
        object.__setattr__(self, "sources", tuple(self.sources))
        if self.safety_margin_seconds < 0:
            raise ValueError("safety_margin_seconds must not be negative")
        if self.lock_timeout_seconds < 0:
            raise ValueError("lock_timeout_seconds must not be negative")


@dataclass(frozen=True)
class SourceLoadResult:
    source: IncrementalSourceConfig
    cursor_before: Optional[str]
    fetch_from: Optional[datetime]
    rows_total: int
    rows_selected: int
    records: Sequence[TimelineSourceRecord]
    max_source_ts: Optional[datetime]
    affected_customer_ids: Sequence[str]
    would_change_customer_ids: Sequence[str]
    skipped_reason: Optional[str] = None

    def to_json_dict(self) -> Mapping[str, Any]:
        return {
            "source": asdict(self.source) | {"path": str(self.source.path)},
            "cursor_before": self.cursor_before,
            "fetch_from": self.fetch_from.isoformat() if self.fetch_from else None,
            "rows_total": self.rows_total,
            "rows_selected": self.rows_selected,
            "records": len(self.records),
            "max_source_ts": self.max_source_ts.isoformat() if self.max_source_ts else None,
            "affected_customer_ids": list(self.affected_customer_ids),
            "would_change_customer_ids": list(self.would_change_customer_ids),
            "skipped_reason": self.skipped_reason,
            "required": self.source.required,
            "status": "failed" if self.skipped_reason and self.source.required else (
                "skipped" if self.skipped_reason else "ok"
            ),
        }


class JsonlTimelineNormalizer(TimelineNormalizer):
    def __init__(self, source_system: str) -> None:
        self.source_system = normalize_key(source_system, "source_system")

    def normalize(self, record: TimelineSourceRecord) -> TimelineNormalizedBatch:
        payload = dict(record.payload)
        customer_id = optional_string(payload.get("customer_id"))
        if not customer_id:
            return TimelineNormalizedBatch(source_record=record)
        event_at = parse_datetime(payload.get("event_at") or payload.get("created_at") or payload.get("updated_at"), "event_at")
        event_type = TimelineEventType(str(payload.get("event_type") or TimelineEventType.SYSTEM_NOTE.value))
        direction = TimelineDirection(str(payload.get("direction") or TimelineDirection.SYSTEM.value))
        source_id = require_text(payload.get("source_id") or payload.get("id") or record.source_ref, "source_id")
        match_status = IdentityMatchClass(str(payload.get("match_status") or IdentityMatchClass.STRONG_UNIQUE.value))
        event = TimelineEvent(
            tenant_id=record.payload.get("tenant_id") or payload.get("tenant_id") or "foton",
            customer_id=customer_id,
            opportunity_id=optional_string(payload.get("opportunity_id")),
            event_type=event_type,
            event_at=event_at,
            source_system=self.source_system,
            source_id=source_id,
            source_ref=record.source_ref,
            direction=direction,
            actor_name=optional_string(payload.get("actor_name")),
            subject=optional_string(payload.get("subject")),
            text_preview=optional_string(payload.get("text_preview") or payload.get("text")),
            summary=optional_string(payload.get("summary") or payload.get("text")),
            importance=int(payload.get("importance") or 0),
            match_status=match_status,
            confidence=float(payload.get("confidence") or 0.9),
            record={"payload": payload.get("record") if isinstance(payload.get("record"), Mapping) else payload},
            metadata={
                "source_updated_at": normalized_timestamp(payload),
                "brand": optional_string(payload.get("brand")),
                "incremental_source": True,
            },
            created_at=parse_datetime(payload.get("created_at") or payload.get("updated_at") or event_at, "created_at"),
        )
        chunks: list[BotContextChunk] = []
        chunk_text = optional_string(payload.get("bot_context_text"))
        if chunk_text:
            allowed_for_bot = bool(payload.get("allowed_for_bot"))
            requires_manager_review = bool(payload.get("requires_manager_review", not allowed_for_bot))
            chunks.append(
                BotContextChunk(
                    tenant_id=event.tenant_id,
                    customer_id=customer_id,
                    opportunity_id=event.opportunity_id,
                    event_id=event.event_id,
                    source_system=self.source_system,
                    source_ref=event.source_ref,
                    chunk_type=str(payload.get("chunk_type") or "incremental_context"),
                    text=chunk_text,
                    summary=optional_string(payload.get("bot_context_summary")) or chunk_text[:160],
                    event_at=event.event_at,
                    freshness_score=float(payload.get("freshness_score") or 0.5),
                    relevance_tags=tuple(str(item) for item in payload.get("relevance_tags") or ()),
                    allowed_for_bot=allowed_for_bot,
                    requires_manager_review=requires_manager_review,
                    created_at=event.created_at,
                )
            )
        return TimelineNormalizedBatch(source_record=record, events=(event,), bot_context_chunks=tuple(chunks))


class MailArchiveStage2IncrementalNormalizer(TimelineNormalizer):
    source_system = "mail_archive_stage2"

    def __init__(self, *, tenant_id: str) -> None:
        self.tenant_id = normalize_key(tenant_id, "tenant_id")

    def normalize(self, record: TimelineSourceRecord) -> TimelineNormalizedBatch:
        payload = dict(record.payload)
        message_sha = optional_string(
            payload.get("message_sha256")
            or payload.get("sha256")
            or payload.get("sha")
            or payload.get("source_id")
            or payload.get("id")
        )
        if not message_sha:
            message_sha = stable_digest(
                {
                    "source_ref": record.source_ref,
                    "subject": payload.get("subject"),
                    "event_at": normalized_timestamp(payload),
                }
            )
        event_at = parse_datetime(normalized_timestamp(payload), "event_at")
        customer_id = optional_string(payload.get("customer_id") or payload.get("resolved_customer_id"))
        subject = optional_string(payload.get("subject")) or "Email message"
        summary = optional_string(
            payload.get("summary")
            or payload.get("thread_summary")
            or payload.get("text_preview")
            or payload.get("body_preview")
            or payload.get("full_clean_text")
            or subject
        )
        preview = optional_string(payload.get("text_preview") or payload.get("body_preview") or summary)
        direction = mail_direction(payload)
        match_status = IdentityMatchClass(
            optional_string(payload.get("match_status"))
            or (IdentityMatchClass.STRONG_UNIQUE.value if customer_id else IdentityMatchClass.UNMATCHED.value)
        )
        confidence = float(payload.get("confidence") if payload.get("confidence") is not None else (0.9 if customer_id else 0.0))
        pending_attribution = payload.get("pending_attribution")
        if pending_attribution is None:
            pending_attribution = not bool(customer_id)
        metadata = {
            "source_updated_at": normalized_timestamp(payload),
            "brand": optional_string(payload.get("brand")),
            "summary_status": optional_string(payload.get("summary_status")) or "needs_summary_later",
            "needs_summary_later": bool(payload.get("needs_summary_later", True)),
            "incremental_source": True,
            "pending_attribution": bool(pending_attribution),
        }
        if payload.get("pending_reason"):
            metadata["pending_reason"] = optional_string(payload.get("pending_reason"))
        if payload.get("fresh_relink") is not None:
            metadata["fresh_relink"] = bool(payload.get("fresh_relink"))
        if isinstance(payload.get("mail_link_enrich"), Mapping):
            metadata["mail_link_enrich"] = dict(payload["mail_link_enrich"])
        event = TimelineEvent(
            tenant_id=self.tenant_id,
            customer_id=customer_id,
            event_type=TimelineEventType.EMAIL_MESSAGE,
            event_at=event_at,
            source_system=self.source_system,
            source_id=message_sha,
            source_ref=record.source_ref,
            direction=direction,
            subject=subject,
            text_preview=preview[:240] if preview else None,
            summary=summary[:1200] if summary else None,
            match_status=match_status,
            confidence=confidence,
            record={"payload": payload},
            metadata=metadata,
            created_at=event_at,
        )
        chunks: list[BotContextChunk] = []
        chunk_text = optional_string(payload.get("bot_context_text") or summary)
        if customer_id and chunk_text:
            chunks.append(
                BotContextChunk(
                    tenant_id=self.tenant_id,
                    customer_id=customer_id,
                    event_id=event.event_id,
                    source_system=self.source_system,
                    source_ref=event.source_ref,
                    chunk_type="email_message",
                    text=chunk_text,
                    summary=chunk_text[:500],
                    event_at=event.event_at,
                    freshness_score=0.7,
                    relevance_tags=("email", "mail_archive_stage2", "manager_only"),
                    allowed_for_bot=False,
                    requires_manager_review=True,
                    metadata={
                        "message_sha256": message_sha,
                        "needs_summary_later": bool(payload.get("needs_summary_later", True)),
                    },
                    created_at=event.created_at,
                )
            )
        return TimelineNormalizedBatch(source_record=record, events=(event,), bot_context_chunks=tuple(chunks))


class AmoEventNormalizer(TimelineNormalizer):
    source_system = "amocrm_event"

    def __init__(self, *, tenant_id: str) -> None:
        self.tenant_id = normalize_key(tenant_id, "tenant_id")

    def normalize(self, record: TimelineSourceRecord) -> TimelineNormalizedBatch:
        payload = dict(record.payload)
        customer_id = optional_string(payload.get("customer_id"))
        if not customer_id:
            return TimelineNormalizedBatch(source_record=record)
        event_at = parse_datetime(payload.get("event_at") or payload.get("created_at"), "event_at")
        amo_event_type = optional_string(payload.get("amo_event_type") or payload.get("type")) or "amo_event"
        entity_type = optional_string(payload.get("entity_type")) or "unknown"
        entity_id = optional_string(payload.get("entity_id")) or "unknown"
        event_id = require_text(payload.get("event_id") or payload.get("id") or record.source_ref, "event_id")
        source_id = f"{event_id}:{stable_digest({'body': payload.get('text_preview') or payload.get('summary') or '', 'updated_at': payload.get('updated_at') or payload.get('created_at')})[:12]}"
        direction = direction_for_amo_event_type(amo_event_type)
        source_ref = f"amocrm:event:{event_id}"
        subject = optional_string(payload.get("subject")) or amo_event_type
        body_status = optional_string(payload.get("source_body_status")) or "event_only"
        text = optional_string(payload.get("text_preview") or payload.get("summary")) or f"AMO event: {amo_event_type}"
        event = TimelineEvent(
            tenant_id=self.tenant_id,
            customer_id=customer_id,
            opportunity_id=optional_string(payload.get("opportunity_id")),
            event_type=TimelineEventType.AMO_NOTE,
            event_at=event_at,
            source_system=self.source_system,
            source_id=source_id,
            source_ref=source_ref,
            direction=direction,
            actor_name=optional_string(payload.get("actor_name")),
            actor_ref=optional_string(payload.get("actor_ref")),
            subject=subject,
            text_preview=text[:240],
            summary=optional_string(payload.get("summary")) or text[:240],
            match_status=IdentityMatchClass.STRONG_UNIQUE,
            confidence=float(payload.get("confidence") or 0.75),
            record={
                "entity_type": entity_type,
                "entity_id": entity_id,
                "amo_event_type": amo_event_type,
                "source_body_status": body_status,
                "payload": payload.get("record") if isinstance(payload.get("record"), Mapping) else payload,
            },
            metadata={
                "source_body_status": body_status,
                "source_updated_at": normalized_timestamp(payload),
                "incremental_source": True,
            },
            created_at=event_at,
        )
        chunk = BotContextChunk(
            tenant_id=self.tenant_id,
            customer_id=customer_id,
            opportunity_id=event.opportunity_id,
            event_id=event.event_id,
            source_ref=event.source_ref,
            source_system=self.source_system,
            chunk_type="amo_event_raw",
            text=text,
            summary=text[:160],
            event_at=event.event_at,
            freshness_score=0.5,
            relevance_tags=("amocrm", "event", amo_event_type, body_status),
            allowed_for_bot=False,
            requires_manager_review=True,
            metadata={"source_body_status": body_status},
            created_at=event.created_at,
        )
        return TimelineNormalizedBatch(source_record=record, events=(event,), bot_context_chunks=(chunk,))


def normalizer_for_source(source: IncrementalSourceConfig) -> TimelineNormalizer:
    if source.normalizer == "amo_snapshot":
        return AmoSnapshotNormalizer(tenant_id=source.tenant_id)
    if source.normalizer == "amo_event":
        return AmoEventNormalizer(tenant_id=source.tenant_id)
    if source.normalizer == "mango_processed_summary":
        return MangoCallSummaryNormalizer(tenant_id=source.tenant_id)
    if source.normalizer == "mail_archive_stage2":
        return MailArchiveStage2IncrementalNormalizer(tenant_id=source.tenant_id)
    if source.normalizer == "tallanto_snapshot":
        # BLOCK C: reuses the existing TallantoSnapshotNormalizer / import
        # contract (mango_mvp.customer_timeline.ingestion) -- see
        # mango_mvp.customer_timeline.tallanto_cards_sync, the real daily
        # Tallanto Contact sync step that feeds this source kind.
        return TallantoSnapshotNormalizer(tenant_id=source.tenant_id)
    if source.normalizer == "jsonl":
        return JsonlTimelineNormalizer(source.source_system)
    raise ValueError(f"unsupported incremental normalizer: {source.normalizer}")


def direction_for_amo_event_type(event_type: str) -> TimelineDirection:
    value = event_type.casefold()
    if value.startswith("incoming"):
        return TimelineDirection.INBOUND
    if value.startswith("outgoing"):
        return TimelineDirection.OUTBOUND
    if value in {"common_note_added", "common_note_deleted", "common_note_updated"}:
        return TimelineDirection.INTERNAL
    return TimelineDirection.SYSTEM


def run_nightly_incremental(config: NightlyIncrementalConfig) -> Mapping[str, Any]:
    started = datetime.now(timezone.utc)
    phase_started = time.monotonic()
    report: dict[str, Any] = {
        "schema_version": NIGHTLY_INCREMENTAL_SCHEMA_VERSION,
        "started_at": started.isoformat(),
        "timeline_db": str(config.timeline_db),
        "tenant_id": config.tenant_id,
        "safety_margin_seconds": config.safety_margin_seconds,
        "sources": [],
        "source_errors": [],
        "phase_seconds": {},
        "safety": {
            "writes_amo": False,
            "writes_tallanto": False,
            "network_calls": False,
            "writes_customer_timeline": True,
        },
    }
    config.journal_path.parent.mkdir(parents=True, exist_ok=True)
    with single_run_lock(config.timeline_db, timeout_seconds=config.lock_timeout_seconds) as lock_info:
        report["lock"] = lock_info
        with CustomerTimelineSQLiteStore(config.timeline_db, allowed_root=config.allowed_root) as store:
            affected: set[str] = set()
            would_change: set[str] = set()
            import_reports: list[Mapping[str, Any]] = []
            cursor_updates: list[Mapping[str, Any]] = []
            for source in config.sources:
                try:
                    loaded = load_incremental_jsonl_source(
                        store,
                        source,
                        safety_margin_seconds=config.safety_margin_seconds,
                    )
                except Exception as exc:  # fail-soft per source; the service gate decides whether this is blocking.
                    reason = f"source_exception:{type(exc).__name__}"
                    if not source.preserve_cursor:
                        update_source_failure_cursor(store, source, skipped_reason=reason, actor=config.actor)
                    report["sources"].append(source_failure_report(source, reason=reason, error=exc))
                    report["source_errors"].append(source_error(source, reason=reason, error=exc))
                    continue
                report["sources"].append(loaded.to_json_dict())
                if loaded.skipped_reason:
                    if not source.preserve_cursor:
                        update_source_failure_cursor(store, source, skipped_reason=loaded.skipped_reason, actor=config.actor)
                    report["source_errors"].append(source_error(source, reason=loaded.skipped_reason))
                    continue
                if not loaded.records:
                    affected.update(loaded.affected_customer_ids)
                    continue
                try:
                    imported = TimelineImportService(store).import_records(
                        loaded.records,
                        normalizer=normalizer_for_source(source),
                        tenant_id=source.tenant_id,
                        source_ref=source.effective_source_ref,
                        idempotency_key=stable_digest(
                            {
                                "schema_version": NIGHTLY_INCREMENTAL_SCHEMA_VERSION,
                                "source": source.effective_source_ref,
                                "records": [record.payload_hash for record in loaded.records],
                            }
                        ),
                        dry_run=False,
                        actor=config.actor,
                    )
                except Exception as exc:  # keep other sources running if one importer breaks.
                    reason = f"import_exception:{type(exc).__name__}"
                    if not source.preserve_cursor:
                        update_source_failure_cursor(store, source, skipped_reason=reason, actor=config.actor)
                    report["source_errors"].append(source_error(source, reason=reason, error=exc))
                    continue
                affected.update(loaded.affected_customer_ids)
                would_change.update(loaded.would_change_customer_ids)
                imported_payload = imported.to_json_dict()
                import_reports.append(imported_payload)
                if not imported.validation_ok:
                    reason = f"import_validation_failed:rejected_count={imported.rejected_count}"
                    if not source.preserve_cursor:
                        update_source_failure_cursor(store, source, skipped_reason=reason, actor=config.actor)
                    report["source_errors"].append(source_error(source, reason=reason))
                    continue
                if loaded.max_source_ts is not None and not source.preserve_cursor:
                    cursor_ts = loaded.max_source_ts - timedelta(seconds=config.safety_margin_seconds)
                    existing_cursor = store.get_ingestion_cursor(source.tenant_id, source.source_system)
                    persisted_cursor_ts = cursor_ts
                    if existing_cursor is not None and existing_cursor.last_cursor_ts > persisted_cursor_ts:
                        persisted_cursor_ts = existing_cursor.last_cursor_ts
                    cursor = store.upsert_ingestion_cursor(
                        source.tenant_id,
                        source.source_system,
                        last_cursor_ts=persisted_cursor_ts,
                        metadata=merge_cursor_metadata(
                            existing_cursor.metadata if existing_cursor else {},
                            source,
                            last_status="ok",
                            last_cursor_ts=cursor_ts,
                            max_source_ts=loaded.max_source_ts,
                        )
                        | {"max_source_ts": loaded.max_source_ts.isoformat(), "consecutive_failures": 0},
                        actor=config.actor,
                        ingestion_run_id=imported.run_id,
                    )
                    cursor_updates.append(cursor.to_json_dict())
            report["phase_seconds"]["ingest"] = round(time.monotonic() - phase_started, 3)
            selected_customers = sorted(would_change)
            report["affected_customer_ids"] = sorted(affected)
            report["changed_customer_ids"] = selected_customers
            report["affected_customer_count"] = len(affected)
            report["changed_customer_count"] = len(selected_customers)
            report["imports"] = import_reports
            report["cursor_updates"] = cursor_updates
            failed_required_sources = [
                str(item.get("source"))
                for item in report["source_errors"]
                if item.get("required") is True
            ]
            report["failed_required_sources"] = failed_required_sources
            report["overall_status"] = "ok" if not report["source_errors"] else (
                "partial" if failed_required_sources else "ok_with_skipped_optional"
            )
            report["gate_passed"] = not failed_required_sources
    recalc_started = time.monotonic()
    report["rebuild"] = rebuild_affected_outputs(config, customer_ids=report["changed_customer_ids"])
    report["phase_seconds"]["rebuild"] = round(time.monotonic() - recalc_started, 3)
    finished = datetime.now(timezone.utc)
    report["finished_at"] = finished.isoformat()
    report["duration_seconds"] = round((finished - started).total_seconds(), 3)
    append_jsonl(config.journal_path, report)
    return report


def source_error(
    source: IncrementalSourceConfig,
    *,
    reason: str,
    error: Exception | None = None,
) -> Mapping[str, Any]:
    payload: dict[str, Any] = {
        "source": source.name,
        "source_system": source.source_system,
        "required": source.required,
        "reason": reason,
    }
    if error is not None:
        payload["error_type"] = type(error).__name__
        payload["error_message"] = safe_error_message(error)
    return payload


def source_failure_report(
    source: IncrementalSourceConfig,
    *,
    reason: str,
    error: Exception | None = None,
) -> Mapping[str, Any]:
    payload: dict[str, Any] = {
        "source": asdict(source) | {"path": str(source.path)},
        "cursor_before": None,
        "fetch_from": None,
        "rows_total": 0,
        "rows_selected": 0,
        "records": 0,
        "max_source_ts": None,
        "affected_customer_ids": [],
        "would_change_customer_ids": [],
        "skipped_reason": reason,
        "required": source.required,
        "status": "failed" if source.required else "skipped",
    }
    if error is not None:
        payload["error_type"] = type(error).__name__
        payload["error_message"] = safe_error_message(error)
    return payload


def safe_error_message(error: Exception) -> str:
    text = " ".join(str(error or "").replace("\n", " ").split())
    if len(text) > 240:
        text = text[:240].rstrip()
    return text


def load_incremental_jsonl_source(
    store: CustomerTimelineSQLiteStore,
    source: IncrementalSourceConfig,
    *,
    safety_margin_seconds: int,
) -> SourceLoadResult:
    cursor = store.get_ingestion_cursor(source.tenant_id, source.source_system)
    fetch_from = None if source.ignore_cursor else (
        cursor_ts_for_source_ref(cursor.metadata if cursor else None, source) if cursor else None
    )
    if not source.ignore_cursor and fetch_from is None and cursor is not None and not has_source_ref_cursors(cursor.metadata):
        fetch_from = cursor.last_cursor_ts
    if not source.path.exists():
        return SourceLoadResult(
            source=source,
            cursor_before=cursor.last_cursor_ts.isoformat() if cursor else None,
            fetch_from=fetch_from,
            rows_total=0,
            rows_selected=0,
            records=(),
            max_source_ts=None,
            affected_customer_ids=(),
            would_change_customer_ids=(),
            skipped_reason="source_unavailable",
        )
    rows = read_jsonl(source.path)
    selected_rows = []
    max_ts: Optional[datetime] = None
    affected: set[str] = set()
    would_change: set[str] = set()
    records: list[TimelineSourceRecord] = []
    normalizer = normalizer_for_source(source)
    for row in rows:
        ts = parse_datetime(normalized_timestamp(row), "source_timestamp")
        max_ts = ts if max_ts is None else max(max_ts, ts)
        if fetch_from is not None and ts < fetch_from:
            continue
        selected_rows.append(row)
        customer_id = optional_string(row.get("customer_id"))
        if customer_id:
            affected.add(customer_id)
        payload = {**row, "tenant_id": source.tenant_id}
        source_ref = str(row.get("source_ref") or f"{source.effective_source_ref}:{row.get('source_id') or row.get('id')}")
        record = TimelineSourceRecord(
            source_system=source.source_system,
            source_ref=source_ref,
            payload=payload,
            source_path=str(source.path),
            observed_at=ts,
        )
        records.append(record)
        batch = normalizer.normalize(record)
        for event in batch.events:
            if event.customer_id and event_would_change(store, event):
                would_change.add(event.customer_id)
    return SourceLoadResult(
        source=source,
        cursor_before=cursor.last_cursor_ts.isoformat() if cursor else None,
        fetch_from=fetch_from,
        rows_total=len(rows),
        rows_selected=len(selected_rows),
        records=tuple(records),
        max_source_ts=max_ts,
        affected_customer_ids=tuple(sorted(affected)),
        would_change_customer_ids=tuple(sorted(would_change)),
    )


def update_source_failure_cursor(
    store: CustomerTimelineSQLiteStore,
    source: IncrementalSourceConfig,
    *,
    skipped_reason: str,
    actor: str,
) -> None:
    cursor = store.get_ingestion_cursor(source.tenant_id, source.source_system)
    failures = int(((cursor.metadata if cursor else {}) or {}).get("consecutive_failures") or 0) + 1
    last_cursor = cursor.last_cursor_ts if cursor else datetime.fromtimestamp(0, timezone.utc)
    metadata = merge_cursor_metadata(
        cursor.metadata if cursor else {},
        source,
        last_status="skipped",
        skipped_reason=skipped_reason,
        last_cursor_ts=last_cursor,
    )
    metadata["consecutive_failures"] = failures
    metadata["alert"] = failures >= 2
    store.upsert_ingestion_cursor(
        source.tenant_id,
        source.source_system,
        last_cursor_ts=last_cursor,
        metadata=metadata,
        actor=actor,
    )


def event_would_change(store: CustomerTimelineSQLiteStore, event: TimelineEvent) -> bool:
    row = store._fetch_one(  # noqa: SLF001 - local low-level check avoids a duplicate import pass.
        "SELECT record_hash FROM timeline_events WHERE dedupe_key = ?",
        (event.dedupe_key,),
    )
    if row is None:
        return True
    return str(row["record_hash"]) != stable_digest(event.to_json_dict())


def rebuild_affected_outputs(config: NightlyIncrementalConfig, *, customer_ids: Sequence[str]) -> Mapping[str, Any]:
    selected = tuple(dict.fromkeys(str(item).strip() for item in customer_ids if str(item).strip()))
    if not selected:
        return {
            "selected_customer_count": 0,
            "profiles": None,
            "bot_safe_summary": None,
        }
    profiles_report = None
    if config.profile_rebuild is not None:
        profiles_report = CustomerProfileBuilder(
            CustomerProfileBuildOptions(
                timeline_db=config.timeline_db,
                profiles_db=config.profile_rebuild.profiles_db,
                master_calls_db=config.profile_rebuild.master_calls_db,
                tenant_id=config.tenant_id,
                customer_ids=selected,
                build_id=config.profile_rebuild.build_id,
            )
        ).build()
    bot_safe_report = None
    if config.bot_safe_rebuild is not None:
        bot_safe_report = build_bot_safe_summaries(
            BotSafeSummaryBuildConfig(
                timeline_db=config.timeline_db,
                allowed_root=config.bot_safe_rebuild.allowed_root,
                tenant_id=config.tenant_id,
                apply=config.bot_safe_rebuild.apply,
                limit=config.bot_safe_rebuild.limit,
                customer_ids=selected,
            )
        ).to_json_dict()
    return {
        "selected_customer_count": len(selected),
        "selected_customer_ids": list(selected),
        "profiles": profiles_report,
        "bot_safe_summary": bot_safe_report
        or {
            "status": "deferred_pending_phase0_builder",
            "customer_ids": list(selected),
            "note": "Final Phase 0 builder will be wired after D3 integration; current interface passes affected customer_ids.",
        },
    }


@contextmanager
def single_run_lock(db_path: Path, *, timeout_seconds: float) -> Iterator[Mapping[str, Any]]:
    lock_path = db_path.with_suffix(db_path.suffix + ".nightly.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    waited = 0.0
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                waited = time.monotonic() - started
                break
            except BlockingIOError:
                waited = time.monotonic() - started
                if waited >= timeout_seconds:
                    raise TimeoutError(f"nightly incremental lock timeout: {lock_path}")
                time.sleep(DEFAULT_LOCK_POLL_SECONDS)
        yield {"path": str(lock_path), "waited_seconds": round(waited, 3)}
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def read_jsonl(path: Path) -> tuple[Mapping[str, Any], ...]:
    rows: list[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            parsed = json.loads(text)
            if not isinstance(parsed, Mapping):
                raise ValueError(f"JSONL row must be an object: {path}")
            rows.append(parsed)
    return tuple(rows)


def normalized_timestamp(row: Mapping[str, Any]) -> str:
    return str(
        row.get("snapshot_at")
        or row.get("captured_at")
        or row.get("updated_at")
        or row.get("created_at")
        or row.get("event_at")
        or row.get("date_last")
        or row.get("date_first")
        or row.get("date_iso")
        or row.get("message_date_iso")
        or row.get("first_ingested_at")
        or ""
    ).strip()


def mail_direction(row: Mapping[str, Any]) -> TimelineDirection:
    raw = str(row.get("direction") or row.get("message_kind") or row.get("mailbox") or row.get("folder") or "").casefold()
    if any(token in raw for token in ("sent", "out", "исход")):
        return TimelineDirection.OUTBOUND
    if any(token in raw for token in ("internal", "draft")):
        return TimelineDirection.INTERNAL
    return TimelineDirection.INBOUND


def cursor_ts_for_source_ref(metadata: Mapping[str, Any] | None, source: IncrementalSourceConfig) -> datetime | None:
    if not isinstance(metadata, Mapping):
        return None
    source_refs = metadata.get("source_refs")
    if not isinstance(source_refs, Mapping):
        return None
    state = source_refs.get(source.effective_source_ref)
    if not isinstance(state, Mapping):
        return None
    raw = optional_string(state.get("last_cursor_ts"))
    if not raw:
        return None
    try:
        return parse_datetime(raw, "source_ref_cursor")
    except ValueError:
        return None


def has_source_ref_cursors(metadata: Mapping[str, Any] | None) -> bool:
    return isinstance(metadata, Mapping) and isinstance(metadata.get("source_refs"), Mapping)


def merge_cursor_metadata(
    metadata: Mapping[str, Any] | None,
    source: IncrementalSourceConfig,
    *,
    last_status: str,
    last_cursor_ts: datetime,
    max_source_ts: datetime | None = None,
    skipped_reason: str | None = None,
) -> dict[str, Any]:
    merged = dict(metadata or {})
    source_refs = dict(merged.get("source_refs") or {})
    state = dict(source_refs.get(source.effective_source_ref) or {})
    state["last_status"] = last_status
    state["last_cursor_ts"] = last_cursor_ts.isoformat()
    if max_source_ts is not None:
        state["max_source_ts"] = max_source_ts.isoformat()
    if skipped_reason:
        state["skipped_reason"] = skipped_reason
    else:
        state.pop("skipped_reason", None)
    source_refs[source.effective_source_ref] = state
    merged["source_refs"] = source_refs
    merged["source_ref"] = source.effective_source_ref
    merged["last_status"] = last_status
    if skipped_reason:
        merged["skipped_reason"] = skipped_reason
    else:
        merged.pop("skipped_reason", None)
    return merged


def parse_datetime(value: Any, field_name: str) -> datetime:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} must be a non-empty ISO timestamp")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def optional_string(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    return text or None


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def summarize_report(report: Mapping[str, Any]) -> Mapping[str, Any]:
    source_statuses = Counter(str(item.get("status") or ("skipped" if item.get("skipped_reason") else "ok")) for item in report.get("sources", ()))
    return {
        "schema_version": report.get("schema_version"),
        "overall_status": report.get("overall_status"),
        "gate_passed": report.get("gate_passed"),
        "failed_required_sources": report.get("failed_required_sources"),
        "duration_seconds": report.get("duration_seconds"),
        "source_statuses": dict(source_statuses),
        "affected_customer_count": report.get("affected_customer_count"),
        "changed_customer_count": report.get("changed_customer_count"),
        "phase_seconds": report.get("phase_seconds"),
        "safety": report.get("safety"),
    }
