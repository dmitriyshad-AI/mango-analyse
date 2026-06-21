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

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", normalize_key(self.name, "source name"))
        object.__setattr__(self, "source_system", normalize_key(self.source_system, "source_system"))
        object.__setattr__(self, "tenant_id", normalize_key(self.tenant_id, "tenant_id"))
        object.__setattr__(self, "path", Path(self.path))
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
                loaded = load_incremental_jsonl_source(store, source, safety_margin_seconds=config.safety_margin_seconds)
                report["sources"].append(loaded.to_json_dict())
                if loaded.skipped_reason:
                    update_source_failure_cursor(store, source, skipped_reason=loaded.skipped_reason, actor=config.actor)
                    report["source_errors"].append({"source": source.name, "reason": loaded.skipped_reason})
                    continue
                affected.update(loaded.affected_customer_ids)
                would_change.update(loaded.would_change_customer_ids)
                if not loaded.records:
                    continue
                imported = TimelineImportService(store).import_records(
                    loaded.records,
                    normalizer=JsonlTimelineNormalizer(source.source_system),
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
                import_reports.append(imported.to_json_dict())
                if loaded.max_source_ts is not None:
                    cursor_ts = loaded.max_source_ts - timedelta(seconds=config.safety_margin_seconds)
                    cursor = store.upsert_ingestion_cursor(
                        source.tenant_id,
                        source.source_system,
                        last_cursor_ts=cursor_ts,
                        metadata={
                            "max_source_ts": loaded.max_source_ts.isoformat(),
                            "source_ref": source.effective_source_ref,
                            "last_status": "ok",
                            "consecutive_failures": 0,
                        },
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
    recalc_started = time.monotonic()
    report["rebuild"] = rebuild_affected_outputs(config, customer_ids=report["changed_customer_ids"])
    report["phase_seconds"]["rebuild"] = round(time.monotonic() - recalc_started, 3)
    finished = datetime.now(timezone.utc)
    report["finished_at"] = finished.isoformat()
    report["duration_seconds"] = round((finished - started).total_seconds(), 3)
    append_jsonl(config.journal_path, report)
    return report


def load_incremental_jsonl_source(
    store: CustomerTimelineSQLiteStore,
    source: IncrementalSourceConfig,
    *,
    safety_margin_seconds: int,
) -> SourceLoadResult:
    cursor = store.get_ingestion_cursor(source.tenant_id, source.source_system)
    fetch_from = cursor.last_cursor_ts if cursor else None
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
    normalizer = JsonlTimelineNormalizer(source.source_system)
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
    store.upsert_ingestion_cursor(
        source.tenant_id,
        source.source_system,
        last_cursor_ts=last_cursor,
        metadata={
            **(dict(cursor.metadata) if cursor else {}),
            "last_status": "skipped",
            "skipped_reason": skipped_reason,
            "consecutive_failures": failures,
            "alert": failures >= 2,
        },
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
    return str(row.get("updated_at") or row.get("created_at") or row.get("event_at") or "").strip()


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
    source_statuses = Counter("skipped" if item.get("skipped_reason") else "ok" for item in report.get("sources", ()))
    return {
        "schema_version": report.get("schema_version"),
        "duration_seconds": report.get("duration_seconds"),
        "source_statuses": dict(source_statuses),
        "affected_customer_count": report.get("affected_customer_count"),
        "changed_customer_count": report.get("changed_customer_count"),
        "phase_seconds": report.get("phase_seconds"),
        "safety": report.get("safety"),
    }
