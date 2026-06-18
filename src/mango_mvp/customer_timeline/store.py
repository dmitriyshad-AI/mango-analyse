from __future__ import annotations

import fcntl
import json
import re
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence

from mango_mvp.customer_timeline.contracts import (
    BotContextChunk,
    CustomerIdentity,
    CustomerOpportunity,
    DerivedSignal,
    EventArtifact,
    IdentityLink,
    TimelineEvent,
)
from mango_mvp.customer_timeline.ids import (
    normalize_key,
    optional_text,
    require_text,
    require_timezone,
    stable_digest,
    stable_prefixed_id,
)
from mango_mvp.customer_timeline.safety import (
    customer_timeline_safety_contract,
    guard_customer_timeline_output_path,
)


CUSTOMER_TIMELINE_SQLITE_SCHEMA_VERSION = "customer_timeline_sqlite_v1"
CUSTOMER_TIMELINE_SQLITE_MIGRATION_ID = "20260618_002_derived_signal_status"

RUNTIME_DB_FILENAMES = {
    "ai_office.db",
    "runtime.db",
    "mango_mvp.db",
    "stable_runtime.db",
    "mango_runtime.db",
    "amo_runtime.db",
    "calls.db",
    "transcripts.db",
    "mango_product_appliance.sqlite",
    "mango_product_appliance.db",
    "mail_archive.sqlite",
    "tallanto_email_identity_map.sqlite",
    "channel_product.sqlite",
    "telegram_history_channel.sqlite",
}
FORBIDDEN_PERSISTED_PAYLOAD_KEYS = {
    "raw_payload",
    "provider_raw_payload",
    "webhook_payload",
    "telegram_update",
    "telegram_raw_update",
    "telegram_update_payload",
    "telegram_raw_payload",
    "telegram_message",
    "telegram_message_payload",
    "telegram_raw_message",
    "whatsapp_update",
    "whatsapp_raw_update",
    "whatsapp_update_payload",
    "whatsapp_raw_payload",
    "whatsapp_message_payload",
    "whatsapp_raw_message",
    "wappi_payload",
    "wappi_raw_payload",
    "wappi_message_payload",
    "wappi_raw_message",
    "raw_update",
    "raw_message",
    "raw_messages",
    "updates",
    "callback_query",
    "edited_message",
    "business_message",
    "channel_post",
    "edited_channel_post",
    "inline_query",
    "chosen_inline_result",
    "poll",
    "pre_checkout_query",
    "crm_dialog_payload",
    "tallanto_raw_payload",
    "tallanto_payload",
    "tallanto_api_response",
    "tallanto_response",
    "raw_finance",
    "raw_finances",
    "raw_abonement",
    "raw_abonements",
    "most_finances_payload",
    "most_abonements_payload",
    "most_class_payload",
    "email_raw_body",
    "raw_body",
    "raw_file",
    "file_bytes",
    "attachment_bytes",
    "audio_bytes",
}
ALLOWED_SEARCH_SCOPES = {"events", "bot_context", "signals"}
_SEARCH_TOKEN_RE = re.compile(r"[\w@.+-]+", re.UNICODE)


Clock = Callable[[], datetime]


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class CustomerTimelineSQLiteOpenResult:
    db_path: str
    allowed_root: str
    schema_version: str
    read_only: bool
    fts_enabled: bool

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CustomerTimelineStoreWriteResult:
    record_type: str
    record_id: str
    created: bool
    status: str
    record_hash: str
    audit_id: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "record_type", normalize_key(self.record_type, "record_type"))
        object.__setattr__(self, "record_id", require_text(self.record_id, "record_id"))
        object.__setattr__(self, "status", normalize_key(self.status, "status"))
        object.__setattr__(self, "record_hash", require_text(self.record_hash, "record_hash"))
        object.__setattr__(self, "audit_id", optional_text(self.audit_id))

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CustomerTimelineIngestionRun:
    tenant_id: str
    source_system: str
    source_ref: str
    run_kind: str
    idempotency_key: str
    run_id: Optional[str] = None
    status: str = "started"
    started_at: datetime = field(default_factory=now_utc)
    finished_at: Optional[datetime] = None
    input_hash: Optional[str] = None
    accepted_count: int = 0
    rejected_count: int = 0
    output_ref: Optional[str] = None
    error: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        tenant_id = normalize_key(self.tenant_id, "tenant_id")
        source_system = normalize_key(self.source_system, "source_system")
        source_ref = require_text(self.source_ref, "source_ref")
        run_kind = normalize_key(self.run_kind, "run_kind")
        key = require_text(self.idempotency_key, "idempotency_key")
        status = normalize_key(self.status, "status")
        require_timezone(self.started_at, "started_at")
        if self.finished_at is not None:
            require_timezone(self.finished_at, "finished_at")
            if self.finished_at < self.started_at:
                raise ValueError("finished_at must be greater than or equal to started_at")
        if self.accepted_count < 0 or self.rejected_count < 0:
            raise ValueError("ingestion counters must not be negative")
        run_id = optional_text(self.run_id) or stable_prefixed_id(
            "timeline_ingestion_run",
            {
                "tenant_id": tenant_id,
                "source_system": source_system,
                "source_ref": source_ref,
                "run_kind": run_kind,
                "idempotency_key": key,
            },
        )
        object.__setattr__(self, "tenant_id", tenant_id)
        object.__setattr__(self, "source_system", source_system)
        object.__setattr__(self, "source_ref", source_ref)
        object.__setattr__(self, "run_kind", run_kind)
        object.__setattr__(self, "idempotency_key", key)
        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "input_hash", optional_text(self.input_hash))
        object.__setattr__(self, "output_ref", optional_text(self.output_ref))
        object.__setattr__(self, "error", optional_text(self.error))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_json_dict(self) -> Mapping[str, Any]:
        return {
            "schema_version": CUSTOMER_TIMELINE_SQLITE_SCHEMA_VERSION,
            "run_id": self.run_id,
            "tenant_id": self.tenant_id,
            "source_system": self.source_system,
            "source_ref": self.source_ref,
            "run_kind": self.run_kind,
            "idempotency_key": self.idempotency_key,
            "status": self.status,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "input_hash": self.input_hash,
            "accepted_count": self.accepted_count,
            "rejected_count": self.rejected_count,
            "output_ref": self.output_ref,
            "error": self.error,
            "metadata": dict(self.metadata),
        }

    def finished(
        self,
        *,
        status: str,
        finished_at: datetime,
        accepted_count: Optional[int] = None,
        rejected_count: Optional[int] = None,
        output_ref: Optional[str] = None,
        error: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "CustomerTimelineIngestionRun":
        return CustomerTimelineIngestionRun(
            tenant_id=self.tenant_id,
            source_system=self.source_system,
            source_ref=self.source_ref,
            run_kind=self.run_kind,
            idempotency_key=self.idempotency_key,
            run_id=self.run_id,
            status=status,
            started_at=self.started_at,
            finished_at=finished_at,
            input_hash=self.input_hash,
            accepted_count=self.accepted_count if accepted_count is None else accepted_count,
            rejected_count=self.rejected_count if rejected_count is None else rejected_count,
            output_ref=self.output_ref if output_ref is None else output_ref,
            error=self.error if error is None else error,
            metadata=self.metadata if metadata is None else metadata,
        )


@dataclass(frozen=True)
class CustomerTimelineAuditEntry:
    audit_id: str
    tenant_id: str
    action: str
    entity_type: str
    entity_id: Optional[str]
    actor: str
    created_at: datetime
    ingestion_run_id: Optional[str] = None
    before_hash: Optional[str] = None
    after_hash: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "audit_id", require_text(self.audit_id, "audit_id"))
        object.__setattr__(self, "tenant_id", normalize_key(self.tenant_id, "tenant_id"))
        object.__setattr__(self, "action", normalize_key(self.action, "action"))
        object.__setattr__(self, "entity_type", normalize_key(self.entity_type, "entity_type"))
        object.__setattr__(self, "entity_id", optional_text(self.entity_id))
        object.__setattr__(self, "actor", require_text(self.actor, "actor"))
        require_timezone(self.created_at, "created_at")
        object.__setattr__(self, "ingestion_run_id", optional_text(self.ingestion_run_id))
        object.__setattr__(self, "before_hash", optional_text(self.before_hash))
        object.__setattr__(self, "after_hash", optional_text(self.after_hash))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_json_dict(self) -> Mapping[str, Any]:
        return {
            "schema_version": CUSTOMER_TIMELINE_SQLITE_SCHEMA_VERSION,
            "audit_id": self.audit_id,
            "tenant_id": self.tenant_id,
            "action": self.action,
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "actor": self.actor,
            "created_at": self.created_at.isoformat(),
            "ingestion_run_id": self.ingestion_run_id,
            "before_hash": self.before_hash,
            "after_hash": self.after_hash,
            "metadata": dict(self.metadata),
        }


class CustomerTimelineSQLiteStore:
    """Local read model for the unified customer timeline.

    The store writes only to its configured SQLite file under `allowed_root`.
    It never reads or mutates runtime DBs, never calls CRM/Tallanto/channel APIs,
    and keeps large/raw source files on disk by reference rather than in SQLite.
    """

    def __init__(
        self,
        db_path: Path | str,
        *,
        allowed_root: Optional[Path | str] = None,
        read_only: bool = False,
        clock: Optional[Clock] = None,
    ) -> None:
        raw_path = Path(db_path).expanduser()
        root = Path(allowed_root).expanduser() if allowed_root is not None else raw_path.parent
        self.db_path = guard_customer_timeline_output_path(raw_path, root)
        self.allowed_root = Path(root).resolve(strict=False)
        guard_customer_timeline_sqlite_path(self.db_path)
        self.read_only = bool(read_only)
        self._clock = clock or now_utc
        self._bulk_write_depth = 0
        self._bulk_write_dirty = False
        self._writer_lock_path: Optional[Path] = None
        self._writer_lock_handle: Any = None
        if not self.read_only:
            self._acquire_writer_lock()
        try:
            self._con = self._connect()
            if not self.read_only:
                self.bootstrap()
            self._fts_enabled = self._detect_existing_fts()
        except Exception:
            self._release_writer_lock()
            raise

    @classmethod
    def open_read_only(
        cls,
        db_path: Path | str,
        *,
        allowed_root: Optional[Path | str] = None,
        clock: Optional[Clock] = None,
    ) -> "CustomerTimelineSQLiteStore":
        return cls(db_path, allowed_root=allowed_root, read_only=True, clock=clock)

    def close(self) -> None:
        try:
            self._con.close()
        finally:
            self._release_writer_lock()

    def __enter__(self) -> "CustomerTimelineSQLiteStore":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    @contextmanager
    def bulk_write(self) -> Iterator["CustomerTimelineSQLiteStore"]:
        """Defer per-record commits until the end of a local batch write."""
        self._ensure_writable()
        outermost = self._bulk_write_depth == 0
        self._bulk_write_depth += 1
        try:
            yield self
        except Exception:
            if outermost:
                self._con.rollback()
                self._bulk_write_dirty = False
            raise
        else:
            if outermost and self._bulk_write_dirty:
                self._con.commit()
        finally:
            self._bulk_write_depth -= 1
            if outermost:
                self._bulk_write_dirty = False

    @property
    def open_result(self) -> CustomerTimelineSQLiteOpenResult:
        return CustomerTimelineSQLiteOpenResult(
            db_path=str(self.db_path),
            allowed_root=str(self.allowed_root),
            schema_version=CUSTOMER_TIMELINE_SQLITE_SCHEMA_VERSION,
            read_only=self.read_only,
            fts_enabled=self._fts_enabled,
        )

    def bootstrap(self) -> None:
        self._ensure_writable()
        self._con.executescript(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
              migration_id TEXT PRIMARY KEY,
              schema_version TEXT NOT NULL,
              applied_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS customer_identities (
              customer_id TEXT PRIMARY KEY,
              tenant_id TEXT NOT NULL,
              identity_status TEXT NOT NULL,
              display_name TEXT,
              primary_phone TEXT,
              primary_email TEXT,
              first_seen_at TEXT,
              last_seen_at TEXT,
              touch_count INTEGER NOT NULL,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              record_hash TEXT NOT NULL,
              record_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS identity_links (
              link_id TEXT PRIMARY KEY,
              tenant_id TEXT NOT NULL,
              customer_id TEXT,
              link_type TEXT NOT NULL,
              link_value TEXT NOT NULL,
              source_system TEXT NOT NULL,
              source_ref TEXT NOT NULL,
              match_class TEXT NOT NULL,
              confidence REAL,
              first_seen_at TEXT,
              last_seen_at TEXT,
              record_hash TEXT NOT NULL,
              record_json TEXT NOT NULL,
              UNIQUE(tenant_id, link_type, link_value, source_system, source_ref)
            );

            CREATE TABLE IF NOT EXISTS customer_opportunities (
              opportunity_id TEXT PRIMARY KEY,
              tenant_id TEXT NOT NULL,
              customer_id TEXT NOT NULL,
              opportunity_type TEXT NOT NULL,
              source_system TEXT NOT NULL,
              source_id TEXT NOT NULL,
              title TEXT,
              status TEXT,
              opened_at TEXT,
              closed_at TEXT,
              confidence REAL,
              record_hash TEXT NOT NULL,
              record_json TEXT NOT NULL,
              UNIQUE(tenant_id, source_system, source_id, opportunity_type)
            );

            CREATE TABLE IF NOT EXISTS timeline_events (
              event_id TEXT PRIMARY KEY,
              dedupe_key TEXT NOT NULL UNIQUE,
              tenant_id TEXT NOT NULL,
              customer_id TEXT,
              opportunity_id TEXT,
              event_type TEXT NOT NULL,
              event_at TEXT NOT NULL,
              source_system TEXT NOT NULL,
              source_id TEXT NOT NULL,
              source_ref TEXT,
              direction TEXT NOT NULL,
              match_status TEXT NOT NULL,
              confidence REAL,
              importance INTEGER NOT NULL,
              subject TEXT,
              text_preview TEXT,
              summary TEXT,
              created_at TEXT NOT NULL,
              record_hash TEXT NOT NULL,
              record_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS event_artifacts (
              artifact_id TEXT PRIMARY KEY,
              tenant_id TEXT NOT NULL,
              event_id TEXT NOT NULL,
              artifact_type TEXT NOT NULL,
              path TEXT NOT NULL,
              sha256 TEXT,
              size_bytes INTEGER,
              mime_type TEXT,
              source_system TEXT NOT NULL,
              source_ref TEXT NOT NULL,
              extraction_status TEXT NOT NULL,
              created_at TEXT NOT NULL,
              record_hash TEXT NOT NULL,
              record_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS derived_signals (
              signal_id TEXT PRIMARY KEY,
              tenant_id TEXT NOT NULL,
              customer_id TEXT,
              opportunity_id TEXT,
              event_id TEXT,
              signal_type TEXT NOT NULL,
              severity TEXT NOT NULL,
              status TEXT NOT NULL DEFAULT 'active',
              expires_at TEXT,
              confidence REAL,
              requires_manager_review INTEGER NOT NULL,
              created_at TEXT NOT NULL,
              record_hash TEXT NOT NULL,
              record_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS bot_context_chunks (
              chunk_id TEXT PRIMARY KEY,
              tenant_id TEXT NOT NULL,
              customer_id TEXT NOT NULL,
              opportunity_id TEXT,
              event_id TEXT,
              source_system TEXT,
              source_ref TEXT,
              chunk_type TEXT NOT NULL,
              event_at TEXT,
              freshness_score REAL,
              allowed_for_bot INTEGER NOT NULL,
              requires_manager_review INTEGER NOT NULL,
              ordinal INTEGER NOT NULL,
              created_at TEXT NOT NULL,
              record_hash TEXT NOT NULL,
              record_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS ingestion_runs (
              run_id TEXT PRIMARY KEY,
              tenant_id TEXT NOT NULL,
              source_system TEXT NOT NULL,
              source_ref TEXT NOT NULL,
              run_kind TEXT NOT NULL,
              idempotency_key TEXT NOT NULL,
              status TEXT NOT NULL,
              started_at TEXT NOT NULL,
              finished_at TEXT,
              input_hash TEXT,
              accepted_count INTEGER NOT NULL,
              rejected_count INTEGER NOT NULL,
              output_ref TEXT,
              error TEXT,
              record_hash TEXT NOT NULL,
              record_json TEXT NOT NULL,
              UNIQUE(tenant_id, source_system, run_kind, idempotency_key)
            );

            CREATE TABLE IF NOT EXISTS timeline_conflicts (
              conflict_id TEXT PRIMARY KEY,
              tenant_id TEXT NOT NULL,
              conflict_type TEXT NOT NULL,
              severity TEXT NOT NULL,
              status TEXT NOT NULL,
              created_at TEXT NOT NULL,
              resolved_at TEXT,
              record_hash TEXT NOT NULL,
              record_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS customer_id_mappings (
              mapping_id TEXT PRIMARY KEY,
              tenant_id TEXT NOT NULL,
              old_customer_id TEXT NOT NULL,
              new_customer_id TEXT NOT NULL,
              mapping_kind TEXT NOT NULL,
              resolution_status TEXT NOT NULL,
              reason TEXT NOT NULL,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              record_hash TEXT NOT NULL,
              record_json TEXT NOT NULL,
              UNIQUE(tenant_id, old_customer_id, new_customer_id)
            );

            CREATE TABLE IF NOT EXISTS audit_log (
              seq INTEGER PRIMARY KEY AUTOINCREMENT,
              audit_id TEXT NOT NULL UNIQUE,
              tenant_id TEXT NOT NULL,
              action TEXT NOT NULL,
              entity_type TEXT NOT NULL,
              entity_id TEXT,
              actor TEXT NOT NULL,
              created_at TEXT NOT NULL,
              ingestion_run_id TEXT,
              before_hash TEXT,
              after_hash TEXT,
              record_json TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS ix_customer_identities_tenant_status
              ON customer_identities(tenant_id, identity_status, updated_at);
            CREATE INDEX IF NOT EXISTS ix_customer_identities_phone
              ON customer_identities(tenant_id, primary_phone);
            CREATE INDEX IF NOT EXISTS ix_customer_identities_email
              ON customer_identities(tenant_id, primary_email);
            CREATE INDEX IF NOT EXISTS ix_identity_links_lookup
              ON identity_links(tenant_id, link_type, link_value);
            CREATE INDEX IF NOT EXISTS ix_identity_links_customer
              ON identity_links(tenant_id, customer_id, match_class);
            CREATE INDEX IF NOT EXISTS ix_opportunities_customer_time
              ON customer_opportunities(tenant_id, customer_id, opened_at);
            CREATE INDEX IF NOT EXISTS ix_timeline_events_customer_time
              ON timeline_events(tenant_id, customer_id, event_at);
            CREATE INDEX IF NOT EXISTS ix_timeline_events_opportunity_time
              ON timeline_events(tenant_id, opportunity_id, event_at);
            CREATE INDEX IF NOT EXISTS ix_timeline_events_source
              ON timeline_events(tenant_id, source_system, source_id, event_type);
            CREATE INDEX IF NOT EXISTS ix_timeline_events_type_time
              ON timeline_events(tenant_id, event_type, event_at);
            CREATE INDEX IF NOT EXISTS ix_artifacts_event
              ON event_artifacts(tenant_id, event_id, artifact_type);
            CREATE INDEX IF NOT EXISTS ix_artifacts_sha256
              ON event_artifacts(tenant_id, sha256);
            CREATE INDEX IF NOT EXISTS ix_signals_customer_type
              ON derived_signals(tenant_id, customer_id, signal_type);
            CREATE INDEX IF NOT EXISTS ix_signals_event
              ON derived_signals(tenant_id, event_id, severity);
            CREATE INDEX IF NOT EXISTS ix_chunks_customer_event_time
              ON bot_context_chunks(tenant_id, customer_id, event_at);
            CREATE INDEX IF NOT EXISTS ix_chunks_allowed
              ON bot_context_chunks(tenant_id, allowed_for_bot, requires_manager_review);
            CREATE INDEX IF NOT EXISTS ix_ingestion_runs_source
              ON ingestion_runs(tenant_id, source_system, status, started_at);
            CREATE INDEX IF NOT EXISTS ix_timeline_conflicts_status
              ON timeline_conflicts(tenant_id, status, severity, created_at);
            CREATE INDEX IF NOT EXISTS ix_customer_id_mappings_old
              ON customer_id_mappings(tenant_id, old_customer_id, resolution_status);
            CREATE INDEX IF NOT EXISTS ix_customer_id_mappings_new
              ON customer_id_mappings(tenant_id, new_customer_id, resolution_status);
            CREATE INDEX IF NOT EXISTS ix_audit_log_entity
              ON audit_log(tenant_id, entity_type, entity_id, created_at);
            CREATE INDEX IF NOT EXISTS ix_audit_log_ingestion
              ON audit_log(tenant_id, ingestion_run_id, created_at);
            """
        )
        self._apply_schema_migrations()
        if sqlite_fts5_available(self._con):
            self._bootstrap_fts()
        self._con.execute(
            """
            INSERT OR IGNORE INTO schema_migrations (migration_id, schema_version, applied_at)
            VALUES (?, ?, ?)
            """,
            (CUSTOMER_TIMELINE_SQLITE_MIGRATION_ID, CUSTOMER_TIMELINE_SQLITE_SCHEMA_VERSION, self._now().isoformat()),
        )
        self._commit()
        self._fts_enabled = self._detect_existing_fts()

    def _apply_schema_migrations(self) -> None:
        derived_signal_columns = {
            row["name"]
            for row in self._con.execute("PRAGMA table_info(derived_signals)").fetchall()
        }
        if "status" not in derived_signal_columns:
            self._con.execute("ALTER TABLE derived_signals ADD COLUMN status TEXT NOT NULL DEFAULT 'active'")
        if "expires_at" not in derived_signal_columns:
            self._con.execute("ALTER TABLE derived_signals ADD COLUMN expires_at TEXT")
        self._con.execute(
            """
            CREATE INDEX IF NOT EXISTS ix_signals_customer_status_expiry
              ON derived_signals(tenant_id, customer_id, status, expires_at)
            """
        )

    def upsert_customer(
        self,
        identity: CustomerIdentity,
        *,
        actor: str = "system",
        ingestion_run_id: Optional[str] = None,
    ) -> CustomerTimelineStoreWriteResult:
        self._ensure_writable()
        if not isinstance(identity, CustomerIdentity):
            raise TypeError("identity must be CustomerIdentity")
        payload = identity.to_json_dict()
        return self._upsert_record(
            table="customer_identities",
            key_column="customer_id",
            key_value=identity.customer_id,
            record_type="customer_identity",
            tenant_id=identity.tenant_id,
            payload=payload,
            columns={
                "tenant_id": identity.tenant_id,
                "identity_status": identity.identity_status.value,
                "display_name": identity.display_name,
                "primary_phone": identity.primary_phone,
                "primary_email": identity.primary_email,
                "first_seen_at": identity.first_seen_at.isoformat() if identity.first_seen_at else None,
                "last_seen_at": identity.last_seen_at.isoformat() if identity.last_seen_at else None,
                "touch_count": identity.touch_count,
                "created_at": identity.created_at.isoformat(),
                "updated_at": identity.updated_at.isoformat(),
            },
            actor=actor,
            ingestion_run_id=ingestion_run_id,
        )

    def upsert_identity_link(
        self,
        link: IdentityLink,
        *,
        actor: str = "system",
        ingestion_run_id: Optional[str] = None,
    ) -> CustomerTimelineStoreWriteResult:
        self._ensure_writable()
        if not isinstance(link, IdentityLink):
            raise TypeError("link must be IdentityLink")
        if link.customer_id:
            self._assert_customer_exists(link.tenant_id, link.customer_id)
        return self._upsert_record(
            table="identity_links",
            key_column="link_id",
            key_value=link.link_id,
            record_type="identity_link",
            tenant_id=link.tenant_id,
            payload=link.to_json_dict(),
            columns={
                "tenant_id": link.tenant_id,
                "customer_id": link.customer_id,
                "link_type": link.link_type.value,
                "link_value": link.link_value,
                "source_system": link.source_system,
                "source_ref": link.source_ref,
                "match_class": link.match_class.value,
                "confidence": link.confidence,
                "first_seen_at": link.first_seen_at.isoformat() if link.first_seen_at else None,
                "last_seen_at": link.last_seen_at.isoformat() if link.last_seen_at else None,
            },
            actor=actor,
            ingestion_run_id=ingestion_run_id,
        )

    def upsert_opportunity(
        self,
        opportunity: CustomerOpportunity,
        *,
        actor: str = "system",
        ingestion_run_id: Optional[str] = None,
    ) -> CustomerTimelineStoreWriteResult:
        self._ensure_writable()
        if not isinstance(opportunity, CustomerOpportunity):
            raise TypeError("opportunity must be CustomerOpportunity")
        self._assert_customer_exists(opportunity.tenant_id, opportunity.customer_id)
        return self._upsert_record(
            table="customer_opportunities",
            key_column="opportunity_id",
            key_value=opportunity.opportunity_id,
            record_type="customer_opportunity",
            tenant_id=opportunity.tenant_id,
            payload=opportunity.to_json_dict(),
            columns={
                "tenant_id": opportunity.tenant_id,
                "customer_id": opportunity.customer_id,
                "opportunity_type": opportunity.opportunity_type.value,
                "source_system": opportunity.source_system,
                "source_id": opportunity.source_id,
                "title": opportunity.title,
                "status": opportunity.status,
                "opened_at": opportunity.opened_at.isoformat() if opportunity.opened_at else None,
                "closed_at": opportunity.closed_at.isoformat() if opportunity.closed_at else None,
                "confidence": opportunity.confidence,
            },
            actor=actor,
            ingestion_run_id=ingestion_run_id,
        )

    def upsert_event(
        self,
        event: TimelineEvent,
        *,
        actor: str = "system",
        ingestion_run_id: Optional[str] = None,
    ) -> CustomerTimelineStoreWriteResult:
        self._ensure_writable()
        if not isinstance(event, TimelineEvent):
            raise TypeError("event must be TimelineEvent")
        if event.customer_id:
            self._assert_customer_exists(event.tenant_id, event.customer_id)
        if event.opportunity_id:
            self._assert_opportunity_exists(event.tenant_id, event.opportunity_id)
        existing_by_dedupe = self._fetch_one(
            "SELECT event_id, record_hash FROM timeline_events WHERE dedupe_key = ?",
            (event.dedupe_key,),
        )
        payload = event.to_json_dict()
        record_hash = stable_digest(payload)
        if existing_by_dedupe is not None and existing_by_dedupe["event_id"] != event.event_id:
            return CustomerTimelineStoreWriteResult(
                record_type="timeline_event",
                record_id=existing_by_dedupe["event_id"],
                created=False,
                status="duplicate",
                record_hash=existing_by_dedupe["record_hash"],
                audit_id=None,
            )
        result = self._upsert_record(
            table="timeline_events",
            key_column="event_id",
            key_value=event.event_id,
            record_type="timeline_event",
            tenant_id=event.tenant_id,
            payload=payload,
            columns={
                "dedupe_key": event.dedupe_key,
                "tenant_id": event.tenant_id,
                "customer_id": event.customer_id,
                "opportunity_id": event.opportunity_id,
                "event_type": event.event_type.value,
                "event_at": event.event_at.isoformat(),
                "source_system": event.source_system,
                "source_id": event.source_id,
                "source_ref": event.source_ref,
                "direction": event.direction.value,
                "match_status": event.match_status.value,
                "confidence": event.confidence,
                "importance": event.importance,
                "subject": event.subject,
                "text_preview": event.text_preview,
                "summary": event.summary,
                "created_at": event.created_at.isoformat(),
            },
            actor=actor,
            ingestion_run_id=ingestion_run_id,
            commit=False,
        )
        self._sync_event_fts(event, payload, record_hash)
        self._commit()
        return result

    def upsert_artifact(
        self,
        artifact: EventArtifact,
        *,
        actor: str = "system",
        ingestion_run_id: Optional[str] = None,
    ) -> CustomerTimelineStoreWriteResult:
        self._ensure_writable()
        if not isinstance(artifact, EventArtifact):
            raise TypeError("artifact must be EventArtifact")
        self._assert_event_exists(artifact.tenant_id, artifact.event_id)
        return self._upsert_record(
            table="event_artifacts",
            key_column="artifact_id",
            key_value=artifact.artifact_id,
            record_type="event_artifact",
            tenant_id=artifact.tenant_id,
            payload=artifact.to_json_dict(),
            columns={
                "tenant_id": artifact.tenant_id,
                "event_id": artifact.event_id,
                "artifact_type": artifact.artifact_type.value,
                "path": artifact.path,
                "sha256": artifact.sha256,
                "size_bytes": artifact.size_bytes,
                "mime_type": artifact.mime_type,
                "source_system": artifact.source_system,
                "source_ref": artifact.source_ref,
                "extraction_status": artifact.extraction_status.value,
                "created_at": artifact.created_at.isoformat(),
            },
            actor=actor,
            ingestion_run_id=ingestion_run_id,
        )

    def upsert_signal(
        self,
        signal: DerivedSignal,
        *,
        actor: str = "system",
        ingestion_run_id: Optional[str] = None,
    ) -> CustomerTimelineStoreWriteResult:
        self._ensure_writable()
        if not isinstance(signal, DerivedSignal):
            raise TypeError("signal must be DerivedSignal")
        if signal.customer_id:
            self._assert_customer_exists(signal.tenant_id, signal.customer_id)
        if signal.opportunity_id:
            self._assert_opportunity_exists(signal.tenant_id, signal.opportunity_id)
        if signal.event_id:
            self._assert_event_exists(signal.tenant_id, signal.event_id)
        for source_event_id in signal.source_event_ids:
            self._assert_event_exists(signal.tenant_id, source_event_id)
        return self._upsert_record(
            table="derived_signals",
            key_column="signal_id",
            key_value=signal.signal_id,
            record_type="derived_signal",
            tenant_id=signal.tenant_id,
            payload=signal.to_json_dict(),
            columns={
                "tenant_id": signal.tenant_id,
                "customer_id": signal.customer_id,
                "opportunity_id": signal.opportunity_id,
                "event_id": signal.event_id,
                "signal_type": signal.signal_type,
                "severity": signal.severity.value,
                "status": signal.status.value,
                "expires_at": signal.expires_at.isoformat() if signal.expires_at else None,
                "confidence": signal.confidence,
                "requires_manager_review": int(signal.requires_manager_review),
                "created_at": signal.created_at.isoformat(),
            },
            actor=actor,
            ingestion_run_id=ingestion_run_id,
        )

    def upsert_bot_context_chunk(
        self,
        chunk: BotContextChunk,
        *,
        actor: str = "system",
        ingestion_run_id: Optional[str] = None,
    ) -> CustomerTimelineStoreWriteResult:
        self._ensure_writable()
        if not isinstance(chunk, BotContextChunk):
            raise TypeError("chunk must be BotContextChunk")
        self._assert_customer_exists(chunk.tenant_id, chunk.customer_id)
        if chunk.opportunity_id:
            self._assert_opportunity_exists(chunk.tenant_id, chunk.opportunity_id)
        if chunk.event_id:
            self._assert_event_exists(chunk.tenant_id, chunk.event_id)
        payload = chunk.to_json_dict()
        record_hash = stable_digest(scrub_timeline_persisted_json(payload))
        result = self._upsert_record(
            table="bot_context_chunks",
            key_column="chunk_id",
            key_value=chunk.chunk_id,
            record_type="bot_context_chunk",
            tenant_id=chunk.tenant_id,
            payload=payload,
            columns={
                "tenant_id": chunk.tenant_id,
                "customer_id": chunk.customer_id,
                "opportunity_id": chunk.opportunity_id,
                "event_id": chunk.event_id,
                "source_system": chunk.source_system,
                "source_ref": chunk.source_ref,
                "chunk_type": chunk.chunk_type,
                "event_at": chunk.event_at.isoformat() if chunk.event_at else None,
                "freshness_score": chunk.freshness_score,
                "allowed_for_bot": int(chunk.allowed_for_bot),
                "requires_manager_review": int(chunk.requires_manager_review),
                "ordinal": chunk.ordinal,
                "created_at": chunk.created_at.isoformat(),
            },
            actor=actor,
            ingestion_run_id=ingestion_run_id,
            commit=False,
        )
        self._sync_chunk_fts(chunk, record_hash)
        self._commit()
        return result

    def start_ingestion_run(
        self,
        *,
        tenant_id: str,
        source_system: str,
        source_ref: str,
        run_kind: str,
        idempotency_key: str,
        input_hash: Optional[str] = None,
        started_at: Optional[datetime] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        actor: str = "system",
    ) -> CustomerTimelineIngestionRun:
        self._ensure_writable()
        run = CustomerTimelineIngestionRun(
            tenant_id=tenant_id,
            source_system=source_system,
            source_ref=source_ref,
            run_kind=run_kind,
            idempotency_key=idempotency_key,
            input_hash=input_hash,
            started_at=started_at or self._now(),
            metadata=metadata or {},
        )
        existing = self.get_ingestion_run(run.run_id)
        if existing is not None:
            return ingestion_run_from_json(existing)
        self._upsert_ingestion_run(run, actor=actor)
        return run

    def upsert_customer_identity(
        self,
        identity: CustomerIdentity,
        *,
        actor: str = "system",
        ingestion_run_id: Optional[str] = None,
    ) -> CustomerTimelineStoreWriteResult:
        return self.upsert_customer(identity, actor=actor, ingestion_run_id=ingestion_run_id)

    def finish_ingestion_run(
        self,
        run_id: str,
        *,
        status: str,
        accepted_count: Optional[int] = None,
        rejected_count: Optional[int] = None,
        output_ref: Optional[str] = None,
        error: Optional[str] = None,
        finished_at: Optional[datetime] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        actor: str = "system",
    ) -> CustomerTimelineIngestionRun:
        self._ensure_writable()
        current = self.get_ingestion_run(run_id)
        if current is None:
            raise KeyError(f"unknown ingestion run: {run_id}")
        run = ingestion_run_from_json(current).finished(
            status=status,
            finished_at=finished_at or self._now(),
            accepted_count=accepted_count,
            rejected_count=rejected_count,
            output_ref=output_ref,
            error=error,
            metadata=metadata,
        )
        self._upsert_ingestion_run(run, actor=actor)
        return run

    def record_conflict(
        self,
        tenant_id: str,
        *,
        conflict_type: str,
        entity_refs: Sequence[str],
        severity: str = "medium",
        status: str = "open",
        summary: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        actor: str = "system",
        ingestion_run_id: Optional[str] = None,
    ) -> CustomerTimelineStoreWriteResult:
        self._ensure_writable()
        tenant = normalize_key(tenant_id, "tenant_id")
        normalized_type = normalize_key(conflict_type, "conflict_type")
        normalized_severity = normalize_key(severity, "severity")
        normalized_status = normalize_key(status, "status")
        refs = tuple(require_text(item, "entity_ref") for item in entity_refs)
        if not refs:
            raise ValueError("entity_refs must not be empty")
        conflict_id = stable_prefixed_id(
            "timeline_conflict",
            {
                "tenant_id": tenant,
                "conflict_type": normalized_type,
                "entity_refs": sorted(refs),
            },
        )
        existing = self._fetch_one("SELECT record_json FROM timeline_conflicts WHERE conflict_id = ?", (conflict_id,))
        existing_payload = json_loads(existing["record_json"]) if existing is not None else {}
        created_at = (
            parse_datetime(existing_payload["created_at"], "created_at")
            if existing_payload.get("created_at")
            else self._now()
        )
        payload = {
            "schema_version": CUSTOMER_TIMELINE_SQLITE_SCHEMA_VERSION,
            "tenant_id": tenant,
            "conflict_id": conflict_id,
            "conflict_type": normalized_type,
            "severity": normalized_severity,
            "status": normalized_status,
            "entity_refs": list(refs),
            "summary": optional_text(summary),
            "metadata": dict(metadata or {}),
            "created_at": created_at.isoformat(),
            "resolved_at": existing_payload.get("resolved_at"),
        }
        return self._upsert_record(
            table="timeline_conflicts",
            key_column="conflict_id",
            key_value=conflict_id,
            record_type="timeline_conflict",
            tenant_id=tenant,
            payload=payload,
            columns={
                "tenant_id": tenant,
                "conflict_type": normalized_type,
                "severity": normalized_severity,
                "status": normalized_status,
                "created_at": created_at.isoformat(),
                "resolved_at": existing_payload.get("resolved_at"),
            },
            actor=actor,
            ingestion_run_id=ingestion_run_id,
        )

    def record_customer_id_mapping(
        self,
        tenant_id: str,
        *,
        old_customer_id: str,
        new_customer_id: str,
        reason: str,
        mapping_kind: str = "alias",
        resolution_status: str = "active",
        source_refs: Sequence[str] = (),
        metadata: Optional[Mapping[str, Any]] = None,
        actor: str = "system",
        ingestion_run_id: Optional[str] = None,
    ) -> CustomerTimelineStoreWriteResult:
        self._ensure_writable()
        tenant = normalize_key(tenant_id, "tenant_id")
        old_id = require_text(old_customer_id, "old_customer_id")
        new_id = require_text(new_customer_id, "new_customer_id")
        normalized_reason = normalize_key(reason, "reason")
        normalized_kind = normalize_key(mapping_kind, "mapping_kind")
        normalized_status = normalize_key(resolution_status, "resolution_status")
        self._assert_customer_exists(tenant, new_id)
        refs = tuple(require_text(item, "source_ref") for item in source_refs)
        mapping_id = stable_prefixed_id(
            "customer_id_mapping",
            {
                "tenant_id": tenant,
                "old_customer_id": old_id,
                "new_customer_id": new_id,
            },
        )
        existing = self._fetch_one(
            """
            SELECT record_json FROM customer_id_mappings
            WHERE tenant_id = ? AND old_customer_id = ? AND resolution_status = 'active'
            """,
            (tenant, old_id),
        )
        existing_payload = json_loads(existing["record_json"]) if existing is not None else {}
        if (
            existing_payload
            and existing_payload.get("mapping_id") != mapping_id
            and existing_payload.get("new_customer_id") != new_id
            and existing_payload.get("mapping_kind") != "split"
            and normalized_kind != "split"
        ):
            raise ValueError(
                f"old_customer_id already has active mapping: {old_id} -> {existing_payload.get('new_customer_id')}"
            )
        created_at = (
            parse_datetime(existing_payload["created_at"], "created_at")
            if existing_payload.get("mapping_id") == mapping_id and existing_payload.get("created_at")
            else self._now()
        )
        updated_at = (
            parse_datetime(existing_payload["updated_at"], "updated_at")
            if existing_payload.get("mapping_id") == mapping_id and existing_payload.get("updated_at")
            else self._now()
        )
        payload = {
            "schema_version": CUSTOMER_TIMELINE_SQLITE_SCHEMA_VERSION,
            "mapping_id": mapping_id,
            "tenant_id": tenant,
            "old_customer_id": old_id,
            "new_customer_id": new_id,
            "mapping_kind": normalized_kind,
            "resolution_status": normalized_status,
            "reason": normalized_reason,
            "source_refs": list(refs),
            "ingestion_run_id": ingestion_run_id,
            "metadata": dict(metadata or {}),
            "created_at": created_at.isoformat(),
            "updated_at": updated_at.isoformat(),
        }
        return self._upsert_record(
            table="customer_id_mappings",
            key_column="mapping_id",
            key_value=mapping_id,
            record_type="customer_id_mapping",
            tenant_id=tenant,
            payload=payload,
            columns={
                "tenant_id": tenant,
                "old_customer_id": old_id,
                "new_customer_id": new_id,
                "mapping_kind": normalized_kind,
                "resolution_status": normalized_status,
                "reason": normalized_reason,
                "created_at": created_at.isoformat(),
                "updated_at": updated_at.isoformat(),
            },
            actor=actor,
            ingestion_run_id=ingestion_run_id,
        )

    def append_audit_log(
        self,
        tenant_id: str,
        *,
        action: str,
        entity_type: str,
        entity_id: Optional[str] = None,
        actor: str = "system",
        ingestion_run_id: Optional[str] = None,
        before_hash: Optional[str] = None,
        after_hash: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> CustomerTimelineAuditEntry:
        self._ensure_writable()
        return self._append_audit_log(
            tenant_id=normalize_key(tenant_id, "tenant_id"),
            action=action,
            entity_type=entity_type,
            entity_id=entity_id,
            actor=actor,
            ingestion_run_id=ingestion_run_id,
            before_hash=before_hash,
            after_hash=after_hash,
            metadata=metadata or {},
            now=self._now(),
        )

    def get_customer(self, tenant_id: str, customer_id: str) -> Optional[Mapping[str, Any]]:
        return self._get_record(
            "customer_identities",
            "tenant_id = ? AND customer_id = ?",
            (normalize_key(tenant_id, "tenant_id"), require_text(customer_id, "customer_id")),
        )

    def get_event(self, tenant_id: str, event_id: str) -> Optional[Mapping[str, Any]]:
        return self._get_record(
            "timeline_events",
            "tenant_id = ? AND event_id = ?",
            (normalize_key(tenant_id, "tenant_id"), require_text(event_id, "event_id")),
        )

    def get_event_by_source(
        self,
        tenant_id: str,
        source_system: str,
        source_id: str,
        event_type: str,
    ) -> Optional[Mapping[str, Any]]:
        return self._get_record(
            "timeline_events",
            "tenant_id = ? AND source_system = ? AND source_id = ? AND event_type = ?",
            (
                normalize_key(tenant_id, "tenant_id"),
                normalize_key(source_system, "source_system"),
                require_text(source_id, "source_id"),
                normalize_key(event_type, "event_type"),
            ),
        )

    def get_ingestion_run(self, run_id: str) -> Optional[Mapping[str, Any]]:
        return self._get_record("ingestion_runs", "run_id = ?", (require_text(run_id, "run_id"),))

    def list_customers(
        self,
        tenant_id: str,
        *,
        q: Optional[str] = None,
        identity_status: Optional[str] = None,
        updated_since: Optional[datetime] = None,
        limit: int = 50,
        cursor: Optional[str] = None,
    ) -> Mapping[str, Any]:
        tenant = normalize_key(tenant_id, "tenant_id")
        clauses = ["tenant_id = ?"]
        params: list[Any] = [tenant]
        if identity_status:
            clauses.append("identity_status = ?")
            params.append(normalize_key(identity_status, "identity_status"))
        if updated_since is not None:
            require_timezone(updated_since, "updated_since")
            clauses.append("updated_at >= ?")
            params.append(updated_since.isoformat())
        if q:
            like = f"%{q.strip()}%"
            clauses.append("(display_name LIKE ? OR primary_phone LIKE ? OR primary_email LIKE ?)")
            params.extend([like, like, like])
        page_limit, offset = normalize_pagination(limit, cursor)
        rows = self._con.execute(
            f"""
            SELECT record_json FROM customer_identities
            WHERE {' AND '.join(clauses)}
            ORDER BY updated_at DESC, customer_id
            LIMIT ? OFFSET ?
            """,
            (*params, page_limit + 1, offset),
        ).fetchall()
        return page_result(rows, limit=page_limit, offset=offset)

    def list_identity_links(
        self,
        tenant_id: str,
        *,
        customer_id: Optional[str] = None,
        link_type: Optional[str] = None,
        link_value: Optional[str] = None,
        limit: int = 100,
    ) -> tuple[Mapping[str, Any], ...]:
        tenant = normalize_key(tenant_id, "tenant_id")
        clauses = ["tenant_id = ?"]
        params: list[Any] = [tenant]
        if customer_id:
            clauses.append("customer_id = ?")
            params.append(require_text(customer_id, "customer_id"))
        if link_type:
            clauses.append("link_type = ?")
            params.append(normalize_key(link_type, "link_type"))
        if link_value:
            clauses.append("link_value = ?")
            params.append(require_text(link_value, "link_value"))
        rows = self._con.execute(
            f"""
            SELECT record_json FROM identity_links
            WHERE {' AND '.join(clauses)}
            ORDER BY match_class, source_system, source_ref, link_id
            LIMIT ?
            """,
            (*params, checked_limit(limit, "limit"),),
        ).fetchall()
        return tuple(json_loads(row["record_json"]) for row in rows)

    def list_customer_id_mappings(
        self,
        tenant_id: str,
        *,
        old_customer_id: Optional[str] = None,
        new_customer_id: Optional[str] = None,
        resolution_status: Optional[str] = None,
        limit: int = 500,
    ) -> tuple[Mapping[str, Any], ...]:
        tenant = normalize_key(tenant_id, "tenant_id")
        clauses = ["tenant_id = ?"]
        params: list[Any] = [tenant]
        if old_customer_id:
            clauses.append("old_customer_id = ?")
            params.append(require_text(old_customer_id, "old_customer_id"))
        if new_customer_id:
            clauses.append("new_customer_id = ?")
            params.append(require_text(new_customer_id, "new_customer_id"))
        if resolution_status:
            clauses.append("resolution_status = ?")
            params.append(normalize_key(resolution_status, "resolution_status"))
        rows = self._con.execute(
            f"""
            SELECT record_json FROM customer_id_mappings
            WHERE {' AND '.join(clauses)}
            ORDER BY old_customer_id, new_customer_id, mapping_id
            LIMIT ?
            """,
            (*params, checked_limit(limit, "limit"),),
        ).fetchall()
        return tuple(json_loads(row["record_json"]) for row in rows)

    def list_events_by_customer(
        self,
        tenant_id: str,
        customer_id: str,
        *,
        opportunity_id: Optional[str] = None,
        event_types: Sequence[str] = (),
        source_systems: Sequence[str] = (),
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        sort: str = "desc",
        include_artifacts: bool = False,
        include_signals: bool = False,
        limit: int = 50,
        cursor: Optional[str] = None,
    ) -> Mapping[str, Any]:
        tenant = normalize_key(tenant_id, "tenant_id")
        clauses = ["tenant_id = ?", "customer_id = ?"]
        params: list[Any] = [tenant, require_text(customer_id, "customer_id")]
        if opportunity_id:
            clauses.append("opportunity_id = ?")
            params.append(require_text(opportunity_id, "opportunity_id"))
        append_in_clause(clauses, params, "event_type", event_types, normalizer=lambda item: normalize_key(item, "event_type"))
        append_in_clause(
            clauses,
            params,
            "source_system",
            source_systems,
            normalizer=lambda item: normalize_key(item, "source_system"),
        )
        if since is not None:
            require_timezone(since, "since")
            clauses.append("event_at >= ?")
            params.append(since.isoformat())
        if until is not None:
            require_timezone(until, "until")
            clauses.append("event_at <= ?")
            params.append(until.isoformat())
        direction = "ASC" if normalize_key(sort, "sort") == "asc" else "DESC"
        page_limit, offset = normalize_pagination(limit, cursor)
        rows = self._con.execute(
            f"""
            SELECT event_id, record_json FROM timeline_events
            WHERE {' AND '.join(clauses)}
            ORDER BY event_at {direction}, event_id {direction}
            LIMIT ? OFFSET ?
            """,
            (*params, page_limit + 1, offset),
        ).fetchall()
        items = [json_loads(row["record_json"]) for row in rows[:page_limit]]
        if include_artifacts or include_signals:
            items = [
                self._with_event_children(tenant, item, include_artifacts=include_artifacts, include_signals=include_signals)
                for item in items
            ]
        return {
            "items": items,
            "next_cursor": str(offset + page_limit) if len(rows) > page_limit else None,
        }

    def list_ingestion_runs(
        self,
        tenant_id: str,
        *,
        source_system: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
        cursor: Optional[str] = None,
    ) -> Mapping[str, Any]:
        tenant = normalize_key(tenant_id, "tenant_id")
        clauses = ["tenant_id = ?"]
        params: list[Any] = [tenant]
        if source_system:
            clauses.append("source_system = ?")
            params.append(normalize_key(source_system, "source_system"))
        if status:
            clauses.append("status = ?")
            params.append(normalize_key(status, "status"))
        page_limit, offset = normalize_pagination(limit, cursor)
        rows = self._con.execute(
            f"""
            SELECT record_json FROM ingestion_runs
            WHERE {' AND '.join(clauses)}
            ORDER BY started_at DESC, run_id
            LIMIT ? OFFSET ?
            """,
            (*params, page_limit + 1, offset),
        ).fetchall()
        return page_result(rows, limit=page_limit, offset=offset)

    def list_audit_log(
        self,
        tenant_id: str,
        *,
        entity_type: Optional[str] = None,
        entity_id: Optional[str] = None,
        ingestion_run_id: Optional[str] = None,
        actor: Optional[str] = None,
        action: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        limit: int = 100,
        cursor: Optional[str] = None,
    ) -> Mapping[str, Any]:
        tenant = normalize_key(tenant_id, "tenant_id")
        clauses = ["tenant_id = ?"]
        params: list[Any] = [tenant]
        if entity_type:
            clauses.append("entity_type = ?")
            params.append(normalize_key(entity_type, "entity_type"))
        if entity_id:
            clauses.append("entity_id = ?")
            params.append(require_text(entity_id, "entity_id"))
        if ingestion_run_id:
            clauses.append("ingestion_run_id = ?")
            params.append(require_text(ingestion_run_id, "ingestion_run_id"))
        if actor:
            clauses.append("actor = ?")
            params.append(require_text(actor, "actor"))
        if action:
            clauses.append("action = ?")
            params.append(normalize_key(action, "action"))
        if since is not None:
            require_timezone(since, "since")
            clauses.append("created_at >= ?")
            params.append(since.isoformat())
        if until is not None:
            require_timezone(until, "until")
            clauses.append("created_at <= ?")
            params.append(until.isoformat())
        page_limit, offset = normalize_pagination(limit, cursor)
        rows = self._con.execute(
            f"""
            SELECT record_json FROM audit_log
            WHERE {' AND '.join(clauses)}
            ORDER BY seq DESC
            LIMIT ? OFFSET ?
            """,
            (*params, page_limit + 1, offset),
        ).fetchall()
        return page_result(rows, limit=page_limit, offset=offset)

    def search_timeline(
        self,
        tenant_id: str,
        query: str,
        *,
        customer_id: Optional[str] = None,
        opportunity_id: Optional[str] = None,
        scopes: Sequence[str] = ("events", "bot_context", "signals"),
        event_types: Sequence[str] = (),
        source_systems: Sequence[str] = (),
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        allowed_for_bot: Optional[bool] = None,
        mode: str = "auto",
        include_highlights: bool = True,
        limit: int = 25,
        cursor: Optional[str] = None,
    ) -> Mapping[str, Any]:
        tenant = normalize_key(tenant_id, "tenant_id")
        text = require_text(query, "query")
        normalized_mode = normalize_key(mode, "mode")
        normalized_scopes = tuple(normalize_key(scope, "search scope") for scope in scopes)
        unknown_scopes = set(normalized_scopes) - ALLOWED_SEARCH_SCOPES
        if unknown_scopes:
            raise ValueError(f"unsupported search scopes: {sorted(unknown_scopes)}")
        page_limit, offset = normalize_pagination(limit, cursor)
        if normalized_mode not in {"auto", "fts", "fallback"}:
            raise ValueError("mode must be auto, fts, or fallback")
        use_fts = self._fts_enabled and normalized_mode in {"auto", "fts"}
        if normalized_mode == "fts" and not self._fts_enabled:
            raise RuntimeError("timeline FTS search requested but FTS5 is not available")
        if use_fts:
            hits = self._search_fts(
                tenant=tenant,
                query=text,
                scopes=normalized_scopes,
                customer_id=customer_id,
                opportunity_id=opportunity_id,
                event_types=event_types,
                source_systems=source_systems,
                since=since,
                until=until,
                allowed_for_bot=allowed_for_bot,
                include_highlights=include_highlights,
                limit=page_limit + 1,
                offset=offset,
            )
            backend = "fts5"
        else:
            hits = self._search_fallback(
                tenant=tenant,
                query=text,
                scopes=normalized_scopes,
                customer_id=customer_id,
                opportunity_id=opportunity_id,
                event_types=event_types,
                source_systems=source_systems,
                since=since,
                until=until,
                allowed_for_bot=allowed_for_bot,
                limit=page_limit + 1,
                offset=offset,
            )
            backend = "fallback_like"
        return {
            "schema_version": CUSTOMER_TIMELINE_SQLITE_SCHEMA_VERSION,
            "backend": backend,
            "query": text,
            "items": hits[:page_limit],
            "next_cursor": str(offset + page_limit) if len(hits) > page_limit else None,
        }

    def summary(self) -> Mapping[str, Any]:
        counts = {table: self._table_count(table) for table in REQUIRED_TABLES}
        event_status_counts = self._counts_by("timeline_events", "match_status")
        signal_severity_counts = self._counts_by("derived_signals", "severity")
        return {
            "schema_version": CUSTOMER_TIMELINE_SQLITE_SCHEMA_VERSION,
            "backend": "sqlite",
            "read_only": self.read_only,
            "fts_enabled": self._fts_enabled,
            "db_path": str(self.db_path),
            "counts": counts,
            "event_status_counts": event_status_counts,
            "signal_severity_counts": signal_severity_counts,
            "soft_integrity": {
                "events_without_customer": self._scalar_int(
                    "SELECT COUNT(*) FROM timeline_events WHERE customer_id IS NULL OR customer_id = ''"
                ),
                "event_customer_missing": self._scalar_int(
                    """
                    SELECT COUNT(*)
                    FROM timeline_events e
                    LEFT JOIN customer_identities c
                      ON c.tenant_id = e.tenant_id AND c.customer_id = e.customer_id
                    WHERE e.customer_id IS NOT NULL AND e.customer_id != '' AND c.customer_id IS NULL
                    """
                ),
                "artifacts_without_event": self._scalar_int(
                    """
                    SELECT COUNT(*)
                    FROM event_artifacts a
                    LEFT JOIN timeline_events e
                      ON e.tenant_id = a.tenant_id AND e.event_id = a.event_id
                    WHERE e.event_id IS NULL
                    """
                ),
                "bot_chunks_blocked_for_bot": self._scalar_int(
                    """
                    SELECT COUNT(*)
                    FROM bot_context_chunks
                    WHERE allowed_for_bot = 0 OR requires_manager_review = 1
                    """
                ),
            },
            "safety": customer_timeline_sqlite_safety_contract(),
            "validation_ok": True,
        }

    def snapshot(self) -> Mapping[str, Any]:
        return {
            "schema_version": CUSTOMER_TIMELINE_SQLITE_SCHEMA_VERSION,
            "open_result": self.open_result.to_json_dict(),
            "summary": self.summary(),
            "safety": customer_timeline_sqlite_safety_contract(),
        }

    def _connect(self) -> sqlite3.Connection:
        if self.read_only:
            uri = f"file:{self.db_path}?mode=ro"
            con = sqlite3.connect(uri, uri=True, timeout=15)
            con.execute("PRAGMA query_only = ON")
        else:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            con = sqlite3.connect(self.db_path, timeout=30)
            con.execute("PRAGMA journal_mode = WAL")
            con.execute("PRAGMA busy_timeout = 30000")
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON")
        return con

    def _acquire_writer_lock(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.db_path.with_suffix(self.db_path.suffix + ".writer.lock")
        handle = lock_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.close()
            raise RuntimeError(f"customer timeline writer lock is already held: {lock_path}") from exc
        self._writer_lock_path = lock_path
        self._writer_lock_handle = handle

    def _release_writer_lock(self) -> None:
        handle = self._writer_lock_handle
        if handle is None:
            return
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
            self._writer_lock_handle = None
            self._writer_lock_path = None

    def _ensure_writable(self) -> None:
        if self.read_only:
            raise PermissionError("CustomerTimelineSQLiteStore is opened in read-only mode")

    def _now(self) -> datetime:
        value = self._clock()
        require_timezone(value, "clock value")
        return value

    def _fetch_one(self, query: str, params: Sequence[Any] = ()) -> Optional[sqlite3.Row]:
        return self._con.execute(query, tuple(params)).fetchone()

    def _upsert_record(
        self,
        *,
        table: str,
        key_column: str,
        key_value: str,
        record_type: str,
        tenant_id: str,
        payload: Mapping[str, Any],
        columns: Mapping[str, Any],
        actor: str,
        ingestion_run_id: Optional[str],
        commit: bool = True,
    ) -> CustomerTimelineStoreWriteResult:
        key = require_text(key_value, key_column)
        safe_payload = scrub_timeline_persisted_json(payload)
        payload_json = json_dumps(safe_payload)
        record_hash = stable_digest(safe_payload)
        existing = self._fetch_one(
            f"SELECT record_hash, record_json FROM {table} WHERE {key_column} = ?",
            (key,),
        )
        if existing is not None and existing["record_hash"] == record_hash:
            return CustomerTimelineStoreWriteResult(record_type, key, False, "duplicate", record_hash)
        before_hash = existing["record_hash"] if existing is not None else None
        action = "updated" if existing is not None else "created"
        all_columns = {key_column: key, **dict(columns), "record_hash": record_hash, "record_json": payload_json}
        column_names = tuple(all_columns)
        placeholders = ", ".join("?" for _ in column_names)
        update_assignments = ", ".join(
            f"{column} = excluded.{column}" for column in column_names if column != key_column
        )
        self._con.execute(
            f"""
            INSERT INTO {table} ({', '.join(column_names)})
            VALUES ({placeholders})
            ON CONFLICT({key_column}) DO UPDATE SET {update_assignments}
            """,
            tuple(all_columns.values()),
        )
        audit = self._append_audit_log(
            tenant_id=tenant_id,
            action=f"{record_type}_{action}",
            entity_type=record_type,
            entity_id=key,
            actor=actor,
            ingestion_run_id=ingestion_run_id,
            before_hash=before_hash,
            after_hash=record_hash,
            metadata={"table": table},
            now=self._now(),
        )
        if commit:
            self._commit()
        return CustomerTimelineStoreWriteResult(
            record_type=record_type,
            record_id=key,
            created=existing is None,
            status=action,
            record_hash=record_hash,
            audit_id=audit.audit_id,
        )

    def _upsert_ingestion_run(self, run: CustomerTimelineIngestionRun, *, actor: str) -> CustomerTimelineStoreWriteResult:
        return self._upsert_record(
            table="ingestion_runs",
            key_column="run_id",
            key_value=run.run_id,
            record_type="ingestion_run",
            tenant_id=run.tenant_id,
            payload=run.to_json_dict(),
            columns={
                "tenant_id": run.tenant_id,
                "source_system": run.source_system,
                "source_ref": run.source_ref,
                "run_kind": run.run_kind,
                "idempotency_key": run.idempotency_key,
                "status": run.status,
                "started_at": run.started_at.isoformat(),
                "finished_at": run.finished_at.isoformat() if run.finished_at else None,
                "input_hash": run.input_hash,
                "accepted_count": run.accepted_count,
                "rejected_count": run.rejected_count,
                "output_ref": run.output_ref,
                "error": run.error,
            },
            actor=actor,
            ingestion_run_id=run.run_id,
        )

    def _commit(self) -> None:
        if self._bulk_write_depth > 0:
            self._bulk_write_dirty = True
            return
        self._con.commit()

    def _append_audit_log(
        self,
        *,
        tenant_id: str,
        action: str,
        entity_type: str,
        entity_id: Optional[str],
        actor: str,
        ingestion_run_id: Optional[str],
        before_hash: Optional[str],
        after_hash: Optional[str],
        metadata: Mapping[str, Any],
        now: datetime,
    ) -> CustomerTimelineAuditEntry:
        seq = self._next_audit_seq()
        audit = CustomerTimelineAuditEntry(
            audit_id=stable_prefixed_id(
                "timeline_audit",
                {
                    "seq": seq,
                    "tenant_id": tenant_id,
                    "action": action,
                    "entity_type": entity_type,
                    "entity_id": entity_id,
                    "created_at": now.isoformat(),
                },
            ),
            tenant_id=tenant_id,
            action=action,
            entity_type=entity_type,
            entity_id=entity_id,
            actor=actor,
            created_at=now,
            ingestion_run_id=ingestion_run_id,
            before_hash=before_hash,
            after_hash=after_hash,
            metadata=scrub_timeline_persisted_json(metadata),
        )
        self._con.execute(
            """
            INSERT INTO audit_log (
              seq, audit_id, tenant_id, action, entity_type, entity_id, actor,
              created_at, ingestion_run_id, before_hash, after_hash, record_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                seq,
                audit.audit_id,
                audit.tenant_id,
                audit.action,
                audit.entity_type,
                audit.entity_id,
                audit.actor,
                audit.created_at.isoformat(),
                audit.ingestion_run_id,
                audit.before_hash,
                audit.after_hash,
                json_dumps(audit.to_json_dict()),
            ),
        )
        return audit

    def _next_audit_seq(self) -> int:
        row = self._con.execute("SELECT COALESCE(MAX(seq), 0) + 1 AS next_seq FROM audit_log").fetchone()
        return int(row["next_seq"])

    def _get_record(self, table: str, where_sql: str, params: Sequence[Any]) -> Optional[Mapping[str, Any]]:
        row = self._fetch_one(f"SELECT record_json FROM {table} WHERE {where_sql}", params)
        return None if row is None else json_loads(row["record_json"])

    def _assert_customer_exists(self, tenant_id: str, customer_id: str) -> None:
        if self.get_customer(tenant_id, customer_id) is None:
            raise ValueError(f"customer does not exist for tenant {tenant_id}: {customer_id}")

    def _assert_opportunity_exists(self, tenant_id: str, opportunity_id: str) -> None:
        row = self._fetch_one(
            "SELECT 1 FROM customer_opportunities WHERE tenant_id = ? AND opportunity_id = ?",
            (normalize_key(tenant_id, "tenant_id"), require_text(opportunity_id, "opportunity_id")),
        )
        if row is None:
            raise ValueError(f"opportunity does not exist for tenant {tenant_id}: {opportunity_id}")

    def _assert_event_exists(self, tenant_id: str, event_id: str) -> None:
        if self.get_event(tenant_id, event_id) is None:
            raise ValueError(f"event does not exist for tenant {tenant_id}: {event_id}")

    def _with_event_children(
        self,
        tenant_id: str,
        event: Mapping[str, Any],
        *,
        include_artifacts: bool,
        include_signals: bool,
    ) -> Mapping[str, Any]:
        event_id = require_text(event.get("event_id"), "event_id")
        result = dict(event)
        if include_artifacts:
            rows = self._con.execute(
                """
                SELECT record_json FROM event_artifacts
                WHERE tenant_id = ? AND event_id = ?
                ORDER BY artifact_type, artifact_id
                """,
                (tenant_id, event_id),
            ).fetchall()
            result["artifacts"] = [json_loads(row["record_json"]) for row in rows]
        if include_signals:
            rows = self._con.execute(
                """
                SELECT record_json FROM derived_signals
                WHERE tenant_id = ? AND event_id = ?
                ORDER BY severity DESC, signal_type, signal_id
                """,
                (tenant_id, event_id),
            ).fetchall()
            result["signals"] = [json_loads(row["record_json"]) for row in rows]
        return result

    def _bootstrap_fts(self) -> None:
        self._con.executescript(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS timeline_event_fts USING fts5(
              tenant_id UNINDEXED,
              event_id UNINDEXED,
              customer_id UNINDEXED,
              opportunity_id UNINDEXED,
              event_type UNINDEXED,
              source_system UNINDEXED,
              event_at UNINDEXED,
              subject,
              text_preview,
              summary,
              record_text
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS bot_context_chunk_fts USING fts5(
              tenant_id UNINDEXED,
              chunk_id UNINDEXED,
              customer_id UNINDEXED,
              opportunity_id UNINDEXED,
              event_id UNINDEXED,
              event_at UNINDEXED,
              text,
              summary
            );
            """
        )

    def _detect_existing_fts(self) -> bool:
        row = self._fetch_one(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'timeline_event_fts'"
        )
        return row is not None

    def _sync_event_fts(self, event: TimelineEvent, payload: Mapping[str, Any], record_hash: str) -> None:
        if not self._fts_enabled and not self._detect_existing_fts():
            return
        self._con.execute("DELETE FROM timeline_event_fts WHERE event_id = ?", (event.event_id,))
        self._con.execute(
            """
            INSERT INTO timeline_event_fts (
              tenant_id, event_id, customer_id, opportunity_id, event_type,
              source_system, event_at, subject, text_preview, summary, record_text
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.tenant_id,
                event.event_id,
                event.customer_id,
                event.opportunity_id,
                event.event_type.value,
                event.source_system,
                event.event_at.isoformat(),
                event.subject,
                event.text_preview,
                event.summary,
                json_dumps({"hash": record_hash, "record": payload.get("record") or {}}),
            ),
        )
        self._fts_enabled = True

    def _sync_chunk_fts(self, chunk: BotContextChunk, record_hash: str) -> None:
        if not self._fts_enabled and not self._detect_existing_fts():
            return
        self._con.execute("DELETE FROM bot_context_chunk_fts WHERE chunk_id = ?", (chunk.chunk_id,))
        self._con.execute(
            """
            INSERT INTO bot_context_chunk_fts (
              tenant_id, chunk_id, customer_id, opportunity_id, event_id,
              event_at, text, summary
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                chunk.tenant_id,
                chunk.chunk_id,
                chunk.customer_id,
                chunk.opportunity_id,
                chunk.event_id,
                chunk.event_at.isoformat() if chunk.event_at else "",
                chunk.text,
                f"{chunk.summary or ''} {record_hash}",
            ),
        )
        self._fts_enabled = True

    def _search_fts(
        self,
        *,
        tenant: str,
        query: str,
        scopes: Sequence[str],
        customer_id: Optional[str],
        opportunity_id: Optional[str],
        event_types: Sequence[str],
        source_systems: Sequence[str],
        since: Optional[datetime],
        until: Optional[datetime],
        allowed_for_bot: Optional[bool],
        include_highlights: bool,
        limit: int,
        offset: int,
    ) -> list[Mapping[str, Any]]:
        hits: list[Mapping[str, Any]] = []
        fts_query = build_fts_query(query)
        if "events" in scopes:
            event_clauses = ["f.tenant_id = ?", "timeline_event_fts MATCH ?"]
            event_params: list[Any] = [tenant, fts_query]
            self._append_event_filters(
                event_clauses,
                event_params,
                customer_id=customer_id,
                opportunity_id=opportunity_id,
                event_types=event_types,
                source_systems=source_systems,
                since=since,
                until=until,
                table_alias="e",
            )
            rows = self._con.execute(
                f"""
                SELECT e.event_id, e.event_at, e.record_json,
                       {'snippet(timeline_event_fts, 8, "[", "]", "...", 12)' if include_highlights else 'NULL'} AS highlight
                FROM timeline_event_fts f
                JOIN timeline_events e ON e.event_id = f.event_id AND e.tenant_id = f.tenant_id
                WHERE {' AND '.join(event_clauses)}
                ORDER BY e.event_at DESC, e.event_id DESC
                LIMIT ? OFFSET ?
                """,
                (*event_params, limit, offset),
            ).fetchall()
            hits.extend(search_hit_from_row("event", row) for row in rows)
        if "bot_context" in scopes:
            chunk_clauses = ["f.tenant_id = ?", "bot_context_chunk_fts MATCH ?"]
            chunk_params: list[Any] = [tenant, fts_query]
            self._append_chunk_filters(
                chunk_clauses,
                chunk_params,
                customer_id=customer_id,
                opportunity_id=opportunity_id,
                since=since,
                until=until,
                allowed_for_bot=allowed_for_bot,
                table_alias="c",
            )
            rows = self._con.execute(
                f"""
                SELECT c.chunk_id AS event_id,
                       COALESCE(c.event_at, c.created_at) AS event_at,
                       c.record_json,
                       {'snippet(bot_context_chunk_fts, 6, "[", "]", "...", 12)' if include_highlights else 'NULL'} AS highlight
                FROM bot_context_chunk_fts f
                JOIN bot_context_chunks c ON c.chunk_id = f.chunk_id AND c.tenant_id = f.tenant_id
                WHERE {' AND '.join(chunk_clauses)}
                ORDER BY COALESCE(c.event_at, c.created_at) DESC, c.chunk_id DESC
                LIMIT ? OFFSET ?
                """,
                (*chunk_params, limit, offset),
            ).fetchall()
            hits.extend(search_hit_from_row("bot_context", row) for row in rows)
        if "signals" in scopes:
            signal_hits = self._search_fallback(
                tenant=tenant,
                query=query,
                scopes=("signals",),
                customer_id=customer_id,
                opportunity_id=opportunity_id,
                event_types=(),
                source_systems=(),
                since=None,
                until=None,
                allowed_for_bot=None,
                limit=limit,
                offset=offset,
            )
            hits.extend(signal_hits)
        return sorted(hits, key=lambda item: (item["event_at"] or "", item["id"]), reverse=True)[:limit]

    def _search_fallback(
        self,
        *,
        tenant: str,
        query: str,
        scopes: Sequence[str],
        customer_id: Optional[str],
        opportunity_id: Optional[str],
        event_types: Sequence[str],
        source_systems: Sequence[str],
        since: Optional[datetime],
        until: Optional[datetime],
        allowed_for_bot: Optional[bool],
        limit: int,
        offset: int,
    ) -> list[Mapping[str, Any]]:
        pattern = f"%{query.strip()}%"
        hits: list[Mapping[str, Any]] = []
        if "events" in scopes:
            clauses = ["tenant_id = ?", "(subject LIKE ? OR text_preview LIKE ? OR summary LIKE ? OR record_json LIKE ?)"]
            params: list[Any] = [tenant, pattern, pattern, pattern, pattern]
            self._append_event_filters(
                clauses,
                params,
                customer_id=customer_id,
                opportunity_id=opportunity_id,
                event_types=event_types,
                source_systems=source_systems,
                since=since,
                until=until,
            )
            rows = self._con.execute(
                f"""
                SELECT event_id, event_at, record_json, NULL AS highlight
                FROM timeline_events
                WHERE {' AND '.join(clauses)}
                ORDER BY event_at DESC, event_id DESC
                LIMIT ? OFFSET ?
                """,
                (*params, limit, offset),
            ).fetchall()
            hits.extend(search_hit_from_row("event", row) for row in rows)
        if "bot_context" in scopes:
            clauses = ["tenant_id = ?", "record_json LIKE ?"]
            params = [tenant, pattern]
            self._append_chunk_filters(
                clauses,
                params,
                customer_id=customer_id,
                opportunity_id=opportunity_id,
                since=since,
                until=until,
                allowed_for_bot=allowed_for_bot,
            )
            rows = self._con.execute(
                f"""
                SELECT chunk_id AS event_id, COALESCE(event_at, created_at) AS event_at, record_json, NULL AS highlight
                FROM bot_context_chunks
                WHERE {' AND '.join(clauses)}
                ORDER BY COALESCE(event_at, created_at) DESC, chunk_id DESC
                LIMIT ? OFFSET ?
                """,
                (*params, limit, offset),
            ).fetchall()
            hits.extend(search_hit_from_row("bot_context", row) for row in rows)
        if "signals" in scopes:
            clauses = ["tenant_id = ?", "record_json LIKE ?"]
            params = [tenant, pattern]
            if customer_id:
                clauses.append("customer_id = ?")
                params.append(require_text(customer_id, "customer_id"))
            if opportunity_id:
                clauses.append("opportunity_id = ?")
                params.append(require_text(opportunity_id, "opportunity_id"))
            rows = self._con.execute(
                f"""
                SELECT signal_id AS event_id, created_at AS event_at, record_json, NULL AS highlight
                FROM derived_signals
                WHERE {' AND '.join(clauses)}
                ORDER BY created_at DESC, signal_id DESC
                LIMIT ? OFFSET ?
                """,
                (*params, limit, offset),
            ).fetchall()
            hits.extend(search_hit_from_row("signal", row) for row in rows)
        return sorted(hits, key=lambda item: (item["event_at"] or "", item["id"]), reverse=True)[:limit]

    def _append_event_filters(
        self,
        clauses: list[str],
        params: list[Any],
        *,
        customer_id: Optional[str],
        opportunity_id: Optional[str],
        event_types: Sequence[str],
        source_systems: Sequence[str],
        since: Optional[datetime],
        until: Optional[datetime],
        table_alias: Optional[str] = None,
    ) -> None:
        prefix = f"{table_alias}." if table_alias else ""
        if customer_id:
            clauses.append(f"{prefix}customer_id = ?")
            params.append(require_text(customer_id, "customer_id"))
        if opportunity_id:
            clauses.append(f"{prefix}opportunity_id = ?")
            params.append(require_text(opportunity_id, "opportunity_id"))
        append_in_clause(
            clauses,
            params,
            f"{prefix}event_type",
            event_types,
            normalizer=lambda item: normalize_key(item, "event_type"),
        )
        append_in_clause(
            clauses,
            params,
            f"{prefix}source_system",
            source_systems,
            normalizer=lambda item: normalize_key(item, "source_system"),
        )
        if since is not None:
            require_timezone(since, "since")
            clauses.append(f"{prefix}event_at >= ?")
            params.append(since.isoformat())
        if until is not None:
            require_timezone(until, "until")
            clauses.append(f"{prefix}event_at <= ?")
            params.append(until.isoformat())

    def _append_chunk_filters(
        self,
        clauses: list[str],
        params: list[Any],
        *,
        customer_id: Optional[str],
        opportunity_id: Optional[str],
        since: Optional[datetime],
        until: Optional[datetime],
        allowed_for_bot: Optional[bool],
        table_alias: Optional[str] = None,
    ) -> None:
        prefix = f"{table_alias}." if table_alias else ""
        if customer_id:
            clauses.append(f"{prefix}customer_id = ?")
            params.append(require_text(customer_id, "customer_id"))
        if opportunity_id:
            clauses.append(f"{prefix}opportunity_id = ?")
            params.append(require_text(opportunity_id, "opportunity_id"))
        if since is not None:
            require_timezone(since, "since")
            clauses.append(f"COALESCE({prefix}event_at, {prefix}created_at) >= ?")
            params.append(since.isoformat())
        if until is not None:
            require_timezone(until, "until")
            clauses.append(f"COALESCE({prefix}event_at, {prefix}created_at) <= ?")
            params.append(until.isoformat())
        if allowed_for_bot is not None:
            clauses.append(f"{prefix}allowed_for_bot = ?")
            params.append(int(bool(allowed_for_bot)))

    def _table_count(self, table_name: str) -> int:
        return self._scalar_int(f"SELECT COUNT(*) FROM {table_name}")

    def _scalar_int(self, query: str, params: Sequence[Any] = ()) -> int:
        row = self._con.execute(query, tuple(params)).fetchone()
        return int(row[0] if row is not None else 0)

    def _counts_by(self, table_name: str, column_name: str) -> Mapping[str, int]:
        rows = self._con.execute(
            f"SELECT {column_name} AS key, COUNT(*) AS total FROM {table_name} GROUP BY {column_name} ORDER BY {column_name}"
        ).fetchall()
        return {str(row["key"]): int(row["total"]) for row in rows}


REQUIRED_TABLES = (
    "schema_migrations",
    "customer_identities",
    "identity_links",
    "customer_opportunities",
    "timeline_events",
    "event_artifacts",
    "derived_signals",
    "bot_context_chunks",
    "ingestion_runs",
    "timeline_conflicts",
    "customer_id_mappings",
    "audit_log",
)


def guard_customer_timeline_sqlite_path(db_path: Path | str) -> Path:
    candidate = Path(db_path).expanduser()
    if not candidate.name:
        raise ValueError("customer timeline SQLite path must include a filename")
    resolved = candidate.resolve(strict=False)
    if candidate.name in RUNTIME_DB_FILENAMES or resolved.name in RUNTIME_DB_FILENAMES:
        raise ValueError(f"refusing runtime-looking DB filename: {candidate.name}")
    if candidate.suffix not in {".sqlite", ".db", ".sqlite3"}:
        raise ValueError("customer timeline SQLite path must end with .sqlite, .db, or .sqlite3")
    return resolved


def customer_timeline_sqlite_safety_contract() -> Mapping[str, Any]:
    return {
        **customer_timeline_safety_contract(),
        "network_calls": False,
        "write_product_timeline_db": True,
        "read_only_mode_available": True,
        "query_only_read_only_connections": True,
        "stores_raw_payload_by_default": False,
        "stores_raw_files_in_sqlite": False,
        "audit_log_append_only": True,
    }


def sqlite_fts5_available(con: sqlite3.Connection) -> bool:
    try:
        con.execute("CREATE VIRTUAL TABLE temp.__timeline_fts_probe USING fts5(value)")
        con.execute("DROP TABLE temp.__timeline_fts_probe")
    except sqlite3.Error:
        return False
    return True


def json_dumps(value: Mapping[str, Any]) -> str:
    return json.dumps(scrub_timeline_persisted_json(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def json_loads(value: str) -> Mapping[str, Any]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("stored customer timeline JSON record must be an object")
    return parsed


def scrub_timeline_persisted_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            text_key = str(key)
            normalized_key = text_key.strip().lower()
            if normalized_key in FORBIDDEN_PERSISTED_PAYLOAD_KEYS or normalized_key.endswith("_bytes"):
                continue
            result[text_key] = scrub_timeline_persisted_json(item)
        return result
    if isinstance(value, (list, tuple)):
        return [scrub_timeline_persisted_json(item) for item in value]
    return value


def parse_datetime(value: str, field_name: str) -> datetime:
    text = require_text(value, field_name)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def ingestion_run_from_json(payload: Mapping[str, Any]) -> CustomerTimelineIngestionRun:
    return CustomerTimelineIngestionRun(
        tenant_id=payload["tenant_id"],
        source_system=payload["source_system"],
        source_ref=payload["source_ref"],
        run_kind=payload["run_kind"],
        idempotency_key=payload["idempotency_key"],
        run_id=payload["run_id"],
        status=payload["status"],
        started_at=parse_datetime(payload["started_at"], "started_at"),
        finished_at=parse_datetime(payload["finished_at"], "finished_at") if payload.get("finished_at") else None,
        input_hash=payload.get("input_hash"),
        accepted_count=int(payload.get("accepted_count") or 0),
        rejected_count=int(payload.get("rejected_count") or 0),
        output_ref=payload.get("output_ref"),
        error=payload.get("error"),
        metadata=payload.get("metadata") or {},
    )


def checked_limit(limit: int, field_name: str = "limit", *, max_limit: int = 500) -> int:
    value = int(limit)
    if value <= 0:
        raise ValueError(f"{field_name} must be positive")
    return min(value, max_limit)


def normalize_pagination(limit: int, cursor: Optional[str]) -> tuple[int, int]:
    page_limit = checked_limit(limit)
    if cursor is None:
        return page_limit, 0
    offset = int(require_text(cursor, "cursor"))
    if offset < 0:
        raise ValueError("cursor must not be negative")
    return page_limit, offset


def page_result(rows: Sequence[sqlite3.Row], *, limit: int, offset: int) -> Mapping[str, Any]:
    items = [json_loads(row["record_json"]) for row in rows[:limit]]
    return {
        "items": items,
        "next_cursor": str(offset + limit) if len(rows) > limit else None,
    }


def append_in_clause(
    clauses: list[str],
    params: list[Any],
    column_name: str,
    values: Sequence[str],
    *,
    normalizer: Callable[[str], str],
) -> None:
    if not values:
        return
    normalized = [normalizer(item) for item in values]
    placeholders = ", ".join("?" for _ in normalized)
    clauses.append(f"{column_name} IN ({placeholders})")
    params.extend(normalized)


def build_fts_query(query: str) -> str:
    tokens = _SEARCH_TOKEN_RE.findall(require_text(query, "query"))
    if not tokens:
        return f'"{query.strip().replace(chr(34), chr(34) + chr(34))}"'
    return " ".join(f'"{token.replace(chr(34), chr(34) + chr(34))}"' for token in tokens)


def search_hit_from_row(scope: str, row: sqlite3.Row) -> Mapping[str, Any]:
    record = json_loads(row["record_json"])
    return {
        "scope": scope,
        "id": row["event_id"],
        "event_at": row["event_at"],
        "record": record,
        "highlight": row["highlight"],
    }


__all__ = [
    "CUSTOMER_TIMELINE_SQLITE_MIGRATION_ID",
    "CUSTOMER_TIMELINE_SQLITE_SCHEMA_VERSION",
    "CustomerTimelineAuditEntry",
    "CustomerTimelineIngestionRun",
    "CustomerTimelineSQLiteOpenResult",
    "CustomerTimelineSQLiteStore",
    "CustomerTimelineStoreWriteResult",
    "build_fts_query",
    "customer_timeline_sqlite_safety_contract",
    "guard_customer_timeline_sqlite_path",
    "ingestion_run_from_json",
    "json_dumps",
    "json_loads",
    "scrub_timeline_persisted_json",
    "sqlite_fts5_available",
]
