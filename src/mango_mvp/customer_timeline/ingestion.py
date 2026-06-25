from __future__ import annotations

import csv
import hashlib
import json
import re
import sqlite3
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, Sequence

from mango_mvp.customer_timeline.contracts import (
    ArtifactType,
    BotContextChunk,
    CustomerIdentity,
    CustomerOpportunity,
    DerivedSignal,
    EventArtifact,
    ExtractionStatus,
    IdentityLink,
    IdentityMatchClass,
    IdentityStatus,
    OpportunityType,
    SignalSeverity,
    TimelineDirection,
    TimelineEvent,
    TimelineEventType,
    TimelineParticipant,
)
from mango_mvp.customer_timeline.ids import (
    normalize_email,
    normalize_key,
    optional_text,
    require_text,
    stable_customer_id,
    stable_digest,
)
from mango_mvp.customer_timeline.safety import (
    customer_timeline_safety_contract,
    guard_customer_timeline_output_path,
    is_stable_runtime_path,
)
from mango_mvp.customer_timeline.store import (
    CustomerTimelineSQLiteStore,
    CustomerTimelineStoreWriteResult,
    customer_timeline_sqlite_safety_contract,
    scrub_timeline_persisted_json,
)
from mango_mvp.utils.phone import normalize_phone


CUSTOMER_TIMELINE_INGESTION_SCHEMA_VERSION = "customer_timeline_ingestion_v1"
FORBIDDEN_IMPORT_HINTS = (
    "requests",
    "urllib",
    "imaplib",
    "smtplib",
    "socket",
    "subprocess",
    "mango_mvp.services.transcribe",
    "mango_mvp.services.resolve",
    "mango_mvp.services.analyze",
    "mango_mvp.amocrm_runtime.amo_integration",
    "mango_mvp.amocrm_runtime.tallanto_api",
)
_SQL_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,79}$")
BOT_FORBIDDEN_SOURCE_SYSTEMS = frozenset({"mail_archive", "channel_snapshot", "telegram_history"})


@dataclass(frozen=True)
class TimelineSourceRecord:
    source_system: str
    source_ref: str
    payload: Mapping[str, Any]
    source_path: Optional[str] = None
    observed_at: Optional[datetime] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_system", normalize_key(self.source_system, "source_system"))
        object.__setattr__(self, "source_ref", require_text(self.source_ref, "source_ref"))
        object.__setattr__(self, "payload", dict(self.payload))
        object.__setattr__(self, "source_path", optional_text(self.source_path))

    @property
    def payload_hash(self) -> str:
        return stable_digest(scrub_timeline_persisted_json(dict(self.payload)))

    def to_json_dict(self) -> Mapping[str, Any]:
        return {
            "schema_version": CUSTOMER_TIMELINE_INGESTION_SCHEMA_VERSION,
            "source_system": self.source_system,
            "source_ref": self.source_ref,
            "source_path": self.source_path,
            "observed_at": self.observed_at.isoformat() if self.observed_at else None,
            "payload_hash": self.payload_hash,
            "payload": scrub_timeline_persisted_json(dict(self.payload)),
        }


@dataclass(frozen=True)
class TimelineNormalizedBatch:
    source_record: TimelineSourceRecord
    customers: Sequence[CustomerIdentity] = field(default_factory=tuple)
    identity_links: Sequence[IdentityLink] = field(default_factory=tuple)
    opportunities: Sequence[CustomerOpportunity] = field(default_factory=tuple)
    events: Sequence[TimelineEvent] = field(default_factory=tuple)
    artifacts: Sequence[EventArtifact] = field(default_factory=tuple)
    signals: Sequence[DerivedSignal] = field(default_factory=tuple)
    bot_context_chunks: Sequence[BotContextChunk] = field(default_factory=tuple)
    conflicts: Sequence[Mapping[str, Any]] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.source_record, TimelineSourceRecord):
            raise TypeError("source_record must be TimelineSourceRecord")
        _assert_sequence_type(self.customers, CustomerIdentity, "customers")
        _assert_sequence_type(self.identity_links, IdentityLink, "identity_links")
        _assert_sequence_type(self.opportunities, CustomerOpportunity, "opportunities")
        _assert_sequence_type(self.events, TimelineEvent, "events")
        _assert_sequence_type(self.artifacts, EventArtifact, "artifacts")
        _assert_sequence_type(self.signals, DerivedSignal, "signals")
        _assert_sequence_type(self.bot_context_chunks, BotContextChunk, "bot_context_chunks")
        object.__setattr__(self, "customers", tuple(self.customers))
        object.__setattr__(self, "identity_links", tuple(self.identity_links))
        object.__setattr__(self, "opportunities", tuple(self.opportunities))
        object.__setattr__(self, "events", tuple(self.events))
        object.__setattr__(self, "artifacts", tuple(self.artifacts))
        object.__setattr__(self, "signals", tuple(self.signals))
        object.__setattr__(self, "bot_context_chunks", tuple(self.bot_context_chunks))
        object.__setattr__(self, "conflicts", tuple(dict(item) for item in self.conflicts))
        assert_bot_context_not_allowed_for_restricted_source(self)

    def counts(self) -> Mapping[str, int]:
        return {
            "customers": len(self.customers),
            "identity_links": len(self.identity_links),
            "opportunities": len(self.opportunities),
            "events": len(self.events),
            "artifacts": len(self.artifacts),
            "signals": len(self.signals),
            "bot_context_chunks": len(self.bot_context_chunks),
            "conflicts": len(self.conflicts),
        }


@dataclass(frozen=True)
class CustomerIdResolutionMapping:
    tenant_id: str
    old_customer_id: str
    new_customer_id: str
    reason: str
    source_refs: Sequence[str] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "tenant_id", normalize_key(self.tenant_id, "tenant_id"))
        object.__setattr__(self, "old_customer_id", require_text(self.old_customer_id, "old_customer_id"))
        object.__setattr__(self, "new_customer_id", require_text(self.new_customer_id, "new_customer_id"))
        object.__setattr__(self, "reason", normalize_key(self.reason, "reason"))
        object.__setattr__(self, "source_refs", tuple(require_text(item, "source_ref") for item in self.source_refs))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class CustomerIdResolutionResult:
    batches: Sequence[TimelineNormalizedBatch]
    mappings: Sequence[CustomerIdResolutionMapping]

    def __post_init__(self) -> None:
        object.__setattr__(self, "batches", tuple(self.batches))
        object.__setattr__(self, "mappings", tuple(self.mappings))


@dataclass(frozen=True)
class TimelineImportError:
    source_ref: str
    error_type: str
    message: str

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TimelineImportReport:
    schema_version: str
    dry_run: bool
    run_id: Optional[str]
    source_system: str
    source_ref: str
    input_hash: str
    accepted_count: int
    rejected_count: int
    normalized_counts: Mapping[str, int]
    write_status_counts: Mapping[str, int]
    errors: Sequence[TimelineImportError]
    source_inventory: Sequence[Mapping[str, Any]]
    source_unchanged: bool
    safety: Mapping[str, Any]

    @property
    def validation_ok(self) -> bool:
        return self.rejected_count == 0

    def to_json_dict(self) -> Mapping[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dry_run": self.dry_run,
            "run_id": self.run_id,
            "source_system": self.source_system,
            "source_ref": self.source_ref,
            "input_hash": self.input_hash,
            "accepted_count": self.accepted_count,
            "rejected_count": self.rejected_count,
            "normalized_counts": dict(self.normalized_counts),
            "write_status_counts": dict(self.write_status_counts),
            "errors": [item.to_json_dict() for item in self.errors],
            "source_inventory": [dict(item) for item in self.source_inventory],
            "source_unchanged": self.source_unchanged,
            "validation_ok": self.validation_ok,
            "safety": dict(self.safety),
        }


class TimelineNormalizer(Protocol):
    source_system: str

    def normalize(self, record: TimelineSourceRecord) -> TimelineNormalizedBatch:
        """Convert one read-only source record to customer timeline contracts."""


class TimelineImportService:
    def __init__(self, store: CustomerTimelineSQLiteStore) -> None:
        self.store = store

    def import_records(
        self,
        records: Sequence[TimelineSourceRecord],
        *,
        normalizer: TimelineNormalizer,
        tenant_id: str,
        source_ref: str,
        idempotency_key: Optional[str] = None,
        dry_run: bool = False,
        actor: str = "timeline_importer",
    ) -> TimelineImportReport:
        tenant = normalize_key(tenant_id, "tenant_id")
        source_system = normalize_key(normalizer.source_system, "source_system")
        normalized_source_ref = require_text(source_ref, "source_ref")
        normalized_records = tuple(records)
        _assert_sequence_type(normalized_records, TimelineSourceRecord, "records")
        input_hash = stable_digest(
            {
                "tenant_id": tenant,
                "source_system": source_system,
                "source_ref": normalized_source_ref,
                "records": [record.to_json_dict() for record in normalized_records],
            }
        )
        source_inventory_before = build_source_inventory(normalized_records)
        run_id: Optional[str] = None

        accepted = 0
        errors: list[TimelineImportError] = []
        normalized_counts = zero_normalized_counts()
        write_status_counts: dict[str, int] = {}
        raw_batches: list[TimelineNormalizedBatch] = []
        for record in normalized_records:
            try:
                batch = normalizer.normalize(record)
                raw_batches.append(batch)
                accepted += 1
            except Exception as exc:  # noqa: BLE001 - report per-record import errors, do not hide the batch.
                errors.append(
                    TimelineImportError(
                        source_ref=record.source_ref,
                        error_type=type(exc).__name__,
                        message=str(exc),
                    )
                )
        resolution = resolve_customer_identity_batches(raw_batches, store=self.store if not dry_run else None)
        batches = tuple(resolution.batches)
        for batch in batches:
            merge_counts(normalized_counts, batch.counts())
        normalized_counts["customer_id_mappings"] = len(resolution.mappings)
        inferred_conflicts = infer_identity_conflicts(batches)
        normalized_counts["conflicts"] = normalized_counts.get("conflicts", 0) + len(inferred_conflicts)
        if not dry_run:
            with self.store.bulk_write():
                run = self.store.start_ingestion_run(
                    tenant_id=tenant,
                    source_system=source_system,
                    source_ref=normalized_source_ref,
                    run_kind="timeline_import",
                    idempotency_key=idempotency_key or input_hash,
                    input_hash=input_hash,
                    metadata={
                        "schema_version": CUSTOMER_TIMELINE_INGESTION_SCHEMA_VERSION,
                        "records": len(normalized_records),
                        "safety": timeline_ingestion_safety_contract(),
                    },
                    actor=actor,
                )
                run_id = run.run_id
                for batch in batches:
                    for result in self._apply_batch(batch, actor=actor, ingestion_run_id=run_id):
                        write_status_counts[result.status] = write_status_counts.get(result.status, 0) + 1
                for conflict in inferred_conflicts:
                    result = self.store.record_conflict(
                        conflict["tenant_id"],
                        conflict_type=conflict["conflict_type"],
                        entity_refs=tuple(conflict["entity_refs"]),
                        severity=conflict.get("severity") or "medium",
                        status=conflict.get("status") or "open",
                        summary=conflict.get("summary"),
                        metadata=conflict.get("metadata") or {},
                        actor=actor,
                        ingestion_run_id=run_id,
                    )
                    write_status_counts[result.status] = write_status_counts.get(result.status, 0) + 1
                for mapping in resolution.mappings:
                    result = self.store.record_customer_id_mapping(
                        mapping.tenant_id,
                        old_customer_id=mapping.old_customer_id,
                        new_customer_id=mapping.new_customer_id,
                        reason=mapping.reason,
                        source_refs=mapping.source_refs,
                        metadata=mapping.metadata,
                        actor=actor,
                        ingestion_run_id=run_id,
                    )
                    write_status_counts[result.status] = write_status_counts.get(result.status, 0) + 1
                if run_id:
                    self.store.finish_ingestion_run(
                        run_id,
                        status="completed" if not errors else "completed_with_errors",
                        accepted_count=accepted,
                        rejected_count=len(errors),
                        metadata={
                            "normalized_counts": dict(normalized_counts),
                            "write_status_counts": dict(write_status_counts),
                        },
                        actor=actor,
                    )
        source_inventory_after = build_source_inventory(normalized_records)
        return TimelineImportReport(
            schema_version=CUSTOMER_TIMELINE_INGESTION_SCHEMA_VERSION,
            dry_run=dry_run,
            run_id=run_id,
            source_system=source_system,
            source_ref=normalized_source_ref,
            input_hash=input_hash,
            accepted_count=accepted,
            rejected_count=len(errors),
            normalized_counts=dict(normalized_counts),
            write_status_counts=dict(write_status_counts),
            errors=tuple(errors),
            source_inventory=tuple(source_inventory_after),
            source_unchanged=source_inventory_before == source_inventory_after,
            safety=timeline_ingestion_safety_contract(),
        )

    def _apply_batch(
        self,
        batch: TimelineNormalizedBatch,
        *,
        actor: str,
        ingestion_run_id: Optional[str],
    ) -> tuple[CustomerTimelineStoreWriteResult, ...]:
        results: list[CustomerTimelineStoreWriteResult] = []
        for customer in batch.customers:
            results.append(self.store.upsert_customer(customer, actor=actor, ingestion_run_id=ingestion_run_id))
        for link in batch.identity_links:
            results.append(self.store.upsert_identity_link(link, actor=actor, ingestion_run_id=ingestion_run_id))
        for opportunity in batch.opportunities:
            results.append(self.store.upsert_opportunity(opportunity, actor=actor, ingestion_run_id=ingestion_run_id))
        for event in batch.events:
            results.append(self.store.upsert_event(event, actor=actor, ingestion_run_id=ingestion_run_id))
        for artifact in batch.artifacts:
            results.append(self.store.upsert_artifact(artifact, actor=actor, ingestion_run_id=ingestion_run_id))
        for signal in batch.signals:
            results.append(self.store.upsert_signal(signal, actor=actor, ingestion_run_id=ingestion_run_id))
        for chunk in batch.bot_context_chunks:
            results.append(self.store.upsert_bot_context_chunk(chunk, actor=actor, ingestion_run_id=ingestion_run_id))
        for conflict in batch.conflicts:
            results.append(
                self.store.record_conflict(
                    conflict["tenant_id"],
                    conflict_type=conflict["conflict_type"],
                    entity_refs=tuple(conflict["entity_refs"]),
                    severity=conflict.get("severity") or "medium",
                    status=conflict.get("status") or "open",
                    summary=conflict.get("summary"),
                    metadata=conflict.get("metadata") or {},
                    actor=actor,
                    ingestion_run_id=ingestion_run_id,
                )
            )
        return tuple(results)


class AmoSnapshotNormalizer:
    source_system = "amocrm_snapshot"

    def __init__(self, *, tenant_id: str) -> None:
        self.tenant_id = normalize_key(tenant_id, "tenant_id")

    def normalize(self, record: TimelineSourceRecord) -> TimelineNormalizedBatch:
        payload = record.payload
        entity_id = require_text(first_value(payload, ("entity_id", "id", "contact_id", "lead_id")), "entity_id")
        entity_type = normalize_key(first_value(payload, ("entity_type", "type")) or "contact", "entity_type")
        source_ref = f"amocrm:{entity_type}:{entity_id}"
        phone = safe_phone(first_value(payload, ("phone", "primary_phone", "Телефон", "Телефон клиента")))
        email = safe_email(first_value(payload, ("email", "primary_email", "Email", "E-mail")))
        match_class = identity_match_class_from_payload(payload)
        display_name = optional_text(first_value(payload, ("name", "display_name", "title", "contact_name", "lead_name")))
        event_at = parse_source_datetime(first_value(payload, ("updated_at", "created_at", "event_at")), record.observed_at)
        customer = CustomerIdentity(
            tenant_id=self.tenant_id,
            identity_status=identity_status_from_match(phone=phone, email=email, match_class=match_class),
            display_name=display_name,
            primary_phone=phone,
            primary_email=email,
            source_ref=source_ref,
            first_seen_at=event_at,
            last_seen_at=event_at,
            touch_count=1,
            summary={"source_system": self.source_system, "entity_type": entity_type},
            metadata={"source_ref": source_ref, "payload_hash": record.payload_hash},
            created_at=event_at,
            updated_at=event_at,
        )
        links = identity_links_for_customer(
            customer,
            source_system=self.source_system,
            source_ref=source_ref,
            phone=phone,
            email=email,
            match_class=match_class,
        )
        link_type = "amo_lead_id" if entity_type in {"lead", "deal", "amo_deal"} else "amo_contact_id"
        links.append(
            IdentityLink(
                tenant_id=self.tenant_id,
                customer_id=customer.customer_id,
                link_type=link_type,
                link_value=entity_id,
                source_system=self.source_system,
                source_ref=source_ref,
                match_class=IdentityMatchClass.STRONG_UNIQUE,
                confidence=1.0,
                first_seen_at=event_at,
                last_seen_at=event_at,
            )
        )
        opportunities: list[CustomerOpportunity] = []
        opportunity_id: Optional[str] = None
        if entity_type in {"lead", "deal", "amo_deal"}:
            opportunity = CustomerOpportunity(
                tenant_id=self.tenant_id,
                customer_id=customer.customer_id,
                opportunity_type=OpportunityType.AMO_DEAL,
                source_system=self.source_system,
                source_id=entity_id,
                title=display_name or f"amoCRM {entity_type} {entity_id}",
                status=optional_text(first_value(payload, ("status", "stage", "pipeline_status"))),
                product_context={"pipeline": first_value(payload, ("pipeline", "pipeline_name"))},
                opened_at=event_at,
                confidence=0.9,
                evidence={"source_ref": source_ref},
            )
            opportunities.append(opportunity)
            opportunity_id = opportunity.opportunity_id
        event = TimelineEvent(
            tenant_id=self.tenant_id,
            customer_id=customer.customer_id,
            opportunity_id=opportunity_id,
            event_type=TimelineEventType.AMO_DEAL_STAGE if opportunity_id else TimelineEventType.AMO_CONTACT_SNAPSHOT,
            event_at=event_at,
            source_system=self.source_system,
            source_id=entity_id,
            source_ref=source_ref,
            direction=TimelineDirection.SYSTEM,
            subject=display_name or f"amoCRM {entity_type}",
            text_preview=compact_text(first_value(payload, ("text_preview", "note", "summary"))),
            summary=compact_text(first_value(payload, ("summary", "status", "stage"))),
            match_status=IdentityMatchClass.STRONG_UNIQUE if phone or email else IdentityMatchClass.INFERRED,
            confidence=0.9 if phone or email else 0.6,
            record={"entity_type": entity_type, "payload": scrub_timeline_persisted_json(dict(payload))},
            created_at=event_at,
        )
        conflicts = conflict_from_payload(self.tenant_id, payload, source_ref)
        return TimelineNormalizedBatch(
            source_record=record,
            customers=(customer,),
            identity_links=tuple(links),
            opportunities=tuple(opportunities),
            events=(event,),
            conflicts=conflicts,
        )


class TallantoSnapshotNormalizer:
    source_system = "tallanto_snapshot"

    def __init__(self, *, tenant_id: str) -> None:
        self.tenant_id = normalize_key(tenant_id, "tenant_id")

    def normalize(self, record: TimelineSourceRecord) -> TimelineNormalizedBatch:
        payload = record.payload
        entity_id = require_text(
            first_value(payload, ("entity_id", "student_id", "tallanto_id", "id", "Contact_ID")),
            "tallanto entity_id",
        )
        source_ref = f"tallanto:student:{entity_id}"
        phone = safe_phone(first_value(payload, ("phone", "primary_phone", "Телефон", "Phone")))
        email = safe_email(first_value(payload, ("email", "primary_email", "Email", "E-mail")))
        match_class = identity_match_class_from_payload(payload)
        display_name = optional_text(first_value(payload, ("name", "display_name", "student_name", "ФИО", "Name")))
        event_at = parse_source_datetime(first_value(payload, ("updated_at", "created_at", "event_at")), record.observed_at)
        customer = CustomerIdentity(
            tenant_id=self.tenant_id,
            identity_status=identity_status_from_match(phone=phone, email=email, match_class=match_class),
            display_name=display_name,
            primary_phone=phone,
            primary_email=email,
            source_ref=source_ref,
            first_seen_at=event_at,
            last_seen_at=event_at,
            touch_count=1,
            summary={"source_system": self.source_system},
            metadata={"source_ref": source_ref, "payload_hash": record.payload_hash},
            created_at=event_at,
            updated_at=event_at,
        )
        links = identity_links_for_customer(
            customer,
            source_system=self.source_system,
            source_ref=source_ref,
            phone=phone,
            email=email,
            match_class=match_class,
        )
        links.append(
            IdentityLink(
                tenant_id=self.tenant_id,
                customer_id=customer.customer_id,
                link_type="tallanto_student_id",
                link_value=entity_id,
                source_system=self.source_system,
                source_ref=source_ref,
                match_class=IdentityMatchClass.STRONG_UNIQUE,
                confidence=1.0,
                first_seen_at=event_at,
                last_seen_at=event_at,
            )
        )
        opportunity = CustomerOpportunity(
            tenant_id=self.tenant_id,
            customer_id=customer.customer_id,
            opportunity_type=OpportunityType.TALLANTO_COURSE,
            source_system=self.source_system,
            source_id=entity_id,
            title=optional_text(first_value(payload, ("course", "group", "product", "title"))) or "Tallanto student",
            status=optional_text(first_value(payload, ("status", "state", "payment_status"))),
            product_context={
                "course": first_value(payload, ("course", "product")),
                "group": first_value(payload, ("group", "group_name")),
            },
            opened_at=event_at,
            confidence=0.8,
            evidence={"source_ref": source_ref},
        )
        event = TimelineEvent(
            tenant_id=self.tenant_id,
            customer_id=customer.customer_id,
            opportunity_id=opportunity.opportunity_id,
            event_type=TimelineEventType.TALLANTO_STUDENT_SNAPSHOT,
            event_at=event_at,
            source_system=self.source_system,
            source_id=entity_id,
            source_ref=source_ref,
            direction=TimelineDirection.SYSTEM,
            subject=display_name or "Tallanto student snapshot",
            text_preview=compact_text(first_value(payload, ("summary", "comment", "note"))),
            summary=compact_text(first_value(payload, ("status", "course", "group"))),
            match_status=IdentityMatchClass.STRONG_UNIQUE if phone or email else IdentityMatchClass.INFERRED,
            confidence=0.85 if phone or email else 0.6,
            record={"payload": scrub_timeline_persisted_json(dict(payload))},
            created_at=event_at,
        )
        return TimelineNormalizedBatch(
            source_record=record,
            customers=(customer,),
            identity_links=tuple(links),
            opportunities=(opportunity,),
            events=(event,),
            conflicts=conflict_from_payload(self.tenant_id, payload, source_ref),
        )


class ChannelMessageNormalizer:
    source_system = "channel_snapshot"

    def __init__(self, *, tenant_id: str) -> None:
        self.tenant_id = normalize_key(tenant_id, "tenant_id")

    def normalize(self, record: TimelineSourceRecord) -> TimelineNormalizedBatch:
        payload = record.payload
        message_payload = payload.get("message") if isinstance(payload.get("message"), Mapping) else payload
        channel = normalize_key(message_payload.get("channel") or payload.get("channel") or "unknown_channel", "channel")
        thread_id = require_text(
            message_payload.get("channel_thread_id") or payload.get("channel_thread_id") or payload.get("session_key"),
            "channel_thread_id",
        )
        message_id = require_text(
            message_payload.get("channel_message_id")
            or message_payload.get("idempotency_key")
            or payload.get("idempotency_key")
            or record.source_ref,
            "channel_message_id",
        )
        user_id = require_text(message_payload.get("channel_user_id") or payload.get("channel_user_id") or thread_id, "channel_user_id")
        event_at = parse_source_datetime(message_payload.get("received_at") or payload.get("created_at"), record.observed_at)
        direction = TimelineDirection(message_payload.get("direction") or "inbound")
        source_ref = f"channel:{channel}:{thread_id}:{message_id}"
        customer = CustomerIdentity(
            tenant_id=self.tenant_id,
            identity_status=IdentityStatus.PARTIAL,
            display_name=optional_text(message_payload.get("display_name") or payload.get("display_name")),
            source_ref=f"channel:{channel}:{thread_id}:{user_id}",
            first_seen_at=event_at,
            last_seen_at=event_at,
            touch_count=1,
            summary={"source_system": self.source_system, "channel": channel},
            metadata={"channel": channel, "thread_id": thread_id, "user_id": user_id},
            created_at=event_at,
            updated_at=event_at,
        )
        link_type = channel_link_type(channel)
        links = [
            IdentityLink(
                tenant_id=self.tenant_id,
                customer_id=customer.customer_id,
                link_type=link_type,
                link_value=user_id if link_type != "channel_session_id" else f"{channel}:{thread_id}",
                source_system=self.source_system,
                source_ref=source_ref,
                match_class=IdentityMatchClass.INFERRED,
                confidence=0.6,
                first_seen_at=event_at,
                last_seen_at=event_at,
            ),
            IdentityLink(
                tenant_id=self.tenant_id,
                customer_id=customer.customer_id,
                link_type="channel_session_id",
                link_value=f"{channel}:{thread_id}",
                source_system=self.source_system,
                source_ref=source_ref,
                match_class=IdentityMatchClass.INFERRED,
                confidence=0.7,
                first_seen_at=event_at,
                last_seen_at=event_at,
            ),
        ]
        event_type = channel_event_type(channel)
        event = TimelineEvent(
            tenant_id=self.tenant_id,
            customer_id=customer.customer_id,
            event_type=event_type,
            event_at=event_at,
            source_system=self.source_system,
            source_id=f"{channel}:{message_id}",
            source_ref=source_ref,
            direction=direction,
            participants=(
                TimelineParticipant(role="client", ref=user_id, channel=channel)
                if direction == TimelineDirection.INBOUND
                else TimelineParticipant(role="manager", ref=user_id, channel=channel),
            ),
            subject=f"{channel} message",
            text_preview=compact_text(message_payload.get("text"), limit=240),
            summary=compact_text(message_payload.get("text"), limit=240),
            match_status=IdentityMatchClass.INFERRED,
            confidence=0.6,
            record={"message": scrub_timeline_persisted_json(dict(message_payload))},
            created_at=event_at,
        )
        chunks: tuple[BotContextChunk, ...] = ()
        text = compact_text(message_payload.get("text"), limit=500)
        if text:
            chunks = (
                BotContextChunk(
                    tenant_id=self.tenant_id,
                    customer_id=customer.customer_id,
                    event_id=event.event_id,
                    source_ref=source_ref,
                    source_system=self.source_system,
                    chunk_type="channel_message",
                    text=text,
                    summary=compact_text(text, limit=160),
                    event_at=event_at,
                    freshness_score=0.7,
                    relevance_tags=(channel, "message"),
                    allowed_for_bot=False,
                    requires_manager_review=True,
                    created_at=event_at,
                ),
            )
        return TimelineNormalizedBatch(
            source_record=record,
            customers=(customer,),
            identity_links=tuple(links),
            events=(event,),
            bot_context_chunks=chunks,
        )


class MailMessageNormalizer:
    source_system = "mail_archive"

    def __init__(self, *, tenant_id: str) -> None:
        self.tenant_id = normalize_key(tenant_id, "tenant_id")

    def normalize(self, record: TimelineSourceRecord) -> TimelineNormalizedBatch:
        payload = record.payload
        message_id = require_text(first_value(payload, ("message_sha256", "sha256", "message_id", "id", "uid")) or record.source_ref, "message_id")
        from_email = safe_email(first_value(payload, ("from_email", "from", "sender", "sender_email")))
        to_email = safe_email(first_value(payload, ("to_email", "to", "recipient", "recipient_email")))
        direction = infer_mail_direction(payload, from_email)
        customer_email = from_email if direction == TimelineDirection.INBOUND else to_email
        event_at = parse_source_datetime(
            first_value(payload, ("date", "date_last", "date_first", "sent_at", "received_at", "event_at")),
            record.observed_at,
        )
        source_ref = f"mail:{message_id}"
        resolved_customer_id = optional_text(
            first_value(
                payload,
                (
                    "resolved_customer_id",
                    "customer_id_resolved",
                    "fresh_relink_customer_id",
                    "_resolved_customer_id",
                ),
            )
        )
        resolved_tallanto_id = optional_text(
            first_value(payload, ("resolved_tallanto_id", "tallanto_id", "fresh_relink_tallanto_id", "_resolved_tallanto_id"))
        )
        resolved_customer_exists = truthy_flag(first_value(payload, ("resolved_customer_exists", "_resolved_customer_exists")))
        if not resolved_customer_id:
            return TimelineNormalizedBatch(
                source_record=record,
                conflicts=(
                    pending_attribution_conflict(
                        self.tenant_id,
                        payload,
                        source_ref,
                        message_id=message_id,
                    ),
                ),
            )
        customer = CustomerIdentity(
            tenant_id=self.tenant_id,
            customer_id=resolved_customer_id,
            identity_status=IdentityStatus.STRONG if resolved_tallanto_id or customer_email else IdentityStatus.PARTIAL,
            display_name=optional_text(first_value(payload, ("name", "display_name", "from_name", "to_name"))),
            primary_email=customer_email,
            source_ref=f"tallanto:student:{resolved_tallanto_id}" if resolved_tallanto_id else source_ref,
            first_seen_at=event_at,
            last_seen_at=event_at,
            touch_count=1,
            summary={"source_system": self.source_system, "direction": direction.value, "identity_authority": "fresh_relink"},
            metadata={
                "message_id": message_id,
                "payload_hash": record.payload_hash,
                "identity_authority": "fresh_relink",
                "resolved_tallanto_id_present": bool(resolved_tallanto_id),
            },
            created_at=event_at,
            updated_at=event_at,
        )
        links = identity_links_for_customer(
            customer,
            source_system=self.source_system,
            source_ref=source_ref,
            email=customer_email,
        )
        if resolved_tallanto_id:
            links.append(
                IdentityLink(
                    tenant_id=self.tenant_id,
                    customer_id=customer.customer_id,
                    link_type="tallanto_student_id",
                    link_value=resolved_tallanto_id,
                    source_system=self.source_system,
                    source_ref=source_ref,
                    match_class=IdentityMatchClass.STRONG_UNIQUE,
                    confidence=1.0,
                    first_seen_at=event_at,
                    last_seen_at=event_at,
                )
            )
        thread_id = optional_text(first_value(payload, ("thread_id", "conversation_id", "subject"))) or message_id
        mail_thread_source_id = f"{thread_id}:{customer.customer_id}"
        opportunity = CustomerOpportunity(
            tenant_id=self.tenant_id,
            customer_id=customer.customer_id,
            opportunity_type=OpportunityType.MAIL_THREAD,
            source_system=self.source_system,
            source_id=mail_thread_source_id,
            title=compact_text(first_value(payload, ("subject", "thread_subject")), limit=120) or "Email thread",
            status="open",
            opened_at=event_at,
            confidence=0.7,
            evidence={"source_ref": source_ref},
        )
        event = TimelineEvent(
            tenant_id=self.tenant_id,
            customer_id=customer.customer_id,
            opportunity_id=opportunity.opportunity_id,
            event_type=TimelineEventType.EMAIL_MESSAGE,
            event_at=event_at,
            source_system=self.source_system,
            source_id=message_id,
            source_ref=source_ref,
            direction=direction,
            participants=mail_participants(from_email, to_email),
            subject=compact_text(first_value(payload, ("subject", "thread_subject")), limit=160),
            text_preview=compact_text(first_value(payload, ("text_preview", "snippet", "body_preview")), limit=240),
            summary=compact_text(first_value(payload, ("summary", "text_preview", "snippet")), limit=240),
            match_status=IdentityMatchClass.STRONG_UNIQUE,
            confidence=0.95 if resolved_tallanto_id else 0.8,
            record={"message": scrub_timeline_persisted_json(dict(payload))},
            created_at=event_at,
        )
        artifacts = mail_artifacts(self.tenant_id, event, payload, event_at)
        return TimelineNormalizedBatch(
            source_record=record,
            customers=() if resolved_customer_exists else (customer,),
            identity_links=tuple(links),
            opportunities=(opportunity,),
            events=(event,),
            artifacts=artifacts,
            conflicts=conflict_from_payload(self.tenant_id, payload, source_ref),
        )


class MangoCallSummaryNormalizer:
    source_system = "mango_processed_summary"

    def __init__(self, *, tenant_id: str) -> None:
        self.tenant_id = normalize_key(tenant_id, "tenant_id")

    def normalize(self, record: TimelineSourceRecord) -> TimelineNormalizedBatch:
        payload = record.payload
        call_id = require_text(first_value(payload, ("provider_call_id", "call_id", "recording_id", "id")), "call_id")
        phone = safe_phone(first_value(payload, ("phone", "client_phone", "normalized_phone", "Телефон клиента")))
        event_at = parse_source_datetime(first_value(payload, ("call_at", "started_at", "event_at", "date")), record.observed_at)
        source_ref = f"mango:{call_id}"
        strict_existing_identity = truthy_flag(
            first_value(payload, ("identity_resolved_by_increment", "existing_timeline_identity"))
        ) or normalize_key(first_value(payload, ("identity_authority", "identity_resolution_authority")) or "legacy", "identity_authority") in {
            "existing_timeline",
            "existing_timeline_increment",
        }
        match_class = identity_match_class_from_payload(payload)
        resolved_customer_id = optional_text(first_value(payload, ("customer_id", "resolved_customer_id")))
        customer_id: Optional[str]
        customers: tuple[CustomerIdentity, ...]
        links: tuple[IdentityLink, ...]
        if strict_existing_identity:
            customer_id = resolved_customer_id if match_class == IdentityMatchClass.STRONG_UNIQUE else None
            customers = ()
            links = ()
        else:
            customer = CustomerIdentity(
                tenant_id=self.tenant_id,
                identity_status=IdentityStatus.STRONG if phone else IdentityStatus.PARTIAL,
                display_name=optional_text(first_value(payload, ("client_name", "name"))),
                primary_phone=phone,
                source_ref=source_ref,
                first_seen_at=event_at,
                last_seen_at=event_at,
                touch_count=1,
                summary={"source_system": self.source_system},
                metadata={"call_id": call_id, "payload_hash": record.payload_hash},
                created_at=event_at,
                updated_at=event_at,
            )
            customer_id = customer.customer_id
            legacy_links = identity_links_for_customer(
                customer,
                source_system=self.source_system,
                source_ref=source_ref,
                phone=phone,
            )
            if phone:
                legacy_links.append(
                    IdentityLink(
                        tenant_id=self.tenant_id,
                        customer_id=customer.customer_id,
                        link_type="mango_client_phone",
                        link_value=phone,
                        source_system=self.source_system,
                        source_ref=source_ref,
                        match_class=IdentityMatchClass.STRONG_UNIQUE,
                        confidence=0.95,
                        first_seen_at=event_at,
                        last_seen_at=event_at,
                    )
                )
            customers = (customer,)
            links = tuple(legacy_links)
            match_class = IdentityMatchClass.STRONG_UNIQUE if phone else IdentityMatchClass.INFERRED
        call_type = normalize_key(first_value(payload, ("call_type", "call_quality_type")) or "unknown", "call_type")
        is_non_conversation = call_type == "non_conversation"
        summary = None if is_non_conversation else compact_text(first_value(payload, ("summary", "insight_summary", "analysis_summary")), limit=500)
        event = TimelineEvent(
            tenant_id=self.tenant_id,
            customer_id=customer_id,
            event_type=TimelineEventType.MANGO_CALL,
            event_at=event_at,
            source_system=self.source_system,
            source_id=call_id,
            source_ref=source_ref,
            direction=TimelineDirection(first_value(payload, ("direction",)) or "inbound"),
            actor_name=optional_text(first_value(payload, ("manager_name", "sales_manager"))),
            actor_ref=optional_text(first_value(payload, ("manager_id", "employee_id"))),
            subject=compact_text(first_value(payload, ("subject", "product", "topic")), limit=160),
            text_preview=compact_text(first_value(payload, ("text_preview", "transcript_excerpt")), limit=240),
            summary=summary,
            importance=int_or_default(first_value(payload, ("importance", "priority_score")), 0),
            match_status=match_class,
            confidence=float_or_default(first_value(payload, ("confidence",)), 0.75 if customer_id else 0.55),
            record={"call": scrub_timeline_persisted_json(dict(payload))},
            metadata={
                "call_type": call_type,
                "identity_authority": optional_text(first_value(payload, ("identity_authority", "identity_resolution_authority"))),
            },
            created_at=event_at,
        )
        artifacts = mango_artifacts(self.tenant_id, event, payload, event_at)
        signals = () if strict_existing_identity and (not customer_id or is_non_conversation) else mango_signals(self.tenant_id, event, payload, event_at)
        chunks: tuple[BotContextChunk, ...] = ()
        if event.summary and event.customer_id and not is_non_conversation:
            chunks = (
                BotContextChunk(
                    tenant_id=self.tenant_id,
                    customer_id=event.customer_id,
                    event_id=event.event_id,
                    source_ref=source_ref,
                    source_system=self.source_system,
                    chunk_type="mango_call_summary",
                    text=event.summary,
                    summary=compact_text(event.summary, limit=160),
                    event_at=event_at,
                    freshness_score=0.8,
                    relevance_tags=("mango", "call"),
                    allowed_for_bot=False,
                    requires_manager_review=True,
                    created_at=event_at,
                ),
            )
        return TimelineNormalizedBatch(
            source_record=record,
            customers=customers,
            identity_links=links,
            events=(event,),
            artifacts=artifacts,
            signals=signals,
            bot_context_chunks=chunks,
            conflicts=(
                (
                    mango_pending_attribution_conflict(
                        self.tenant_id,
                        payload,
                        source_ref,
                        phone=phone,
                        match_class=match_class,
                    ),
                )
                if strict_existing_identity and not customer_id
                else ()
            )
            + conflict_from_payload(self.tenant_id, payload, source_ref),
        )


def load_local_source_records(
    path: Path | str,
    *,
    allowed_root: Path | str,
    source_system: str,
    source_ref_prefix: Optional[str] = None,
    csv_encoding: str = "utf-8-sig",
    csv_delimiter: Optional[str] = None,
    observed_at: Optional[datetime] = None,
) -> tuple[TimelineSourceRecord, ...]:
    source_path = guard_customer_timeline_source_path(path, allowed_root)
    suffix = source_path.suffix.casefold()
    if suffix == ".json":
        rows = rows_from_json(source_path)
    elif suffix == ".jsonl":
        rows = rows_from_jsonl(source_path)
    elif suffix == ".csv":
        rows = rows_from_csv(source_path, encoding=csv_encoding, delimiter=csv_delimiter)
    else:
        raise ValueError("timeline source path must be .json, .jsonl, or .csv")
    prefix = source_ref_prefix or source_path.name
    normalized_source = normalize_key(source_system, "source_system")
    return tuple(
        TimelineSourceRecord(
            source_system=normalized_source,
            source_ref=f"{prefix}#{idx}",
            payload=row,
            source_path=str(source_path),
            observed_at=observed_at,
        )
        for idx, row in enumerate(rows, start=1)
    )


def load_sqlite_source_records(
    path: Path | str,
    *,
    allowed_root: Path | str,
    source_system: str,
    table_name: str,
    source_ref_column: Optional[str] = None,
    where_sql: Optional[str] = None,
    params: Sequence[Any] = (),
    limit: Optional[int] = None,
    observed_at: Optional[datetime] = None,
) -> tuple[TimelineSourceRecord, ...]:
    source_path = guard_customer_timeline_source_path(path, allowed_root)
    if source_path.suffix.casefold() not in {".sqlite", ".db", ".sqlite3"}:
        raise ValueError("SQLite source path must end with .sqlite, .db, or .sqlite3")
    table = require_sql_identifier(table_name, "table_name")
    ref_column = require_sql_identifier(source_ref_column, "source_ref_column") if source_ref_column else None
    query = f"SELECT * FROM {table}"
    query_params = tuple(params)
    if where_sql:
        if contains_sql_write_keyword(where_sql):
            raise ValueError("where_sql must be read-only")
        query += f" WHERE {where_sql}"
    query += " ORDER BY rowid"
    if limit is not None:
        if int(limit) <= 0:
            raise ValueError("limit must be positive")
        query += " LIMIT ?"
        query_params = (*query_params, int(limit))
    uri = f"file:{source_path}?mode=ro"
    with sqlite3.connect(uri, uri=True, timeout=15) as con:
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA query_only = ON")
        rows = con.execute(query, query_params).fetchall()
    normalized_source = normalize_key(source_system, "source_system")
    records: list[TimelineSourceRecord] = []
    for idx, row in enumerate(rows, start=1):
        payload = dict(row)
        source_ref = str(payload.get(ref_column) or f"{source_path.name}:{table}#{idx}") if ref_column else f"{source_path.name}:{table}#{idx}"
        records.append(
            TimelineSourceRecord(
                source_system=normalized_source,
                source_ref=source_ref,
                payload=payload,
                source_path=str(source_path),
                observed_at=observed_at,
            )
        )
    return tuple(records)


def guard_customer_timeline_source_path(path: Path | str, allowed_root: Path | str) -> Path:
    resolved = guard_customer_timeline_output_path(path, allowed_root)
    if is_stable_runtime_path(resolved):
        raise ValueError(f"customer timeline source must not be under stable_runtime: {resolved}")
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    if not resolved.is_file():
        raise ValueError(f"customer timeline source must be a file: {resolved}")
    return resolved


def build_source_inventory(records: Sequence[TimelineSourceRecord]) -> tuple[Mapping[str, Any], ...]:
    inventory: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        if not record.source_path:
            continue
        path = Path(record.source_path).resolve(strict=False)
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if not path.exists() or not path.is_file():
            inventory.append({"path": key, "exists": False})
            continue
        stat = path.stat()
        inventory.append(
            {
                "path": key,
                "exists": True,
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": file_sha256(path),
            }
        )
    return tuple(inventory)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows_from_json(path: Path) -> tuple[Mapping[str, Any], ...]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return tuple(require_mapping(item, "json row") for item in payload)
    if isinstance(payload, Mapping):
        for key in ("entities", "items", "messages", "events", "rows"):
            value = payload.get(key)
            if isinstance(value, list):
                return tuple(require_mapping(item, f"{key} row") for item in value)
        return (dict(payload),)
    raise ValueError("JSON source must contain an object or array")


def rows_from_jsonl(path: Path) -> tuple[Mapping[str, Any], ...]:
    rows: list[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            rows.append(require_mapping(json.loads(text), f"jsonl line {lineno}"))
    return tuple(rows)


def rows_from_csv(
    path: Path,
    *,
    encoding: str = "utf-8-sig",
    delimiter: Optional[str] = None,
) -> tuple[Mapping[str, Any], ...]:
    try:
        text = path.read_text(encoding=encoding)
    except UnicodeDecodeError:
        text = path.read_text(encoding="cp1251")
    sample = text[:4096]
    resolved_delimiter = delimiter
    if resolved_delimiter is None:
        try:
            resolved_delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t").delimiter
        except csv.Error:
            resolved_delimiter = ","
    reader = csv.DictReader(text.splitlines(), delimiter=resolved_delimiter)
    return tuple(dict(row) for row in reader)


def timeline_ingestion_safety_contract() -> Mapping[str, Any]:
    return {
        **customer_timeline_safety_contract(),
        **customer_timeline_sqlite_safety_contract(),
        "schema_version": CUSTOMER_TIMELINE_INGESTION_SCHEMA_VERSION,
        "read_local_files_only": True,
        "source_reads_are_read_only": True,
        "network_calls": False,
        "subprocess_calls": False,
        "write_crm": False,
        "write_tallanto": False,
        "send_email": False,
        "send_messenger": False,
        "run_asr": False,
        "run_ra": False,
        "source_sqlite_mode": "mode=ro",
        "source_sqlite_query_only": True,
        "source_db_attached_to_writer": False,
        "raw_payload_scrubbed": True,
        "idempotent_upsert": True,
        "identity_conflicts_auto_merge": False,
        "blocked_import_hints": FORBIDDEN_IMPORT_HINTS,
    }


def identity_links_for_customer(
    customer: CustomerIdentity,
    *,
    source_system: str,
    source_ref: str,
    phone: Optional[str] = None,
    email: Optional[str] = None,
    match_class: IdentityMatchClass = IdentityMatchClass.STRONG_UNIQUE,
) -> list[IdentityLink]:
    links: list[IdentityLink] = []
    confidence = 0.9 if match_class == IdentityMatchClass.STRONG_UNIQUE else 0.6
    if phone:
        links.append(
            IdentityLink(
                tenant_id=customer.tenant_id,
                customer_id=customer.customer_id,
                link_type="phone",
                link_value=phone,
                source_system=source_system,
                source_ref=source_ref,
                match_class=match_class,
                confidence=confidence,
                first_seen_at=customer.first_seen_at,
                last_seen_at=customer.last_seen_at,
            )
        )
    if email:
        links.append(
            IdentityLink(
                tenant_id=customer.tenant_id,
                customer_id=customer.customer_id,
                link_type="email",
                link_value=email,
                source_system=source_system,
                source_ref=source_ref,
                match_class=match_class,
                confidence=confidence,
                first_seen_at=customer.first_seen_at,
                last_seen_at=customer.last_seen_at,
            )
        )
    return links


def conflict_from_payload(tenant_id: str, payload: Mapping[str, Any], source_ref: str) -> tuple[Mapping[str, Any], ...]:
    marker = optional_text(first_value(payload, ("match_class", "identity_status", "resolution_status", "conflict_type")))
    duplicates = payload.get("duplicates") or payload.get("duplicate_ids") or payload.get("candidate_ids")
    conflict_type: Optional[str] = None
    if marker and marker.casefold() in {"ambiguous", "duplicate", "duplicate_merge_required", "duplicate_merge_required"}:
        conflict_type = marker
    if isinstance(duplicates, Sequence) and not isinstance(duplicates, (str, bytes)) and len(duplicates) > 1:
        conflict_type = conflict_type or "duplicate_candidates"
    if not conflict_type:
        return ()
    identifier = safe_phone(first_value(payload, ("phone", "primary_phone", "Телефон", "Телефон клиента"))) or safe_email(
        first_value(payload, ("email", "primary_email", "Email", "E-mail"))
    )
    refs = [f"identity:{identifier}"] if identifier and isinstance(duplicates, Sequence) and not isinstance(duplicates, (str, bytes)) else [source_ref]
    if isinstance(duplicates, Sequence) and not isinstance(duplicates, (str, bytes)):
        refs.extend(str(item) for item in duplicates)
    return (
        {
            "tenant_id": tenant_id,
            "conflict_type": normalize_key(conflict_type, "conflict_type"),
            "entity_refs": tuple(refs),
            "severity": "medium",
            "status": "open",
            "summary": f"Identity conflict from {source_ref}",
            "metadata": {"source_ref": source_ref, "identifier": identifier},
        },
    )


def pending_attribution_conflict(
    tenant_id: str,
    payload: Mapping[str, Any],
    source_ref: str,
    *,
    message_id: str,
) -> Mapping[str, Any]:
    message_sha256 = optional_text(first_value(payload, ("message_sha256", "sha256"))) or message_id
    return {
        "tenant_id": tenant_id,
        "conflict_type": "pending_attribution",
        "entity_refs": (source_ref, f"message_sha256:{message_sha256}"),
        "severity": "low",
        "status": "open",
        "summary": "Email message has no authoritative fresh relink customer attribution.",
        "metadata": {
            "source_ref": source_ref,
            "message_sha256": message_sha256,
            "relink_decision": optional_text(first_value(payload, ("relink_decision", "decision"))),
            "relink_reason": optional_text(first_value(payload, ("relink_reason", "reason"))),
            "identity_authority": "fresh_relink_required",
        },
    }


def resolve_customer_identity_batches(
    batches: Sequence[TimelineNormalizedBatch],
    *,
    store: Optional[CustomerTimelineSQLiteStore] = None,
) -> CustomerIdResolutionResult:
    normalized_batches = tuple(batches)
    customers_by_id: dict[str, CustomerIdentity] = {}
    links_by_customer: dict[str, list[IdentityLink]] = {}
    source_refs_by_customer: dict[str, set[str]] = {}
    phone_to_customers: dict[tuple[str, str], set[str]] = {}
    phone_to_tallanto_students: dict[tuple[str, str], set[str]] = {}
    existing_customer_ids: set[str] = set()

    for batch in normalized_batches:
        for customer in batch.customers:
            customers_by_id[customer.customer_id] = customer
            refs = source_refs_by_customer.setdefault(customer.customer_id, set())
            if customer.source_ref:
                refs.add(customer.source_ref)
        for link in batch.identity_links:
            if not link.customer_id:
                continue
            links_by_customer.setdefault(link.customer_id, []).append(link)
            source_refs_by_customer.setdefault(link.customer_id, set()).add(link.source_ref)
            if link.link_type.value in {"phone", "mango_client_phone"}:
                phone_to_customers.setdefault((link.tenant_id, link.link_value), set()).add(link.customer_id)
    if store is not None:
        existing_customer_ids = _load_existing_identity_context(
            store=store,
            customers_by_id=customers_by_id,
            links_by_customer=links_by_customer,
            source_refs_by_customer=source_refs_by_customer,
            phone_to_customers=phone_to_customers,
        )

    tallanto_by_customer: dict[str, set[str]] = {}
    for customer_id, links in links_by_customer.items():
        for link in links:
            if link.link_type.value == "tallanto_student_id":
                tallanto_by_customer.setdefault(customer_id, set()).add(link.link_value)
    for phone_key, customer_ids in phone_to_customers.items():
        students: set[str] = set()
        for customer_id in customer_ids:
            students.update(tallanto_by_customer.get(customer_id, set()))
        if students:
            phone_to_tallanto_students[phone_key] = students

    family_phone_keys = {
        phone_key
        for phone_key, student_ids in phone_to_tallanto_students.items()
        if len(student_ids) > 1
    }

    parent = {customer_id: customer_id for customer_id in customers_by_id}

    def find(customer_id: str) -> str:
        while parent[customer_id] != customer_id:
            parent[customer_id] = parent[parent[customer_id]]
            customer_id = parent[customer_id]
        return customer_id

    def union(left: str, right: str) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left == root_right:
            return
        keep, move = sorted((root_left, root_right))
        parent[move] = keep

    for phone_key, customer_ids in sorted(phone_to_customers.items()):
        if phone_key in family_phone_keys or len(customer_ids) < 2:
            continue
        ordered_ids = sorted(customer_ids)
        for customer_id in ordered_ids[1:]:
            union(ordered_ids[0], customer_id)

    groups: dict[str, list[str]] = {}
    for customer_id in sorted(customers_by_id):
        groups.setdefault(find(customer_id), []).append(customer_id)

    new_id_by_old: dict[str, str] = {}
    reason_by_old: dict[str, str] = {}
    merged_customers: dict[str, CustomerIdentity] = {}
    for members in groups.values():
        shared_phone = _single_shared_phone(members, phone_to_customers, family_phone_keys)
        tenant_id = customers_by_id[members[0]].tenant_id
        if len(members) > 1 and shared_phone:
            existing_members = sorted(customer_id for customer_id in members if customer_id in existing_customer_ids)
            new_customer_id = existing_members[0] if existing_members else stable_customer_id(tenant_id=tenant_id, primary_phone=shared_phone)
            reason = "phone_identity_union"
        elif any(_customer_has_family_phone(member, phone_to_customers, family_phone_keys) for member in members):
            new_customer_id = members[0]
            reason = "family_phone_ambiguous"
        else:
            new_customer_id = members[0]
            reason = "unchanged"
        for old_customer_id in members:
            new_id_by_old[old_customer_id] = new_customer_id
            reason_by_old[old_customer_id] = reason
        merged_customers[new_customer_id] = _merge_customers(
            tuple(customers_by_id[customer_id] for customer_id in members),
            new_customer_id=new_customer_id,
            reason=reason,
            source_refs=tuple(sorted(set().union(*(source_refs_by_customer.get(customer_id, set()) for customer_id in members)))),
            links=tuple(link for customer_id in members for link in links_by_customer.get(customer_id, ())),
        )

    resolved_batches: list[TimelineNormalizedBatch] = []
    mappings: list[CustomerIdResolutionMapping] = []
    for batch in normalized_batches:
        opportunity_id_by_old: dict[str, str] = {}
        rewritten_customers: list[CustomerIdentity] = []
        seen_customer_ids: set[str] = set()
        for customer in batch.customers:
            new_customer_id = new_id_by_old[customer.customer_id]
            if new_customer_id not in seen_customer_ids:
                rewritten_customers.append(merged_customers[new_customer_id])
                seen_customer_ids.add(new_customer_id)
            mappings.append(
                CustomerIdResolutionMapping(
                    tenant_id=customer.tenant_id,
                    old_customer_id=customer.customer_id,
                    new_customer_id=new_customer_id,
                    reason=reason_by_old[customer.customer_id],
                    source_refs=tuple(sorted(source_refs_by_customer.get(customer.customer_id, ()))),
                    metadata={
                        "brand_history": _customer_brand_values(customer),
                        "changed": customer.customer_id != new_customer_id,
                    },
                )
            )

        rewritten_links = tuple(
            replace(link, customer_id=new_id_by_old.get(link.customer_id, link.customer_id))
            for link in batch.identity_links
        )
        rewritten_opportunities: list[CustomerOpportunity] = []
        for opportunity in batch.opportunities:
            new_customer_id = new_id_by_old.get(opportunity.customer_id, opportunity.customer_id)
            changed = new_customer_id != opportunity.customer_id
            rewritten = replace(
                opportunity,
                customer_id=new_customer_id,
                opportunity_id=None if changed else opportunity.opportunity_id,
            )
            opportunity_id_by_old[opportunity.opportunity_id] = rewritten.opportunity_id
            rewritten_opportunities.append(rewritten)
        rewritten_events = tuple(
            replace(
                event,
                customer_id=new_id_by_old.get(event.customer_id, event.customer_id) if event.customer_id else None,
                opportunity_id=opportunity_id_by_old.get(event.opportunity_id, event.opportunity_id)
                if event.opportunity_id
                else None,
            )
            for event in batch.events
        )
        rewritten_signals = tuple(
            replace(
                signal,
                customer_id=new_id_by_old.get(signal.customer_id, signal.customer_id) if signal.customer_id else None,
                opportunity_id=opportunity_id_by_old.get(signal.opportunity_id, signal.opportunity_id)
                if signal.opportunity_id
                else None,
                signal_id=None
                if (
                    (signal.customer_id and new_id_by_old.get(signal.customer_id, signal.customer_id) != signal.customer_id)
                    or (signal.opportunity_id and opportunity_id_by_old.get(signal.opportunity_id, signal.opportunity_id) != signal.opportunity_id)
                )
                else signal.signal_id,
            )
            for signal in batch.signals
        )
        rewritten_chunks = tuple(
            replace(
                chunk,
                customer_id=new_id_by_old.get(chunk.customer_id, chunk.customer_id),
                opportunity_id=opportunity_id_by_old.get(chunk.opportunity_id, chunk.opportunity_id)
                if chunk.opportunity_id
                else None,
                chunk_id=None
                if (
                    new_id_by_old.get(chunk.customer_id, chunk.customer_id) != chunk.customer_id
                    or (chunk.opportunity_id and opportunity_id_by_old.get(chunk.opportunity_id, chunk.opportunity_id) != chunk.opportunity_id)
                )
                else chunk.chunk_id,
            )
            for chunk in batch.bot_context_chunks
        )
        resolved_batches.append(
            TimelineNormalizedBatch(
                source_record=batch.source_record,
                customers=tuple(rewritten_customers),
                identity_links=rewritten_links,
                opportunities=tuple(rewritten_opportunities),
                events=rewritten_events,
                artifacts=batch.artifacts,
                signals=rewritten_signals,
                bot_context_chunks=rewritten_chunks,
                conflicts=batch.conflicts,
            )
        )
    return CustomerIdResolutionResult(batches=tuple(resolved_batches), mappings=tuple(mappings))


def _load_existing_identity_context(
    *,
    store: CustomerTimelineSQLiteStore,
    customers_by_id: dict[str, CustomerIdentity],
    links_by_customer: dict[str, list[IdentityLink]],
    source_refs_by_customer: dict[str, set[str]],
    phone_to_customers: dict[tuple[str, str], set[str]],
) -> set[str]:
    existing_customer_ids: set[str] = set()
    identity_queries: set[tuple[str, str, str]] = set()
    for links in links_by_customer.values():
        for link in links:
            if link.link_type.value == "mango_client_phone":
                identity_queries.add((link.tenant_id, "phone", link.link_value))
                identity_queries.add((link.tenant_id, "mango_client_phone", link.link_value))
            elif link.link_type.value in {"phone", "email"}:
                identity_queries.add((link.tenant_id, link.link_type.value, link.link_value))
    for tenant_id, link_type, link_value in sorted(identity_queries):
        for payload in store.list_identity_links(tenant_id, link_type=link_type, link_value=link_value, limit=500):
            customer_id = optional_text(payload.get("customer_id"))
            if not customer_id:
                continue
            existing_customer_ids.add(customer_id)
            if customer_id not in customers_by_id:
                customer_payload = store.get_customer(tenant_id, customer_id)
                if customer_payload:
                    customers_by_id[customer_id] = customer_identity_from_json(customer_payload)
            link = identity_link_from_json(payload)
            links_by_customer.setdefault(customer_id, []).append(link)
            source_refs_by_customer.setdefault(customer_id, set()).add(link.source_ref)
            if link.link_type.value in {"phone", "mango_client_phone"}:
                phone_to_customers.setdefault((link.tenant_id, link.link_value), set()).add(customer_id)
            for tallanto_payload in store.list_identity_links(tenant_id, customer_id=customer_id, link_type="tallanto_student_id", limit=500):
                tallanto_link = identity_link_from_json(tallanto_payload)
                links_by_customer.setdefault(customer_id, []).append(tallanto_link)
                source_refs_by_customer.setdefault(customer_id, set()).add(tallanto_link.source_ref)
    return existing_customer_ids


def customer_identity_from_json(payload: Mapping[str, Any]) -> CustomerIdentity:
    return CustomerIdentity(
        tenant_id=payload["tenant_id"],
        identity_status=payload["identity_status"],
        customer_id=payload["customer_id"],
        display_name=payload.get("display_name"),
        primary_phone=payload.get("primary_phone"),
        primary_email=payload.get("primary_email"),
        source_ref=payload.get("source_ref"),
        first_seen_at=parse_source_datetime(payload.get("first_seen_at")) if payload.get("first_seen_at") else None,
        last_seen_at=parse_source_datetime(payload.get("last_seen_at")) if payload.get("last_seen_at") else None,
        touch_count=int(payload.get("touch_count") or 0),
        summary=payload.get("summary") or {},
        metadata=payload.get("metadata") or {},
        created_at=parse_source_datetime(payload.get("created_at")),
        updated_at=parse_source_datetime(payload.get("updated_at")),
    )


def identity_link_from_json(payload: Mapping[str, Any]) -> IdentityLink:
    return IdentityLink(
        tenant_id=payload["tenant_id"],
        link_id=payload.get("link_id"),
        customer_id=payload.get("customer_id"),
        link_type=payload["link_type"],
        link_value=payload["link_value"],
        source_system=payload["source_system"],
        source_ref=payload["source_ref"],
        match_class=payload.get("match_class") or IdentityMatchClass.STRONG_UNIQUE,
        confidence=payload.get("confidence"),
        evidence=payload.get("evidence") or {},
        first_seen_at=parse_source_datetime(payload.get("first_seen_at")) if payload.get("first_seen_at") else None,
        last_seen_at=parse_source_datetime(payload.get("last_seen_at")) if payload.get("last_seen_at") else None,
    )


def _single_shared_phone(
    customer_ids: Sequence[str],
    phone_to_customers: Mapping[tuple[str, str], set[str]],
    family_phone_keys: set[tuple[str, str]],
) -> Optional[str]:
    member_set = set(customer_ids)
    phones = [
        phone
        for phone_key, owners in phone_to_customers.items()
        for _tenant_id, phone in (phone_key,)
        if phone_key not in family_phone_keys and len(member_set & owners) >= 2
    ]
    unique_phones = sorted(set(phones))
    return unique_phones[0] if len(unique_phones) == 1 else None


def _customer_has_family_phone(
    customer_id: str,
    phone_to_customers: Mapping[tuple[str, str], set[str]],
    family_phone_keys: set[tuple[str, str]],
) -> bool:
    return any(customer_id in phone_to_customers[phone_key] for phone_key in family_phone_keys)


def _merge_customers(
    customers: Sequence[CustomerIdentity],
    *,
    new_customer_id: str,
    reason: str,
    source_refs: Sequence[str],
    links: Sequence[IdentityLink],
) -> CustomerIdentity:
    ordered = tuple(sorted(customers, key=lambda item: (item.first_seen_at or item.created_at, item.customer_id)))
    first = ordered[0]
    phones = sorted({link.link_value for link in links if link.link_type.value in {"phone", "mango_client_phone"}})
    emails = sorted({link.link_value for link in links if link.link_type.value == "email"})
    first_seen = min((item.first_seen_at for item in ordered if item.first_seen_at), default=None)
    last_seen = max((item.last_seen_at for item in ordered if item.last_seen_at), default=None)
    brands = sorted({brand for item in ordered for brand in _customer_brand_values(item)})
    source_systems = sorted({str(item.summary.get("source_system")) for item in ordered if item.summary.get("source_system")})
    source_customer_ids = tuple(item.customer_id for item in ordered)
    summary = dict(first.summary)
    summary.update(
        {
            "identity_resolution": reason,
            "source_customer_ids": list(source_customer_ids),
            "source_systems": source_systems,
        }
    )
    if brands:
        summary["brands"] = brands
    metadata = dict(first.metadata)
    metadata.update(
        {
            "identity_resolution": reason,
            "source_customer_ids": list(source_customer_ids),
            "source_refs": list(source_refs),
        }
    )
    if brands:
        metadata["brands"] = brands
    return CustomerIdentity(
        tenant_id=first.tenant_id,
        identity_status=_merged_identity_status(ordered, primary_phone=phones[0] if phones else None, primary_email=emails[0] if emails else None),
        customer_id=new_customer_id,
        display_name=next((item.display_name for item in ordered if item.display_name), None),
        primary_phone=phones[0] if len(phones) == 1 else first.primary_phone,
        primary_email=emails[0] if len(emails) == 1 else first.primary_email,
        source_ref=f"identity_resolution:{new_customer_id}",
        first_seen_at=first_seen,
        last_seen_at=last_seen,
        touch_count=sum(item.touch_count for item in ordered),
        summary=summary,
        metadata=metadata,
        created_at=min(item.created_at for item in ordered),
        updated_at=max(item.updated_at for item in ordered),
    )


def _merged_identity_status(
    customers: Sequence[CustomerIdentity],
    *,
    primary_phone: Optional[str],
    primary_email: Optional[str],
) -> IdentityStatus:
    statuses = {item.identity_status for item in customers}
    if IdentityStatus.AMBIGUOUS in statuses:
        return IdentityStatus.AMBIGUOUS
    if IdentityStatus.UNMATCHED in statuses and len(statuses) == 1:
        return IdentityStatus.UNMATCHED
    return IdentityStatus.STRONG if primary_phone or primary_email else IdentityStatus.PARTIAL


def _customer_brand_values(customer: CustomerIdentity) -> tuple[str, ...]:
    values: set[str] = set()
    for source in (customer.summary, customer.metadata):
        raw = source.get("brands") or source.get("brand")
        if isinstance(raw, str):
            values.update(brand_values_from_text(raw))
        elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            for item in raw:
                values.update(brand_values_from_text(item))
    return tuple(sorted(values))


def brand_values_from_text(value: Any) -> tuple[str, ...]:
    text = optional_text(value)
    if not text:
        return ()
    values: set[str] = set()
    for raw_part in re.split(r"[,;|/]+", text):
        part = raw_part.strip().casefold()
        if not part or part == "unknown":
            continue
        if part in {"foton", "фотон"} or "фотон" in part:
            values.add("foton")
        elif part in {"unpk", "унпк"} or "унпк" in part or "мфти" in part:
            values.add("unpk")
        else:
            try:
                values.add(normalize_key(part, "brand"))
            except ValueError:
                continue
    return tuple(sorted(values))


def infer_identity_conflicts(batches: Sequence[TimelineNormalizedBatch]) -> tuple[Mapping[str, Any], ...]:
    by_identity_value: dict[tuple[str, str, str], set[str]] = {}
    refs_by_identity_value: dict[tuple[str, str, str], set[str]] = {}
    for batch in batches:
        for link in batch.identity_links:
            if link.link_type.value not in {"phone", "email", "mango_client_phone"}:
                continue
            if not link.customer_id:
                continue
            key = (link.tenant_id, "phone" if link.link_type.value == "mango_client_phone" else link.link_type.value, link.link_value)
            by_identity_value.setdefault(key, set()).add(link.customer_id)
            refs_by_identity_value.setdefault(key, set()).add(link.source_ref)
    grouped: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}
    for (tenant_id, link_type, link_value), customer_ids in sorted(by_identity_value.items()):
        if len(customer_ids) < 2:
            continue
        group_key = (tenant_id, tuple(sorted(customer_ids)))
        item = grouped.setdefault(group_key, {"identifiers": [], "source_refs": set()})
        item["identifiers"].append(f"{link_type}:{link_value}")
        item["source_refs"].update(refs_by_identity_value.get((tenant_id, link_type, link_value), set()))
    conflicts: list[Mapping[str, Any]] = []
    for (tenant_id, customer_ids), item in sorted(grouped.items()):
        identifiers = tuple(sorted(item["identifiers"]))
        refs = list(identifiers)
        refs.extend(customer_ids)
        refs.extend(sorted(item["source_refs"]))
        conflicts.append(
            {
                "tenant_id": tenant_id,
                "conflict_type": "ambiguous_identity",
                "entity_refs": tuple(refs),
                "severity": "medium",
                "status": "open",
                "summary": f"Multiple customers share identifiers: {', '.join(identifiers)}",
                "metadata": {"identifiers": identifiers},
            }
        )
    return tuple(conflicts)


def identity_match_class_from_payload(payload: Mapping[str, Any]) -> IdentityMatchClass:
    marker = optional_text(first_value(payload, ("match_class", "identity_status", "resolution_status", "conflict_type")))
    if not marker:
        return IdentityMatchClass.STRONG_UNIQUE
    normalized = marker.strip().casefold()
    if "ambiguous" in normalized:
        return IdentityMatchClass.AMBIGUOUS
    if "duplicate" in normalized:
        return IdentityMatchClass.DUPLICATE
    if "manual" in normalized:
        return IdentityMatchClass.MANUAL
    if "inferred" in normalized:
        return IdentityMatchClass.INFERRED
    if "unmatched" in normalized:
        return IdentityMatchClass.UNMATCHED
    return IdentityMatchClass.STRONG_UNIQUE


def identity_status_from_match(
    *,
    phone: Optional[str],
    email: Optional[str],
    match_class: IdentityMatchClass,
) -> IdentityStatus:
    if match_class in {IdentityMatchClass.AMBIGUOUS, IdentityMatchClass.DUPLICATE}:
        return IdentityStatus.AMBIGUOUS
    if match_class == IdentityMatchClass.UNMATCHED:
        return IdentityStatus.UNMATCHED
    return IdentityStatus.STRONG if phone or email else IdentityStatus.PARTIAL


def mango_pending_attribution_conflict(
    tenant_id: str,
    payload: Mapping[str, Any],
    source_ref: str,
    *,
    phone: Optional[str],
    match_class: IdentityMatchClass,
) -> Mapping[str, Any]:
    reason = optional_text(first_value(payload, ("identity_resolution_reason", "relink_reason"))) or match_class.value
    refs = [source_ref, f"match_class:{match_class.value}", f"reason:{reason}"]
    if phone:
        refs.append(f"phone_hash:{stable_digest({'phone': phone})[:16]}")
    return {
        "tenant_id": tenant_id,
        "conflict_type": "pending_attribution",
        "entity_refs": tuple(refs),
        "severity": "medium" if match_class == IdentityMatchClass.AMBIGUOUS else "low",
        "status": "open",
        "summary": "Mango call has no single authoritative existing customer attribution.",
        "metadata": {
            "source_system": "mango_processed_summary",
            "identity_authority": "existing_timeline_increment",
            "match_class": match_class.value,
            "reason": reason,
            "phone_hash": stable_digest({"phone": phone})[:16] if phone else None,
        },
    }


def mail_artifacts(
    tenant_id: str,
    event: TimelineEvent,
    payload: Mapping[str, Any],
    event_at: datetime,
) -> tuple[EventArtifact, ...]:
    artifacts: list[EventArtifact] = []
    eml_path = optional_text(first_value(payload, ("eml_path", "raw_eml_path", "path")))
    if eml_path:
        artifacts.append(
            EventArtifact(
                tenant_id=tenant_id,
                event_id=event.event_id,
                artifact_type=ArtifactType.RAW_EMAIL_EML,
                path=eml_path,
                sha256=optional_text(first_value(payload, ("sha256", "eml_sha256"))),
                size_bytes=int_or_none(first_value(payload, ("size_bytes", "eml_size_bytes"))),
                mime_type="message/rfc822",
                source_system="mail_archive",
                source_ref=event.source_ref,
                extraction_status=ExtractionStatus.NOT_NEEDED,
                created_at=event_at,
            )
        )
    return tuple(artifacts)


def mango_artifacts(
    tenant_id: str,
    event: TimelineEvent,
    payload: Mapping[str, Any],
    event_at: datetime,
) -> tuple[EventArtifact, ...]:
    artifacts: list[EventArtifact] = []
    for field_name, artifact_type in (
        ("audio_path", ArtifactType.CALL_AUDIO),
        ("transcript_path", ArtifactType.CALL_TRANSCRIPT_JSON),
        ("analysis_path", ArtifactType.ANALYSIS_JSON),
    ):
        path = optional_text(payload.get(field_name))
        if not path:
            continue
        artifacts.append(
            EventArtifact(
                tenant_id=tenant_id,
                event_id=event.event_id,
                artifact_type=artifact_type,
                path=path,
                sha256=optional_text(payload.get(f"{field_name}_sha256")),
                size_bytes=int_or_none(payload.get(f"{field_name}_size_bytes")),
                mime_type=optional_text(payload.get(f"{field_name}_mime_type")) or "application/octet-stream",
                source_system="mango_processed_summary",
                source_ref=event.source_ref,
                extraction_status=ExtractionStatus.EXTRACTED,
                created_at=event_at,
            )
        )
    return tuple(artifacts)


def mango_signals(
    tenant_id: str,
    event: TimelineEvent,
    payload: Mapping[str, Any],
    event_at: datetime,
) -> tuple[DerivedSignal, ...]:
    signal_text = compact_text(first_value(payload, ("recommended_action", "next_step", "signal")), limit=240)
    if not signal_text:
        return ()
    return (
        DerivedSignal(
            tenant_id=tenant_id,
            customer_id=event.customer_id,
            opportunity_id=event.opportunity_id,
            event_id=event.event_id,
            source_event_ids=(event.event_id,),
            signal_type=normalize_key(payload.get("signal_type") or "sales_next_step", "signal_type"),
            severity=SignalSeverity(payload.get("severity") or "medium"),
            evidence_text=signal_text,
            confidence=float_or_default(payload.get("signal_confidence"), 0.7),
            recommended_action=signal_text,
            requires_manager_review=True,
            created_at=event_at,
        ),
    )


def mail_participants(from_email: Optional[str], to_email: Optional[str]) -> tuple[TimelineParticipant, ...]:
    participants: list[TimelineParticipant] = []
    if from_email:
        participants.append(TimelineParticipant(role="sender", ref=from_email, channel="email"))
    if to_email:
        participants.append(TimelineParticipant(role="recipient", ref=to_email, channel="email"))
    return tuple(participants)


def channel_link_type(channel: str) -> str:
    if channel.startswith("telegram") or channel == "tg":
        return "telegram_user_id"
    if channel.startswith("whatsapp") or channel in {"wa", "wappi"}:
        return "whatsapp_user_id"
    if channel.startswith("max"):
        return "max_user_id"
    if "web" in channel or "site" in channel:
        return "web_chat_user_id"
    return "channel_session_id"


def channel_event_type(channel: str) -> TimelineEventType:
    if channel.startswith("telegram") or channel == "tg":
        return TimelineEventType.TELEGRAM_MESSAGE
    if channel.startswith("whatsapp") or channel in {"wa", "wappi"}:
        return TimelineEventType.WHATSAPP_MESSAGE
    if channel.startswith("max"):
        return TimelineEventType.MAX_MESSAGE
    return TimelineEventType.WEB_CHAT_MESSAGE


def infer_mail_direction(payload: Mapping[str, Any], from_email: Optional[str]) -> TimelineDirection:
    explicit = optional_text(first_value(payload, ("direction", "mail_direction")))
    if explicit:
        return TimelineDirection(explicit)
    if from_email and from_email.endswith(("@kmipt.ru", "@cdpofoton.ru", "@fotonai.online")):
        return TimelineDirection.OUTBOUND
    return TimelineDirection.INBOUND


def first_value(payload: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in payload and payload[key] not in (None, ""):
            return payload[key]
    custom = payload.get("custom_fields")
    if isinstance(custom, Mapping):
        for key in keys:
            if key in custom and custom[key] not in (None, ""):
                return custom[key]
    return None


def parse_source_datetime(value: Any, fallback: Optional[datetime] = None) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        text = optional_text(value)
        if not text:
            parsed = fallback or datetime.now(timezone.utc)
        else:
            normalized = text.replace("Z", "+00:00")
            try:
                if normalized.isdigit():
                    parsed = datetime.fromtimestamp(int(normalized), tz=timezone.utc)
                else:
                    parsed = datetime.fromisoformat(normalized)
            except ValueError:
                parsed = fallback or datetime.now(timezone.utc)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def safe_phone(value: Any) -> Optional[str]:
    text = optional_text(value)
    if not text:
        return None
    normalized = normalize_phone(text)
    return normalized or None


def safe_email(value: Any) -> Optional[str]:
    text = optional_text(value)
    if not text:
        return None
    normalized = normalize_email(text)
    return normalized or None


def compact_text(value: Any, *, limit: int = 500) -> Optional[str]:
    text = optional_text(value)
    if not text:
        return None
    compacted = " ".join(text.split())
    return compacted[:limit]


def int_or_none(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def int_or_default(value: Any, default: int) -> int:
    parsed = int_or_none(value)
    return default if parsed is None else parsed


def float_or_default(value: Any, default: float) -> float:
    if value in (None, ""):
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return min(1.0, max(0.0, parsed))


def require_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return dict(value)


def require_sql_identifier(value: Any, field_name: str) -> str:
    text = require_text(value, field_name)
    if not _SQL_IDENTIFIER_RE.match(text):
        raise ValueError(f"{field_name} must be a simple SQL identifier")
    return text


def contains_sql_write_keyword(sql: str) -> bool:
    normalized = f" {sql.casefold()} "
    return any(f" {keyword} " in normalized for keyword in ("insert", "update", "delete", "drop", "alter", "create", "replace", "vacuum", "pragma", "attach"))


def zero_normalized_counts() -> dict[str, int]:
    return {
        "customers": 0,
        "identity_links": 0,
        "opportunities": 0,
        "events": 0,
        "artifacts": 0,
        "signals": 0,
        "bot_context_chunks": 0,
        "conflicts": 0,
    }


def merge_counts(target: dict[str, int], source: Mapping[str, int]) -> None:
    for key, value in source.items():
        target[key] = target.get(key, 0) + int(value)


def assert_bot_context_not_allowed_for_restricted_source(batch: TimelineNormalizedBatch) -> None:
    source_system = batch.source_record.source_system
    if source_system not in BOT_FORBIDDEN_SOURCE_SYSTEMS:
        return
    if contains_allowed_for_bot_true(batch.source_record.payload):
        raise ValueError(f"{source_system} source records must be loaded with allowed_for_bot=False")
    for event in batch.events:
        if event.source_system in BOT_FORBIDDEN_SOURCE_SYSTEMS and contains_allowed_for_bot_true(event.record):
            raise ValueError(f"{event.source_system} timeline events must be loaded with allowed_for_bot=False")
    for chunk in batch.bot_context_chunks:
        if chunk.source_system in BOT_FORBIDDEN_SOURCE_SYSTEMS and chunk.allowed_for_bot:
            raise ValueError(f"{chunk.source_system} bot context chunks must be loaded with allowed_for_bot=False")


def contains_allowed_for_bot_true(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key == "allowed_for_bot" and truthy_allowed_for_bot(item):
                return True
            if contains_allowed_for_bot_true(item):
                return True
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(contains_allowed_for_bot_true(item) for item in value)
    return False


def truthy_allowed_for_bot(value: Any) -> bool:
    return truthy_flag(value)


def truthy_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value in (None, 0):
        return False
    text = str(value).strip().casefold()
    return text in {"1", "true", "yes", "y", "да", "allowed"}


def _assert_sequence_type(values: Sequence[Any], expected_type: type, field_name: str) -> None:
    if any(not isinstance(item, expected_type) for item in values):
        raise TypeError(f"{field_name} must contain {expected_type.__name__} items")


__all__ = [
    "CUSTOMER_TIMELINE_INGESTION_SCHEMA_VERSION",
    "AmoSnapshotNormalizer",
    "ChannelMessageNormalizer",
    "MailMessageNormalizer",
    "MangoCallSummaryNormalizer",
    "TallantoSnapshotNormalizer",
    "TimelineImportError",
    "TimelineImportReport",
    "TimelineImportService",
    "TimelineNormalizedBatch",
    "TimelineNormalizer",
    "TimelineSourceRecord",
    "build_source_inventory",
    "file_sha256",
    "guard_customer_timeline_source_path",
    "infer_identity_conflicts",
    "load_local_source_records",
    "load_sqlite_source_records",
    "rows_from_csv",
    "timeline_ingestion_safety_contract",
]
