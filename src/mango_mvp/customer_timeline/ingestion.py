from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
import re
from dataclasses import asdict, dataclass, field
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
        if not dry_run:
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

        accepted = 0
        errors: list[TimelineImportError] = []
        normalized_counts = zero_normalized_counts()
        write_status_counts: dict[str, int] = {}
        batches: list[TimelineNormalizedBatch] = []
        for record in normalized_records:
            try:
                batch = normalizer.normalize(record)
                batches.append(batch)
                merge_counts(normalized_counts, batch.counts())
                accepted += 1
            except Exception as exc:  # noqa: BLE001 - report per-record import errors, do not hide the batch.
                errors.append(
                    TimelineImportError(
                        source_ref=record.source_ref,
                        error_type=type(exc).__name__,
                        message=str(exc),
                    )
                )
        inferred_conflicts = infer_identity_conflicts(batches)
        normalized_counts["conflicts"] = normalized_counts.get("conflicts", 0) + len(inferred_conflicts)
        if not dry_run:
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
        if not dry_run and run_id:
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
        message_id = require_text(first_value(payload, ("message_id", "id", "uid")) or record.source_ref, "message_id")
        from_email = safe_email(first_value(payload, ("from_email", "from", "sender", "sender_email")))
        to_email = safe_email(first_value(payload, ("to_email", "to", "recipient", "recipient_email")))
        direction = infer_mail_direction(payload, from_email)
        customer_email = from_email if direction == TimelineDirection.INBOUND else to_email
        event_at = parse_source_datetime(first_value(payload, ("date", "sent_at", "received_at", "event_at")), record.observed_at)
        source_ref = f"mail:{message_id}"
        customer = CustomerIdentity(
            tenant_id=self.tenant_id,
            identity_status=IdentityStatus.STRONG if customer_email else IdentityStatus.PARTIAL,
            display_name=optional_text(first_value(payload, ("name", "display_name", "from_name", "to_name"))),
            primary_email=customer_email,
            source_ref=source_ref,
            first_seen_at=event_at,
            last_seen_at=event_at,
            touch_count=1,
            summary={"source_system": self.source_system, "direction": direction.value},
            metadata={"message_id": message_id, "payload_hash": record.payload_hash},
            created_at=event_at,
            updated_at=event_at,
        )
        links = identity_links_for_customer(
            customer,
            source_system=self.source_system,
            source_ref=source_ref,
            email=customer_email,
        )
        thread_id = optional_text(first_value(payload, ("thread_id", "conversation_id", "subject"))) or message_id
        opportunity = CustomerOpportunity(
            tenant_id=self.tenant_id,
            customer_id=customer.customer_id,
            opportunity_type=OpportunityType.MAIL_THREAD,
            source_system=self.source_system,
            source_id=thread_id,
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
            match_status=IdentityMatchClass.STRONG_UNIQUE if customer_email else IdentityMatchClass.INFERRED,
            confidence=0.75 if customer_email else 0.5,
            record={"message": scrub_timeline_persisted_json(dict(payload))},
            created_at=event_at,
        )
        artifacts = mail_artifacts(self.tenant_id, event, payload, event_at)
        return TimelineNormalizedBatch(
            source_record=record,
            customers=(customer,),
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
        links = identity_links_for_customer(
            customer,
            source_system=self.source_system,
            source_ref=source_ref,
            phone=phone,
        )
        if phone:
            links.append(
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
        event = TimelineEvent(
            tenant_id=self.tenant_id,
            customer_id=customer.customer_id,
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
            summary=compact_text(first_value(payload, ("summary", "insight_summary", "analysis_summary")), limit=500),
            importance=int_or_default(first_value(payload, ("importance", "priority_score")), 0),
            match_status=IdentityMatchClass.STRONG_UNIQUE if phone else IdentityMatchClass.INFERRED,
            confidence=float_or_default(first_value(payload, ("confidence",)), 0.75 if phone else 0.5),
            record={"call": scrub_timeline_persisted_json(dict(payload))},
            created_at=event_at,
        )
        artifacts = mango_artifacts(self.tenant_id, event, payload, event_at)
        signals = mango_signals(self.tenant_id, event, payload, event_at)
        chunks: tuple[BotContextChunk, ...] = ()
        if event.summary:
            chunks = (
                BotContextChunk(
                    tenant_id=self.tenant_id,
                    customer_id=customer.customer_id,
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
            customers=(customer,),
            identity_links=tuple(links),
            events=(event,),
            artifacts=artifacts,
            signals=signals,
            bot_context_chunks=chunks,
            conflicts=conflict_from_payload(self.tenant_id, payload, source_ref),
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
    if channel.startswith("max"):
        return "max_user_id"
    if "web" in channel or "site" in channel:
        return "web_chat_user_id"
    return "channel_session_id"


def channel_event_type(channel: str) -> TimelineEventType:
    if channel.startswith("telegram") or channel == "tg":
        return TimelineEventType.TELEGRAM_MESSAGE
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
