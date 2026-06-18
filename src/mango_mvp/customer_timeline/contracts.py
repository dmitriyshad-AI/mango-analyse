from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Optional, Sequence

from mango_mvp.customer_timeline.ids import (
    normalize_email,
    normalize_identity_value,
    normalize_key,
    optional_text,
    require_confidence,
    require_ordered_datetimes,
    require_text,
    require_timezone,
    stable_artifact_id,
    stable_chunk_id,
    stable_customer_id,
    stable_event_id,
    stable_identity_link_id,
    stable_opportunity_id,
    stable_signal_id,
)
from mango_mvp.customer_timeline.safety import customer_timeline_safety_contract


CUSTOMER_TIMELINE_CONTRACTS_SCHEMA_VERSION = "customer_timeline_contracts_v1"
_SHA256_RE = re.compile(r"^[a-fA-F0-9]{64}$")


class IdentityStatus(str, Enum):
    STRONG = "strong"
    PARTIAL = "partial"
    AMBIGUOUS = "ambiguous"
    UNMATCHED = "unmatched"


class IdentityLinkType(str, Enum):
    PHONE = "phone"
    EMAIL = "email"
    AMO_CONTACT_ID = "amo_contact_id"
    AMO_LEAD_ID = "amo_lead_id"
    TALLANTO_STUDENT_ID = "tallanto_student_id"
    TALLANTO_PARENT_REF = "tallanto_parent_ref"
    MANGO_CLIENT_PHONE = "mango_client_phone"
    TELEGRAM_USER_ID = "telegram_user_id"
    TELEGRAM_USERNAME = "telegram_username"
    WHATSAPP_USER_ID = "whatsapp_user_id"
    WHATSAPP_PHONE = "whatsapp_phone"
    MAX_USER_ID = "max_user_id"
    WEB_CHAT_USER_ID = "web_chat_user_id"
    CHANNEL_SESSION_ID = "channel_session_id"


class IdentityMatchClass(str, Enum):
    STRONG_UNIQUE = "strong_unique"
    DUPLICATE = "duplicate"
    AMBIGUOUS = "ambiguous"
    INFERRED = "inferred"
    MANUAL = "manual"
    UNMATCHED = "unmatched"


class OpportunityType(str, Enum):
    AMO_DEAL = "amo_deal"
    TALLANTO_COURSE = "tallanto_course"
    SERVICE_CASE = "service_case"
    RENEWAL = "renewal"
    MAIL_THREAD = "mail_thread"
    TELEGRAM_DIALOG = "telegram_dialog"
    UNKNOWN = "unknown"


class TimelineDirection(str, Enum):
    INBOUND = "inbound"
    OUTBOUND = "outbound"
    INTERNAL = "internal"
    SYSTEM = "system"


class TimelineEventType(str, Enum):
    MANGO_CALL = "mango_call"
    CALL_TRANSCRIPT = "call_transcript"
    CALL_ANALYSIS = "call_analysis"
    EMAIL_MESSAGE = "email_message"
    EMAIL_ATTACHMENT = "email_attachment"
    TELEGRAM_MESSAGE = "telegram_message"
    TELEGRAM_DIALOG = "telegram_dialog"
    WHATSAPP_MESSAGE = "whatsapp_message"
    MAX_MESSAGE = "max_message"
    WEB_CHAT_MESSAGE = "web_chat_message"
    BOT_DRAFT = "bot_draft"
    BOT_ACTION = "bot_action"
    AMO_CONTACT_SNAPSHOT = "amo_contact_snapshot"
    AMO_DEAL_STAGE = "amo_deal_stage"
    AMO_NOTE = "amo_note"
    AMO_TASK = "amo_task"
    TALLANTO_STUDENT_SNAPSHOT = "tallanto_student_snapshot"
    TALLANTO_PAYMENT = "tallanto_payment"
    TALLANTO_ABONEMENT = "tallanto_abonement"
    TALLANTO_GROUP = "tallanto_group"
    MANAGER_ACTION = "manager_action"
    SYSTEM_NOTE = "system_note"


class ArtifactType(str, Enum):
    CALL_AUDIO = "call_audio"
    CALL_TRANSCRIPT_JSON = "call_transcript_json"
    RAW_EMAIL_EML = "raw_email_eml"
    MAIL_ATTACHMENT = "mail_attachment"
    ATTACHMENT_TEXT = "attachment_text"
    API_RAW_JSON = "api_raw_json"
    ANALYSIS_JSON = "analysis_json"
    REPORT_FILE = "report_file"


class ExtractionStatus(str, Enum):
    NOT_NEEDED = "not_needed"
    PENDING = "pending"
    EXTRACTED = "extracted"
    FAILED = "failed"
    QUARANTINED = "quarantined"


class SignalSeverity(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class TimelineParticipant:
    role: str
    ref: str
    name: Optional[str] = None
    channel: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", normalize_key(self.role, "participant role"))
        object.__setattr__(self, "ref", require_text(self.ref, "participant ref"))
        object.__setattr__(self, "name", optional_text(self.name))
        object.__setattr__(self, "channel", optional_text(self.channel))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CustomerIdentity:
    tenant_id: str
    identity_status: IdentityStatus | str
    customer_id: Optional[str] = None
    display_name: Optional[str] = None
    primary_phone: Optional[str] = None
    primary_email: Optional[str] = None
    source_ref: Optional[str] = None
    first_seen_at: Optional[datetime] = None
    last_seen_at: Optional[datetime] = None
    touch_count: int = 0
    summary: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=now_utc)
    updated_at: datetime = field(default_factory=now_utc)

    def __post_init__(self) -> None:
        tenant_id = normalize_key(self.tenant_id, "tenant_id")
        status = IdentityStatus(self.identity_status)
        phone = normalize_identity_value("phone", self.primary_phone) if self.primary_phone else None
        email = normalize_identity_value("email", self.primary_email) if self.primary_email else None
        if self.primary_email and not email:
            raise ValueError(f"invalid primary_email: {self.primary_email!r}")
        require_ordered_datetimes(
            self.first_seen_at,
            self.last_seen_at,
            start_name="first_seen_at",
            end_name="last_seen_at",
        )
        require_timezone(self.created_at, "created_at")
        require_timezone(self.updated_at, "updated_at")
        if self.updated_at < self.created_at:
            raise ValueError("updated_at must be greater than or equal to created_at")
        if self.touch_count < 0:
            raise ValueError("touch_count must not be negative")
        customer_id = optional_text(self.customer_id) or stable_customer_id(
            tenant_id=tenant_id,
            primary_phone=phone,
            primary_email=email,
            source_ref=self.source_ref,
        )
        object.__setattr__(self, "tenant_id", tenant_id)
        object.__setattr__(self, "identity_status", status)
        object.__setattr__(self, "customer_id", customer_id)
        object.__setattr__(self, "display_name", optional_text(self.display_name))
        object.__setattr__(self, "primary_phone", phone)
        object.__setattr__(self, "primary_email", email or None)
        object.__setattr__(self, "source_ref", optional_text(self.source_ref))
        object.__setattr__(self, "summary", dict(self.summary))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_json_dict(self) -> Mapping[str, Any]:
        return {
            "schema_version": CUSTOMER_TIMELINE_CONTRACTS_SCHEMA_VERSION,
            "tenant_id": self.tenant_id,
            "customer_id": self.customer_id,
            "display_name": self.display_name,
            "identity_status": self.identity_status.value,
            "first_seen_at": self.first_seen_at.isoformat() if self.first_seen_at else None,
            "last_seen_at": self.last_seen_at.isoformat() if self.last_seen_at else None,
            "touch_count": self.touch_count,
            "primary_phone": self.primary_phone,
            "primary_email": self.primary_email,
            "source_ref": self.source_ref,
            "summary": dict(self.summary),
            "metadata": dict(self.metadata),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass(frozen=True)
class IdentityLink:
    tenant_id: str
    link_type: IdentityLinkType | str
    link_value: str
    source_system: str
    source_ref: str
    customer_id: Optional[str] = None
    link_id: Optional[str] = None
    match_class: IdentityMatchClass | str = IdentityMatchClass.STRONG_UNIQUE
    confidence: Optional[float] = None
    evidence: Mapping[str, Any] = field(default_factory=dict)
    first_seen_at: Optional[datetime] = None
    last_seen_at: Optional[datetime] = None

    def __post_init__(self) -> None:
        tenant_id = normalize_key(self.tenant_id, "tenant_id")
        link_type = IdentityLinkType(self.link_type)
        link_value = normalize_identity_value(link_type.value, self.link_value)
        source_system = normalize_key(self.source_system, "source_system")
        source_ref = require_text(self.source_ref, "source_ref")
        require_ordered_datetimes(
            self.first_seen_at,
            self.last_seen_at,
            start_name="first_seen_at",
            end_name="last_seen_at",
        )
        object.__setattr__(self, "tenant_id", tenant_id)
        object.__setattr__(self, "link_type", link_type)
        object.__setattr__(self, "link_value", link_value)
        object.__setattr__(self, "source_system", source_system)
        object.__setattr__(self, "source_ref", source_ref)
        object.__setattr__(self, "customer_id", optional_text(self.customer_id))
        object.__setattr__(
            self,
            "link_id",
            optional_text(self.link_id)
            or stable_identity_link_id(
                tenant_id=tenant_id,
                link_type=link_type.value,
                link_value=link_value,
                source_system=source_system,
                source_ref=source_ref,
            ),
        )
        object.__setattr__(self, "match_class", IdentityMatchClass(self.match_class))
        object.__setattr__(self, "confidence", require_confidence(self.confidence))
        object.__setattr__(self, "evidence", dict(self.evidence))

    def to_json_dict(self) -> Mapping[str, Any]:
        return {
            "schema_version": CUSTOMER_TIMELINE_CONTRACTS_SCHEMA_VERSION,
            "tenant_id": self.tenant_id,
            "link_id": self.link_id,
            "customer_id": self.customer_id,
            "link_type": self.link_type.value,
            "link_value": self.link_value,
            "source_system": self.source_system,
            "source_ref": self.source_ref,
            "match_class": self.match_class.value,
            "confidence": self.confidence,
            "evidence": dict(self.evidence),
            "first_seen_at": self.first_seen_at.isoformat() if self.first_seen_at else None,
            "last_seen_at": self.last_seen_at.isoformat() if self.last_seen_at else None,
        }


@dataclass(frozen=True)
class CustomerOpportunity:
    tenant_id: str
    customer_id: str
    opportunity_type: OpportunityType | str
    source_system: str
    source_id: str
    opportunity_id: Optional[str] = None
    title: Optional[str] = None
    status: Optional[str] = None
    product_context: Mapping[str, Any] = field(default_factory=dict)
    opened_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None
    confidence: Optional[float] = None
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        tenant_id = normalize_key(self.tenant_id, "tenant_id")
        customer_id = require_text(self.customer_id, "customer_id")
        opportunity_type = OpportunityType(self.opportunity_type)
        source_system = normalize_key(self.source_system, "source_system")
        source_id = require_text(self.source_id, "source_id")
        require_ordered_datetimes(
            self.opened_at,
            self.closed_at,
            start_name="opened_at",
            end_name="closed_at",
        )
        object.__setattr__(self, "tenant_id", tenant_id)
        object.__setattr__(self, "customer_id", customer_id)
        object.__setattr__(self, "opportunity_type", opportunity_type)
        object.__setattr__(self, "source_system", source_system)
        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(
            self,
            "opportunity_id",
            optional_text(self.opportunity_id)
            or stable_opportunity_id(
                tenant_id=tenant_id,
                customer_id=customer_id,
                opportunity_type=opportunity_type.value,
                source_system=source_system,
                source_id=source_id,
            ),
        )
        object.__setattr__(self, "title", optional_text(self.title))
        object.__setattr__(self, "status", optional_text(self.status))
        object.__setattr__(self, "product_context", dict(self.product_context))
        object.__setattr__(self, "confidence", require_confidence(self.confidence))
        object.__setattr__(self, "evidence", dict(self.evidence))

    def to_json_dict(self) -> Mapping[str, Any]:
        return {
            "schema_version": CUSTOMER_TIMELINE_CONTRACTS_SCHEMA_VERSION,
            "tenant_id": self.tenant_id,
            "customer_id": self.customer_id,
            "opportunity_id": self.opportunity_id,
            "opportunity_type": self.opportunity_type.value,
            "source_system": self.source_system,
            "source_id": self.source_id,
            "title": self.title,
            "status": self.status,
            "product_context": dict(self.product_context),
            "opened_at": self.opened_at.isoformat() if self.opened_at else None,
            "closed_at": self.closed_at.isoformat() if self.closed_at else None,
            "confidence": self.confidence,
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True)
class TimelineEvent:
    tenant_id: str
    event_type: TimelineEventType | str
    event_at: datetime
    source_system: str
    source_id: str
    direction: TimelineDirection | str
    customer_id: Optional[str] = None
    opportunity_id: Optional[str] = None
    event_id: Optional[str] = None
    source_ref: Optional[str] = None
    source_refs: Sequence[str] = field(default_factory=tuple)
    participants: Sequence[TimelineParticipant] = field(default_factory=tuple)
    actor_name: Optional[str] = None
    actor_ref: Optional[str] = None
    subject: Optional[str] = None
    text_preview: Optional[str] = None
    summary: Optional[str] = None
    stage_before: Optional[str] = None
    stage_after: Optional[str] = None
    importance: int = 0
    match_status: IdentityMatchClass | str = IdentityMatchClass.UNMATCHED
    confidence: Optional[float] = None
    record: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=now_utc)

    def __post_init__(self) -> None:
        tenant_id = normalize_key(self.tenant_id, "tenant_id")
        event_type = TimelineEventType(self.event_type)
        source_system = normalize_key(self.source_system, "source_system")
        source_id = require_text(self.source_id, "source_id")
        direction = TimelineDirection(self.direction)
        require_timezone(self.event_at, "event_at")
        require_timezone(self.created_at, "created_at")
        participants = tuple(self.participants)
        if any(not isinstance(item, TimelineParticipant) for item in participants):
            raise TypeError("participants must contain TimelineParticipant items")
        if self.importance < 0:
            raise ValueError("importance must not be negative")
        object.__setattr__(self, "tenant_id", tenant_id)
        object.__setattr__(self, "event_type", event_type)
        object.__setattr__(self, "source_system", source_system)
        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "customer_id", optional_text(self.customer_id))
        object.__setattr__(self, "opportunity_id", optional_text(self.opportunity_id))
        object.__setattr__(
            self,
            "event_id",
            optional_text(self.event_id)
            or stable_event_id(
                tenant_id=tenant_id,
                source_system=source_system,
                source_id=source_id,
                event_type=event_type.value,
            ),
        )
        source_ref = optional_text(self.source_ref) or source_id
        source_refs = tuple(require_text(item, "source_ref") for item in self.source_refs)
        if source_ref not in source_refs:
            source_refs = (source_ref,) + source_refs
        object.__setattr__(self, "source_ref", source_ref)
        object.__setattr__(self, "source_refs", source_refs)
        object.__setattr__(self, "participants", participants)
        object.__setattr__(self, "actor_name", optional_text(self.actor_name))
        object.__setattr__(self, "actor_ref", optional_text(self.actor_ref))
        object.__setattr__(self, "subject", optional_text(self.subject))
        object.__setattr__(self, "text_preview", optional_text(self.text_preview))
        object.__setattr__(self, "summary", optional_text(self.summary))
        object.__setattr__(self, "stage_before", optional_text(self.stage_before))
        object.__setattr__(self, "stage_after", optional_text(self.stage_after))
        object.__setattr__(self, "match_status", IdentityMatchClass(self.match_status))
        object.__setattr__(self, "confidence", require_confidence(self.confidence))
        object.__setattr__(self, "record", dict(self.record))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def dedupe_key(self) -> str:
        return f"{self.tenant_id}:{self.source_system}:{self.event_type.value}:{self.source_id}"

    def to_json_dict(self) -> Mapping[str, Any]:
        return {
            "schema_version": CUSTOMER_TIMELINE_CONTRACTS_SCHEMA_VERSION,
            "tenant_id": self.tenant_id,
            "event_id": self.event_id,
            "customer_id": self.customer_id,
            "opportunity_id": self.opportunity_id,
            "event_type": self.event_type.value,
            "event_at": self.event_at.isoformat(),
            "source_system": self.source_system,
            "source_id": self.source_id,
            "source_ref": self.source_ref,
            "source_refs": list(self.source_refs),
            "direction": self.direction.value,
            "participants": [item.to_json_dict() for item in self.participants],
            "actor_name": self.actor_name,
            "actor_ref": self.actor_ref,
            "subject": self.subject,
            "text_preview": self.text_preview,
            "summary": self.summary,
            "stage_before": self.stage_before,
            "stage_after": self.stage_after,
            "importance": self.importance,
            "match_status": self.match_status.value,
            "confidence": self.confidence,
            "record": dict(self.record),
            "metadata": dict(self.metadata),
            "created_at": self.created_at.isoformat(),
            "dedupe_key": self.dedupe_key,
        }


@dataclass(frozen=True)
class EventArtifact:
    tenant_id: str
    event_id: str
    artifact_type: ArtifactType | str
    path: str
    source_system: str
    source_ref: str
    artifact_id: Optional[str] = None
    sha256: Optional[str] = None
    size_bytes: Optional[int] = None
    mime_type: Optional[str] = None
    extraction_status: ExtractionStatus | str = ExtractionStatus.NOT_NEEDED
    metadata: Mapping[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=now_utc)

    def __post_init__(self) -> None:
        tenant_id = normalize_key(self.tenant_id, "tenant_id")
        event_id = require_text(self.event_id, "event_id")
        artifact_type = ArtifactType(self.artifact_type)
        path = require_text(self.path, "path")
        source_system = normalize_key(self.source_system, "source_system")
        source_ref = require_text(self.source_ref, "source_ref")
        if self.size_bytes is not None and self.size_bytes < 0:
            raise ValueError("size_bytes must not be negative")
        sha256 = optional_text(self.sha256)
        if sha256 is not None and not _SHA256_RE.match(sha256):
            raise ValueError("sha256 must be a 64-character hex digest")
        require_timezone(self.created_at, "created_at")
        object.__setattr__(self, "tenant_id", tenant_id)
        object.__setattr__(self, "event_id", event_id)
        object.__setattr__(self, "artifact_type", artifact_type)
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "source_system", source_system)
        object.__setattr__(self, "source_ref", source_ref)
        object.__setattr__(
            self,
            "artifact_id",
            optional_text(self.artifact_id)
            or stable_artifact_id(
                event_id=event_id,
                tenant_id=tenant_id,
                artifact_type=artifact_type.value,
                path=path,
                sha256=sha256,
            ),
        )
        object.__setattr__(self, "sha256", sha256.lower() if sha256 else None)
        object.__setattr__(self, "mime_type", optional_text(self.mime_type))
        object.__setattr__(self, "extraction_status", ExtractionStatus(self.extraction_status))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_json_dict(self) -> Mapping[str, Any]:
        return {
            "schema_version": CUSTOMER_TIMELINE_CONTRACTS_SCHEMA_VERSION,
            "artifact_id": self.artifact_id,
            "tenant_id": self.tenant_id,
            "event_id": self.event_id,
            "artifact_type": self.artifact_type.value,
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "mime_type": self.mime_type,
            "source_system": self.source_system,
            "source_ref": self.source_ref,
            "extraction_status": self.extraction_status.value,
            "metadata": dict(self.metadata),
            "created_at": self.created_at.isoformat(),
        }


@dataclass(frozen=True)
class DerivedSignal:
    tenant_id: str
    signal_type: str
    severity: SignalSeverity | str
    evidence_text: str
    signal_id: Optional[str] = None
    customer_id: Optional[str] = None
    opportunity_id: Optional[str] = None
    event_id: Optional[str] = None
    source_event_ids: Sequence[str] = field(default_factory=tuple)
    confidence: Optional[float] = None
    recommended_action: Optional[str] = None
    requires_manager_review: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=now_utc)

    def __post_init__(self) -> None:
        tenant_id = normalize_key(self.tenant_id, "tenant_id")
        signal_type = normalize_key(self.signal_type, "signal_type")
        evidence_text = require_text(self.evidence_text, "evidence_text")
        require_timezone(self.created_at, "created_at")
        source_event_ids = tuple(require_text(item, "source_event_id") for item in self.source_event_ids)
        event_id = optional_text(self.event_id)
        if event_id and event_id not in source_event_ids:
            source_event_ids = (event_id,) + source_event_ids
        object.__setattr__(self, "tenant_id", tenant_id)
        object.__setattr__(self, "customer_id", optional_text(self.customer_id))
        object.__setattr__(self, "signal_type", signal_type)
        object.__setattr__(self, "severity", SignalSeverity(self.severity))
        object.__setattr__(self, "evidence_text", evidence_text)
        object.__setattr__(self, "opportunity_id", optional_text(self.opportunity_id))
        object.__setattr__(self, "event_id", event_id)
        object.__setattr__(self, "source_event_ids", source_event_ids)
        object.__setattr__(self, "confidence", require_confidence(self.confidence))
        object.__setattr__(self, "recommended_action", optional_text(self.recommended_action))
        object.__setattr__(
            self,
            "signal_id",
            optional_text(self.signal_id)
            or stable_signal_id(
                tenant_id=tenant_id,
                customer_id=self.customer_id,
                signal_type=signal_type,
                source_event_ids=source_event_ids,
                evidence_text=evidence_text if not source_event_ids else None,
            ),
        )
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_json_dict(self) -> Mapping[str, Any]:
        return {
            "schema_version": CUSTOMER_TIMELINE_CONTRACTS_SCHEMA_VERSION,
            "signal_id": self.signal_id,
            "tenant_id": self.tenant_id,
            "customer_id": self.customer_id,
            "opportunity_id": self.opportunity_id,
            "event_id": self.event_id,
            "source_event_ids": list(self.source_event_ids),
            "signal_type": self.signal_type,
            "severity": self.severity.value,
            "confidence": self.confidence,
            "evidence_text": self.evidence_text,
            "recommended_action": self.recommended_action,
            "requires_manager_review": self.requires_manager_review,
            "metadata": dict(self.metadata),
            "created_at": self.created_at.isoformat(),
        }


@dataclass(frozen=True)
class BotContextChunk:
    tenant_id: str
    customer_id: str
    chunk_type: str
    text: str
    chunk_id: Optional[str] = None
    opportunity_id: Optional[str] = None
    event_id: Optional[str] = None
    source_ref: Optional[str] = None
    ordinal: int = 0
    source_system: Optional[str] = None
    summary: Optional[str] = None
    event_at: Optional[datetime] = None
    freshness_score: Optional[float] = None
    relevance_tags: Sequence[str] = field(default_factory=tuple)
    allowed_for_bot: bool = True
    requires_manager_review: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=now_utc)

    def __post_init__(self) -> None:
        tenant_id = normalize_key(self.tenant_id, "tenant_id")
        customer_id = require_text(self.customer_id, "customer_id")
        chunk_type = normalize_key(self.chunk_type, "chunk_type")
        text = require_text(self.text, "text")
        if self.event_at is not None:
            require_timezone(self.event_at, "event_at")
        require_timezone(self.created_at, "created_at")
        if self.ordinal < 0:
            raise ValueError("ordinal must not be negative")
        if self.requires_manager_review and self.allowed_for_bot:
            raise ValueError("bot context chunks requiring manager review must not be allowed_for_bot")
        tags = tuple(normalize_key(item, "relevance tag") for item in self.relevance_tags)
        object.__setattr__(self, "tenant_id", tenant_id)
        object.__setattr__(self, "customer_id", customer_id)
        object.__setattr__(self, "chunk_type", chunk_type)
        object.__setattr__(self, "text", text)
        object.__setattr__(self, "opportunity_id", optional_text(self.opportunity_id))
        object.__setattr__(self, "event_id", optional_text(self.event_id))
        object.__setattr__(self, "source_ref", optional_text(self.source_ref))
        object.__setattr__(self, "source_system", normalize_key(self.source_system, "source_system") if self.source_system else None)
        object.__setattr__(self, "summary", optional_text(self.summary))
        object.__setattr__(self, "freshness_score", require_confidence(self.freshness_score, "freshness_score"))
        object.__setattr__(self, "relevance_tags", tags)
        object.__setattr__(
            self,
            "chunk_id",
            optional_text(self.chunk_id)
            or stable_chunk_id(
                tenant_id=tenant_id,
                customer_id=customer_id,
                chunk_type=chunk_type,
                event_id=self.event_id,
                source_ref=self.source_ref,
                ordinal=self.ordinal,
            ),
        )
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_json_dict(self) -> Mapping[str, Any]:
        return {
            "schema_version": CUSTOMER_TIMELINE_CONTRACTS_SCHEMA_VERSION,
            "chunk_id": self.chunk_id,
            "tenant_id": self.tenant_id,
            "customer_id": self.customer_id,
            "opportunity_id": self.opportunity_id,
            "event_id": self.event_id,
            "source_ref": self.source_ref,
            "source_system": self.source_system,
            "chunk_type": self.chunk_type,
            "text": self.text,
            "summary": self.summary,
            "event_at": self.event_at.isoformat() if self.event_at else None,
            "freshness_score": self.freshness_score,
            "relevance_tags": list(self.relevance_tags),
            "ordinal": self.ordinal,
            "allowed_for_bot": self.allowed_for_bot,
            "requires_manager_review": self.requires_manager_review,
            "metadata": dict(self.metadata),
            "created_at": self.created_at.isoformat(),
        }


def customer_timeline_contract_inventory() -> Mapping[str, Any]:
    return {
        "schema_version": CUSTOMER_TIMELINE_CONTRACTS_SCHEMA_VERSION,
        "contracts": [
            "CustomerIdentity",
            "IdentityLink",
            "CustomerOpportunity",
            "TimelineEvent",
            "EventArtifact",
            "DerivedSignal",
            "BotContextChunk",
        ],
        "event_types": [item.value for item in TimelineEventType],
        "identity_link_types": [item.value for item in IdentityLinkType],
        "artifact_types": [item.value for item in ArtifactType],
        "safety": customer_timeline_safety_contract(),
    }


def dedupe_timeline_events(events: Sequence[TimelineEvent]) -> tuple[TimelineEvent, ...]:
    result: list[TimelineEvent] = []
    seen: set[str] = set()
    for event in events:
        if not isinstance(event, TimelineEvent):
            raise TypeError("events must contain TimelineEvent items")
        if event.dedupe_key in seen:
            continue
        seen.add(event.dedupe_key)
        result.append(event)
    return tuple(result)
