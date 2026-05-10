from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence

from mango_mvp.channels.actions import recommended_action_to_agent_proposal
from mango_mvp.channels.contracts import (
    ChannelMessage,
    ChannelSession,
    RecommendedAction,
    normalize_key,
    optional_text,
    require_text,
    require_timezone,
    stable_digest,
)
from mango_mvp.channels.preview_service import ChannelDraftPreview, ChannelPreviewService


CHANNEL_STORAGE_SCHEMA_VERSION = "channel_storage_v1"

DRAFT_STATUS_NEEDS_REVIEW = "needs_review"
DRAFT_STATUS_APPROVED = "approved"
DRAFT_STATUS_REJECTED = "rejected"
DRAFT_STATUS_MOCK_SENT = "mock_sent"
DRAFT_STATUS_FAILED = "failed"
DRAFT_STATUS_SUPERSEDED = "superseded"
DRAFT_STATUS_SENT = "sent"

ACTION_STATUS_PROPOSED = "proposed"
ACTION_STATUS_APPROVED = "approved"
ACTION_STATUS_REJECTED = "rejected"
ACTION_STATUS_BLOCKED = "blocked"
ACTION_STATUS_DISMISSED = "dismissed"
ACTION_STATUS_MOCK_EXECUTED = "mock_executed"
ACTION_STATUS_FAILED = "failed"
ACTION_STATUS_EXECUTED = "executed"

ALLOWED_DRAFT_STATUSES = {
    DRAFT_STATUS_NEEDS_REVIEW,
    DRAFT_STATUS_APPROVED,
    DRAFT_STATUS_REJECTED,
    DRAFT_STATUS_MOCK_SENT,
    DRAFT_STATUS_FAILED,
    DRAFT_STATUS_SUPERSEDED,
}
ALLOWED_ACTION_STATUSES = {
    ACTION_STATUS_PROPOSED,
    ACTION_STATUS_APPROVED,
    ACTION_STATUS_REJECTED,
    ACTION_STATUS_BLOCKED,
    ACTION_STATUS_DISMISSED,
    ACTION_STATUS_MOCK_EXECUTED,
    ACTION_STATUS_FAILED,
}
LIVE_EXECUTION_STATUSES = {
    DRAFT_STATUS_SENT,
    ACTION_STATUS_EXECUTED,
    "live_sent",
    "sent",
}
DRAFT_STATUS_TRANSITIONS = {
    DRAFT_STATUS_NEEDS_REVIEW: {
        DRAFT_STATUS_APPROVED,
        DRAFT_STATUS_REJECTED,
        DRAFT_STATUS_FAILED,
        DRAFT_STATUS_SUPERSEDED,
    },
    DRAFT_STATUS_APPROVED: {DRAFT_STATUS_MOCK_SENT, DRAFT_STATUS_FAILED, DRAFT_STATUS_SUPERSEDED},
    DRAFT_STATUS_REJECTED: set(),
    DRAFT_STATUS_MOCK_SENT: set(),
    DRAFT_STATUS_FAILED: set(),
    DRAFT_STATUS_SUPERSEDED: set(),
}
ACTION_STATUS_TRANSITIONS = {
    ACTION_STATUS_PROPOSED: {
        ACTION_STATUS_APPROVED,
        ACTION_STATUS_REJECTED,
        ACTION_STATUS_BLOCKED,
        ACTION_STATUS_DISMISSED,
        ACTION_STATUS_FAILED,
    },
    ACTION_STATUS_APPROVED: {ACTION_STATUS_MOCK_EXECUTED, ACTION_STATUS_FAILED},
    ACTION_STATUS_REJECTED: set(),
    ACTION_STATUS_BLOCKED: set(),
    ACTION_STATUS_DISMISSED: set(),
    ACTION_STATUS_MOCK_EXECUTED: set(),
    ACTION_STATUS_FAILED: set(),
}

Clock = Callable[[], datetime]


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


class ChannelStore(Protocol):
    def upsert_message(
        self,
        message: ChannelMessage,
        *,
        session: Optional[ChannelSession] = None,
        actor: str = "system",
    ) -> "ChannelStoreWriteResult":
        """Store a normalized channel message idempotently."""

    def upsert_preview(self, preview: ChannelDraftPreview, *, actor: str = "system") -> "ChannelPreviewStoreResult":
        """Store a draft preview and its recommended actions idempotently."""

    def transition_draft(
        self,
        draft_id: str,
        status: str,
        *,
        actor: str,
        reason: Optional[str] = None,
        manager_context: Optional[Mapping[str, Any]] = None,
    ) -> "ChannelDraftRecord":
        """Move a draft through the approval lifecycle without sending anything live."""

    def transition_action(
        self,
        action_idempotency_key: str,
        status: str,
        *,
        actor: str,
        reason: Optional[str] = None,
    ) -> "ChannelActionRecord":
        """Move a recommended action through the review lifecycle without executing it live."""


@dataclass(frozen=True)
class ChannelStoreWriteResult:
    record_type: str
    record_id: str
    created: bool
    status: str
    history_event_id: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "record_type", normalize_key(self.record_type, "record_type"))
        object.__setattr__(self, "record_id", require_text(self.record_id, "record_id"))
        object.__setattr__(self, "status", normalize_key(self.status, "status"))
        object.__setattr__(self, "history_event_id", optional_text(self.history_event_id))

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ChannelPreviewStoreResult:
    draft_id: str
    preview_idempotency_key: str
    created: bool
    message_created: bool
    actions_total: int
    actions_created: int
    history_events_created: int
    status: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "draft_id", require_text(self.draft_id, "draft_id"))
        object.__setattr__(
            self,
            "preview_idempotency_key",
            require_text(self.preview_idempotency_key, "preview_idempotency_key"),
        )
        object.__setattr__(self, "status", validate_draft_status(self.status))
        if self.actions_total < 0 or self.actions_created < 0 or self.history_events_created < 0:
            raise ValueError("preview store counters must not be negative")

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ChannelMessageRecord:
    message: ChannelMessage
    session_key: str
    inserted_at: datetime
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.message, ChannelMessage):
            raise TypeError("message must be ChannelMessage")
        object.__setattr__(self, "session_key", require_text(self.session_key, "session_key"))
        require_timezone(self.inserted_at, "inserted_at")
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def idempotency_key(self) -> str:
        return self.message.idempotency_key

    def to_json_dict(self, *, include_raw_payload: bool = False) -> Mapping[str, Any]:
        message_payload = self.message.to_json_dict()
        if not include_raw_payload:
            message_payload = dict(message_payload)
            message_payload.pop("raw_payload", None)
        return {
            "schema_version": CHANNEL_STORAGE_SCHEMA_VERSION,
            "idempotency_key": self.idempotency_key,
            "session_key": self.session_key,
            "inserted_at": self.inserted_at.isoformat(),
            "message": message_payload,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ChannelDraftRecord:
    preview: ChannelDraftPreview
    status: str = DRAFT_STATUS_NEEDS_REVIEW
    created_at: datetime = field(default_factory=now_utc)
    updated_at: datetime = field(default_factory=now_utc)
    approved_by: Optional[str] = None
    rejected_by: Optional[str] = None
    status_reason: Optional[str] = None
    manager_context: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.preview, ChannelDraftPreview):
            raise TypeError("preview must be ChannelDraftPreview")
        object.__setattr__(self, "status", validate_draft_status(self.status))
        require_timezone(self.created_at, "created_at")
        require_timezone(self.updated_at, "updated_at")
        object.__setattr__(self, "approved_by", optional_text(self.approved_by))
        object.__setattr__(self, "rejected_by", optional_text(self.rejected_by))
        object.__setattr__(self, "status_reason", optional_text(self.status_reason))
        object.__setattr__(self, "manager_context", dict(self.manager_context))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def draft_id(self) -> str:
        return self.preview.draft_id

    @property
    def idempotency_key(self) -> str:
        return self.preview.idempotency_key

    @property
    def session_key(self) -> str:
        return self.preview.session.session_key

    def with_status(
        self,
        status: str,
        *,
        actor: str,
        reason: Optional[str],
        manager_context: Optional[Mapping[str, Any]],
        now: datetime,
    ) -> "ChannelDraftRecord":
        normalized_status = validate_draft_status(status)
        actor_text = require_text(actor, "actor")
        approved_by = actor_text if normalized_status in {DRAFT_STATUS_APPROVED, DRAFT_STATUS_MOCK_SENT} else self.approved_by
        rejected_by = actor_text if normalized_status == DRAFT_STATUS_REJECTED else self.rejected_by
        return ChannelDraftRecord(
            preview=self.preview,
            status=normalized_status,
            created_at=self.created_at,
            updated_at=now,
            approved_by=approved_by,
            rejected_by=rejected_by,
            status_reason=reason,
            manager_context=manager_context if manager_context is not None else self.manager_context,
            metadata=self.metadata,
        )

    def to_json_dict(self, *, include_raw_payload: bool = False) -> Mapping[str, Any]:
        return {
            "schema_version": CHANNEL_STORAGE_SCHEMA_VERSION,
            "draft_id": self.draft_id,
            "idempotency_key": self.idempotency_key,
            "session_key": self.session_key,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "approved_by": self.approved_by,
            "rejected_by": self.rejected_by,
            "status_reason": self.status_reason,
            "manager_context": dict(self.manager_context),
            "preview": self.preview.to_json_dict(include_raw_payload=include_raw_payload),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ChannelActionRecord:
    action: RecommendedAction
    draft_id: str
    session_key: str
    source_message_idempotency_key: str
    status: str = ACTION_STATUS_PROPOSED
    created_at: datetime = field(default_factory=now_utc)
    updated_at: datetime = field(default_factory=now_utc)
    status_reason: Optional[str] = None
    actor: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.action, RecommendedAction):
            raise TypeError("action must be RecommendedAction")
        object.__setattr__(self, "draft_id", require_text(self.draft_id, "draft_id"))
        object.__setattr__(self, "session_key", require_text(self.session_key, "session_key"))
        object.__setattr__(
            self,
            "source_message_idempotency_key",
            require_text(self.source_message_idempotency_key, "source_message_idempotency_key"),
        )
        object.__setattr__(self, "status", validate_action_status(self.status))
        require_timezone(self.created_at, "created_at")
        require_timezone(self.updated_at, "updated_at")
        object.__setattr__(self, "status_reason", optional_text(self.status_reason))
        object.__setattr__(self, "actor", optional_text(self.actor))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def idempotency_key(self) -> str:
        return self.action.idempotency_key or ""

    def with_status(self, status: str, *, actor: str, reason: Optional[str], now: datetime) -> "ChannelActionRecord":
        return ChannelActionRecord(
            action=self.action,
            draft_id=self.draft_id,
            session_key=self.session_key,
            source_message_idempotency_key=self.source_message_idempotency_key,
            status=validate_action_status(status),
            created_at=self.created_at,
            updated_at=now,
            status_reason=reason,
            actor=require_text(actor, "actor"),
            metadata=self.metadata,
        )

    def to_json_dict(self) -> Mapping[str, Any]:
        return {
            "schema_version": CHANNEL_STORAGE_SCHEMA_VERSION,
            "idempotency_key": self.idempotency_key,
            "draft_id": self.draft_id,
            "session_key": self.session_key,
            "source_message_idempotency_key": self.source_message_idempotency_key,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "status_reason": self.status_reason,
            "actor": self.actor,
            "action": self.action.to_json_dict(),
            "agent_action_proposal": asdict(recommended_action_to_agent_proposal(self.action)),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ChannelHistoryEvent:
    event_id: str
    event_type: str
    entity_type: str
    entity_id: str
    session_key: Optional[str]
    created_at: datetime
    actor: str
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_id", require_text(self.event_id, "event_id"))
        object.__setattr__(self, "event_type", normalize_key(self.event_type, "event_type"))
        object.__setattr__(self, "entity_type", normalize_key(self.entity_type, "entity_type"))
        object.__setattr__(self, "entity_id", require_text(self.entity_id, "entity_id"))
        object.__setattr__(self, "session_key", optional_text(self.session_key))
        require_timezone(self.created_at, "created_at")
        object.__setattr__(self, "actor", require_text(self.actor, "actor"))
        object.__setattr__(self, "payload", dict(self.payload))

    def to_json_dict(self) -> Mapping[str, Any]:
        return {
            "schema_version": CHANNEL_STORAGE_SCHEMA_VERSION,
            "event_id": self.event_id,
            "event_type": self.event_type,
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "session_key": self.session_key,
            "created_at": self.created_at.isoformat(),
            "actor": self.actor,
            "payload": dict(self.payload),
        }


class ChannelMemoryStore:
    """Reference channel store for tests and future UI/approval workflow.

    The store is intentionally process-local: it does not touch runtime DB,
    CRM, channel APIs, filesystem, or Telegram. Persistent storage can be
    added later behind the same `ChannelStore` protocol.
    """

    def __init__(self, *, clock: Optional[Clock] = None) -> None:
        self._clock = clock or now_utc
        self._sessions: dict[str, ChannelSession] = {}
        self._messages: dict[str, ChannelMessageRecord] = {}
        self._drafts_by_id: dict[str, ChannelDraftRecord] = {}
        self._draft_ids_by_preview_key: dict[str, str] = {}
        self._actions: dict[str, ChannelActionRecord] = {}
        self._history: list[ChannelHistoryEvent] = []
        self._event_seq = 0

    def upsert_session(self, session: ChannelSession, *, actor: str = "system") -> ChannelStoreWriteResult:
        now = self._now()
        key = session.session_key
        existing = self._sessions.get(key)
        self._sessions[key] = session
        if existing is None:
            event = self._append_history(
                event_type="session_created",
                entity_type="channel_session",
                entity_id=key,
                session_key=key,
                actor=actor,
                payload={"channel": session.channel, "channel_thread_id": session.channel_thread_id},
                now=now,
            )
            return ChannelStoreWriteResult("channel_session", key, True, "created", event.event_id)
        if existing.to_json_dict() == session.to_json_dict():
            return ChannelStoreWriteResult("channel_session", key, False, "duplicate", None)
        event = self._append_history(
            event_type="session_updated",
            entity_type="channel_session",
            entity_id=key,
            session_key=key,
            actor=actor,
            payload={"channel": session.channel, "channel_thread_id": session.channel_thread_id},
            now=now,
        )
        return ChannelStoreWriteResult("channel_session", key, False, "updated", event.event_id)

    def upsert_message(
        self,
        message: ChannelMessage,
        *,
        session: Optional[ChannelSession] = None,
        actor: str = "system",
    ) -> ChannelStoreWriteResult:
        resolved_session = session or ChannelSession.from_message(message)
        ensure_message_session_match(message, resolved_session)
        self._sessions[resolved_session.session_key] = resolved_session
        key = message.idempotency_key
        existing = self._messages.get(key)
        if existing is not None:
            return ChannelStoreWriteResult("channel_message", key, False, "duplicate", None)
        now = self._now()
        self._messages[key] = ChannelMessageRecord(
            message=message,
            session_key=resolved_session.session_key,
            inserted_at=now,
            metadata={"source": "channel_store"},
        )
        event = self._append_history(
            event_type="message_received" if message.direction.value == "inbound" else "message_recorded",
            entity_type="channel_message",
            entity_id=key,
            session_key=resolved_session.session_key,
            actor=actor,
            payload={
                "channel": message.channel,
                "channel_thread_id": message.channel_thread_id,
                "direction": message.direction.value,
            },
            now=now,
        )
        return ChannelStoreWriteResult("channel_message", key, True, "created", event.event_id)

    def upsert_preview(self, preview: ChannelDraftPreview, *, actor: str = "system") -> ChannelPreviewStoreResult:
        history_before = len(self._history)
        message_result = self.upsert_message(preview.source_message, session=preview.session, actor=actor)
        preview_key = preview.idempotency_key
        existing_draft_id = self._draft_ids_by_preview_key.get(preview_key)
        if existing_draft_id is not None:
            existing = self._drafts_by_id[existing_draft_id]
            return ChannelPreviewStoreResult(
                draft_id=existing.draft_id,
                preview_idempotency_key=preview_key,
                created=False,
                message_created=message_result.created,
                actions_total=len(existing.preview.reply.recommended_actions),
                actions_created=0,
                history_events_created=len(self._history) - history_before,
                status=existing.status,
            )

        now = self._now()
        draft = ChannelDraftRecord(
            preview=preview,
            status=DRAFT_STATUS_NEEDS_REVIEW,
            created_at=now,
            updated_at=now,
            manager_context=build_manager_visible_context(preview),
            metadata={"source": "channel_preview_service"},
        )
        self._drafts_by_id[draft.draft_id] = draft
        self._draft_ids_by_preview_key[preview_key] = draft.draft_id
        self._append_history(
            event_type="draft_created",
            entity_type="channel_draft",
            entity_id=draft.draft_id,
            session_key=draft.session_key,
            actor=actor,
            payload={
                "preview_idempotency_key": preview_key,
                "status": draft.status,
                "recommended_actions": len(preview.reply.recommended_actions),
            },
            now=now,
        )
        actions_created = 0
        for action in preview.reply.recommended_actions:
            if self._upsert_action(action, draft=draft, actor=actor):
                actions_created += 1
        return ChannelPreviewStoreResult(
            draft_id=draft.draft_id,
            preview_idempotency_key=preview_key,
            created=True,
            message_created=message_result.created,
            actions_total=len(preview.reply.recommended_actions),
            actions_created=actions_created,
            history_events_created=len(self._history) - history_before,
            status=draft.status,
        )

    def transition_draft(
        self,
        draft_id: str,
        status: str,
        *,
        actor: str,
        reason: Optional[str] = None,
        manager_context: Optional[Mapping[str, Any]] = None,
    ) -> ChannelDraftRecord:
        normalized_status = validate_draft_status(status)
        key = require_text(draft_id, "draft_id")
        current = self._drafts_by_id.get(key)
        if current is None:
            raise KeyError(f"unknown draft_id: {key}")
        reason_text = optional_text(reason)
        if (
            current.status == normalized_status
            and current.status_reason == reason_text
            and (manager_context is None or dict(manager_context) == dict(current.manager_context))
        ):
            return current
        validate_draft_transition(current.status, normalized_status)
        now = self._now()
        updated = current.with_status(
            normalized_status,
            actor=actor,
            reason=reason_text,
            manager_context=manager_context,
            now=now,
        )
        self._drafts_by_id[key] = updated
        self._append_history(
            event_type="draft_status_changed",
            entity_type="channel_draft",
            entity_id=key,
            session_key=updated.session_key,
            actor=actor,
            payload={
                "from_status": current.status,
                "to_status": updated.status,
                "reason": reason_text,
                "live_send_executed": False,
            },
            now=now,
        )
        return updated

    def transition_action(
        self,
        action_idempotency_key: str,
        status: str,
        *,
        actor: str,
        reason: Optional[str] = None,
    ) -> ChannelActionRecord:
        normalized_status = validate_action_status(status)
        key = require_text(action_idempotency_key, "action_idempotency_key")
        current = self._actions.get(key)
        if current is None:
            raise KeyError(f"unknown action idempotency key: {key}")
        reason_text = optional_text(reason)
        if current.status == normalized_status and current.status_reason == reason_text and current.actor == actor:
            return current
        validate_action_transition(current.status, normalized_status)
        now = self._now()
        updated = current.with_status(normalized_status, actor=actor, reason=reason_text, now=now)
        self._actions[key] = updated
        self._append_history(
            event_type="action_status_changed",
            entity_type="recommended_action",
            entity_id=key,
            session_key=updated.session_key,
            actor=actor,
            payload={
                "from_status": current.status,
                "to_status": updated.status,
                "reason": reason_text,
                "live_execution_performed": False,
            },
            now=now,
        )
        return updated

    def get_session(self, session_key: str) -> Optional[ChannelSession]:
        return self._sessions.get(require_text(session_key, "session_key"))

    def get_message(self, idempotency_key: str) -> Optional[ChannelMessageRecord]:
        return self._messages.get(require_text(idempotency_key, "idempotency_key"))

    def get_draft(self, draft_id: str) -> Optional[ChannelDraftRecord]:
        return self._drafts_by_id.get(require_text(draft_id, "draft_id"))

    def get_action(self, idempotency_key: str) -> Optional[ChannelActionRecord]:
        return self._actions.get(require_text(idempotency_key, "idempotency_key"))

    def list_drafts(
        self,
        *,
        status: Optional[str] = None,
        session_key: Optional[str] = None,
    ) -> tuple[ChannelDraftRecord, ...]:
        normalized_status = validate_draft_status(status) if status is not None else None
        session_filter = optional_text(session_key)
        rows = self._drafts_by_id.values()
        if normalized_status:
            rows = (row for row in rows if row.status == normalized_status)
        if session_filter:
            rows = (row for row in rows if row.session_key == session_filter)
        return tuple(sorted(rows, key=lambda row: (row.created_at, row.draft_id)))

    def list_actions(
        self,
        *,
        status: Optional[str] = None,
        session_key: Optional[str] = None,
        draft_id: Optional[str] = None,
    ) -> tuple[ChannelActionRecord, ...]:
        normalized_status = validate_action_status(status) if status is not None else None
        session_filter = optional_text(session_key)
        draft_filter = optional_text(draft_id)
        rows = self._actions.values()
        if normalized_status:
            rows = (row for row in rows if row.status == normalized_status)
        if session_filter:
            rows = (row for row in rows if row.session_key == session_filter)
        if draft_filter:
            rows = (row for row in rows if row.draft_id == draft_filter)
        return tuple(sorted(rows, key=lambda row: (row.created_at, row.idempotency_key)))

    def list_history(
        self,
        *,
        session_key: Optional[str] = None,
        entity_type: Optional[str] = None,
        entity_id: Optional[str] = None,
    ) -> tuple[ChannelHistoryEvent, ...]:
        session_filter = optional_text(session_key)
        entity_type_filter = normalize_key(entity_type, "entity_type") if entity_type is not None else None
        entity_id_filter = optional_text(entity_id)
        rows = self._history
        if session_filter:
            rows = [row for row in rows if row.session_key == session_filter]
        if entity_type_filter:
            rows = [row for row in rows if row.entity_type == entity_type_filter]
        if entity_id_filter:
            rows = [row for row in rows if row.entity_id == entity_id_filter]
        return tuple(rows)

    def snapshot(self, *, include_raw_payload: bool = False) -> Mapping[str, Any]:
        return {
            "schema_version": CHANNEL_STORAGE_SCHEMA_VERSION,
            "summary": self.summary(),
            "sessions": [session.to_json_dict() for session in self._sessions.values()],
            "messages": [
                record.to_json_dict(include_raw_payload=include_raw_payload)
                for record in sorted(self._messages.values(), key=lambda row: (row.inserted_at, row.idempotency_key))
            ],
            "drafts": [
                record.to_json_dict(include_raw_payload=include_raw_payload)
                for record in sorted(self._drafts_by_id.values(), key=lambda row: (row.created_at, row.draft_id))
            ],
            "actions": [record.to_json_dict() for record in self.list_actions()],
            "history": [event.to_json_dict() for event in self._history],
            "safety": channel_store_safety_contract(),
        }

    def summary(self) -> Mapping[str, Any]:
        draft_status_counts: dict[str, int] = {}
        for draft in self._drafts_by_id.values():
            draft_status_counts[draft.status] = draft_status_counts.get(draft.status, 0) + 1
        action_status_counts: dict[str, int] = {}
        for action in self._actions.values():
            action_status_counts[action.status] = action_status_counts.get(action.status, 0) + 1
        return {
            "schema_version": CHANNEL_STORAGE_SCHEMA_VERSION,
            "sessions": len(self._sessions),
            "messages": len(self._messages),
            "drafts": len(self._drafts_by_id),
            "actions": len(self._actions),
            "history_events": len(self._history),
            "draft_status_counts": draft_status_counts,
            "action_status_counts": action_status_counts,
            "validation_ok": True,
            "blocked": 0,
            "warnings": 0,
        }

    def _upsert_action(self, action: RecommendedAction, *, draft: ChannelDraftRecord, actor: str) -> bool:
        key = require_text(action.idempotency_key, "action.idempotency_key")
        if key in self._actions:
            return False
        now = self._now()
        record = ChannelActionRecord(
            action=action,
            draft_id=draft.draft_id,
            session_key=draft.session_key,
            source_message_idempotency_key=draft.preview.source_message.idempotency_key,
            status=ACTION_STATUS_PROPOSED,
            created_at=now,
            updated_at=now,
            metadata={"source": "channel_preview_recommended_action"},
        )
        self._actions[key] = record
        self._append_history(
            event_type="action_proposed",
            entity_type="recommended_action",
            entity_id=key,
            session_key=draft.session_key,
            actor=actor,
            payload={
                "draft_id": draft.draft_id,
                "action_type": action.action_type,
                "requires_approval": action.requires_approval,
            },
            now=now,
        )
        return True

    def _append_history(
        self,
        *,
        event_type: str,
        entity_type: str,
        entity_id: str,
        session_key: Optional[str],
        actor: str,
        payload: Mapping[str, Any],
        now: datetime,
    ) -> ChannelHistoryEvent:
        self._event_seq += 1
        event = ChannelHistoryEvent(
            event_id=stable_history_event_id(
                seq=self._event_seq,
                event_type=event_type,
                entity_type=entity_type,
                entity_id=entity_id,
                created_at=now,
            ),
            event_type=event_type,
            entity_type=entity_type,
            entity_id=entity_id,
            session_key=session_key,
            created_at=now,
            actor=actor,
            payload=payload,
        )
        self._history.append(event)
        return event

    def _now(self) -> datetime:
        value = self._clock()
        require_timezone(value, "clock value")
        return value


def build_and_store_channel_draft_preview(
    store: ChannelStore,
    message: ChannelMessage,
    *,
    service: Optional[ChannelPreviewService] = None,
    session: Optional[ChannelSession] = None,
    context: Optional[Mapping[str, Any]] = None,
    actor: str = "system",
) -> tuple[ChannelDraftPreview, ChannelPreviewStoreResult]:
    preview = (service or ChannelPreviewService()).build_preview(message, session=session, context=context)
    result = store.upsert_preview(preview, actor=actor)
    return preview, result


def build_manager_visible_context(preview: ChannelDraftPreview) -> Mapping[str, Any]:
    message = preview.source_message
    return {
        "channel": message.channel,
        "channel_thread_id": message.channel_thread_id,
        "channel_user_id": message.channel_user_id,
        "source_message_id": message.channel_message_id,
        "source_message_text": message.text,
        "draft_text": preview.reply.text,
        "requires_approval": preview.reply.requires_approval,
        "recommended_action_count": len(preview.reply.recommended_actions),
        "safety_flags": list(preview.reply.safety_flags),
        "blocked_reasons": list(preview.blocked_reasons),
    }


def channel_store_safety_contract() -> Mapping[str, bool]:
    return {
        "network_calls": False,
        "llm_calls": False,
        "live_send": False,
        "write_crm": False,
        "write_tallanto": False,
        "write_runtime_db": False,
        "run_asr": False,
        "run_ra": False,
        "stores_raw_payload_by_default": False,
    }


def validate_draft_status(status: str) -> str:
    normalized = normalize_key(status, "draft status")
    if normalized in LIVE_EXECUTION_STATUSES:
        raise ValueError("real live-send draft statuses are not allowed in channel storage")
    if normalized not in ALLOWED_DRAFT_STATUSES:
        raise ValueError(f"unsupported draft status: {status!r}")
    return normalized


def validate_action_status(status: str) -> str:
    normalized = normalize_key(status, "action status")
    if normalized in LIVE_EXECUTION_STATUSES:
        raise ValueError("real action execution statuses are not allowed in channel storage")
    if normalized not in ALLOWED_ACTION_STATUSES:
        raise ValueError(f"unsupported action status: {status!r}")
    return normalized


def validate_draft_transition(current_status: str, next_status: str) -> None:
    current = validate_draft_status(current_status)
    next_value = validate_draft_status(next_status)
    allowed = DRAFT_STATUS_TRANSITIONS[current]
    if next_value not in allowed:
        raise ValueError(f"unsupported draft transition: {current} -> {next_value}")


def validate_action_transition(current_status: str, next_status: str) -> None:
    current = validate_action_status(current_status)
    next_value = validate_action_status(next_status)
    allowed = ACTION_STATUS_TRANSITIONS[current]
    if next_value not in allowed:
        raise ValueError(f"unsupported action transition: {current} -> {next_value}")


def ensure_message_session_match(message: ChannelMessage, session: ChannelSession) -> None:
    if message.channel != session.channel:
        raise ValueError("session channel must match source message channel")
    if message.channel_thread_id != session.channel_thread_id:
        raise ValueError("session thread must match source message thread")


def stable_history_event_id(
    *,
    seq: int,
    event_type: str,
    entity_type: str,
    entity_id: str,
    created_at: datetime,
) -> str:
    digest = stable_digest(
        {
            "seq": seq,
            "event_type": event_type,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "created_at": created_at.isoformat(),
        }
    )
    return f"channel_history:{seq:08d}:{digest[:24]}"
