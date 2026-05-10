from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Optional, Sequence

from mango_mvp.channels.contracts import normalize_key, optional_text, require_text, require_timezone, stable_digest
from mango_mvp.channels.signals import SignalDecision
from mango_mvp.channels.storage import ChannelActionRecord, ChannelDraftRecord


CHANNEL_FEEDBACK_SCHEMA_VERSION = "channel_feedback_v1"

FEEDBACK_MANAGER_DRAFT_APPROVED = "manager_draft_approved"
FEEDBACK_MANAGER_DRAFT_REJECTED = "manager_draft_rejected"
FEEDBACK_MANAGER_DRAFT_EDITED = "manager_draft_edited"
FEEDBACK_ACTION_ACCEPTED = "action_accepted"
FEEDBACK_ACTION_DISMISSED = "action_dismissed"
FEEDBACK_CLIENT_REPLIED = "client_replied"
FEEDBACK_CLIENT_NO_REPLY = "client_no_reply"
FEEDBACK_FOLLOW_UP_DONE = "follow_up_done"
FEEDBACK_FOLLOW_UP_OVERDUE = "follow_up_overdue"
FEEDBACK_LEAD_MOVED = "lead_moved"
FEEDBACK_LEAD_WON = "lead_won"
FEEDBACK_LEAD_LOST = "lead_lost"
FEEDBACK_ROP_ANSWER_GOOD = "rop_answer_good"
FEEDBACK_ROP_ANSWER_RISKY = "rop_answer_risky"

DRAFT_FEEDBACK_EVENTS = {
    FEEDBACK_MANAGER_DRAFT_APPROVED,
    FEEDBACK_MANAGER_DRAFT_REJECTED,
    FEEDBACK_MANAGER_DRAFT_EDITED,
}
ACTION_FEEDBACK_EVENTS = {
    FEEDBACK_ACTION_ACCEPTED,
    FEEDBACK_ACTION_DISMISSED,
}
CLIENT_FEEDBACK_EVENTS = {
    FEEDBACK_CLIENT_REPLIED,
    FEEDBACK_CLIENT_NO_REPLY,
}
FOLLOW_UP_FEEDBACK_EVENTS = {
    FEEDBACK_FOLLOW_UP_DONE,
    FEEDBACK_FOLLOW_UP_OVERDUE,
}
READ_ONLY_LEAD_OUTCOME_EVENTS = {
    FEEDBACK_LEAD_MOVED,
    FEEDBACK_LEAD_WON,
    FEEDBACK_LEAD_LOST,
}
ROP_FEEDBACK_EVENTS = {
    FEEDBACK_ROP_ANSWER_GOOD,
    FEEDBACK_ROP_ANSWER_RISKY,
}
ALLOWED_FEEDBACK_EVENTS = (
    FEEDBACK_MANAGER_DRAFT_APPROVED,
    FEEDBACK_MANAGER_DRAFT_REJECTED,
    FEEDBACK_MANAGER_DRAFT_EDITED,
    FEEDBACK_ACTION_ACCEPTED,
    FEEDBACK_ACTION_DISMISSED,
    FEEDBACK_CLIENT_REPLIED,
    FEEDBACK_CLIENT_NO_REPLY,
    FEEDBACK_FOLLOW_UP_DONE,
    FEEDBACK_FOLLOW_UP_OVERDUE,
    FEEDBACK_LEAD_MOVED,
    FEEDBACK_LEAD_WON,
    FEEDBACK_LEAD_LOST,
    FEEDBACK_ROP_ANSWER_GOOD,
    FEEDBACK_ROP_ANSWER_RISKY,
)
POSITIVE_FEEDBACK_EVENTS = {
    FEEDBACK_MANAGER_DRAFT_APPROVED,
    FEEDBACK_ACTION_ACCEPTED,
    FEEDBACK_CLIENT_REPLIED,
    FEEDBACK_FOLLOW_UP_DONE,
    FEEDBACK_LEAD_WON,
    FEEDBACK_ROP_ANSWER_GOOD,
}
RISK_FEEDBACK_EVENTS = {
    FEEDBACK_MANAGER_DRAFT_REJECTED,
    FEEDBACK_ACTION_DISMISSED,
    FEEDBACK_CLIENT_NO_REPLY,
    FEEDBACK_FOLLOW_UP_OVERDUE,
    FEEDBACK_LEAD_LOST,
    FEEDBACK_ROP_ANSWER_RISKY,
}
EXTERNAL_EFFECT_METADATA_KEYS = {
    "live_send",
    "live_send_executed",
    "message_sent",
    "channel_api_called",
    "crm_write",
    "crm_write_executed",
    "write_crm",
    "write_tallanto",
    "tallanto_write_executed",
    "runtime_db_write",
    "runtime_db_write_executed",
    "network_calls",
    "network_call_executed",
    "asr_executed",
    "run_asr",
    "ra_executed",
    "run_ra",
}
FORBIDDEN_RAW_METADATA_KEYS = {
    "raw_payload",
    "provider_raw_payload",
    "webhook_payload",
    "telegram_update",
}

Clock = Callable[[], datetime]


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class FeedbackEvent:
    event_type: str
    session_key: str
    actor: str
    occurred_at: datetime = field(default_factory=now_utc)
    entity_type: Optional[str] = None
    entity_id: Optional[str] = None
    message_idempotency_key: Optional[str] = None
    decision_id: Optional[str] = None
    draft_id: Optional[str] = None
    action_idempotency_key: Optional[str] = None
    source_system: Optional[str] = None
    imported_read_only: bool = False
    value: Optional[str] = None
    score: Optional[float] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    idempotency_key: Optional[str] = None

    def __post_init__(self) -> None:
        event_type = validate_feedback_event_type(self.event_type)
        session_key = require_text(self.session_key, "session_key")
        actor = require_text(self.actor, "actor")
        require_timezone(self.occurred_at, "occurred_at")

        message_key = optional_text(self.message_idempotency_key)
        decision_id = optional_text(self.decision_id)
        draft_id = optional_text(self.draft_id)
        action_key = optional_text(self.action_idempotency_key)
        source_system = normalize_key(self.source_system, "source_system") if self.source_system else None
        value = optional_text(self.value)
        metadata = dict(self.metadata)
        validate_feedback_metadata(metadata)

        if event_type in DRAFT_FEEDBACK_EVENTS and not draft_id:
            raise ValueError("draft feedback events require draft_id")
        if event_type in ACTION_FEEDBACK_EVENTS and not action_key:
            raise ValueError("action feedback events require action_idempotency_key")
        if event_type == FEEDBACK_MANAGER_DRAFT_EDITED and not value:
            raise ValueError("manager_draft_edited requires edited draft text in value")
        if event_type in READ_ONLY_LEAD_OUTCOME_EVENTS:
            if not self.imported_read_only:
                raise ValueError("lead outcome feedback must be imported as read-only")
            if not source_system:
                raise ValueError("lead outcome feedback requires source_system")
            if not optional_text(self.entity_id):
                raise ValueError("lead outcome feedback requires entity_id")

        score = None if self.score is None else float(self.score)
        if score is not None and not 0 <= score <= 1:
            raise ValueError("feedback score must be between 0 and 1")

        entity_type, entity_id = resolve_feedback_entity(
            event_type=event_type,
            entity_type=self.entity_type,
            entity_id=self.entity_id,
            decision_id=decision_id,
            message_idempotency_key=message_key,
            draft_id=draft_id,
            action_idempotency_key=action_key,
        )
        key = optional_text(self.idempotency_key) or stable_feedback_event_id(
            event_type=event_type,
            session_key=session_key,
            actor=actor,
            occurred_at=self.occurred_at,
            entity_type=entity_type,
            entity_id=entity_id,
            message_idempotency_key=message_key,
            decision_id=decision_id,
            draft_id=draft_id,
            action_idempotency_key=action_key,
            source_system=source_system,
            imported_read_only=self.imported_read_only,
            value=value,
            score=score,
            metadata=metadata,
        )

        object.__setattr__(self, "event_type", event_type)
        object.__setattr__(self, "session_key", session_key)
        object.__setattr__(self, "actor", actor)
        object.__setattr__(self, "entity_type", entity_type)
        object.__setattr__(self, "entity_id", entity_id)
        object.__setattr__(self, "message_idempotency_key", message_key)
        object.__setattr__(self, "decision_id", decision_id)
        object.__setattr__(self, "draft_id", draft_id)
        object.__setattr__(self, "action_idempotency_key", action_key)
        object.__setattr__(self, "source_system", source_system)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "idempotency_key", key)

    @property
    def sentiment_bucket(self) -> str:
        if self.event_type in POSITIVE_FEEDBACK_EVENTS:
            return "positive"
        if self.event_type in RISK_FEEDBACK_EVENTS:
            return "risk"
        return "neutral"

    def to_json_dict(self) -> Mapping[str, Any]:
        return {
            "schema_version": CHANNEL_FEEDBACK_SCHEMA_VERSION,
            "event_type": self.event_type,
            "session_key": self.session_key,
            "actor": self.actor,
            "occurred_at": self.occurred_at.isoformat(),
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "message_idempotency_key": self.message_idempotency_key,
            "decision_id": self.decision_id,
            "draft_id": self.draft_id,
            "action_idempotency_key": self.action_idempotency_key,
            "source_system": self.source_system,
            "imported_read_only": self.imported_read_only,
            "value": self.value,
            "score": self.score,
            "metadata": dict(self.metadata),
            "sentiment_bucket": self.sentiment_bucket,
            "idempotency_key": self.idempotency_key,
        }


@dataclass(frozen=True)
class FeedbackStoreResult:
    event_id: str
    created: bool
    status: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_id", require_text(self.event_id, "event_id"))
        object.__setattr__(self, "status", normalize_key(self.status, "status"))

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ChannelFeedbackReport:
    report_id: str
    session_key: Optional[str]
    decision_id: Optional[str]
    created_at: datetime
    metrics: Mapping[str, Any]
    events: Sequence[FeedbackEvent] = field(default_factory=tuple)
    safety: Mapping[str, bool] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "report_id", require_text(self.report_id, "report_id"))
        object.__setattr__(self, "session_key", optional_text(self.session_key))
        object.__setattr__(self, "decision_id", optional_text(self.decision_id))
        require_timezone(self.created_at, "created_at")
        events = tuple(self.events)
        if any(not isinstance(item, FeedbackEvent) for item in events):
            raise TypeError("events must contain FeedbackEvent items")
        object.__setattr__(self, "events", events)
        object.__setattr__(self, "metrics", dict(self.metrics))
        object.__setattr__(self, "safety", dict(self.safety or feedback_loop_safety_contract()))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_json_dict(self) -> Mapping[str, Any]:
        return {
            "schema_version": CHANNEL_FEEDBACK_SCHEMA_VERSION,
            "report_id": self.report_id,
            "session_key": self.session_key,
            "decision_id": self.decision_id,
            "created_at": self.created_at.isoformat(),
            "metrics": dict(self.metrics),
            "events": [event.to_json_dict() for event in self.events],
            "safety": dict(self.safety),
            "metadata": dict(self.metadata),
        }


class ChannelFeedbackMemoryStore:
    """Process-local feedback loop store for channel revenue learning.

    It records product-level feedback facts only. It does not call channel APIs,
    CRM, Tallanto, ASR, R+A, runtime DB, or filesystem.
    """

    def __init__(self, *, clock: Optional[Clock] = None) -> None:
        self._clock = clock or now_utc
        self._events: dict[str, FeedbackEvent] = {}

    def record_event(self, event: FeedbackEvent) -> FeedbackStoreResult:
        key = require_text(event.idempotency_key, "event.idempotency_key")
        if key in self._events:
            return FeedbackStoreResult(event_id=key, created=False, status="duplicate")
        self._events[key] = event
        return FeedbackStoreResult(event_id=key, created=True, status="created")

    def record_many(self, events: Sequence[FeedbackEvent]) -> tuple[FeedbackStoreResult, ...]:
        return tuple(self.record_event(event) for event in events)

    def get_event(self, idempotency_key: str) -> Optional[FeedbackEvent]:
        return self._events.get(require_text(idempotency_key, "idempotency_key"))

    def list_events(
        self,
        *,
        session_key: Optional[str] = None,
        event_type: Optional[str] = None,
        decision_id: Optional[str] = None,
        entity_type: Optional[str] = None,
    ) -> tuple[FeedbackEvent, ...]:
        session_filter = optional_text(session_key)
        event_filter = validate_feedback_event_type(event_type) if event_type is not None else None
        decision_filter = optional_text(decision_id)
        entity_filter = normalize_key(entity_type, "entity_type") if entity_type is not None else None
        rows = self._events.values()
        if session_filter:
            rows = (event for event in rows if event.session_key == session_filter)
        if event_filter:
            rows = (event for event in rows if event.event_type == event_filter)
        if decision_filter:
            rows = (event for event in rows if event.decision_id == decision_filter)
        if entity_filter:
            rows = (event for event in rows if event.entity_type == entity_filter)
        return tuple(sorted(rows, key=lambda event: (event.occurred_at, event.idempotency_key)))

    def summary(self, *, session_key: Optional[str] = None) -> Mapping[str, Any]:
        return summarize_feedback_events(self.list_events(session_key=session_key))

    def build_report(
        self,
        *,
        session_key: Optional[str] = None,
        decision: Optional[SignalDecision] = None,
    ) -> ChannelFeedbackReport:
        resolved_session_key = optional_text(session_key) or (decision.session_key if decision else None)
        decision_id = decision.decision_id if decision else None
        events = self.list_events(session_key=resolved_session_key)
        if decision_id:
            events = tuple(event for event in events if event.decision_id in {None, decision_id})
        return build_feedback_loop_report(
            events=events,
            session_key=resolved_session_key,
            decision_id=decision_id,
            created_at=self._now(),
            metadata={
                "decision_present": decision is not None,
                "store": "channel_feedback_memory_store",
            },
        )

    def snapshot(self) -> Mapping[str, Any]:
        events = self.list_events()
        return {
            "schema_version": CHANNEL_FEEDBACK_SCHEMA_VERSION,
            "summary": summarize_feedback_events(events),
            "events": [event.to_json_dict() for event in events],
            "safety": feedback_loop_safety_contract(),
        }

    def _now(self) -> datetime:
        value = self._clock()
        require_timezone(value, "clock value")
        return value


def build_manager_draft_feedback_event(
    draft: ChannelDraftRecord,
    event_type: str,
    *,
    actor: str,
    occurred_at: Optional[datetime] = None,
    edited_text: Optional[str] = None,
    reason: Optional[str] = None,
    decision_id: Optional[str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> FeedbackEvent:
    if not isinstance(draft, ChannelDraftRecord):
        raise TypeError("draft must be ChannelDraftRecord")
    normalized = validate_feedback_event_type(event_type)
    if normalized not in DRAFT_FEEDBACK_EVENTS:
        raise ValueError("event_type must be a draft feedback event")
    payload = {
        "draft_status": draft.status,
        "source_message_idempotency_key": draft.preview.source_message.idempotency_key,
        "reason": optional_text(reason),
        **dict(metadata or {}),
    }
    return FeedbackEvent(
        event_type=normalized,
        session_key=draft.session_key,
        actor=actor,
        occurred_at=occurred_at or now_utc(),
        decision_id=decision_id,
        draft_id=draft.draft_id,
        message_idempotency_key=draft.preview.source_message.idempotency_key,
        value=edited_text,
        metadata=payload,
    )


def build_action_feedback_event(
    action: ChannelActionRecord,
    event_type: str,
    *,
    actor: str,
    occurred_at: Optional[datetime] = None,
    reason: Optional[str] = None,
    decision_id: Optional[str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> FeedbackEvent:
    if not isinstance(action, ChannelActionRecord):
        raise TypeError("action must be ChannelActionRecord")
    normalized = validate_feedback_event_type(event_type)
    if normalized not in ACTION_FEEDBACK_EVENTS:
        raise ValueError("event_type must be an action feedback event")
    payload = {
        "action_type": action.action.action_type,
        "action_status": action.status,
        "draft_id": action.draft_id,
        "reason": optional_text(reason),
        **dict(metadata or {}),
    }
    return FeedbackEvent(
        event_type=normalized,
        session_key=action.session_key,
        actor=actor,
        occurred_at=occurred_at or now_utc(),
        decision_id=decision_id,
        draft_id=action.draft_id,
        action_idempotency_key=action.idempotency_key,
        message_idempotency_key=action.source_message_idempotency_key,
        metadata=payload,
    )


def build_decision_feedback_event(
    decision: SignalDecision,
    event_type: str,
    *,
    actor: str,
    occurred_at: Optional[datetime] = None,
    value: Optional[str] = None,
    score: Optional[float] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> FeedbackEvent:
    if not isinstance(decision, SignalDecision):
        raise TypeError("decision must be SignalDecision")
    normalized = validate_feedback_event_type(event_type)
    if normalized in READ_ONLY_LEAD_OUTCOME_EVENTS:
        raise ValueError("lead outcomes must use build_read_only_lead_outcome_event")
    return FeedbackEvent(
        event_type=normalized,
        session_key=decision.session_key,
        actor=actor,
        occurred_at=occurred_at or now_utc(),
        decision_id=decision.decision_id,
        message_idempotency_key=decision.message_idempotency_key,
        value=value,
        score=score,
        metadata={
            "signal_types": tuple(signal.signal_type for signal in decision.signals),
            "recommended_action_types": tuple(decision.recommended_action_types),
            **dict(metadata or {}),
        },
    )


def build_read_only_lead_outcome_event(
    *,
    session_key: str,
    lead_id: str,
    event_type: str,
    source_system: str,
    actor: str,
    occurred_at: Optional[datetime] = None,
    decision_id: Optional[str] = None,
    value: Optional[str] = None,
    score: Optional[float] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> FeedbackEvent:
    normalized = validate_feedback_event_type(event_type)
    if normalized not in READ_ONLY_LEAD_OUTCOME_EVENTS:
        raise ValueError("event_type must be a read-only lead outcome event")
    return FeedbackEvent(
        event_type=normalized,
        session_key=session_key,
        actor=actor,
        occurred_at=occurred_at or now_utc(),
        entity_type="crm_lead",
        entity_id=lead_id,
        decision_id=decision_id,
        source_system=source_system,
        imported_read_only=True,
        value=value,
        score=score,
        metadata={
            "source_mode": "read_only_import",
            **dict(metadata or {}),
        },
    )


def build_feedback_loop_report(
    *,
    events: Sequence[FeedbackEvent],
    session_key: Optional[str] = None,
    decision_id: Optional[str] = None,
    created_at: Optional[datetime] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> ChannelFeedbackReport:
    event_items = tuple(events)
    if any(not isinstance(event, FeedbackEvent) for event in event_items):
        raise TypeError("events must contain FeedbackEvent items")
    sorted_events = tuple(sorted(event_items, key=lambda event: (event.occurred_at, event.idempotency_key)))
    resolved_session_key = optional_text(session_key)
    resolved_decision_id = optional_text(decision_id)
    if resolved_session_key and any(event.session_key != resolved_session_key for event in sorted_events):
        raise ValueError("all report events must match session_key")
    if resolved_decision_id:
        decision_event_ids = {event.decision_id for event in sorted_events if event.decision_id}
        if decision_event_ids and any(item != resolved_decision_id for item in decision_event_ids):
            raise ValueError("decision-linked report events must match decision_id")
    now = created_at or now_utc()
    require_timezone(now, "created_at")
    metrics = summarize_feedback_events(sorted_events)
    report_id = stable_feedback_report_id(
        session_key=resolved_session_key,
        decision_id=resolved_decision_id,
        event_ids=tuple(event.idempotency_key for event in sorted_events),
        created_at=now,
    )
    return ChannelFeedbackReport(
        report_id=report_id,
        session_key=resolved_session_key,
        decision_id=resolved_decision_id,
        created_at=now,
        metrics=metrics,
        events=sorted_events,
        safety=feedback_loop_safety_contract(),
        metadata=dict(metadata or {}),
    )


def summarize_feedback_events(events: Sequence[FeedbackEvent]) -> Mapping[str, Any]:
    event_items = tuple(events)
    if any(not isinstance(event, FeedbackEvent) for event in event_items):
        raise TypeError("events must contain FeedbackEvent items")
    type_counts = Counter(event.event_type for event in event_items)
    session_counts = Counter(event.session_key for event in event_items)
    positive_count = sum(1 for event in event_items if event.event_type in POSITIVE_FEEDBACK_EVENTS)
    risk_count = sum(1 for event in event_items if event.event_type in RISK_FEEDBACK_EVENTS)
    manager_total = type_counts[FEEDBACK_MANAGER_DRAFT_APPROVED] + type_counts[FEEDBACK_MANAGER_DRAFT_REJECTED]
    action_total = type_counts[FEEDBACK_ACTION_ACCEPTED] + type_counts[FEEDBACK_ACTION_DISMISSED]
    client_total = type_counts[FEEDBACK_CLIENT_REPLIED] + type_counts[FEEDBACK_CLIENT_NO_REPLY]
    follow_up_total = type_counts[FEEDBACK_FOLLOW_UP_DONE] + type_counts[FEEDBACK_FOLLOW_UP_OVERDUE]
    lead_closed_total = type_counts[FEEDBACK_LEAD_WON] + type_counts[FEEDBACK_LEAD_LOST]
    rop_total = type_counts[FEEDBACK_ROP_ANSWER_GOOD] + type_counts[FEEDBACK_ROP_ANSWER_RISKY]
    return {
        "schema_version": CHANNEL_FEEDBACK_SCHEMA_VERSION,
        "total_events": len(event_items),
        "event_type_counts": dict(sorted(type_counts.items())),
        "session_counts": dict(sorted(session_counts.items())),
        "unique_sessions": len(session_counts),
        "unique_decisions": len({event.decision_id for event in event_items if event.decision_id}),
        "manager_review": {
            "approved": type_counts[FEEDBACK_MANAGER_DRAFT_APPROVED],
            "rejected": type_counts[FEEDBACK_MANAGER_DRAFT_REJECTED],
            "edited": type_counts[FEEDBACK_MANAGER_DRAFT_EDITED],
            "approval_rate": safe_rate(type_counts[FEEDBACK_MANAGER_DRAFT_APPROVED], manager_total),
        },
        "action_review": {
            "accepted": type_counts[FEEDBACK_ACTION_ACCEPTED],
            "dismissed": type_counts[FEEDBACK_ACTION_DISMISSED],
            "acceptance_rate": safe_rate(type_counts[FEEDBACK_ACTION_ACCEPTED], action_total),
        },
        "client_engagement": {
            "replied": type_counts[FEEDBACK_CLIENT_REPLIED],
            "no_reply": type_counts[FEEDBACK_CLIENT_NO_REPLY],
            "reply_rate": safe_rate(type_counts[FEEDBACK_CLIENT_REPLIED], client_total),
        },
        "follow_up": {
            "done": type_counts[FEEDBACK_FOLLOW_UP_DONE],
            "overdue": type_counts[FEEDBACK_FOLLOW_UP_OVERDUE],
            "completion_rate": safe_rate(type_counts[FEEDBACK_FOLLOW_UP_DONE], follow_up_total),
        },
        "lead_outcomes": {
            "moved": type_counts[FEEDBACK_LEAD_MOVED],
            "won": type_counts[FEEDBACK_LEAD_WON],
            "lost": type_counts[FEEDBACK_LEAD_LOST],
            "win_rate": safe_rate(type_counts[FEEDBACK_LEAD_WON], lead_closed_total),
        },
        "rop_quality": {
            "good": type_counts[FEEDBACK_ROP_ANSWER_GOOD],
            "risky": type_counts[FEEDBACK_ROP_ANSWER_RISKY],
            "good_rate": safe_rate(type_counts[FEEDBACK_ROP_ANSWER_GOOD], rop_total),
        },
        "positive_events": positive_count,
        "risk_events": risk_count,
        "validation_ok": True,
        "blocked": 0,
        "warnings": risk_count,
    }


def feedback_loop_safety_contract() -> Mapping[str, bool]:
    return {
        "network_calls": False,
        "llm_calls": False,
        "rag_used": False,
        "live_send": False,
        "write_crm": False,
        "write_tallanto": False,
        "write_runtime_db": False,
        "run_asr": False,
        "run_ra": False,
        "lead_outcomes_are_read_only_imports": True,
        "stores_raw_payload_by_default": False,
    }


def validate_feedback_event_type(event_type: str) -> str:
    normalized = normalize_key(event_type, "feedback event type")
    if normalized not in ALLOWED_FEEDBACK_EVENTS:
        raise ValueError(f"unsupported feedback event type: {event_type!r}")
    return normalized


def validate_feedback_metadata(metadata: Mapping[str, Any]) -> None:
    for key, value in metadata.items():
        normalized_key = str(key).strip().lower()
        if normalized_key in FORBIDDEN_RAW_METADATA_KEYS:
            raise ValueError(f"feedback metadata must not include raw payload key: {key!r}")
        if normalized_key in EXTERNAL_EFFECT_METADATA_KEYS and metadata_value_is_true(value):
            raise ValueError(f"feedback metadata must not claim external side effect: {key!r}")


def resolve_feedback_entity(
    *,
    event_type: str,
    entity_type: Optional[str],
    entity_id: Optional[str],
    decision_id: Optional[str],
    message_idempotency_key: Optional[str],
    draft_id: Optional[str],
    action_idempotency_key: Optional[str],
) -> tuple[str, str]:
    provided_type = optional_text(entity_type)
    provided_id = optional_text(entity_id)
    if bool(provided_type) != bool(provided_id):
        raise ValueError("entity_type and entity_id must be provided together")
    if provided_type and provided_id:
        return normalize_key(provided_type, "entity_type"), provided_id
    if event_type in ACTION_FEEDBACK_EVENTS and action_idempotency_key:
        return "recommended_action", action_idempotency_key
    if event_type in DRAFT_FEEDBACK_EVENTS and draft_id:
        return "channel_draft", draft_id
    if decision_id:
        return "signal_decision", decision_id
    if message_idempotency_key:
        return "channel_message", message_idempotency_key
    return "channel_session", "current"


def stable_feedback_event_id(
    *,
    event_type: str,
    session_key: str,
    actor: str,
    occurred_at: datetime,
    entity_type: str,
    entity_id: str,
    message_idempotency_key: Optional[str],
    decision_id: Optional[str],
    draft_id: Optional[str],
    action_idempotency_key: Optional[str],
    source_system: Optional[str],
    imported_read_only: bool,
    value: Optional[str],
    score: Optional[float],
    metadata: Mapping[str, Any],
) -> str:
    digest = stable_digest(
        {
            "schema_version": CHANNEL_FEEDBACK_SCHEMA_VERSION,
            "event_type": event_type,
            "session_key": session_key,
            "actor": actor,
            "occurred_at": occurred_at.isoformat(),
            "entity_type": entity_type,
            "entity_id": entity_id,
            "message_idempotency_key": message_idempotency_key,
            "decision_id": decision_id,
            "draft_id": draft_id,
            "action_idempotency_key": action_idempotency_key,
            "source_system": source_system,
            "imported_read_only": imported_read_only,
            "value": value,
            "score": score,
            "metadata": dict(metadata),
        }
    )
    return f"channel_feedback:{event_type}:{digest[:32]}"


def stable_feedback_report_id(
    *,
    session_key: Optional[str],
    decision_id: Optional[str],
    event_ids: Sequence[str],
    created_at: datetime,
) -> str:
    digest = stable_digest(
        {
            "schema_version": CHANNEL_FEEDBACK_SCHEMA_VERSION,
            "session_key": session_key,
            "decision_id": decision_id,
            "event_ids": list(event_ids),
            "created_at": created_at.isoformat(),
        }
    )
    return f"channel_feedback_report:{digest[:32]}"


def safe_rate(numerator: int, denominator: int) -> Optional[float]:
    if denominator <= 0:
        return None
    return round(float(numerator) / float(denominator), 4)


def metadata_value_is_true(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "done", "executed", "sent", "called"}
    return bool(value)
