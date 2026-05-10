from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Optional, Protocol, Sequence


CHANNEL_CONTRACTS_SCHEMA_VERSION = "channel_contracts_v1"
_KEY_RE = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,79}$")


class ChannelDirection(str, Enum):
    INBOUND = "inbound"
    OUTBOUND = "outbound"
    INTERNAL = "internal"


@dataclass(frozen=True)
class ChannelAttachment:
    kind: str
    uri: str
    content_type: Optional[str] = None
    size_bytes: Optional[int] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", normalize_key(self.kind, "attachment kind"))
        object.__setattr__(self, "uri", require_text(self.uri, "attachment uri"))
        if self.size_bytes is not None and self.size_bytes < 0:
            raise ValueError("attachment size_bytes must not be negative")
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ChannelMessage:
    channel: str
    channel_message_id: str
    channel_thread_id: str
    channel_user_id: str
    direction: ChannelDirection
    text: str = ""
    received_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    attachments: Sequence[ChannelAttachment] = field(default_factory=tuple)
    raw_payload: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        channel = normalize_key(self.channel, "channel")
        message_id = require_text(self.channel_message_id, "channel_message_id")
        thread_id = require_text(self.channel_thread_id, "channel_thread_id")
        user_id = require_text(self.channel_user_id, "channel_user_id")
        direction = ChannelDirection(self.direction)
        require_timezone(self.received_at, "received_at")
        attachments = tuple(self.attachments)
        if any(not isinstance(item, ChannelAttachment) for item in attachments):
            raise TypeError("attachments must contain ChannelAttachment items")
        text = str(self.text or "").strip()
        if not text and not attachments:
            raise ValueError("ChannelMessage requires text or at least one attachment")
        object.__setattr__(self, "channel", channel)
        object.__setattr__(self, "channel_message_id", message_id)
        object.__setattr__(self, "channel_thread_id", thread_id)
        object.__setattr__(self, "channel_user_id", user_id)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "text", text)
        object.__setattr__(self, "attachments", attachments)
        object.__setattr__(self, "raw_payload", dict(self.raw_payload))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def idempotency_key(self) -> str:
        return stable_message_idempotency_key(
            channel=self.channel,
            channel_thread_id=self.channel_thread_id,
            channel_message_id=self.channel_message_id,
            direction=self.direction.value,
        )

    def to_json_dict(self) -> Mapping[str, Any]:
        return {
            "schema_version": CHANNEL_CONTRACTS_SCHEMA_VERSION,
            "channel": self.channel,
            "channel_message_id": self.channel_message_id,
            "channel_thread_id": self.channel_thread_id,
            "channel_user_id": self.channel_user_id,
            "direction": self.direction.value,
            "text": self.text,
            "received_at": self.received_at.isoformat(),
            "attachments": [item.to_json_dict() for item in self.attachments],
            "raw_payload": dict(self.raw_payload),
            "metadata": dict(self.metadata),
            "idempotency_key": self.idempotency_key,
        }


@dataclass(frozen=True)
class ChannelSession:
    channel: str
    channel_thread_id: str
    normalized_customer_id: Optional[str] = None
    crm_contact_id: Optional[str] = None
    state: Mapping[str, Any] = field(default_factory=dict)
    context_summary: Optional[str] = None
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def from_message(
        cls,
        message: ChannelMessage,
        *,
        normalized_customer_id: Optional[str] = None,
        crm_contact_id: Optional[str] = None,
        state: Optional[Mapping[str, Any]] = None,
        context_summary: Optional[str] = None,
        updated_at: Optional[datetime] = None,
    ) -> "ChannelSession":
        return cls(
            channel=message.channel,
            channel_thread_id=message.channel_thread_id,
            normalized_customer_id=normalized_customer_id,
            crm_contact_id=crm_contact_id,
            state=state or {},
            context_summary=context_summary,
            updated_at=updated_at or message.received_at,
        )

    def __post_init__(self) -> None:
        object.__setattr__(self, "channel", normalize_key(self.channel, "channel"))
        object.__setattr__(self, "channel_thread_id", require_text(self.channel_thread_id, "channel_thread_id"))
        object.__setattr__(self, "normalized_customer_id", optional_text(self.normalized_customer_id))
        object.__setattr__(self, "crm_contact_id", optional_text(self.crm_contact_id))
        object.__setattr__(self, "context_summary", optional_text(self.context_summary))
        require_timezone(self.updated_at, "updated_at")
        object.__setattr__(self, "state", dict(self.state))

    @property
    def session_key(self) -> str:
        digest = stable_digest({"channel": self.channel, "thread": self.channel_thread_id})
        return f"channel_session:{self.channel}:{digest[:24]}"

    def to_json_dict(self) -> Mapping[str, Any]:
        return {
            "schema_version": CHANNEL_CONTRACTS_SCHEMA_VERSION,
            "channel": self.channel,
            "channel_thread_id": self.channel_thread_id,
            "normalized_customer_id": self.normalized_customer_id,
            "crm_contact_id": self.crm_contact_id,
            "state": dict(self.state),
            "context_summary": self.context_summary,
            "updated_at": self.updated_at.isoformat(),
            "session_key": self.session_key,
        }


@dataclass(frozen=True)
class ReplyButton:
    label: str
    action: str
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "label", require_text(self.label, "button label"))
        object.__setattr__(self, "action", normalize_key(self.action, "button action"))
        object.__setattr__(self, "payload", dict(self.payload))

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RecommendedAction:
    action_type: str
    target_system: str
    entity_type: str
    entity_id: Optional[str] = None
    title: str = ""
    summary: str = ""
    payload: Mapping[str, Any] = field(default_factory=dict)
    confidence: Optional[float] = None
    requires_approval: bool = True
    idempotency_key: Optional[str] = None

    def __post_init__(self) -> None:
        action_type = normalize_key(self.action_type, "action_type")
        target_system = normalize_key(self.target_system, "target_system")
        entity_type = normalize_key(self.entity_type, "entity_type")
        confidence = self.confidence
        if confidence is not None and not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        payload = dict(self.payload)
        title = str(self.title or "").strip()
        summary = str(self.summary or "").strip()
        entity_id = optional_text(self.entity_id)
        key = optional_text(self.idempotency_key)
        if key is None:
            key = stable_action_idempotency_key(
                action_type=action_type,
                target_system=target_system,
                entity_type=entity_type,
                entity_id=entity_id,
                payload=payload,
            )
        object.__setattr__(self, "action_type", action_type)
        object.__setattr__(self, "target_system", target_system)
        object.__setattr__(self, "entity_type", entity_type)
        object.__setattr__(self, "entity_id", entity_id)
        object.__setattr__(self, "title", title)
        object.__setattr__(self, "summary", summary)
        object.__setattr__(self, "payload", payload)
        object.__setattr__(self, "idempotency_key", key)

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BotReply:
    text: str
    buttons: Sequence[ReplyButton] = field(default_factory=tuple)
    attachments: Sequence[ChannelAttachment] = field(default_factory=tuple)
    recommended_actions: Sequence[RecommendedAction] = field(default_factory=tuple)
    confidence: Optional[float] = None
    requires_approval: bool = True
    safety_flags: Sequence[str] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        text = str(self.text or "").strip()
        buttons = tuple(self.buttons)
        attachments = tuple(self.attachments)
        actions = tuple(self.recommended_actions)
        if not text and not attachments and not actions:
            raise ValueError("BotReply requires text, attachment, or recommended action")
        if any(not isinstance(item, ReplyButton) for item in buttons):
            raise TypeError("buttons must contain ReplyButton items")
        if any(not isinstance(item, ChannelAttachment) for item in attachments):
            raise TypeError("attachments must contain ChannelAttachment items")
        if any(not isinstance(item, RecommendedAction) for item in actions):
            raise TypeError("recommended_actions must contain RecommendedAction items")
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        object.__setattr__(self, "text", text)
        object.__setattr__(self, "buttons", buttons)
        object.__setattr__(self, "attachments", attachments)
        object.__setattr__(self, "recommended_actions", actions)
        object.__setattr__(self, "safety_flags", tuple(normalize_key(item, "safety flag") for item in self.safety_flags))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_json_dict(self) -> Mapping[str, Any]:
        return {
            "schema_version": CHANNEL_CONTRACTS_SCHEMA_VERSION,
            "text": self.text,
            "buttons": [item.to_json_dict() for item in self.buttons],
            "attachments": [item.to_json_dict() for item in self.attachments],
            "recommended_actions": [item.to_json_dict() for item in self.recommended_actions],
            "confidence": self.confidence,
            "requires_approval": self.requires_approval,
            "safety_flags": list(self.safety_flags),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ChannelRenderedReply:
    channel: str
    channel_thread_id: str
    payload: Mapping[str, Any]
    idempotency_key: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "channel", normalize_key(self.channel, "channel"))
        object.__setattr__(self, "channel_thread_id", require_text(self.channel_thread_id, "channel_thread_id"))
        object.__setattr__(self, "payload", dict(self.payload))
        object.__setattr__(self, "idempotency_key", require_text(self.idempotency_key, "idempotency_key"))


@dataclass(frozen=True)
class SendResult:
    channel: str
    idempotency_key: str
    sent: bool = False
    status: str = "blocked"
    external_message_id: Optional[str] = None
    error: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "channel", normalize_key(self.channel, "channel"))
        object.__setattr__(self, "idempotency_key", require_text(self.idempotency_key, "idempotency_key"))
        object.__setattr__(self, "status", normalize_key(self.status, "send status"))
        object.__setattr__(self, "external_message_id", optional_text(self.external_message_id))
        object.__setattr__(self, "error", optional_text(self.error))
        object.__setattr__(self, "metadata", dict(self.metadata))


class ChannelAdapter(Protocol):
    channel_name: str

    def parse_inbound(self, raw_update: Mapping[str, Any]) -> Sequence[ChannelMessage]:
        """Normalize a channel-specific payload into channel-neutral messages."""

    def render_reply(
        self,
        session: ChannelSession,
        reply: BotReply,
    ) -> ChannelRenderedReply:
        """Convert a channel-neutral reply to an adapter-specific payload without sending it."""

    def send(self, rendered: ChannelRenderedReply, *, live_send_enabled: bool = False) -> SendResult:
        """Send a rendered reply only when a future explicit policy enables live send."""


def stable_message_idempotency_key(
    *,
    channel: str,
    channel_thread_id: str,
    channel_message_id: str,
    direction: str,
) -> str:
    payload = {
        "channel": normalize_key(channel, "channel"),
        "thread": require_text(channel_thread_id, "channel_thread_id"),
        "message": require_text(channel_message_id, "channel_message_id"),
        "direction": ChannelDirection(direction).value,
    }
    digest = stable_digest(payload)
    return f"channel_message:{payload['channel']}:{payload['direction']}:{digest[:32]}"


def dedupe_channel_messages(messages: Sequence[ChannelMessage]) -> tuple[ChannelMessage, ...]:
    result: list[ChannelMessage] = []
    seen: set[str] = set()
    for message in messages:
        if not isinstance(message, ChannelMessage):
            raise TypeError("messages must contain ChannelMessage items")
        key = message.idempotency_key
        if key in seen:
            continue
        seen.add(key)
        result.append(message)
    return tuple(result)


def stable_action_idempotency_key(
    *,
    action_type: str,
    target_system: str,
    entity_type: str,
    entity_id: Optional[str],
    payload: Mapping[str, Any],
) -> str:
    normalized = {
        "action_type": normalize_key(action_type, "action_type"),
        "target_system": normalize_key(target_system, "target_system"),
        "entity_type": normalize_key(entity_type, "entity_type"),
        "entity_id": optional_text(entity_id),
        "payload": dict(payload),
    }
    digest = stable_digest(normalized)
    return f"recommended_action:{normalized['action_type']}:{digest[:32]}"


def stable_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def normalize_key(value: str, field_name: str) -> str:
    normalized = str(value or "").strip().lower()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    if not _KEY_RE.match(normalized):
        raise ValueError(f"{field_name} contains unsupported characters: {value!r}")
    return normalized


def require_text(value: Any, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} must not be empty")
    return text


def optional_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def require_timezone(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
