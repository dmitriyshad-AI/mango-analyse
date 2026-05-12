from __future__ import annotations

from collections.abc import Sequence as SequenceABC
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Optional, Sequence

from mango_mvp.channels.contracts import ChannelMessage, optional_text, require_text, require_timezone, stable_digest
from mango_mvp.channels.telegram_adapter import (
    TELEGRAM_BUSINESS_CHANNEL,
    TelegramReadOnlyAdapter,
    ensure_mapping,
    mapping_or_none,
    telegram_datetime,
)


TELEGRAM_BUSINESS_RUNTIME_SCHEMA_VERSION = "telegram_business_runtime_v1"
TELEGRAM_BUSINESS_UPDATE_TYPES = {
    "business_connection",
    "business_message",
    "edited_business_message",
    "deleted_business_messages",
}

Clock = Callable[[], datetime]


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class TelegramBusinessConnectionRecord:
    business_connection_id: str
    user_chat_id: Optional[str]
    user_id: Optional[str]
    date: Optional[datetime]
    can_reply: bool
    is_enabled: bool
    rights: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "business_connection_id",
            require_text(self.business_connection_id, "business_connection_id"),
        )
        object.__setattr__(self, "user_chat_id", optional_text(self.user_chat_id))
        object.__setattr__(self, "user_id", optional_text(self.user_id))
        if self.date is not None:
            require_timezone(self.date, "date")
        object.__setattr__(self, "can_reply", bool(self.can_reply))
        object.__setattr__(self, "is_enabled", bool(self.is_enabled))
        object.__setattr__(self, "rights", scrub_telegram_business_payload(dict(self.rights)))
        object.__setattr__(self, "metadata", scrub_telegram_business_payload(dict(self.metadata)))

    @property
    def idempotency_key(self) -> str:
        digest = stable_digest(
            {
                "business_connection_id": self.business_connection_id,
                "user_chat_id": self.user_chat_id,
                "user_id": self.user_id,
                "date": self.date.isoformat() if self.date else None,
                "can_reply": self.can_reply,
                "is_enabled": self.is_enabled,
            }
        )
        return f"telegram_business_connection:{digest[:32]}"

    def to_json_dict(self) -> Mapping[str, Any]:
        return {
            "schema_version": TELEGRAM_BUSINESS_RUNTIME_SCHEMA_VERSION,
            "business_connection_id": self.business_connection_id,
            "user_chat_id": self.user_chat_id,
            "user_id": self.user_id,
            "date": self.date.isoformat() if self.date else None,
            "can_reply": self.can_reply,
            "is_enabled": self.is_enabled,
            "rights": dict(self.rights),
            "metadata": dict(self.metadata),
            "idempotency_key": self.idempotency_key,
        }


@dataclass(frozen=True)
class TelegramBusinessUpdateRecord:
    update_id: str
    update_type: str
    received_at: datetime
    business_connection_id: Optional[str] = None
    chat_id: Optional[str] = None
    message_ids: Sequence[str] = field(default_factory=tuple)
    message_idempotency_keys: Sequence[str] = field(default_factory=tuple)
    connection: Optional[TelegramBusinessConnectionRecord] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        update_type = require_text(self.update_type, "update_type")
        if update_type not in TELEGRAM_BUSINESS_UPDATE_TYPES:
            raise ValueError(f"unsupported Telegram Business update_type: {update_type!r}")
        require_timezone(self.received_at, "received_at")
        if self.connection is not None and not isinstance(self.connection, TelegramBusinessConnectionRecord):
            raise TypeError("connection must be TelegramBusinessConnectionRecord")
        object.__setattr__(self, "update_id", require_text(self.update_id, "update_id"))
        object.__setattr__(self, "update_type", update_type)
        object.__setattr__(self, "business_connection_id", optional_text(self.business_connection_id))
        object.__setattr__(self, "chat_id", optional_text(self.chat_id))
        object.__setattr__(self, "message_ids", tuple(require_text(item, "message_id") for item in self.message_ids))
        object.__setattr__(
            self,
            "message_idempotency_keys",
            tuple(require_text(item, "message_idempotency_key") for item in self.message_idempotency_keys),
        )
        object.__setattr__(self, "metadata", scrub_telegram_business_payload(dict(self.metadata)))

    @property
    def idempotency_key(self) -> str:
        digest = stable_digest(
            {
                "update_id": self.update_id,
                "update_type": self.update_type,
                "business_connection_id": self.business_connection_id,
                "chat_id": self.chat_id,
                "message_ids": list(self.message_ids),
            }
        )
        return f"telegram_business_update:{self.update_type}:{digest[:32]}"

    def to_json_dict(self) -> Mapping[str, Any]:
        return {
            "schema_version": TELEGRAM_BUSINESS_RUNTIME_SCHEMA_VERSION,
            "update_id": self.update_id,
            "update_type": self.update_type,
            "received_at": self.received_at.isoformat(),
            "business_connection_id": self.business_connection_id,
            "chat_id": self.chat_id,
            "message_ids": list(self.message_ids),
            "message_idempotency_keys": list(self.message_idempotency_keys),
            "connection": self.connection.to_json_dict() if self.connection else None,
            "metadata": dict(self.metadata),
            "idempotency_key": self.idempotency_key,
            "safety": telegram_business_runtime_safety_contract(),
        }


@dataclass(frozen=True)
class TelegramBusinessRuntimeResult:
    update_record: TelegramBusinessUpdateRecord
    messages: Sequence[ChannelMessage] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.update_record, TelegramBusinessUpdateRecord):
            raise TypeError("update_record must be TelegramBusinessUpdateRecord")
        messages = tuple(self.messages)
        if any(not isinstance(item, ChannelMessage) for item in messages):
            raise TypeError("messages must contain ChannelMessage items")
        object.__setattr__(self, "messages", messages)

    def to_json_dict(self, *, include_message_text: bool = False) -> Mapping[str, Any]:
        return {
            "schema_version": TELEGRAM_BUSINESS_RUNTIME_SCHEMA_VERSION,
            "update_record": self.update_record.to_json_dict(),
            "messages": [
                project_business_message_for_report(item, include_text=include_message_text) for item in self.messages
            ],
            "messages_total": len(self.messages),
            "safety": telegram_business_runtime_safety_contract(),
        }


@dataclass(frozen=True)
class TelegramBusinessRuntimeStoreResult:
    update_idempotency_key: str
    created: bool
    status: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "update_idempotency_key", require_text(self.update_idempotency_key, "update_idempotency_key"))
        object.__setattr__(self, "status", require_text(self.status, "status"))

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


class TelegramBusinessRuntimeMemoryStore:
    """Process-local idempotency store for Business update tests and dry-runs."""

    def __init__(self) -> None:
        self._updates: dict[str, TelegramBusinessUpdateRecord] = {}

    def upsert_update(self, record: TelegramBusinessUpdateRecord) -> TelegramBusinessRuntimeStoreResult:
        if not isinstance(record, TelegramBusinessUpdateRecord):
            raise TypeError("record must be TelegramBusinessUpdateRecord")
        key = record.idempotency_key
        if key in self._updates:
            return TelegramBusinessRuntimeStoreResult(key, False, "duplicate")
        self._updates[key] = record
        return TelegramBusinessRuntimeStoreResult(key, True, "created")

    def get_update(self, idempotency_key: str) -> Optional[TelegramBusinessUpdateRecord]:
        return self._updates.get(require_text(idempotency_key, "idempotency_key"))

    def summary(self) -> Mapping[str, Any]:
        by_type: dict[str, int] = {}
        for record in self._updates.values():
            by_type[record.update_type] = by_type.get(record.update_type, 0) + 1
        return {
            "schema_version": TELEGRAM_BUSINESS_RUNTIME_SCHEMA_VERSION,
            "updates": len(self._updates),
            "by_type": by_type,
            "safety": telegram_business_runtime_safety_contract(),
        }


class TelegramBusinessRuntime:
    """Read-only Telegram Business runtime parser with no Telegram API transport."""

    def __init__(self, *, adapter: Optional[TelegramReadOnlyAdapter] = None, clock: Optional[Clock] = None) -> None:
        self.adapter = adapter or TelegramReadOnlyAdapter()
        self._clock = clock or now_utc

    def process_update(self, raw_update: Mapping[str, Any]) -> TelegramBusinessRuntimeResult:
        update = ensure_mapping(raw_update, "raw_update")
        update_type = telegram_business_update_type(update)
        messages = tuple(message for message in self.adapter.parse_inbound(update) if message.channel == TELEGRAM_BUSINESS_CHANNEL)
        record = build_telegram_business_update_record(update, update_type=update_type, messages=messages, now=self._now())
        return TelegramBusinessRuntimeResult(update_record=record, messages=messages)

    def _now(self) -> datetime:
        value = self._clock()
        require_timezone(value, "clock value")
        return value


def telegram_business_update_type(update: Mapping[str, Any]) -> str:
    present = [key for key in TELEGRAM_BUSINESS_UPDATE_TYPES if key in update]
    if len(present) != 1:
        raise ValueError("Telegram Business update must contain exactly one supported business update field")
    return present[0]


def build_telegram_business_update_record(
    update: Mapping[str, Any],
    *,
    update_type: str,
    messages: Sequence[ChannelMessage],
    now: datetime,
) -> TelegramBusinessUpdateRecord:
    require_timezone(now, "now")
    update_id = require_text(update.get("update_id"), "update_id")
    connection: Optional[TelegramBusinessConnectionRecord] = None
    business_connection_id: Optional[str] = None
    chat_id: Optional[str] = None
    message_ids: tuple[str, ...] = ()

    if update_type == "business_connection":
        payload = ensure_mapping(update.get("business_connection"), "business_connection")
        connection = parse_business_connection(payload)
        business_connection_id = connection.business_connection_id
        chat_id = connection.user_chat_id
    elif update_type in {"business_message", "edited_business_message"}:
        payload = ensure_mapping(update.get(update_type), update_type)
        business_connection_id = optional_text(payload.get("business_connection_id")) or optional_text(
            update.get("business_connection_id")
        )
        chat = mapping_or_none(payload.get("chat")) or {}
        chat_id = optional_text(chat.get("id"))
        message_ids = tuple(optional_text(message.channel_message_id) or "" for message in messages)
    elif update_type == "deleted_business_messages":
        payload = ensure_mapping(update.get("deleted_business_messages"), "deleted_business_messages")
        business_connection_id = require_text(payload.get("business_connection_id"), "business_connection_id")
        chat = ensure_mapping(payload.get("chat"), "deleted_business_messages.chat")
        chat_id = require_text(chat.get("id"), "deleted_business_messages.chat.id")
        raw_ids = payload.get("message_ids") or ()
        if not isinstance(raw_ids, SequenceABC) or isinstance(raw_ids, (str, bytes)):
            raise ValueError("deleted_business_messages.message_ids must be a sequence")
        message_ids = tuple(require_text(item, "deleted_business_messages.message_id") for item in raw_ids)

    return TelegramBusinessUpdateRecord(
        update_id=update_id,
        update_type=update_type,
        received_at=now,
        business_connection_id=business_connection_id,
        chat_id=chat_id,
        message_ids=message_ids,
        message_idempotency_keys=tuple(message.idempotency_key for message in messages),
        connection=connection,
        metadata={
            "parser_mode": "read_only",
            "telegram_update_id": update_id,
            "raw_payload_persisted": False,
            "message_count": len(messages),
            "deleted_message_count": len(message_ids) if update_type == "deleted_business_messages" else 0,
        },
    )


def parse_business_connection(payload: Mapping[str, Any]) -> TelegramBusinessConnectionRecord:
    user = mapping_or_none(payload.get("user")) or {}
    rights = mapping_or_none(payload.get("rights")) or {}
    return TelegramBusinessConnectionRecord(
        business_connection_id=require_text(payload.get("id"), "business_connection.id"),
        user_chat_id=optional_text(payload.get("user_chat_id")),
        user_id=optional_text(user.get("id")),
        date=telegram_datetime(payload.get("date")) if payload.get("date") not in (None, "") else None,
        can_reply=bool(payload.get("can_reply")),
        is_enabled=bool(payload.get("is_enabled", True)),
        rights=rights,
        metadata={
            "user_is_bot": bool(user.get("is_bot")) if user else None,
            "has_user": bool(user),
            "has_rights": bool(rights),
        },
    )


def project_business_message_for_report(message: ChannelMessage, *, include_text: bool = False) -> Mapping[str, Any]:
    if message.channel != TELEGRAM_BUSINESS_CHANNEL:
        raise ValueError(f"expected {TELEGRAM_BUSINESS_CHANNEL}, got {message.channel!r}")
    return {
        "channel": message.channel,
        "channel_message_id": message.channel_message_id,
        "channel_thread_id": message.channel_thread_id,
        "channel_user_id": message.channel_user_id,
        "direction": message.direction.value,
        "received_at": message.received_at.isoformat(),
        "text": message.text if include_text else None,
        "text_redacted": not include_text,
        "text_length": len(message.text),
        "attachment_count": len(message.attachments),
        "idempotency_key": message.idempotency_key,
        "metadata": scrub_telegram_business_payload(dict(message.metadata)),
    }


SENSITIVE_KEY_PARTS = (
    "token",
    "secret",
    "password",
    "authorization",
    "api_key",
    "apikey",
    "hash",
    "пароль",
    "токен",
    "ключ",
)


def scrub_telegram_business_payload(value: Any) -> Any:
    if isinstance(value, Mapping):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if any(part in key_text.casefold() for part in SENSITIVE_KEY_PARTS):
                cleaned[key_text] = "[REDACTED]"
            else:
                cleaned[key_text] = scrub_telegram_business_payload(item)
        return cleaned
    if isinstance(value, list):
        return [scrub_telegram_business_payload(item) for item in value]
    if isinstance(value, tuple):
        return tuple(scrub_telegram_business_payload(item) for item in value)
    return value


def telegram_business_runtime_safety_contract() -> Mapping[str, bool]:
    return {
        "network_calls": False,
        "telegram_api_called": False,
        "live_send": False,
        "tdlib_used": False,
        "write_crm": False,
        "write_tallanto": False,
        "write_runtime_db": False,
        "run_asr": False,
        "run_ra": False,
        "stores_raw_payload_by_default": False,
    }


__all__ = [
    "TELEGRAM_BUSINESS_RUNTIME_SCHEMA_VERSION",
    "TELEGRAM_BUSINESS_UPDATE_TYPES",
    "TelegramBusinessConnectionRecord",
    "TelegramBusinessRuntime",
    "TelegramBusinessRuntimeMemoryStore",
    "TelegramBusinessRuntimeResult",
    "TelegramBusinessRuntimeStoreResult",
    "TelegramBusinessUpdateRecord",
    "build_telegram_business_update_record",
    "parse_business_connection",
    "project_business_message_for_report",
    "scrub_telegram_business_payload",
    "telegram_business_runtime_safety_contract",
    "telegram_business_update_type",
]
