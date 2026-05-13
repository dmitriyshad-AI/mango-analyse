from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence

from mango_mvp.channels.contracts import (
    ChannelMessage,
    normalize_key,
    optional_text,
    require_text,
    require_timezone,
    stable_digest,
)
from mango_mvp.channels.preview_service import ChannelDraftPreview
from mango_mvp.channels.storage import ChannelDraftRecord
from mango_mvp.channels.telegram_adapter import TELEGRAM_BUSINESS_CHANNEL


TELEGRAM_NATIVE_DRAFT_SCHEMA_VERSION = "telegram_native_draft_v1"

NATIVE_DRAFT_OPERATION_SAVE = "save_draft"
NATIVE_DRAFT_OPERATION_CLEAR = "clear_draft"
NATIVE_DRAFT_OPERATION_GET = "get_draft_state"
NATIVE_DRAFT_OPERATIONS = {
    NATIVE_DRAFT_OPERATION_SAVE,
    NATIVE_DRAFT_OPERATION_CLEAR,
    NATIVE_DRAFT_OPERATION_GET,
}

NATIVE_DRAFT_STATUS_SAVED = "saved"
NATIVE_DRAFT_STATUS_UNCHANGED = "unchanged"
NATIVE_DRAFT_STATUS_CLEARED = "cleared"
NATIVE_DRAFT_STATUS_EMPTY = "empty"
NATIVE_DRAFT_STATUS_BLOCKED = "blocked"
NATIVE_DRAFT_STATUS_CONFLICT = "conflict"
NATIVE_DRAFT_STATUS_STALE = "stale"

MAX_NATIVE_DRAFT_TEXT_LENGTH = 4096

Clock = Callable[[], datetime]


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class TelegramNativeDraftConfig:
    enabled: bool = False
    kill_switch: bool = True
    allowed_chat_ids: Sequence[str] = field(default_factory=tuple)
    tdlib_database_dir: Optional[str] = None
    api_id_present: bool = False
    api_hash_present: bool = False
    database_encryption_key_present: bool = False
    phone_number_present: bool = False
    test_account_first_required: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "enabled", bool(self.enabled))
        object.__setattr__(self, "kill_switch", bool(self.kill_switch))
        object.__setattr__(self, "allowed_chat_ids", tuple(str(item).strip() for item in self.allowed_chat_ids if str(item).strip()))
        object.__setattr__(self, "tdlib_database_dir", optional_text(self.tdlib_database_dir))
        object.__setattr__(self, "api_id_present", bool(self.api_id_present))
        object.__setattr__(self, "api_hash_present", bool(self.api_hash_present))
        object.__setattr__(
            self,
            "database_encryption_key_present",
            bool(self.database_encryption_key_present),
        )
        object.__setattr__(self, "phone_number_present", bool(self.phone_number_present))
        object.__setattr__(self, "test_account_first_required", bool(self.test_account_first_required))

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "TelegramNativeDraftConfig":
        source = os.environ if env is None else env
        return cls(
            enabled=env_bool(source, "CHANNEL_TELEGRAM_NATIVE_DRAFTS_ENABLED", default=False),
            kill_switch=env_bool(source, "CHANNEL_TELEGRAM_NATIVE_DRAFT_KILL_SWITCH", default=True),
            allowed_chat_ids=split_env_list(source.get("CHANNEL_TELEGRAM_NATIVE_DRAFT_ALLOWED_CHAT_IDS")),
            tdlib_database_dir=source.get("TDLIB_DATABASE_DIR"),
            api_id_present=bool(optional_text(source.get("TDLIB_API_ID"))),
            api_hash_present=bool(optional_text(source.get("TDLIB_API_HASH"))),
            database_encryption_key_present=bool(optional_text(source.get("TDLIB_DATABASE_ENCRYPTION_KEY"))),
            phone_number_present=bool(optional_text(source.get("TDLIB_PHONE_NUMBER"))),
        )

    def to_json_dict(self) -> Mapping[str, Any]:
        return {
            "schema_version": TELEGRAM_NATIVE_DRAFT_SCHEMA_VERSION,
            "enabled": self.enabled,
            "kill_switch": self.kill_switch,
            "allowed_chat_ids": list(self.allowed_chat_ids),
            "tdlib_database_dir_present": bool(self.tdlib_database_dir),
            "api_id_present": self.api_id_present,
            "api_hash_present": self.api_hash_present,
            "database_encryption_key_present": self.database_encryption_key_present,
            "phone_number_present": self.phone_number_present,
            "test_account_first_required": self.test_account_first_required,
            "safety": telegram_native_draft_safety_contract(),
        }


@dataclass(frozen=True)
class TelegramNativeDraftIntent:
    operation: str
    chat_id: str
    text: Optional[str] = None
    draft_id: Optional[str] = None
    channel_thread_id: Optional[str] = None
    source_message_idempotency_key: Optional[str] = None
    reply_to_message_id: Optional[str] = None
    allow_overwrite_manager_draft: bool = False
    actor: str = "native_draft_orchestrator"
    created_at: datetime = field(default_factory=now_utc)
    idempotency_key: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        operation = normalize_operation(self.operation)
        text = optional_text(self.text)
        if operation == NATIVE_DRAFT_OPERATION_SAVE:
            if not text:
                raise ValueError("save_draft intent requires non-empty text")
            if len(text) > MAX_NATIVE_DRAFT_TEXT_LENGTH:
                raise ValueError("native draft text exceeds Telegram-safe limit")
            draft_id = require_text(self.draft_id, "draft_id")
        else:
            if text is not None:
                raise ValueError(f"{operation} intent must not include text")
            draft_id = optional_text(self.draft_id)
        require_timezone(self.created_at, "created_at")
        chat_id = require_text(self.chat_id, "chat_id")
        key = optional_text(self.idempotency_key)
        payload = {
            "operation": operation,
            "chat_id": chat_id,
            "text_hash": draft_text_hash(text),
            "draft_id": draft_id,
            "channel_thread_id": optional_text(self.channel_thread_id),
            "source_message_idempotency_key": optional_text(self.source_message_idempotency_key),
            "reply_to_message_id": optional_text(self.reply_to_message_id),
            "allow_overwrite_manager_draft": bool(self.allow_overwrite_manager_draft),
        }
        if key is None:
            key = f"telegram_native_draft_intent:{operation}:{stable_digest(payload)[:32]}"
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "chat_id", chat_id)
        object.__setattr__(self, "text", text)
        object.__setattr__(self, "draft_id", draft_id)
        object.__setattr__(self, "channel_thread_id", optional_text(self.channel_thread_id))
        object.__setattr__(
            self,
            "source_message_idempotency_key",
            optional_text(self.source_message_idempotency_key),
        )
        object.__setattr__(self, "reply_to_message_id", optional_text(self.reply_to_message_id))
        object.__setattr__(self, "allow_overwrite_manager_draft", bool(self.allow_overwrite_manager_draft))
        object.__setattr__(self, "actor", require_text(self.actor, "actor"))
        object.__setattr__(self, "idempotency_key", key)
        object.__setattr__(self, "metadata", scrub_native_draft_payload(dict(self.metadata)))

    @property
    def text_hash(self) -> Optional[str]:
        return draft_text_hash(self.text)

    @property
    def text_length(self) -> int:
        return len(self.text or "")

    def to_json_dict(self, *, include_text: bool = False) -> Mapping[str, Any]:
        payload = {
            "schema_version": TELEGRAM_NATIVE_DRAFT_SCHEMA_VERSION,
            "operation": self.operation,
            "chat_id": self.chat_id,
            "text": self.text if include_text else None,
            "text_redacted": bool(self.text and not include_text),
            "text_hash": self.text_hash,
            "text_length": self.text_length,
            "draft_id": self.draft_id,
            "channel_thread_id": self.channel_thread_id,
            "source_message_idempotency_key": self.source_message_idempotency_key,
            "reply_to_message_id": self.reply_to_message_id,
            "allow_overwrite_manager_draft": self.allow_overwrite_manager_draft,
            "actor": self.actor,
            "created_at": self.created_at.isoformat(),
            "idempotency_key": self.idempotency_key,
            "metadata": dict(self.metadata),
        }
        return payload


@dataclass(frozen=True)
class TelegramNativeDraftState:
    chat_id: str
    text: Optional[str] = None
    owner: str = "empty"
    reply_to_message_id: Optional[str] = None
    last_intent_idempotency_key: Optional[str] = None
    last_draft_id: Optional[str] = None
    last_written_hash: Optional[str] = None
    updated_at: datetime = field(default_factory=now_utc)
    stale: bool = False
    stale_reason: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        owner = normalize_key(self.owner, "native draft owner")
        if owner not in {"empty", "mango", "manager", "unknown"}:
            raise ValueError(f"unsupported native draft owner: {owner!r}")
        text = optional_text(self.text)
        require_timezone(self.updated_at, "updated_at")
        object.__setattr__(self, "chat_id", require_text(self.chat_id, "chat_id"))
        object.__setattr__(self, "text", text)
        object.__setattr__(self, "owner", owner)
        object.__setattr__(self, "reply_to_message_id", optional_text(self.reply_to_message_id))
        object.__setattr__(self, "last_intent_idempotency_key", optional_text(self.last_intent_idempotency_key))
        object.__setattr__(self, "last_draft_id", optional_text(self.last_draft_id))
        object.__setattr__(self, "last_written_hash", optional_text(self.last_written_hash) or draft_text_hash(text))
        object.__setattr__(self, "stale", bool(self.stale))
        object.__setattr__(self, "stale_reason", optional_text(self.stale_reason))
        object.__setattr__(self, "metadata", scrub_native_draft_payload(dict(self.metadata)))

    @classmethod
    def empty(cls, chat_id: str, *, updated_at: Optional[datetime] = None) -> "TelegramNativeDraftState":
        return cls(chat_id=chat_id, owner="empty", updated_at=updated_at or now_utc())

    @property
    def text_hash(self) -> Optional[str]:
        return draft_text_hash(self.text)

    @property
    def text_length(self) -> int:
        return len(self.text or "")

    @property
    def is_empty(self) -> bool:
        return not bool(self.text)

    def to_json_dict(self, *, include_text: bool = False) -> Mapping[str, Any]:
        return {
            "schema_version": TELEGRAM_NATIVE_DRAFT_SCHEMA_VERSION,
            "chat_id": self.chat_id,
            "text": self.text if include_text else None,
            "text_redacted": bool(self.text and not include_text),
            "text_hash": self.text_hash,
            "text_length": self.text_length,
            "owner": self.owner,
            "reply_to_message_id": self.reply_to_message_id,
            "last_intent_idempotency_key": self.last_intent_idempotency_key,
            "last_draft_id": self.last_draft_id,
            "last_written_hash": self.last_written_hash,
            "updated_at": self.updated_at.isoformat(),
            "stale": self.stale,
            "stale_reason": self.stale_reason,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class TelegramNativeDraftResult:
    operation: str
    chat_id: str
    status: str
    idempotency_key: str
    changed: bool = False
    blocked_reason: Optional[str] = None
    conflict_flags: Sequence[str] = field(default_factory=tuple)
    state: Optional[TelegramNativeDraftState] = None
    audit_event: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        operation = normalize_operation(self.operation)
        status = normalize_key(self.status, "native draft status")
        if status in {"sent", "live_sent", "executed"}:
            raise ValueError("native draft result must not use live-send statuses")
        if self.state is not None and not isinstance(self.state, TelegramNativeDraftState):
            raise TypeError("state must be TelegramNativeDraftState")
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "chat_id", require_text(self.chat_id, "chat_id"))
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "idempotency_key", require_text(self.idempotency_key, "idempotency_key"))
        object.__setattr__(self, "changed", bool(self.changed))
        object.__setattr__(self, "blocked_reason", optional_text(self.blocked_reason))
        object.__setattr__(self, "conflict_flags", tuple(normalize_key(item, "conflict_flag") for item in self.conflict_flags))
        object.__setattr__(self, "audit_event", scrub_native_draft_payload(dict(self.audit_event)))
        object.__setattr__(self, "metadata", scrub_native_draft_payload(dict(self.metadata)))

    @classmethod
    def blocked(
        cls,
        intent: TelegramNativeDraftIntent,
        *,
        reason: str,
        state: Optional[TelegramNativeDraftState] = None,
        conflict_flags: Sequence[str] = (),
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "TelegramNativeDraftResult":
        return cls(
            operation=intent.operation,
            chat_id=intent.chat_id,
            status=NATIVE_DRAFT_STATUS_BLOCKED,
            idempotency_key=intent.idempotency_key or "",
            blocked_reason=reason,
            conflict_flags=conflict_flags,
            state=state,
            audit_event=build_native_draft_audit_event(
                "native_draft_blocked",
                intent=intent,
                state=state,
                reason=reason,
                conflict_flags=conflict_flags,
            ),
            metadata=metadata or {},
        )

    def to_json_dict(self, *, include_text: bool = False) -> Mapping[str, Any]:
        return {
            "schema_version": TELEGRAM_NATIVE_DRAFT_SCHEMA_VERSION,
            "operation": self.operation,
            "chat_id": self.chat_id,
            "status": self.status,
            "idempotency_key": self.idempotency_key,
            "changed": self.changed,
            "blocked_reason": self.blocked_reason,
            "conflict_flags": list(self.conflict_flags),
            "state": self.state.to_json_dict(include_text=include_text) if self.state else None,
            "audit_event": dict(self.audit_event),
            "metadata": dict(self.metadata),
            "safety": telegram_native_draft_safety_contract(),
        }


class TelegramNativeDraftClientProtocol(Protocol):
    def save_draft(self, intent: TelegramNativeDraftIntent) -> TelegramNativeDraftResult:
        """Write a native Telegram draft. Must not send a message."""

    def clear_draft(
        self,
        chat_id: str,
        *,
        reason: str = "manual_clear",
        idempotency_key: Optional[str] = None,
    ) -> TelegramNativeDraftResult:
        """Clear a Mango-owned native draft. Must not send a message."""

    def get_draft_state(self, chat_id: str) -> TelegramNativeDraftState:
        """Read draft state without changing Telegram."""


class FakeTelegramNativeDraftClient:
    """Deterministic fake TDLib draft writer for tests and offline dry-runs."""

    def __init__(self, *, clock: Optional[Clock] = None) -> None:
        self._clock = clock or now_utc
        self._states: dict[str, TelegramNativeDraftState] = {}
        self.operations: list[TelegramNativeDraftResult] = []

    def save_draft(self, intent: TelegramNativeDraftIntent) -> TelegramNativeDraftResult:
        if not isinstance(intent, TelegramNativeDraftIntent):
            raise TypeError("intent must be TelegramNativeDraftIntent")
        if intent.operation != NATIVE_DRAFT_OPERATION_SAVE:
            raise ValueError("save_draft requires save_draft intent")
        current = self.get_draft_state(intent.chat_id)
        if current.owner == "manager" and not current.is_empty and not intent.allow_overwrite_manager_draft:
            return self._record(
                TelegramNativeDraftResult(
                    operation=intent.operation,
                    chat_id=intent.chat_id,
                    status=NATIVE_DRAFT_STATUS_CONFLICT,
                    idempotency_key=intent.idempotency_key or "",
                    changed=False,
                    blocked_reason="manager_owned_draft_present",
                    conflict_flags=("manager_draft_present",),
                    state=current,
                    audit_event=build_native_draft_audit_event(
                        "native_draft_conflict",
                        intent=intent,
                        state=current,
                        reason="manager_owned_draft_present",
                        conflict_flags=("manager_draft_present",),
                    ),
                    metadata={"telegram_api_called": False, "network_calls": False, "live_send": False},
                )
            )
        if current.owner == "mango" and current.text_hash == intent.text_hash and not current.stale:
            return self._record(
                TelegramNativeDraftResult(
                    operation=intent.operation,
                    chat_id=intent.chat_id,
                    status=NATIVE_DRAFT_STATUS_UNCHANGED,
                    idempotency_key=intent.idempotency_key or "",
                    changed=False,
                    state=current,
                    audit_event=build_native_draft_audit_event("native_draft_unchanged", intent=intent, state=current),
                    metadata={"telegram_api_called": False, "network_calls": False, "live_send": False},
                )
            )
        state = TelegramNativeDraftState(
            chat_id=intent.chat_id,
            text=intent.text,
            owner="mango",
            reply_to_message_id=intent.reply_to_message_id,
            last_intent_idempotency_key=intent.idempotency_key,
            last_draft_id=intent.draft_id,
            last_written_hash=intent.text_hash,
            updated_at=self._now(),
            stale=False,
            metadata={
                "source": "fake_telegram_native_draft_client",
                "channel_thread_id": intent.channel_thread_id,
                "source_message_idempotency_key": intent.source_message_idempotency_key,
            },
        )
        self._states[intent.chat_id] = state
        return self._record(
            TelegramNativeDraftResult(
                operation=intent.operation,
                chat_id=intent.chat_id,
                status=NATIVE_DRAFT_STATUS_SAVED,
                idempotency_key=intent.idempotency_key or "",
                changed=True,
                state=state,
                audit_event=build_native_draft_audit_event("native_draft_saved", intent=intent, state=state),
                metadata={"telegram_api_called": False, "network_calls": False, "live_send": False},
            )
        )

    def clear_draft(
        self,
        chat_id: str,
        *,
        reason: str = "manual_clear",
        idempotency_key: Optional[str] = None,
    ) -> TelegramNativeDraftResult:
        chat = require_text(chat_id, "chat_id")
        current = self.get_draft_state(chat)
        key = optional_text(idempotency_key) or f"telegram_native_draft_clear:{stable_digest({'chat_id': chat, 'reason': reason})[:32]}"
        intent = TelegramNativeDraftIntent(
            operation=NATIVE_DRAFT_OPERATION_CLEAR,
            chat_id=chat,
            actor="fake_telegram_native_draft_client",
            idempotency_key=key,
            metadata={"reason": reason},
        )
        if current.is_empty:
            return self._record(
                TelegramNativeDraftResult(
                    operation=NATIVE_DRAFT_OPERATION_CLEAR,
                    chat_id=chat,
                    status=NATIVE_DRAFT_STATUS_EMPTY,
                    idempotency_key=key,
                    changed=False,
                    state=current,
                    audit_event=build_native_draft_audit_event("native_draft_empty", intent=intent, state=current),
                    metadata={"telegram_api_called": False, "network_calls": False, "live_send": False},
                )
            )
        if current.owner != "mango":
            return self._record(
                TelegramNativeDraftResult(
                    operation=NATIVE_DRAFT_OPERATION_CLEAR,
                    chat_id=chat,
                    status=NATIVE_DRAFT_STATUS_CONFLICT,
                    idempotency_key=key,
                    changed=False,
                    blocked_reason="cannot_clear_manager_owned_draft",
                    conflict_flags=("manager_draft_present",),
                    state=current,
                    audit_event=build_native_draft_audit_event(
                        "native_draft_clear_conflict",
                        intent=intent,
                        state=current,
                        reason="cannot_clear_manager_owned_draft",
                        conflict_flags=("manager_draft_present",),
                    ),
                    metadata={"telegram_api_called": False, "network_calls": False, "live_send": False},
                )
            )
        empty = TelegramNativeDraftState.empty(chat, updated_at=self._now())
        self._states[chat] = empty
        return self._record(
            TelegramNativeDraftResult(
                operation=NATIVE_DRAFT_OPERATION_CLEAR,
                chat_id=chat,
                status=NATIVE_DRAFT_STATUS_CLEARED,
                idempotency_key=key,
                changed=True,
                state=empty,
                audit_event=build_native_draft_audit_event("native_draft_cleared", intent=intent, state=empty),
                metadata={"telegram_api_called": False, "network_calls": False, "live_send": False},
            )
        )

    def get_draft_state(self, chat_id: str) -> TelegramNativeDraftState:
        chat = require_text(chat_id, "chat_id")
        return self._states.get(chat) or TelegramNativeDraftState.empty(chat, updated_at=self._now())

    def inject_manager_draft(self, chat_id: str, text: str) -> TelegramNativeDraftState:
        state = TelegramNativeDraftState(
            chat_id=chat_id,
            text=require_text(text, "text"),
            owner="manager",
            updated_at=self._now(),
            metadata={"source": "test_manager_draft"},
        )
        self._states[state.chat_id] = state
        return state

    def mark_stale(self, chat_id: str, *, reason: str) -> TelegramNativeDraftState:
        current = self.get_draft_state(chat_id)
        if current.is_empty:
            return current
        state = TelegramNativeDraftState(
            chat_id=current.chat_id,
            text=current.text,
            owner=current.owner,
            reply_to_message_id=current.reply_to_message_id,
            last_intent_idempotency_key=current.last_intent_idempotency_key,
            last_draft_id=current.last_draft_id,
            last_written_hash=current.last_written_hash,
            updated_at=self._now(),
            stale=True,
            stale_reason=reason,
            metadata=current.metadata,
        )
        self._states[state.chat_id] = state
        return state

    def _record(self, result: TelegramNativeDraftResult) -> TelegramNativeDraftResult:
        self.operations.append(result)
        return result

    def _now(self) -> datetime:
        value = self._clock()
        require_timezone(value, "clock value")
        return value


class TDLibTelegramNativeDraftClient:
    """Draft-only TDLib adapter stub.

    The real TDLib transport is intentionally not implemented until explicit
    credentials and account access are provided. This class proves the public
    contract while guaranteeing no network calls in the current MVP slice.
    """

    def __init__(self, config: TelegramNativeDraftConfig, *, clock: Optional[Clock] = None) -> None:
        if not isinstance(config, TelegramNativeDraftConfig):
            raise TypeError("config must be TelegramNativeDraftConfig")
        self.config = config
        self._clock = clock or now_utc
        if config.tdlib_database_dir:
            guard_tdlib_database_dir(config.tdlib_database_dir)

    def save_draft(self, intent: TelegramNativeDraftIntent) -> TelegramNativeDraftResult:
        return TelegramNativeDraftResult.blocked(
            intent,
            reason="tdlib_transport_not_configured",
            metadata={"telegram_api_called": False, "network_calls": False, "live_send": False},
        )

    def clear_draft(
        self,
        chat_id: str,
        *,
        reason: str = "manual_clear",
        idempotency_key: Optional[str] = None,
    ) -> TelegramNativeDraftResult:
        intent = TelegramNativeDraftIntent(
            operation=NATIVE_DRAFT_OPERATION_CLEAR,
            chat_id=chat_id,
            actor="tdlib_telegram_native_draft_client",
            idempotency_key=idempotency_key,
            metadata={"reason": reason},
        )
        return TelegramNativeDraftResult.blocked(
            intent,
            reason="tdlib_transport_not_configured",
            metadata={"telegram_api_called": False, "network_calls": False, "live_send": False},
        )

    def get_draft_state(self, chat_id: str) -> TelegramNativeDraftState:
        return TelegramNativeDraftState.empty(chat_id, updated_at=self._now())

    def _now(self) -> datetime:
        value = self._clock()
        require_timezone(value, "clock value")
        return value


class TelegramNativeDraftMemoryStore:
    """Local audit/idempotency store for native draft intents and results."""

    def __init__(self) -> None:
        self._intents: dict[str, TelegramNativeDraftIntent] = {}
        self._results: dict[str, TelegramNativeDraftResult] = {}

    def record_intent(self, intent: TelegramNativeDraftIntent) -> bool:
        key = require_text(intent.idempotency_key, "intent.idempotency_key")
        if key in self._intents:
            return False
        self._intents[key] = intent
        return True

    def record_result(self, result: TelegramNativeDraftResult) -> bool:
        key = require_text(result.idempotency_key, "result.idempotency_key")
        if key in self._results:
            return False
        self._results[key] = result
        return True

    def get_result(self, idempotency_key: str) -> Optional[TelegramNativeDraftResult]:
        return self._results.get(require_text(idempotency_key, "idempotency_key"))

    def snapshot(self, *, include_text: bool = False) -> Mapping[str, Any]:
        return {
            "schema_version": TELEGRAM_NATIVE_DRAFT_SCHEMA_VERSION,
            "intents": [intent.to_json_dict(include_text=include_text) for intent in self._intents.values()],
            "results": [result.to_json_dict(include_text=include_text) for result in self._results.values()],
            "summary": {"intents": len(self._intents), "results": len(self._results)},
            "safety": telegram_native_draft_safety_contract(),
        }


class NativeDraftOrchestrator:
    def __init__(
        self,
        client: TelegramNativeDraftClientProtocol,
        *,
        config: Optional[TelegramNativeDraftConfig] = None,
        store: Optional[TelegramNativeDraftMemoryStore] = None,
    ) -> None:
        self.client = client
        self.config = config or TelegramNativeDraftConfig()
        self.store = store

    def build_intent(
        self,
        draft: ChannelDraftPreview | ChannelDraftRecord,
        *,
        chat_id: Optional[str] = None,
        chat_ref_map: Optional[Mapping[str, str]] = None,
        actor: str = "native_draft_orchestrator",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> TelegramNativeDraftIntent:
        preview = preview_from_draft(draft)
        resolved_chat_id, resolution = resolve_native_draft_chat_ref(
            preview.source_message,
            explicit_chat_id=chat_id,
            chat_ref_map=chat_ref_map,
        )
        return TelegramNativeDraftIntent(
            operation=NATIVE_DRAFT_OPERATION_SAVE,
            chat_id=resolved_chat_id,
            text=preview.reply.text,
            draft_id=preview.draft_id,
            channel_thread_id=preview.session.channel_thread_id,
            source_message_idempotency_key=preview.source_message.idempotency_key,
            actor=actor,
            metadata={
                "source": "native_draft_orchestrator",
                "chat_ref_resolution": resolution,
                "requires_manager_review": True,
                "source_channel": preview.source_message.channel,
                "context_keys": list(preview.metadata.get("context_keys") or ()),
                **dict(metadata or {}),
            },
        )

    def save_from_draft(
        self,
        draft: ChannelDraftPreview | ChannelDraftRecord,
        *,
        chat_id: Optional[str] = None,
        chat_ref_map: Optional[Mapping[str, str]] = None,
        actor: str = "native_draft_orchestrator",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> TelegramNativeDraftResult:
        intent = self.build_intent(
            draft,
            chat_id=chat_id,
            chat_ref_map=chat_ref_map,
            actor=actor,
            metadata=metadata,
        )
        return self.save_intent(intent)

    def save_intent(self, intent: TelegramNativeDraftIntent) -> TelegramNativeDraftResult:
        if self.store:
            self.store.record_intent(intent)
        blocked = self._blocked_by_config(intent)
        if blocked is not None:
            if self.store:
                self.store.record_result(blocked)
            return blocked
        result = self.client.save_draft(intent)
        if self.store:
            self.store.record_result(result)
        return result

    def clear_draft(self, chat_id: str, *, reason: str = "manual_clear") -> TelegramNativeDraftResult:
        intent = TelegramNativeDraftIntent(
            operation=NATIVE_DRAFT_OPERATION_CLEAR,
            chat_id=chat_id,
            actor="native_draft_orchestrator",
            metadata={"reason": reason},
        )
        if self.store:
            self.store.record_intent(intent)
        blocked = self._blocked_by_config(intent, allow_read=False)
        if blocked is not None:
            if self.store:
                self.store.record_result(blocked)
            return blocked
        result = self.client.clear_draft(chat_id, reason=reason, idempotency_key=intent.idempotency_key)
        if self.store:
            self.store.record_result(result)
        return result

    def get_draft_state(self, chat_id: str) -> TelegramNativeDraftState:
        return self.client.get_draft_state(chat_id)

    def mark_stale_after_inbound(self, chat_id: str, message: ChannelMessage) -> TelegramNativeDraftState:
        if hasattr(self.client, "mark_stale"):
            return self.client.mark_stale(chat_id, reason=f"new_inbound:{message.idempotency_key}")  # type: ignore[attr-defined]
        return self.client.get_draft_state(chat_id)

    def reconcile_manual_send(
        self,
        chat_id: str,
        *,
        sent_text: Optional[str],
        last_written_hash: Optional[str],
    ) -> Mapping[str, Any]:
        sent_hash = draft_text_hash(sent_text)
        if sent_hash and last_written_hash and sent_hash == last_written_hash:
            status = "manual_send_observed"
        elif sent_hash:
            status = "manager_sent_modified_text"
        else:
            status = "unknown_send_state"
        return {
            "schema_version": TELEGRAM_NATIVE_DRAFT_SCHEMA_VERSION,
            "chat_id": require_text(chat_id, "chat_id"),
            "status": status,
            "sent_text_hash": sent_hash,
            "last_written_hash": optional_text(last_written_hash),
            "telegram_send_called": False,
            "live_send": False,
        }

    def _blocked_by_config(
        self,
        intent: TelegramNativeDraftIntent,
        *,
        allow_read: bool = False,
    ) -> Optional[TelegramNativeDraftResult]:
        if not self.config.enabled:
            return TelegramNativeDraftResult.blocked(
                intent,
                reason="native_drafts_disabled",
                metadata={"telegram_api_called": False, "network_calls": False, "live_send": False},
            )
        if self.config.kill_switch and not (allow_read and intent.operation == NATIVE_DRAFT_OPERATION_GET):
            return TelegramNativeDraftResult.blocked(
                intent,
                reason="native_draft_kill_switch",
                metadata={"telegram_api_called": False, "network_calls": False, "live_send": False},
            )
        if self.config.allowed_chat_ids and intent.chat_id not in self.config.allowed_chat_ids:
            return TelegramNativeDraftResult.blocked(
                intent,
                reason="chat_not_allowlisted",
                metadata={"telegram_api_called": False, "network_calls": False, "live_send": False},
            )
        return None


def build_native_draft_intent_from_channel_draft(
    draft: ChannelDraftPreview | ChannelDraftRecord,
    *,
    chat_id: Optional[str] = None,
    chat_ref_map: Optional[Mapping[str, str]] = None,
    actor: str = "native_draft_orchestrator",
    metadata: Optional[Mapping[str, Any]] = None,
) -> TelegramNativeDraftIntent:
    return NativeDraftOrchestrator(
        FakeTelegramNativeDraftClient(),
        config=TelegramNativeDraftConfig(enabled=True, kill_switch=False),
    ).build_intent(
        draft,
        chat_id=chat_id,
        chat_ref_map=chat_ref_map,
        actor=actor,
        metadata=metadata,
    )


def preview_from_draft(draft: ChannelDraftPreview | ChannelDraftRecord) -> ChannelDraftPreview:
    if isinstance(draft, ChannelDraftPreview):
        return draft
    if isinstance(draft, ChannelDraftRecord):
        return draft.preview
    raise TypeError("draft must be ChannelDraftPreview or ChannelDraftRecord")


def resolve_native_draft_chat_ref(
    message: ChannelMessage,
    *,
    explicit_chat_id: Optional[str] = None,
    chat_ref_map: Optional[Mapping[str, str]] = None,
) -> tuple[str, Mapping[str, Any]]:
    if explicit_chat_id:
        return require_text(explicit_chat_id, "chat_id"), {"source": "explicit_chat_id", "verified": True}
    mapping = dict(chat_ref_map or {})
    if message.channel_thread_id in mapping:
        return require_text(mapping[message.channel_thread_id], "mapped_chat_id"), {
            "source": "chat_ref_map",
            "verified": True,
        }
    metadata_chat_id = optional_text(message.metadata.get("telegram_chat_id"))
    if metadata_chat_id:
        return metadata_chat_id, {"source": "telegram_metadata_chat_id", "verified": False}
    if message.channel == TELEGRAM_BUSINESS_CHANNEL and ":" in message.channel_thread_id:
        return message.channel_thread_id.split(":", 1)[1], {
            "source": "business_thread_suffix",
            "verified": False,
        }
    return message.channel_thread_id, {"source": "channel_thread_id", "verified": False}


def build_native_draft_audit_event(
    event_type: str,
    *,
    intent: TelegramNativeDraftIntent,
    state: Optional[TelegramNativeDraftState] = None,
    reason: Optional[str] = None,
    conflict_flags: Sequence[str] = (),
) -> Mapping[str, Any]:
    return {
        "schema_version": TELEGRAM_NATIVE_DRAFT_SCHEMA_VERSION,
        "event_type": normalize_key(event_type, "event_type"),
        "operation": intent.operation,
        "chat_id": intent.chat_id,
        "draft_id": intent.draft_id,
        "channel_thread_id": intent.channel_thread_id,
        "source_message_idempotency_key": intent.source_message_idempotency_key,
        "idempotency_key": intent.idempotency_key,
        "text_hash": intent.text_hash,
        "text_length": intent.text_length,
        "state_text_hash": state.text_hash if state else None,
        "state_text_length": state.text_length if state else None,
        "state_owner": state.owner if state else None,
        "reason": optional_text(reason),
        "conflict_flags": list(conflict_flags),
        "live_send": False,
        "telegram_api_called": False,
        "network_calls": False,
    }


def build_native_draft_manager_summary(
    draft: ChannelDraftPreview | ChannelDraftRecord,
    result: TelegramNativeDraftResult,
    *,
    identity_status: Optional[str] = None,
    source_refs: Sequence[str] = (),
) -> Mapping[str, Any]:
    preview = preview_from_draft(draft)
    return {
        "schema_version": TELEGRAM_NATIVE_DRAFT_SCHEMA_VERSION,
        "draft_id": preview.draft_id,
        "channel": preview.source_message.channel,
        "channel_thread_id": preview.source_message.channel_thread_id,
        "channel_user_id": preview.source_message.channel_user_id,
        "source_message_id": preview.source_message.channel_message_id,
        "identity_status": optional_text(identity_status) or "unknown",
        "source_refs": [str(item) for item in source_refs if str(item).strip()],
        "exact_draft_text": preview.reply.text,
        "warnings": sorted(
            set(preview.blocked_reasons)
            | set(preview.reply.safety_flags)
            | set(result.conflict_flags)
            | ({result.blocked_reason} if result.blocked_reason else set())
        ),
        "native_draft_status": result.status,
        "native_draft_blocked_reason": result.blocked_reason,
        "native_draft_conflict_flags": list(result.conflict_flags),
        "native_draft_written_at": result.state.updated_at.isoformat() if result.state else None,
        "native_draft_state": result.state.to_json_dict(include_text=False) if result.state else None,
        "actions": {
            "available": ("open_telegram_chat", "inspect_context", "reply_manually"),
            "client_send_button_available": False,
            "native_draft_send_available": False,
        },
        "safety": telegram_native_draft_safety_contract(),
    }


def normalize_operation(value: str) -> str:
    operation = normalize_key(value, "native draft operation")
    if operation not in NATIVE_DRAFT_OPERATIONS:
        raise ValueError(f"unsupported native draft operation: {operation!r}")
    return operation


def draft_text_hash(text: Optional[str]) -> Optional[str]:
    value = optional_text(text)
    if not value:
        return None
    return stable_digest({"draft_text": value})


def env_bool(env: Mapping[str, str], key: str, *, default: bool) -> bool:
    value = optional_text(env.get(key))
    if value is None:
        return default
    return value.casefold() in {"1", "true", "yes", "y", "on"}


def split_env_list(value: Optional[str]) -> tuple[str, ...]:
    text = optional_text(value)
    if not text:
        return ()
    return tuple(item.strip() for item in re.split(r"[,;\s]+", text) if item.strip())


def guard_tdlib_database_dir(path: Path | str, *, repo_root: Optional[Path | str] = None) -> Path:
    raw = Path(path)
    if not raw.is_absolute():
        raise ValueError("TDLib database dir must be an absolute path outside the repository")
    resolved = raw.expanduser().resolve(strict=False)
    if any(part == "stable_runtime" for part in resolved.parts):
        raise ValueError("TDLib database dir must not be under stable_runtime")
    if resolved.name in {"runtime.db", "stable_runtime.db", "tdlib_session", "session", ".env"}:
        raise ValueError("TDLib database dir looks like a runtime/session secret path")
    public_roots = {Path("/tmp").resolve(), Path("/var/tmp").resolve()}
    for public_root in public_roots:
        if is_relative_to(resolved, public_root):
            raise ValueError("TDLib database dir must not be under a public temp path")
    root = Path(repo_root).expanduser().resolve(strict=False) if repo_root else Path.cwd().resolve(strict=False)
    if is_relative_to(resolved, root):
        raise ValueError("TDLib database dir must be outside the git repository")
    return resolved


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


SENSITIVE_KEY_PARTS = (
    "token",
    "secret",
    "password",
    "passwd",
    "authorization",
    "api_key",
    "apikey",
    "api_hash",
    "encryption_key",
    "phone",
    "tdlib_database_dir",
    "database_dir",
    "raw_payload",
    "session",
    "пароль",
    "токен",
    "ключ",
)


def scrub_native_draft_payload(value: Any) -> Any:
    if isinstance(value, Mapping):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if any(part in key_text.casefold() for part in SENSITIVE_KEY_PARTS):
                cleaned[key_text] = "[REDACTED]"
            else:
                cleaned[key_text] = scrub_native_draft_payload(item)
        return cleaned
    if isinstance(value, list):
        return [scrub_native_draft_payload(item) for item in value]
    if isinstance(value, tuple):
        return tuple(scrub_native_draft_payload(item) for item in value)
    if isinstance(value, str) and looks_like_secret_or_local_path(value):
        return "[REDACTED]"
    return value


def looks_like_secret_or_local_path(value: str) -> bool:
    lowered = value.casefold()
    if any(marker in lowered for marker in ("tdlib_api_hash", "tdlib_database_encryption_key", "bot_token")):
        return True
    return value.startswith("/Users/") or value.startswith("/private/") or "/stable_runtime/" in value


def telegram_native_draft_safety_contract() -> Mapping[str, bool]:
    return {
        "native_draft_allowed": True,
        "network_calls": False,
        "telegram_api_called_in_tests": False,
        "live_send": False,
        "send_endpoint": False,
        "raw_tdlib_rpc_endpoint": False,
        "batch_operations": False,
        "write_crm": False,
        "write_tallanto": False,
        "write_runtime_db": False,
        "run_asr": False,
        "run_ra": False,
        "requires_manager_manual_send": True,
    }


__all__ = [
    "MAX_NATIVE_DRAFT_TEXT_LENGTH",
    "NATIVE_DRAFT_OPERATION_CLEAR",
    "NATIVE_DRAFT_OPERATION_GET",
    "NATIVE_DRAFT_OPERATION_SAVE",
    "NATIVE_DRAFT_STATUS_BLOCKED",
    "NATIVE_DRAFT_STATUS_CLEARED",
    "NATIVE_DRAFT_STATUS_CONFLICT",
    "NATIVE_DRAFT_STATUS_EMPTY",
    "NATIVE_DRAFT_STATUS_SAVED",
    "NATIVE_DRAFT_STATUS_STALE",
    "NATIVE_DRAFT_STATUS_UNCHANGED",
    "TELEGRAM_NATIVE_DRAFT_SCHEMA_VERSION",
    "FakeTelegramNativeDraftClient",
    "NativeDraftOrchestrator",
    "TDLibTelegramNativeDraftClient",
    "TelegramNativeDraftClientProtocol",
    "TelegramNativeDraftConfig",
    "TelegramNativeDraftIntent",
    "TelegramNativeDraftMemoryStore",
    "TelegramNativeDraftResult",
    "TelegramNativeDraftState",
    "build_native_draft_audit_event",
    "build_native_draft_intent_from_channel_draft",
    "build_native_draft_manager_summary",
    "draft_text_hash",
    "guard_tdlib_database_dir",
    "resolve_native_draft_chat_ref",
    "scrub_native_draft_payload",
    "telegram_native_draft_safety_contract",
]
