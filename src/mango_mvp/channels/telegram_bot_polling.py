from __future__ import annotations

import logging
import os
import inspect
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Mapping, Optional, Sequence

from mango_mvp.channels.contracts import (
    ChannelDirection,
    ChannelMessage,
    SendResult,
    optional_text,
    require_text,
    require_timezone,
    stable_digest,
)
from mango_mvp.channels.preview_service import ChannelDraftPreview, ChannelPreviewService
from mango_mvp.channels.storage import (
    ChannelMemoryStore,
    ChannelPreviewStoreResult,
    ChannelStore,
    ChannelStoreWriteResult,
    build_and_store_channel_draft_preview,
)
from mango_mvp.channels.telegram_adapter import (
    TELEGRAM_BOT_CHANNEL,
    TelegramReadOnlyAdapter,
    ensure_mapping,
    mapping_or_none,
)


TELEGRAM_BOT_POLLING_SCHEMA_VERSION = "telegram_bot_polling_v1"
DEFAULT_DEBOUNCE_SECONDS = 7
MIN_DEBOUNCE_SECONDS = 5
MAX_DEBOUNCE_SECONDS = 10

POLLING_STATUS_ACCEPTED = "accepted"
POLLING_STATUS_BLOCKED = "blocked"
POLLING_STATUS_DUPLICATE_UPDATE = "duplicate_update"
POLLING_STATUS_IGNORED = "ignored"

Clock = Callable[[], datetime]

LOGGER = logging.getLogger(__name__)


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class TelegramBotPollingConfig:
    enabled: bool = False
    kill_switch: bool = False
    bot_token: Optional[str] = field(default=None, repr=False, compare=False)
    debounce_seconds: int = DEFAULT_DEBOUNCE_SECONDS
    private_chats_only: bool = True
    client_send_enabled: bool = False

    def __post_init__(self) -> None:
        debounce = int(self.debounce_seconds)
        if not MIN_DEBOUNCE_SECONDS <= debounce <= MAX_DEBOUNCE_SECONDS:
            raise ValueError("Telegram pilot debounce must be between 5 and 10 seconds")
        object.__setattr__(self, "enabled", bool(self.enabled))
        object.__setattr__(self, "kill_switch", bool(self.kill_switch))
        object.__setattr__(self, "bot_token", optional_text(self.bot_token))
        object.__setattr__(self, "debounce_seconds", debounce)
        object.__setattr__(self, "private_chats_only", bool(self.private_chats_only))
        object.__setattr__(self, "client_send_enabled", False)

    @classmethod
    def from_env(
        cls,
        env: Optional[Mapping[str, str]] = None,
        *,
        load_dotenv_file: bool = True,
    ) -> "TelegramBotPollingConfig":
        if env is None and load_dotenv_file:
            try:
                from dotenv import load_dotenv
            except ImportError:
                load_dotenv = None
            if load_dotenv is not None:
                load_dotenv()
        source = os.environ if env is None else env
        return cls(
            enabled=env_bool(source, "TELEGRAM_PILOT_ENABLED", default=False),
            kill_switch=env_bool(source, "TELEGRAM_PILOT_KILL_SWITCH", default=False),
            bot_token=source.get("TELEGRAM_BOT_TOKEN"),
            debounce_seconds=env_int(source, "TELEGRAM_PILOT_DEBOUNCE_SECONDS", default=DEFAULT_DEBOUNCE_SECONDS),
            private_chats_only=env_bool(source, "TELEGRAM_PILOT_PRIVATE_ONLY", default=True),
            client_send_enabled=False,
        )

    def blocking_reason(self, *, require_token: bool = False) -> Optional[str]:
        if not self.enabled:
            return "telegram_pilot_disabled"
        if self.kill_switch:
            return "telegram_pilot_kill_switch"
        if require_token and not self.bot_token:
            return "telegram_bot_token_missing"
        return None

    def to_json_dict(self) -> Mapping[str, Any]:
        return {
            "schema_version": TELEGRAM_BOT_POLLING_SCHEMA_VERSION,
            "enabled": self.enabled,
            "kill_switch": self.kill_switch,
            "bot_token_present": bool(self.bot_token),
            "debounce_seconds": self.debounce_seconds,
            "private_chats_only": self.private_chats_only,
            "client_send_enabled": False,
            "safety": telegram_bot_polling_safety_contract(),
        }


@dataclass(frozen=True)
class TelegramDebounceBatch:
    client_key: str
    messages: Sequence[ChannelMessage]
    first_seen_at: datetime
    last_seen_at: datetime
    due_at: datetime

    def __post_init__(self) -> None:
        messages = tuple(self.messages)
        if not messages:
            raise ValueError("debounce batch requires at least one message")
        if any(not isinstance(item, ChannelMessage) for item in messages):
            raise TypeError("debounce batch messages must contain ChannelMessage items")
        require_timezone(self.first_seen_at, "first_seen_at")
        require_timezone(self.last_seen_at, "last_seen_at")
        require_timezone(self.due_at, "due_at")
        object.__setattr__(self, "client_key", require_text(self.client_key, "client_key"))
        object.__setattr__(self, "messages", messages)

    @property
    def source_message_idempotency_keys(self) -> tuple[str, ...]:
        return tuple(message.idempotency_key for message in self.messages)

    def with_message(self, message: ChannelMessage, *, now: datetime, debounce_seconds: int) -> "TelegramDebounceBatch":
        require_timezone(now, "now")
        return TelegramDebounceBatch(
            client_key=self.client_key,
            messages=tuple(self.messages) + (message,),
            first_seen_at=self.first_seen_at,
            last_seen_at=now,
            due_at=now + timedelta(seconds=debounce_seconds),
        )

    def to_json_dict(self) -> Mapping[str, Any]:
        return {
            "schema_version": TELEGRAM_BOT_POLLING_SCHEMA_VERSION,
            "client_key": self.client_key,
            "message_count": len(self.messages),
            "source_message_idempotency_keys": list(self.source_message_idempotency_keys),
            "first_seen_at": self.first_seen_at.isoformat(),
            "last_seen_at": self.last_seen_at.isoformat(),
            "due_at": self.due_at.isoformat(),
        }


@dataclass(frozen=True)
class TelegramDebounceResult:
    client_key: str
    message_count: int
    due_at: datetime
    source_message_idempotency_keys: Sequence[str]

    def __post_init__(self) -> None:
        require_timezone(self.due_at, "due_at")
        object.__setattr__(self, "client_key", require_text(self.client_key, "client_key"))
        object.__setattr__(
            self,
            "source_message_idempotency_keys",
            tuple(require_text(item, "source_message_idempotency_key") for item in self.source_message_idempotency_keys),
        )
        if self.message_count <= 0:
            raise ValueError("message_count must be positive")

    def to_json_dict(self) -> Mapping[str, Any]:
        return {
            "schema_version": TELEGRAM_BOT_POLLING_SCHEMA_VERSION,
            "client_key": self.client_key,
            "message_count": self.message_count,
            "due_at": self.due_at.isoformat(),
            "source_message_idempotency_keys": list(self.source_message_idempotency_keys),
        }


@dataclass(frozen=True)
class TelegramBotDraftResult:
    preview: ChannelDraftPreview
    store_result: ChannelPreviewStoreResult
    source_message_idempotency_keys: Sequence[str]
    debounce_window_seconds: int

    def __post_init__(self) -> None:
        if not isinstance(self.preview, ChannelDraftPreview):
            raise TypeError("preview must be ChannelDraftPreview")
        if not isinstance(self.store_result, ChannelPreviewStoreResult):
            raise TypeError("store_result must be ChannelPreviewStoreResult")
        object.__setattr__(
            self,
            "source_message_idempotency_keys",
            tuple(require_text(item, "source_message_idempotency_key") for item in self.source_message_idempotency_keys),
        )

    @property
    def draft_id(self) -> str:
        return self.store_result.draft_id

    @property
    def created(self) -> bool:
        return self.store_result.created

    def to_json_dict(self, *, include_message_text: bool = False) -> Mapping[str, Any]:
        preview_payload = self.preview.to_json_dict()
        if not include_message_text:
            source = dict(preview_payload["source_message"])
            source["text"] = None
            source["text_redacted"] = True
            preview_payload = dict(preview_payload) | {"source_message": source}
        return {
            "schema_version": TELEGRAM_BOT_POLLING_SCHEMA_VERSION,
            "draft_id": self.draft_id,
            "created": self.created,
            "status": self.store_result.status,
            "debounce_window_seconds": self.debounce_window_seconds,
            "source_message_idempotency_keys": list(self.source_message_idempotency_keys),
            "preview": preview_payload,
        }


@dataclass(frozen=True)
class TelegramBotPollingResult:
    status: str
    update_idempotency_key: Optional[str] = None
    messages: Sequence[ChannelMessage] = field(default_factory=tuple)
    message_results: Sequence[ChannelStoreWriteResult] = field(default_factory=tuple)
    debounce_results: Sequence[TelegramDebounceResult] = field(default_factory=tuple)
    draft_results: Sequence[TelegramBotDraftResult] = field(default_factory=tuple)
    blocked_reason: Optional[str] = None
    skipped_reason: Optional[str] = None
    telegram_api_called: bool = False
    network_calls: bool = False
    client_send_attempted: bool = False

    def __post_init__(self) -> None:
        messages = tuple(self.messages)
        if any(not isinstance(item, ChannelMessage) for item in messages):
            raise TypeError("messages must contain ChannelMessage items")
        object.__setattr__(self, "messages", messages)
        object.__setattr__(self, "message_results", tuple(self.message_results))
        object.__setattr__(self, "debounce_results", tuple(self.debounce_results))
        object.__setattr__(self, "draft_results", tuple(self.draft_results))
        object.__setattr__(self, "update_idempotency_key", optional_text(self.update_idempotency_key))
        object.__setattr__(self, "blocked_reason", optional_text(self.blocked_reason))
        object.__setattr__(self, "skipped_reason", optional_text(self.skipped_reason))

    def to_json_dict(self, *, include_message_text: bool = False) -> Mapping[str, Any]:
        messages = []
        for message in self.messages:
            payload = message.to_json_dict()
            payload.pop("raw_payload", None)
            if not include_message_text:
                payload["text"] = None
                payload["text_redacted"] = True
            messages.append(payload)
        return {
            "schema_version": TELEGRAM_BOT_POLLING_SCHEMA_VERSION,
            "status": self.status,
            "update_idempotency_key": self.update_idempotency_key,
            "messages": messages,
            "message_results": [item.to_json_dict() for item in self.message_results],
            "debounce_results": [item.to_json_dict() for item in self.debounce_results],
            "draft_results": [item.to_json_dict(include_message_text=include_message_text) for item in self.draft_results],
            "blocked_reason": self.blocked_reason,
            "skipped_reason": self.skipped_reason,
            "telegram_api_called": self.telegram_api_called,
            "network_calls": self.network_calls,
            "client_send_attempted": self.client_send_attempted,
            "safety": telegram_bot_polling_safety_contract(),
        }


class TelegramDebounceBuffer:
    def __init__(self, *, debounce_seconds: int) -> None:
        if not MIN_DEBOUNCE_SECONDS <= int(debounce_seconds) <= MAX_DEBOUNCE_SECONDS:
            raise ValueError("Telegram debounce must be between 5 and 10 seconds")
        self.debounce_seconds = int(debounce_seconds)
        self._pending: dict[str, TelegramDebounceBatch] = {}

    def add_message(self, message: ChannelMessage, *, now: datetime) -> TelegramDebounceResult:
        require_timezone(now, "now")
        key = telegram_debounce_client_key(message)
        existing = self._pending.get(key)
        if existing is None:
            batch = TelegramDebounceBatch(
                client_key=key,
                messages=(message,),
                first_seen_at=now,
                last_seen_at=now,
                due_at=now + timedelta(seconds=self.debounce_seconds),
            )
        else:
            batch = existing.with_message(message, now=now, debounce_seconds=self.debounce_seconds)
        self._pending[key] = batch
        return TelegramDebounceResult(
            client_key=key,
            message_count=len(batch.messages),
            due_at=batch.due_at,
            source_message_idempotency_keys=batch.source_message_idempotency_keys,
        )

    def pop_due(self, *, now: datetime) -> tuple[TelegramDebounceBatch, ...]:
        require_timezone(now, "now")
        due_keys = [key for key, batch in self._pending.items() if batch.due_at <= now]
        due = tuple(self._pending.pop(key) for key in due_keys)
        return tuple(sorted(due, key=lambda batch: (batch.due_at, batch.client_key)))

    def pop_all(self) -> tuple[TelegramDebounceBatch, ...]:
        batches = tuple(sorted(self._pending.values(), key=lambda batch: (batch.due_at, batch.client_key)))
        self._pending.clear()
        return batches

    def pending(self) -> tuple[TelegramDebounceBatch, ...]:
        return tuple(sorted(self._pending.values(), key=lambda batch: (batch.due_at, batch.client_key)))


class TelegramBotPollingRuntime:
    """Safe Bot API inbound runtime.

    `process_update` accepts already fetched Bot API updates. It does not call
    Telegram, AMO, Tallanto, ASR, R+A, or any filesystem runtime by itself.
    """

    def __init__(
        self,
        *,
        config: Optional[TelegramBotPollingConfig] = None,
        adapter: Optional[TelegramReadOnlyAdapter] = None,
        channel_store: Optional[ChannelStore] = None,
        preview_service: Optional[ChannelPreviewService] = None,
        clock: Optional[Clock] = None,
    ) -> None:
        self.config = config or TelegramBotPollingConfig.from_env()
        self.adapter = adapter or TelegramReadOnlyAdapter()
        self.channel_store = channel_store or ChannelMemoryStore()
        self.preview_service = preview_service or ChannelPreviewService()
        self._clock = clock or now_utc
        self._seen_update_keys: set[str] = set()
        self._debounce = TelegramDebounceBuffer(debounce_seconds=self.config.debounce_seconds)

    def process_update(
        self,
        raw_update: Mapping[str, Any],
        *,
        actor: str = "telegram_bot_polling",
    ) -> TelegramBotPollingResult:
        blocked = self.config.blocking_reason(require_token=False)
        if blocked:
            return TelegramBotPollingResult(status=POLLING_STATUS_BLOCKED, blocked_reason=blocked)

        update = ensure_mapping(raw_update, "raw_update")
        update_key = telegram_bot_update_idempotency_key(update)
        if update_key in self._seen_update_keys:
            return TelegramBotPollingResult(
                status=POLLING_STATUS_DUPLICATE_UPDATE,
                update_idempotency_key=update_key,
                skipped_reason="duplicate_update",
            )
        self._seen_update_keys.add(update_key)

        parsed_messages = tuple(message for message in self.adapter.parse_inbound(update) if message.channel == TELEGRAM_BOT_CHANNEL)
        accepted_messages: list[ChannelMessage] = []
        message_results: list[ChannelStoreWriteResult] = []
        debounce_results: list[TelegramDebounceResult] = []
        skipped_reason: Optional[str] = None
        now = self._now()

        for message in parsed_messages:
            if not self._accept_message(message):
                skipped_reason = "non_private_or_non_inbound_message"
                continue
            store_result = self.channel_store.upsert_message(message, actor=actor)
            message_results.append(store_result)
            accepted_messages.append(message)
            if store_result.created:
                debounce_results.append(self._debounce.add_message(message, now=now))
            else:
                skipped_reason = "duplicate_message"

        status = POLLING_STATUS_ACCEPTED if accepted_messages else POLLING_STATUS_IGNORED
        LOGGER.info(
            "telegram_bot_update_processed",
            extra={
                "status": status,
                "update_idempotency_key": update_key,
                "messages_total": len(accepted_messages),
                "debounce_pending": len(self._debounce.pending()),
            },
        )
        return TelegramBotPollingResult(
            status=status,
            update_idempotency_key=update_key,
            messages=tuple(accepted_messages),
            message_results=tuple(message_results),
            debounce_results=tuple(debounce_results),
            skipped_reason=skipped_reason,
        )

    def flush_due(
        self,
        *,
        now: Optional[datetime] = None,
        actor: str = "telegram_bot_polling",
    ) -> tuple[TelegramBotDraftResult, ...]:
        blocked = self.config.blocking_reason(require_token=False)
        if blocked:
            return ()
        flush_time = now or self._now()
        require_timezone(flush_time, "now")
        return self._flush_batches(self._debounce.pop_due(now=flush_time), actor=actor)

    def flush_all(self, *, actor: str = "telegram_bot_polling") -> tuple[TelegramBotDraftResult, ...]:
        blocked = self.config.blocking_reason(require_token=False)
        if blocked:
            return ()
        return self._flush_batches(self._debounce.pop_all(), actor=actor)

    def pending_batches(self) -> tuple[TelegramDebounceBatch, ...]:
        return self._debounce.pending()

    def send_client_message(
        self,
        *,
        chat_id: str,
        text: str,
        idempotency_key: Optional[str] = None,
    ) -> SendResult:
        chat = require_text(chat_id, "chat_id")
        require_text(text, "text")
        key = optional_text(idempotency_key) or f"telegram_client_send_blocked:{stable_digest({'chat_id': chat})[:32]}"
        return SendResult(
            channel=TELEGRAM_BOT_CHANNEL,
            idempotency_key=key,
            sent=False,
            status="client_send_disabled",
            error="Telegram pilot phase 1 never sends messages to clients",
            metadata={
                "schema_version": TELEGRAM_BOT_POLLING_SCHEMA_VERSION,
                "telegram_api_called": False,
                "network_calls": False,
                "client_send_enabled": False,
                "live_send": False,
            },
        )

    def assert_can_start_long_polling(self) -> None:
        blocked = self.config.blocking_reason(require_token=True)
        if blocked:
            raise RuntimeError(blocked)

    def start_long_polling(
        self,
        *,
        on_drafts_ready: Optional[Callable[[Sequence[TelegramBotDraftResult]], Any]] = None,
    ) -> None:
        self.assert_can_start_long_polling()
        try:
            from telegram import Update
            from telegram.ext import Application, ContextTypes, MessageHandler, filters
        except ImportError as exc:
            raise RuntimeError("python-telegram-bot is required for live long polling") from exc

        runtime = self

        async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            del context
            runtime.process_update(update.to_dict(), actor="telegram_bot_polling")

        async def flush_job(context: ContextTypes.DEFAULT_TYPE) -> None:
            del context
            draft_results = runtime.flush_due(actor="telegram_bot_polling")
            if draft_results and on_drafts_ready is not None:
                maybe_awaitable = on_drafts_ready(draft_results)
                if inspect.isawaitable(maybe_awaitable):
                    await maybe_awaitable

        application = Application.builder().token(require_text(self.config.bot_token, "TELEGRAM_BOT_TOKEN")).build()
        application.add_handler(MessageHandler(filters.ChatType.PRIVATE, handle_message))
        if application.job_queue is not None:
            application.job_queue.run_repeating(flush_job, interval=1.0, first=1.0)
        application.run_polling(allowed_updates=("message", "edited_message"))

    def _accept_message(self, message: ChannelMessage) -> bool:
        if message.direction != ChannelDirection.INBOUND:
            return False
        if self.config.private_chats_only and message.metadata.get("telegram_chat_type") != "private":
            return False
        return True

    def _flush_batches(
        self,
        batches: Sequence[TelegramDebounceBatch],
        *,
        actor: str,
    ) -> tuple[TelegramBotDraftResult, ...]:
        results: list[TelegramBotDraftResult] = []
        for batch in batches:
            source_message = debounced_channel_message(batch)
            preview, store_result = build_and_store_channel_draft_preview(
                self.channel_store,
                source_message,
                service=self.preview_service,
                actor=actor,
            )
            results.append(
                TelegramBotDraftResult(
                    preview=preview,
                    store_result=store_result,
                    source_message_idempotency_keys=batch.source_message_idempotency_keys,
                    debounce_window_seconds=self.config.debounce_seconds,
                )
            )
        if results:
            LOGGER.info("telegram_bot_debounce_flushed", extra={"drafts_total": len(results)})
        return tuple(results)

    def _now(self) -> datetime:
        value = self._clock()
        require_timezone(value, "clock value")
        return value


def telegram_bot_update_idempotency_key(update: Mapping[str, Any]) -> str:
    update_id = require_text(update.get("update_id"), "update_id")
    update_type = telegram_bot_update_type(update)
    message = mapping_or_none(update.get(update_type)) if update_type else None
    chat = mapping_or_none(message.get("chat")) if message else None
    payload = {
        "source": "telegram_bot_api",
        "update_id": update_id,
        "update_type": update_type or "unsupported",
        "chat_id": optional_text(chat.get("id")) if chat else None,
        "message_id": optional_text(message.get("message_id")) if message else None,
    }
    return f"telegram_bot_update:{stable_digest(payload)[:32]}"


def telegram_bot_update_type(update: Mapping[str, Any]) -> Optional[str]:
    for key in ("message", "edited_message"):
        if key in update:
            return key
    return None


def telegram_debounce_client_key(message: ChannelMessage) -> str:
    payload = {
        "channel": message.channel,
        "channel_thread_id": message.channel_thread_id,
        "channel_user_id": message.channel_user_id,
    }
    return f"telegram_debounce:{stable_digest(payload)[:32]}"


def debounced_channel_message(batch: TelegramDebounceBatch) -> ChannelMessage:
    if len(batch.messages) == 1:
        return batch.messages[0]
    first = batch.messages[0]
    last = batch.messages[-1]
    text = "\n".join(message.text for message in batch.messages if message.text)
    attachments = tuple(attachment for message in batch.messages for attachment in message.attachments)
    digest = stable_digest({"source_message_idempotency_keys": list(batch.source_message_idempotency_keys)})
    return ChannelMessage(
        channel=first.channel,
        channel_message_id=f"debounce:{digest[:24]}",
        channel_thread_id=first.channel_thread_id,
        channel_user_id=first.channel_user_id,
        direction=ChannelDirection.INBOUND,
        text=text,
        received_at=last.received_at,
        attachments=attachments,
        raw_payload={
            "telegram_debounce": {
                "message_count": len(batch.messages),
                "source_message_idempotency_keys": list(batch.source_message_idempotency_keys),
            }
        },
        metadata={
            "schema_version": TELEGRAM_BOT_POLLING_SCHEMA_VERSION,
            "telegram_debounce": True,
            "telegram_debounce_message_count": len(batch.messages),
            "source_message_idempotency_keys": list(batch.source_message_idempotency_keys),
            "telegram_chat_type": first.metadata.get("telegram_chat_type"),
            "telegram_channel": TELEGRAM_BOT_CHANNEL,
        },
    )


def telegram_bot_polling_safety_contract() -> Mapping[str, bool]:
    return {
        "fake_update_processing_network_calls": False,
        "telegram_api_called_by_process_update": False,
        "client_send": False,
        "live_send": False,
        "write_crm": False,
        "write_tallanto": False,
        "write_runtime_db": False,
        "run_asr": False,
        "run_ra": False,
        "long_polling_requires_explicit_start": True,
    }


def env_bool(env: Mapping[str, str], key: str, *, default: bool) -> bool:
    raw = optional_text(env.get(key))
    if raw is None:
        return default
    normalized = raw.casefold()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"invalid boolean env {key}: {raw!r}")


def env_int(env: Mapping[str, str], key: str, *, default: int) -> int:
    raw = optional_text(env.get(key))
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"invalid integer env {key}: {raw!r}") from exc


__all__ = [
    "DEFAULT_DEBOUNCE_SECONDS",
    "MAX_DEBOUNCE_SECONDS",
    "MIN_DEBOUNCE_SECONDS",
    "POLLING_STATUS_ACCEPTED",
    "POLLING_STATUS_BLOCKED",
    "POLLING_STATUS_DUPLICATE_UPDATE",
    "POLLING_STATUS_IGNORED",
    "TELEGRAM_BOT_POLLING_SCHEMA_VERSION",
    "TelegramBotDraftResult",
    "TelegramBotPollingConfig",
    "TelegramBotPollingResult",
    "TelegramBotPollingRuntime",
    "TelegramDebounceBatch",
    "TelegramDebounceBuffer",
    "TelegramDebounceResult",
    "debounced_channel_message",
    "telegram_bot_polling_safety_contract",
    "telegram_bot_update_idempotency_key",
    "telegram_bot_update_type",
    "telegram_debounce_client_key",
]
