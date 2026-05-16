from __future__ import annotations

import os
from collections.abc import Sequence as SequenceABC
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence

from mango_mvp.channels.contracts import optional_text, require_text, require_timezone, stable_digest
from mango_mvp.channels.feedback import (
    FEEDBACK_MANAGER_DRAFT_APPROVED,
    FEEDBACK_MANAGER_DRAFT_REJECTED,
    FeedbackEvent,
    FeedbackStoreResult,
    build_manager_draft_feedback_event,
)
from mango_mvp.channels.preview_service import ChannelDraftPreview
from mango_mvp.channels.storage import ChannelDraftRecord


TELEGRAM_MANAGER_INBOX_SCHEMA_VERSION = "telegram_manager_inbox_v1"
TELEGRAM_MANAGER_CHAT_IDS_ENV = "TELEGRAM_PILOT_MANAGER_CHAT_IDS"

MANAGER_ACTION_ACCEPT = "manager_draft_accept"
MANAGER_ACTION_NEEDS_EDIT = "manager_draft_needs_edit"
MANAGER_ACTION_MANAGER_ONLY = "manager_draft_manager_only"
MANAGER_BUTTON_ACTIONS = (
    MANAGER_ACTION_ACCEPT,
    MANAGER_ACTION_NEEDS_EDIT,
    MANAGER_ACTION_MANAGER_ONLY,
)

MANAGER_DRAFT_STATUS_NEEDS_REVIEW = "needs_review"
MANAGER_DRAFT_STATUS_MARKED_USEFUL = "manager_marked_useful"
MANAGER_DRAFT_STATUS_MARKED_NEEDS_EDIT = "manager_marked_needs_edit"
MANAGER_DRAFT_STATUS_MANAGER_ONLY = "manager_only"
MANAGER_DRAFT_STATUS_BLOCKED = "blocked"
MANAGER_DRAFT_STATUS_FAILED = "failed"

MANAGER_DELIVERY_STATUS_READY = "ready_for_manager_chat"
MANAGER_DELIVERY_STATUS_BLOCKED = "blocked"
MANAGER_START_STATUS_REGISTERED = "registered"
MANAGER_START_STATUS_DUPLICATE = "duplicate"
MANAGER_START_STATUS_BLOCKED = "blocked"
MANAGER_START_STATUS_IGNORED = "ignored"

FORBIDDEN_CLIENT_SEND_BUTTON_MARKERS = (
    "отправить клиенту",
    "отправить_клиенту",
    "send_client",
    "send-to-client",
    "send_to_client",
    "client_send",
    "client-send",
    "send_customer",
    "send_to_customer",
    "customer_send",
)

Clock = Callable[[], datetime]


class FeedbackRecorder(Protocol):
    def record_event(self, event: FeedbackEvent) -> FeedbackStoreResult:
        ...


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class TelegramManagerInboxConfig:
    allowed_manager_chat_ids: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "allowed_manager_chat_ids", parse_manager_chat_ids(self.allowed_manager_chat_ids))

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "TelegramManagerInboxConfig":
        source = os.environ if env is None else env
        return cls(allowed_manager_chat_ids=parse_manager_chat_ids(source.get(TELEGRAM_MANAGER_CHAT_IDS_ENV, "")))

    @property
    def default_manager_chat_id(self) -> Optional[str]:
        return self.allowed_manager_chat_ids[0] if self.allowed_manager_chat_ids else None

    def authorize(self, chat_id: Any) -> "TelegramManagerAuthorization":
        resolved = optional_text(chat_id)
        if not resolved:
            return TelegramManagerAuthorization(
                chat_id=None,
                authorized=False,
                reason="manager_chat_id_missing",
            )
        if not self.allowed_manager_chat_ids:
            return TelegramManagerAuthorization(
                chat_id=resolved,
                authorized=False,
                reason="manager_chat_ids_not_configured",
            )
        if resolved not in set(self.allowed_manager_chat_ids):
            return TelegramManagerAuthorization(
                chat_id=resolved,
                authorized=False,
                reason="manager_chat_not_allowed",
            )
        return TelegramManagerAuthorization(chat_id=resolved, authorized=True, reason="allowed")


@dataclass(frozen=True)
class TelegramManagerAuthorization:
    chat_id: Optional[str]
    authorized: bool
    reason: str

    @property
    def blocked(self) -> bool:
        return not self.authorized

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TelegramManagerChatRegistration:
    chat_id: str
    actor: str
    registered_at: datetime
    source: str = "telegram_start"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "chat_id", require_text(self.chat_id, "chat_id"))
        object.__setattr__(self, "actor", require_text(self.actor, "actor"))
        require_timezone(self.registered_at, "registered_at")
        object.__setattr__(self, "source", require_text(self.source, "source"))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_json_dict(self) -> Mapping[str, Any]:
        return {
            "schema_version": TELEGRAM_MANAGER_INBOX_SCHEMA_VERSION,
            "chat_id": self.chat_id,
            "actor": self.actor,
            "registered_at": self.registered_at.isoformat(),
            "source": self.source,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class TelegramManagerDraftButton:
    label: str
    action: str
    callback_data: str
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        label = require_text(self.label, "button label")
        action = require_text(self.action, "button action")
        if action not in MANAGER_BUTTON_ACTIONS:
            raise ValueError(f"unsupported manager button action: {action!r}")
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "callback_data", require_text(self.callback_data, "callback_data"))
        object.__setattr__(self, "payload", dict(self.payload))
        validate_no_client_send_button(self)

    def to_json_dict(self) -> Mapping[str, Any]:
        return {
            "label": self.label,
            "action": self.action,
            "callback_data": self.callback_data,
            "payload": dict(self.payload),
        }


@dataclass(frozen=True)
class TelegramManagerDraftMessage:
    text: str
    buttons: Sequence[TelegramManagerDraftButton]
    draft_id: str
    source_message_idempotency_key: str
    status: str = MANAGER_DRAFT_STATUS_NEEDS_REVIEW
    safety: Mapping[str, bool] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "text", require_text(self.text, "manager draft text"))
        buttons = tuple(self.buttons)
        validate_manager_buttons_without_client_send(buttons)
        object.__setattr__(self, "buttons", buttons)
        object.__setattr__(self, "draft_id", require_text(self.draft_id, "draft_id"))
        object.__setattr__(
            self,
            "source_message_idempotency_key",
            require_text(self.source_message_idempotency_key, "source_message_idempotency_key"),
        )
        object.__setattr__(self, "status", require_text(self.status, "status"))
        object.__setattr__(self, "safety", dict(self.safety or telegram_manager_inbox_safety_contract()))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_json_dict(self) -> Mapping[str, Any]:
        return {
            "schema_version": TELEGRAM_MANAGER_INBOX_SCHEMA_VERSION,
            "text": self.text,
            "buttons": [button.to_json_dict() for button in self.buttons],
            "draft_id": self.draft_id,
            "source_message_idempotency_key": self.source_message_idempotency_key,
            "status": self.status,
            "safety": dict(self.safety),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class TelegramManagerDraftDelivery:
    status: str
    manager_chat_id: Optional[str]
    message: Optional[TelegramManagerDraftMessage] = None
    rendered_payload: Optional[Mapping[str, Any]] = None
    blocked_reason: Optional[str] = None
    warning: Optional[str] = None
    client_send_attempted: bool = False
    telegram_api_called: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", require_text(self.status, "status"))
        object.__setattr__(self, "manager_chat_id", optional_text(self.manager_chat_id))
        object.__setattr__(self, "blocked_reason", optional_text(self.blocked_reason))
        object.__setattr__(self, "warning", optional_text(self.warning))
        if self.message is not None and not isinstance(self.message, TelegramManagerDraftMessage):
            raise TypeError("message must be TelegramManagerDraftMessage")
        object.__setattr__(self, "rendered_payload", dict(self.rendered_payload) if self.rendered_payload else None)

    @property
    def blocked(self) -> bool:
        return self.status == MANAGER_DELIVERY_STATUS_BLOCKED

    def to_json_dict(self) -> Mapping[str, Any]:
        return {
            "schema_version": TELEGRAM_MANAGER_INBOX_SCHEMA_VERSION,
            "status": self.status,
            "manager_chat_id": self.manager_chat_id,
            "message": self.message.to_json_dict() if self.message else None,
            "rendered_payload": dict(self.rendered_payload) if self.rendered_payload else None,
            "blocked_reason": self.blocked_reason,
            "warning": self.warning,
            "client_send_attempted": self.client_send_attempted,
            "telegram_api_called": self.telegram_api_called,
        }


@dataclass(frozen=True)
class TelegramManagerStartResult:
    status: str
    chat_id: Optional[str]
    authorized: bool
    registered: bool
    reply_text: str
    rendered_payload: Optional[Mapping[str, Any]] = None
    blocked_reason: Optional[str] = None
    client_send_attempted: bool = False
    telegram_api_called: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", require_text(self.status, "status"))
        object.__setattr__(self, "chat_id", optional_text(self.chat_id))
        object.__setattr__(self, "reply_text", require_text(self.reply_text, "reply_text"))
        object.__setattr__(self, "blocked_reason", optional_text(self.blocked_reason))
        object.__setattr__(self, "rendered_payload", dict(self.rendered_payload) if self.rendered_payload else None)

    def to_json_dict(self) -> Mapping[str, Any]:
        return {
            "schema_version": TELEGRAM_MANAGER_INBOX_SCHEMA_VERSION,
            "status": self.status,
            "chat_id": self.chat_id,
            "authorized": self.authorized,
            "registered": self.registered,
            "reply_text": self.reply_text,
            "rendered_payload": dict(self.rendered_payload) if self.rendered_payload else None,
            "blocked_reason": self.blocked_reason,
            "client_send_attempted": self.client_send_attempted,
            "telegram_api_called": self.telegram_api_called,
        }


@dataclass(frozen=True)
class TelegramManagerFeedbackCommand:
    manager_chat_id: str
    draft_id: str
    action: str
    callback_query_id: Optional[str] = None
    reason: Optional[str] = None
    occurred_at: Optional[datetime] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "manager_chat_id", require_text(self.manager_chat_id, "manager_chat_id"))
        object.__setattr__(self, "draft_id", require_text(self.draft_id, "draft_id"))
        action = require_text(self.action, "action")
        if action not in MANAGER_BUTTON_ACTIONS:
            raise ValueError(f"unsupported manager feedback action: {action!r}")
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "callback_query_id", optional_text(self.callback_query_id))
        object.__setattr__(self, "reason", optional_text(self.reason))
        if self.occurred_at is not None:
            require_timezone(self.occurred_at, "occurred_at")


@dataclass(frozen=True)
class TelegramManagerDraftFeedbackState:
    draft_id: str
    status: str
    manager_chat_id: str
    action: str
    updated_at: datetime
    feedback_event_id: Optional[str] = None
    reason: Optional[str] = None
    client_send_attempted: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "draft_id", require_text(self.draft_id, "draft_id"))
        object.__setattr__(self, "status", require_text(self.status, "status"))
        object.__setattr__(self, "manager_chat_id", require_text(self.manager_chat_id, "manager_chat_id"))
        object.__setattr__(self, "action", require_text(self.action, "action"))
        require_timezone(self.updated_at, "updated_at")
        object.__setattr__(self, "feedback_event_id", optional_text(self.feedback_event_id))
        object.__setattr__(self, "reason", optional_text(self.reason))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_json_dict(self) -> Mapping[str, Any]:
        return {
            "schema_version": TELEGRAM_MANAGER_INBOX_SCHEMA_VERSION,
            "draft_id": self.draft_id,
            "status": self.status,
            "manager_chat_id": self.manager_chat_id,
            "action": self.action,
            "updated_at": self.updated_at.isoformat(),
            "feedback_event_id": self.feedback_event_id,
            "reason": self.reason,
            "client_send_attempted": self.client_send_attempted,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class TelegramManagerFeedbackResult:
    status: str
    manager_status: Optional[str]
    feedback_event: Optional[FeedbackEvent] = None
    feedback_store_result: Optional[FeedbackStoreResult] = None
    state: Optional[TelegramManagerDraftFeedbackState] = None
    blocked_reason: Optional[str] = None
    client_send_attempted: bool = False
    telegram_api_called: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", require_text(self.status, "status"))
        object.__setattr__(self, "manager_status", optional_text(self.manager_status))
        object.__setattr__(self, "blocked_reason", optional_text(self.blocked_reason))

    @property
    def blocked(self) -> bool:
        return self.status == MANAGER_DRAFT_STATUS_BLOCKED

    def to_json_dict(self) -> Mapping[str, Any]:
        return {
            "schema_version": TELEGRAM_MANAGER_INBOX_SCHEMA_VERSION,
            "status": self.status,
            "manager_status": self.manager_status,
            "feedback_event": self.feedback_event.to_json_dict() if self.feedback_event else None,
            "feedback_store_result": self.feedback_store_result.to_json_dict() if self.feedback_store_result else None,
            "state": self.state.to_json_dict() if self.state else None,
            "blocked_reason": self.blocked_reason,
            "client_send_attempted": self.client_send_attempted,
            "telegram_api_called": self.telegram_api_called,
        }


class TelegramManagerInboxMemoryStore:
    """Локальное состояние пилота без Telegram API, CRM и runtime-записей."""

    def __init__(self, *, clock: Optional[Clock] = None) -> None:
        self._clock = clock or now_utc
        self._registrations: dict[str, TelegramManagerChatRegistration] = {}
        self._feedback_state_by_draft: dict[str, TelegramManagerDraftFeedbackState] = {}

    def register_manager_chat(
        self,
        chat_id: str,
        *,
        actor: str,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> tuple[TelegramManagerChatRegistration, bool]:
        key = require_text(chat_id, "chat_id")
        existing = self._registrations.get(key)
        if existing is not None:
            return existing, False
        registration = TelegramManagerChatRegistration(
            chat_id=key,
            actor=actor,
            registered_at=self._now(),
            metadata=metadata or {},
        )
        self._registrations[key] = registration
        return registration, True

    def is_registered(self, chat_id: str) -> bool:
        return require_text(chat_id, "chat_id") in self._registrations

    def registered_chat_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._registrations))

    def first_registered_allowed_chat_id(self, config: TelegramManagerInboxConfig) -> Optional[str]:
        allowed = set(config.allowed_manager_chat_ids)
        for chat_id in self.registered_chat_ids():
            if chat_id in allowed:
                return chat_id
        return None

    def record_feedback_state(
        self,
        *,
        draft_id: str,
        status: str,
        manager_chat_id: str,
        action: str,
        feedback_event_id: Optional[str],
        reason: Optional[str],
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> TelegramManagerDraftFeedbackState:
        state = TelegramManagerDraftFeedbackState(
            draft_id=draft_id,
            status=status,
            manager_chat_id=manager_chat_id,
            action=action,
            updated_at=self._now(),
            feedback_event_id=feedback_event_id,
            reason=reason,
            client_send_attempted=False,
            metadata=metadata or {},
        )
        self._feedback_state_by_draft[state.draft_id] = state
        return state

    def get_feedback_state(self, draft_id: str) -> Optional[TelegramManagerDraftFeedbackState]:
        return self._feedback_state_by_draft.get(require_text(draft_id, "draft_id"))

    def snapshot(self) -> Mapping[str, Any]:
        return {
            "schema_version": TELEGRAM_MANAGER_INBOX_SCHEMA_VERSION,
            "registrations": [item.to_json_dict() for item in self._registrations.values()],
            "feedback_states": [item.to_json_dict() for item in self._feedback_state_by_draft.values()],
            "safety": telegram_manager_inbox_safety_contract(),
        }

    def _now(self) -> datetime:
        value = self._clock()
        require_timezone(value, "clock value")
        return value


class TelegramManagerInboxService:
    def __init__(
        self,
        *,
        config: TelegramManagerInboxConfig,
        state_store: Optional[TelegramManagerInboxMemoryStore] = None,
        feedback_store: Optional[Any] = None,
        clock: Optional[Clock] = None,
    ) -> None:
        self.config = config
        self._clock = clock or now_utc
        self.state_store = state_store or TelegramManagerInboxMemoryStore(clock=self._clock)
        self.feedback_store = feedback_store

    def authorize_manager_chat(self, chat_id: Any) -> TelegramManagerAuthorization:
        return self.config.authorize(chat_id)

    def handle_start_update(self, raw_update: Mapping[str, Any]) -> TelegramManagerStartResult:
        parsed = parse_manager_start_update(raw_update)
        if parsed is None:
            return TelegramManagerStartResult(
                status=MANAGER_START_STATUS_IGNORED,
                chat_id=None,
                authorized=False,
                registered=False,
                reply_text="Команда не обработана.",
                blocked_reason="not_start_command",
            )
        authorization = self.authorize_manager_chat(parsed["chat_id"])
        if not authorization.authorized:
            reply_text = "Этот Telegram-чат не разрешен для служебных черновиков менеджера."
            return TelegramManagerStartResult(
                status=MANAGER_START_STATUS_BLOCKED,
                chat_id=authorization.chat_id,
                authorized=False,
                registered=False,
                reply_text=reply_text,
                rendered_payload=render_manager_text_payload(authorization.chat_id, reply_text),
                blocked_reason=authorization.reason,
            )

        registration, created = self.state_store.register_manager_chat(
            authorization.chat_id or "",
            actor=parsed["actor"],
            metadata={
                "telegram_user_id": parsed.get("telegram_user_id"),
                "chat_type": parsed.get("chat_type"),
            },
        )
        reply_text = (
            "Служебный чат менеджера зарегистрирован. "
            "Черновики будут приходить сюда, клиенту автоответ не отправляется."
        )
        return TelegramManagerStartResult(
            status=MANAGER_START_STATUS_REGISTERED if created else MANAGER_START_STATUS_DUPLICATE,
            chat_id=registration.chat_id,
            authorized=True,
            registered=True,
            reply_text=reply_text,
            rendered_payload=render_manager_text_payload(registration.chat_id, reply_text),
        )

    def build_manager_draft_message(
        self,
        draft: ChannelDraftPreview | ChannelDraftRecord,
        *,
        context: Optional[Mapping[str, Any]] = None,
    ) -> TelegramManagerDraftMessage:
        return build_manager_draft_message(draft, context=context)

    def build_delivery(
        self,
        draft: ChannelDraftPreview | ChannelDraftRecord,
        *,
        manager_chat_id: Optional[Any] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> TelegramManagerDraftDelivery:
        target_chat_id = optional_text(manager_chat_id)
        if target_chat_id is None:
            target_chat_id = self.state_store.first_registered_allowed_chat_id(self.config)
        if target_chat_id is None:
            target_chat_id = self.config.default_manager_chat_id
        authorization = self.authorize_manager_chat(target_chat_id)
        if not authorization.authorized:
            return TelegramManagerDraftDelivery(
                status=MANAGER_DELIVERY_STATUS_BLOCKED,
                manager_chat_id=authorization.chat_id,
                blocked_reason=authorization.reason,
                warning=(
                    "manager_chat_id не задан или не разрешен; "
                    "служебный черновик не сформирован для Telegram."
                ),
            )

        message = self.build_manager_draft_message(draft, context=context)
        payload = render_manager_draft_payload(authorization.chat_id or "", message)
        return TelegramManagerDraftDelivery(
            status=MANAGER_DELIVERY_STATUS_READY,
            manager_chat_id=authorization.chat_id,
            message=message,
            rendered_payload=payload,
        )

    def handle_feedback(
        self,
        draft: ChannelDraftRecord,
        command: TelegramManagerFeedbackCommand,
    ) -> TelegramManagerFeedbackResult:
        if not isinstance(draft, ChannelDraftRecord):
            raise TypeError("draft must be ChannelDraftRecord")
        if draft.draft_id != command.draft_id:
            raise ValueError("feedback command draft_id must match draft record")
        authorization = self.authorize_manager_chat(command.manager_chat_id)
        if not authorization.authorized:
            return TelegramManagerFeedbackResult(
                status=MANAGER_DRAFT_STATUS_BLOCKED,
                manager_status=None,
                blocked_reason=authorization.reason,
            )

        manager_status, feedback_event_type, default_reason = manager_feedback_mapping(command.action)
        reason = command.reason or default_reason
        event = build_manager_draft_feedback_event(
            draft,
            feedback_event_type,
            actor=f"telegram_manager:{authorization.chat_id}",
            occurred_at=command.occurred_at or self._now(),
            reason=reason,
            metadata={
                "telegram_manager_action": command.action,
                "telegram_manager_status": manager_status,
                "telegram_callback_query_id": command.callback_query_id,
                "telegram_manager_chat_id": authorization.chat_id,
                "client_send_attempted": False,
                "telegram_api_called": False,
                "network_calls": False,
            },
        )
        feedback_store_result = record_feedback_event(self.feedback_store, event) if self.feedback_store else None
        state = self.state_store.record_feedback_state(
            draft_id=draft.draft_id,
            status=manager_status,
            manager_chat_id=authorization.chat_id or "",
            action=command.action,
            feedback_event_id=event.idempotency_key,
            reason=reason,
            metadata={
                "feedback_event_type": event.event_type,
                "feedback_store_status": feedback_store_result.status if feedback_store_result else None,
            },
        )
        return TelegramManagerFeedbackResult(
            status="recorded",
            manager_status=manager_status,
            feedback_event=event,
            feedback_store_result=feedback_store_result,
            state=state,
        )

    def _now(self) -> datetime:
        value = self._clock()
        require_timezone(value, "clock value")
        return value


def build_manager_draft_message(
    draft: ChannelDraftPreview | ChannelDraftRecord,
    *,
    context: Optional[Mapping[str, Any]] = None,
) -> TelegramManagerDraftMessage:
    preview = resolve_preview(draft)
    context_payload = dict(context or {})
    text = format_manager_draft_text(preview, context=context_payload)
    buttons = build_manager_feedback_buttons(preview.draft_id)
    return TelegramManagerDraftMessage(
        text=text,
        buttons=buttons,
        draft_id=preview.draft_id,
        source_message_idempotency_key=preview.source_message.idempotency_key,
        safety=telegram_manager_inbox_safety_contract(),
        metadata={
            "preview_idempotency_key": preview.idempotency_key,
            "client_send_enabled": False,
            "client_send_button_included": False,
            "context_keys": tuple(sorted(str(key) for key in context_payload)),
        },
    )


def format_manager_draft_text(
    preview: ChannelDraftPreview,
    *,
    context: Optional[Mapping[str, Any]] = None,
) -> str:
    if not isinstance(preview, ChannelDraftPreview):
        raise TypeError("preview must be ChannelDraftPreview")
    context_payload = dict(context or {})
    message = preview.source_message
    topic = first_context_value(context_payload, "found_topic", "question_topic", "topic", "theme")
    rop_decision = first_context_value(
        context_payload,
        "rop_decision",
        "rop_policy_decision",
        "rop_policy",
        "answer_mode",
    )
    required_questions = first_context_value(
        context_payload,
        "bot_must_ask",
        "required_questions",
        "required_question_texts",
        "must_ask",
    )
    risk_flags = merge_unique_texts(
        first_context_value(context_payload, "risk_flags", "safety_flags", "blocked_reasons"),
        preview.reply.safety_flags,
        preview.blocked_reasons,
    )
    crm_recommendations = first_context_value(
        context_payload,
        "crm_recommendations",
        "amo_crm_recommendations",
        "crm_checks",
        "what_to_check_in_crm",
    )
    if crm_recommendations is None:
        crm_recommendations = recommendations_from_actions(preview)
    followup_text = manager_followup_text(context_payload)

    source_lines = (
        f"Канал: {telegram_channel_label(message.channel)}",
        f"Чат клиента: {message.channel_thread_id}",
        f"ID сообщения: {message.channel_message_id}",
    )
    sections = [
        ("Откуда пришел клиент", format_list(source_lines)),
        ("Текст клиента", message.text or "Текст отсутствует, есть вложение."),
        ("Найденная тема", format_value(topic, empty="Тема не определена.")),
        (
            "Решение РОПа",
            format_value(rop_decision, empty="Решение РОПа не найдено в контексте."),
        ),
        (
            "Что бот обязан спросить",
            format_value(required_questions, empty="Обязательные вопросы не указаны."),
        ),
        ("Черновик ответа", preview.reply.text),
        ("Флаги риска", format_value(risk_flags, empty="Явных флагов риска нет.")),
        (
            "Что проверить в AMO/CRM",
            format_value(crm_recommendations, empty="Проверить карточку клиента и сделку."),
        ),
        (
            "Напоминание менеджеру",
            followup_text or "Отдельное напоминание не требуется.",
        ),
        ("Статус", "Клиенту не отправлено. Автоотправка клиенту отключена."),
    ]
    body: list[str] = ["Служебный черновик для менеджера"]
    for title, value in sections:
        body.extend(("", title, value))
    return "\n".join(body).strip()


def build_manager_feedback_buttons(draft_id: str) -> tuple[TelegramManagerDraftButton, ...]:
    draft_key = require_text(draft_id, "draft_id")
    return (
        TelegramManagerDraftButton(
            label="Принято",
            action=MANAGER_ACTION_ACCEPT,
            callback_data=stable_manager_callback_data(MANAGER_ACTION_ACCEPT, draft_key),
            payload={"draft_id": draft_key, "manager_status": MANAGER_DRAFT_STATUS_MARKED_USEFUL},
        ),
        TelegramManagerDraftButton(
            label="Нужно исправить",
            action=MANAGER_ACTION_NEEDS_EDIT,
            callback_data=stable_manager_callback_data(MANAGER_ACTION_NEEDS_EDIT, draft_key),
            payload={"draft_id": draft_key, "manager_status": MANAGER_DRAFT_STATUS_MARKED_NEEDS_EDIT},
        ),
        TelegramManagerDraftButton(
            label="Только менеджер",
            action=MANAGER_ACTION_MANAGER_ONLY,
            callback_data=stable_manager_callback_data(MANAGER_ACTION_MANAGER_ONLY, draft_key),
            payload={"draft_id": draft_key, "manager_status": MANAGER_DRAFT_STATUS_MANAGER_ONLY},
        ),
    )


def render_manager_draft_payload(chat_id: str, message: TelegramManagerDraftMessage) -> Mapping[str, Any]:
    target = require_text(chat_id, "manager chat_id")
    if not isinstance(message, TelegramManagerDraftMessage):
        raise TypeError("message must be TelegramManagerDraftMessage")
    return {
        "schema_version": TELEGRAM_MANAGER_INBOX_SCHEMA_VERSION,
        "method": "sendMessage",
        "chat_id": target,
        "text": message.text,
        "disable_web_page_preview": True,
        "reply_markup": render_manager_reply_markup(message.buttons),
        "button_actions": [button.to_json_dict() for button in message.buttons],
        "target": "manager_service_chat",
        "client_send_enabled": False,
        "client_send_button_included": False,
        "telegram_api_called": False,
        "safety": telegram_manager_inbox_safety_contract(),
    }


def render_manager_text_payload(chat_id: Optional[str], text: str) -> Optional[Mapping[str, Any]]:
    target = optional_text(chat_id)
    if not target:
        return None
    return {
        "schema_version": TELEGRAM_MANAGER_INBOX_SCHEMA_VERSION,
        "method": "sendMessage",
        "chat_id": target,
        "text": require_text(text, "text"),
        "target": "manager_service_chat",
        "client_send_enabled": False,
        "telegram_api_called": False,
        "safety": telegram_manager_inbox_safety_contract(),
    }


def render_manager_reply_markup(buttons: Sequence[TelegramManagerDraftButton]) -> Mapping[str, Any]:
    button_items = tuple(buttons)
    validate_manager_buttons_without_client_send(button_items)
    return {
        "inline_keyboard": [
            [{"text": button.label, "callback_data": button.callback_data}]
            for button in button_items
        ]
    }


def parse_manager_start_update(raw_update: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    if not isinstance(raw_update, Mapping):
        raise ValueError("raw_update must be an object")
    message = raw_update.get("message")
    if not isinstance(message, Mapping):
        return None
    text = str(message.get("text") or "").strip()
    if not is_start_command(text):
        return None
    chat = message.get("chat")
    if not isinstance(chat, Mapping):
        raise ValueError("message.chat must be an object")
    chat_id = require_text(chat.get("id"), "message.chat.id")
    from_user = message.get("from") if isinstance(message.get("from"), Mapping) else {}
    user_id = optional_text(from_user.get("id")) if isinstance(from_user, Mapping) else None
    return {
        "chat_id": chat_id,
        "chat_type": optional_text(chat.get("type")),
        "telegram_user_id": user_id,
        "actor": f"telegram_user:{user_id or chat_id}",
    }


def parse_manager_chat_ids(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        raw_items = value.replace(";", ",").split(",")
    elif isinstance(value, SequenceABC) and not isinstance(value, (bytes, bytearray)):
        raw_items = list(value)
    else:
        raw_items = [value]
    result: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        text = optional_text(item)
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return tuple(result)


def is_start_command(text: str) -> bool:
    first = str(text or "").strip().split(maxsplit=1)[0] if str(text or "").strip() else ""
    command = first.split("@", 1)[0].casefold()
    return command == "/start"


def manager_feedback_mapping(action: str) -> tuple[str, str, str]:
    normalized = require_text(action, "action")
    if normalized == MANAGER_ACTION_ACCEPT:
        return (
            MANAGER_DRAFT_STATUS_MARKED_USEFUL,
            FEEDBACK_MANAGER_DRAFT_APPROVED,
            "Менеджер отметил черновик как полезный.",
        )
    if normalized == MANAGER_ACTION_NEEDS_EDIT:
        return (
            MANAGER_DRAFT_STATUS_MARKED_NEEDS_EDIT,
            FEEDBACK_MANAGER_DRAFT_REJECTED,
            "Менеджер отметил, что черновик нужно исправить.",
        )
    if normalized == MANAGER_ACTION_MANAGER_ONLY:
        return (
            MANAGER_DRAFT_STATUS_MANAGER_ONLY,
            FEEDBACK_MANAGER_DRAFT_REJECTED,
            "Менеджер отметил, что тему должен вести только человек.",
        )
    raise ValueError(f"unsupported manager feedback action: {action!r}")


def record_feedback_event(feedback_store: Any, event: FeedbackEvent) -> FeedbackStoreResult:
    if hasattr(feedback_store, "record_event"):
        return feedback_store.record_event(event)
    if hasattr(feedback_store, "record_feedback_event"):
        return feedback_store.record_feedback_event(event)
    raise TypeError("feedback_store must support record_event or record_feedback_event")


def stable_manager_callback_data(action: str, draft_id: str) -> str:
    suffix = {
        MANAGER_ACTION_ACCEPT: "ok",
        MANAGER_ACTION_NEEDS_EDIT: "fix",
        MANAGER_ACTION_MANAGER_ONLY: "mgr",
    }.get(action)
    if suffix is None:
        raise ValueError(f"unsupported manager action: {action!r}")
    digest = stable_digest({"action": action, "draft_id": require_text(draft_id, "draft_id")})[:18]
    return f"tmgr:v1:{suffix}:{digest}"


def manager_action_from_callback_data(callback_data: str) -> str:
    parts = require_text(callback_data, "callback_data").split(":")
    if len(parts) != 4 or parts[0] != "tmgr" or parts[1] != "v1":
        raise ValueError("unsupported manager callback data")
    suffix = parts[2]
    mapping = {
        "ok": MANAGER_ACTION_ACCEPT,
        "fix": MANAGER_ACTION_NEEDS_EDIT,
        "mgr": MANAGER_ACTION_MANAGER_ONLY,
    }
    if suffix not in mapping:
        raise ValueError("unsupported manager callback action")
    return mapping[suffix]


def resolve_preview(draft: ChannelDraftPreview | ChannelDraftRecord) -> ChannelDraftPreview:
    if isinstance(draft, ChannelDraftRecord):
        return draft.preview
    if isinstance(draft, ChannelDraftPreview):
        return draft
    raise TypeError("draft must be ChannelDraftPreview or ChannelDraftRecord")


def first_context_value(context: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in context and context[key] not in (None, "", (), [], {}):
            return context[key]
    return None


def merge_unique_texts(*values: Any) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        for item in iter_text_items(value):
            if item in seen:
                continue
            seen.add(item)
            result.append(item)
    return tuple(result)


def iter_text_items(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        text = value.strip()
        return (text,) if text else ()
    if isinstance(value, Mapping):
        text = format_value(value, empty="")
        return (text,) if text else ()
    if isinstance(value, SequenceABC) and not isinstance(value, (bytes, bytearray)):
        result: list[str] = []
        for item in value:
            result.extend(iter_text_items(item))
        return tuple(result)
    text = str(value).strip()
    return (text,) if text else ()


def format_value(value: Any, *, empty: str) -> str:
    if value in (None, "", (), [], {}):
        return empty
    if isinstance(value, Mapping):
        items: list[str] = []
        for key, item in value.items():
            if should_skip_manager_context_key(key):
                continue
            rendered = format_value(item, empty="")
            if rendered:
                items.append(f"{key}: {rendered}")
        return format_list(items) if items else empty
    if isinstance(value, SequenceABC) and not isinstance(value, (str, bytes, bytearray)):
        return format_list(str(item).strip() for item in value if str(item).strip()) or empty
    text = str(value).strip()
    return text or empty


def format_list(items: Sequence[Any]) -> str:
    lines = [f"- {str(item).strip()}" for item in items if str(item).strip()]
    return "\n".join(lines)


def recommendations_from_actions(preview: ChannelDraftPreview) -> tuple[str, ...]:
    result: list[str] = []
    for action in preview.reply.recommended_actions:
        if "crm" in action.target_system or "crm" in action.action_type:
            line = action.title or action.summary or action.action_type
            if action.summary and action.summary != line:
                line = f"{line}: {action.summary}"
            result.append(line)
    if not result:
        result.append(
            "Сопоставить клиента в AMO/CRM и проверить актуальную сделку перед ручным ответом."
        )
    return tuple(result)


def manager_followup_text(context: Mapping[str, Any]) -> str:
    required = context.get("manager_followup_required")
    deadline = optional_text(context.get("manager_followup_deadline") or context.get("followup_deadline"))
    schedule = context.get("safe_schedule_template")
    if isinstance(schedule, Mapping):
        if schedule.get("manager_followup_required") is True:
            required = True
        deadline = deadline or optional_text(schedule.get("manager_followup_deadline") or schedule.get("deadline_at"))
    if required is not True:
        return ""
    if deadline:
        return f"Требуется follow-up до {deadline}."
    return "Требуется follow-up: менеджеру нужно вернуться к клиенту после проверки."


def telegram_channel_label(channel: str) -> str:
    labels = {
        "telegram_bot": "Telegram бот",
        "telegram_business": "Telegram Business",
        "telegram_miniapp": "Telegram Mini App",
    }
    return labels.get(channel, channel)


def should_skip_manager_context_key(key: Any) -> bool:
    normalized = str(key).strip().casefold()
    return any(marker in normalized for marker in ("raw", "payload", "token", "secret"))


def validate_manager_buttons_without_client_send(buttons: Sequence[TelegramManagerDraftButton]) -> None:
    for button in buttons:
        if not isinstance(button, TelegramManagerDraftButton):
            raise TypeError("buttons must contain TelegramManagerDraftButton items")
        validate_no_client_send_button(button)


def validate_no_client_send_button(button: TelegramManagerDraftButton) -> None:
    haystack = " ".join(
        (
            button.label,
            button.action,
            button.callback_data,
            " ".join(str(key) for key in button.payload.keys()),
        )
    ).casefold()
    if any(marker in haystack for marker in FORBIDDEN_CLIENT_SEND_BUTTON_MARKERS):
        raise ValueError("manager inbox buttons must not include client-send actions")


def telegram_manager_inbox_safety_contract() -> Mapping[str, bool]:
    return {
        "network_calls": False,
        "telegram_api_called": False,
        "live_send": False,
        "client_send": False,
        "client_send_button_included": False,
        "write_crm": False,
        "write_tallanto": False,
        "write_runtime_db": False,
        "run_asr": False,
        "run_ra": False,
    }
