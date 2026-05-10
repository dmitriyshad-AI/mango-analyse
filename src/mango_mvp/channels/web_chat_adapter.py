from __future__ import annotations

from collections.abc import Sequence as SequenceABC
from datetime import datetime, timezone
from typing import Any, Mapping, Optional, Sequence

from mango_mvp.channels.contracts import (
    BotReply,
    ChannelAttachment,
    ChannelMessage,
    ChannelRenderedReply,
    ChannelSession,
    ReplyButton,
    SendResult,
    normalize_key,
    optional_text,
    require_text,
    stable_digest,
)


WEB_CHAT_ADAPTER_SCHEMA_VERSION = "web_chat_adapter_v1"
SITE_CHAT_CHANNEL = "site_chat"
CRM_CHAT_CHANNEL = "crm_chat"
SUPPORTED_WEB_CHAT_CHANNELS = {SITE_CHAT_CHANNEL, CRM_CHAT_CHANNEL}


class WebChatReadOnlyAdapter:
    """Read-only adapter for site chat and CRM chat style payloads."""

    channel_name = "web_chat"

    def __init__(self, *, default_channel: str = SITE_CHAT_CHANNEL) -> None:
        normalized = normalize_web_chat_channel(default_channel)
        self.default_channel = normalized

    def parse_inbound(self, raw_update: Mapping[str, Any]) -> Sequence[ChannelMessage]:
        update = ensure_mapping(raw_update, "raw_update")
        if is_service_event(update):
            return ()
        channel = normalize_web_chat_channel(update.get("channel") or update.get("source_channel") or self.default_channel)
        message = ensure_mapping(update.get("message"), "message") if isinstance(update.get("message"), Mapping) else update
        text = message_text(message)
        attachments = web_chat_attachments(message)
        if not text and not attachments:
            return ()
        message_id = first_text(
            message,
            ("message_id", "id", "event_id", "external_message_id"),
            default=update.get("event_id") or update.get("id"),
            field_name="message_id",
        )
        thread_id = first_text(
            message,
            ("thread_id", "conversation_id", "chat_id", "lead_id", "session_id"),
            default=update.get("thread_id") or update.get("conversation_id"),
            field_name="thread_id",
        )
        user_id = first_text(
            message,
            ("user_id", "visitor_id", "contact_id", "customer_id", "author_id"),
            default=update.get("user_id") or update.get("visitor_id") or thread_id,
            field_name="user_id",
        )
        return (
            ChannelMessage(
                channel=channel,
                channel_message_id=message_id,
                channel_thread_id=thread_id,
                channel_user_id=user_id,
                direction="inbound",
                text=text,
                received_at=web_chat_datetime(
                    first_optional(message, ("received_at", "created_at", "timestamp", "time", "date"))
                    or first_optional(update, ("received_at", "created_at", "timestamp", "time", "date"))
                ),
                attachments=attachments,
                raw_payload={"web_chat_update": dict(update)},
                metadata=web_chat_message_metadata(update, message, channel=channel),
            ),
        )

    def render_reply(self, session: ChannelSession, reply: BotReply) -> ChannelRenderedReply:
        if session.channel not in SUPPORTED_WEB_CHAT_CHANNELS:
            raise ValueError(f"unsupported web chat channel: {session.channel!r}")
        payload: dict[str, Any] = {
            "schema_version": WEB_CHAT_ADAPTER_SCHEMA_VERSION,
            "channel": session.channel,
            "operation": "draft_reply",
            "thread_id": session.channel_thread_id,
            "text": reply.text,
            "requires_approval": reply.requires_approval,
            "live_send_enabled": False,
            "safety": web_chat_adapter_safety_contract(),
            "source_session": session.to_json_dict(),
            "reply_metadata": dict(reply.metadata),
            "button_actions": [button.to_json_dict() for button in reply.buttons],
        }
        rendered_buttons = render_web_chat_buttons(reply.buttons)
        if rendered_buttons:
            payload["buttons"] = rendered_buttons
        idempotency_key = stable_web_chat_render_idempotency_key(session=session, reply=reply, payload=payload)
        return ChannelRenderedReply(
            channel=session.channel,
            channel_thread_id=session.channel_thread_id,
            payload=payload,
            idempotency_key=idempotency_key,
        )

    def send(self, rendered: ChannelRenderedReply, *, live_send_enabled: bool = False) -> SendResult:
        status = "live_send_disabled" if not live_send_enabled else "live_send_not_implemented"
        error = (
            "Web/CRM chat live send is disabled in read-only adapter"
            if not live_send_enabled
            else "Web/CRM chat live send requires a future controlled-send stage"
        )
        return SendResult(
            channel=rendered.channel,
            idempotency_key=rendered.idempotency_key,
            sent=False,
            status=status,
            error=error,
            metadata={
                "schema_version": WEB_CHAT_ADAPTER_SCHEMA_VERSION,
                "requested_live_send_enabled": bool(live_send_enabled),
                "network_calls": False,
                "chat_api_called": False,
            },
        )


def web_chat_message_metadata(
    update: Mapping[str, Any],
    message: Mapping[str, Any],
    *,
    channel: str,
) -> Mapping[str, Any]:
    metadata = {
        "schema_version": WEB_CHAT_ADAPTER_SCHEMA_VERSION,
        "source_channel": channel,
        "source": optional_text(update.get("source")) or optional_text(update.get("provider")),
        "event_type": optional_text(update.get("event_type")) or optional_text(update.get("type")) or "message",
        "message_type": optional_text(message.get("message_type")) or optional_text(message.get("type")) or "text",
        "source_message_id": optional_text(message.get("message_id")) or optional_text(message.get("id")),
        "source_thread_id": (
            optional_text(message.get("thread_id"))
            or optional_text(message.get("conversation_id"))
            or optional_text(message.get("chat_id"))
        ),
        "crm_lead_id": optional_text(message.get("lead_id")) or optional_text(update.get("lead_id")),
        "crm_contact_id": optional_text(message.get("contact_id")) or optional_text(update.get("contact_id")),
        "page_url": optional_text(message.get("page_url")) or optional_text(update.get("page_url")),
        "has_attachments": bool(web_chat_attachments(message)),
        "attachment_count": len(web_chat_attachments(message)),
        "parser_mode": "read_only",
    }
    return {key: value for key, value in metadata.items() if value is not None}


def web_chat_attachments(message: Mapping[str, Any]) -> tuple[ChannelAttachment, ...]:
    raw_items = message.get("attachments") or message.get("files") or ()
    if not isinstance(raw_items, SequenceABC) or isinstance(raw_items, (str, bytes, bytearray)):
        return ()
    result: list[ChannelAttachment] = []
    for item in raw_items:
        if not isinstance(item, Mapping):
            continue
        uri = optional_text(item.get("uri")) or optional_text(item.get("url")) or optional_text(item.get("file_url"))
        if not uri:
            file_id = optional_text(item.get("file_id")) or optional_text(item.get("id"))
            uri = f"web-chat:file:{file_id}" if file_id else None
        if not uri:
            continue
        result.append(
            ChannelAttachment(
                kind=normalize_attachment_kind(item.get("kind") or item.get("type") or "file"),
                uri=uri,
                content_type=optional_text(item.get("content_type")) or optional_text(item.get("mime_type")),
                size_bytes=parse_optional_int(item.get("size_bytes") or item.get("file_size")),
                metadata={
                    key: value
                    for key, value in {
                        "name": item.get("name") or item.get("file_name"),
                        "source_id": item.get("id") or item.get("file_id"),
                    }.items()
                    if value is not None
                },
            )
        )
    return tuple(result)


def render_web_chat_buttons(buttons: Sequence[ReplyButton]) -> list[Mapping[str, Any]]:
    rendered = []
    for button in buttons:
        payload = dict(button.payload)
        rendered.append(
            {
                "label": button.label,
                "action": button.action,
                "payload": payload,
                "callback_id": stable_digest(button.to_json_dict())[:48],
            }
        )
    return rendered


def stable_web_chat_render_idempotency_key(
    *,
    session: ChannelSession,
    reply: BotReply,
    payload: Mapping[str, Any],
) -> str:
    digest = stable_digest(
        {
            "session": session.to_json_dict(),
            "reply": reply.to_json_dict(),
            "payload": dict(payload),
        }
    )
    return f"web_chat_rendered_reply:{digest[:32]}"


def web_chat_adapter_safety_contract() -> Mapping[str, bool]:
    return {
        "network_calls": False,
        "chat_api_called": False,
        "live_send": False,
        "write_crm": False,
        "write_tallanto": False,
        "write_runtime_db": False,
        "run_asr": False,
        "run_ra": False,
    }


def normalize_web_chat_channel(value: Any) -> str:
    channel = normalize_key(str(value or SITE_CHAT_CHANNEL), "channel")
    if channel not in SUPPORTED_WEB_CHAT_CHANNELS:
        raise ValueError(f"unsupported web chat channel: {value!r}")
    return channel


def normalize_attachment_kind(value: Any) -> str:
    try:
        return normalize_key(str(value or "file"), "attachment kind")
    except ValueError:
        return "file"


def message_text(message: Mapping[str, Any]) -> str:
    return (
        optional_text(message.get("text"))
        or optional_text(message.get("body"))
        or optional_text(message.get("message"))
        or optional_text(message.get("content"))
        or ""
    )


def first_text(
    mapping: Mapping[str, Any],
    keys: Sequence[str],
    *,
    default: Any = None,
    field_name: str,
) -> str:
    value = first_optional(mapping, keys)
    if value is None:
        value = default
    return require_text(value, field_name)


def first_optional(mapping: Mapping[str, Any], keys: Sequence[str]) -> Optional[Any]:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, ""):
            return value
    return None


def web_chat_datetime(value: Any) -> datetime:
    if value in (None, ""):
        return datetime.now(timezone.utc)
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("web chat datetime must be timezone-aware")
        return value
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    text = str(value).strip()
    if text.isdigit():
        return datetime.fromtimestamp(int(text), tz=timezone.utc)
    normalized = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"invalid web chat timestamp: {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("web chat timestamp must be timezone-aware")
    return parsed


def parse_optional_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def is_service_event(update: Mapping[str, Any]) -> bool:
    event_type = optional_text(update.get("event_type")) or optional_text(update.get("type"))
    if event_type in {"typing", "read", "delivered", "presence", "heartbeat"}:
        return True
    message = update.get("message")
    if isinstance(message, Mapping):
        message_type = optional_text(message.get("message_type")) or optional_text(message.get("type"))
        return message_type in {"typing", "read", "delivered", "presence", "heartbeat"}
    return False


def ensure_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return value
