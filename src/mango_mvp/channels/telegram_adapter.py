from __future__ import annotations

import json
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
    require_text,
    stable_digest,
)


TELEGRAM_ADAPTER_SCHEMA_VERSION = "telegram_adapter_v1"
TELEGRAM_BOT_CHANNEL = "telegram_bot"
TELEGRAM_BUSINESS_CHANNEL = "telegram_business"
TELEGRAM_MINIAPP_CHANNEL = "telegram_miniapp"
SUPPORTED_TELEGRAM_CHANNELS = {
    TELEGRAM_BOT_CHANNEL,
    TELEGRAM_BUSINESS_CHANNEL,
    TELEGRAM_MINIAPP_CHANNEL,
}


class TelegramReadOnlyAdapter:
    """Telegram parser/renderer skeleton with live send deliberately disabled."""

    channel_name = "telegram"

    def parse_inbound(self, raw_update: Mapping[str, Any]) -> Sequence[ChannelMessage]:
        update = ensure_mapping(raw_update, "raw_update")
        messages: list[ChannelMessage] = []
        if "message" in update:
            message = ensure_mapping(update["message"], "message")
            if "web_app_data" in message:
                parsed = parse_miniapp_message(update, message)
            else:
                parsed = parse_bot_message(update, message, update_type="message")
            if parsed is not None:
                messages.append(parsed)
        if "edited_message" in update:
            message = ensure_mapping(update["edited_message"], "edited_message")
            parsed = parse_bot_message(update, message, update_type="edited_message")
            if parsed is not None:
                messages.append(parsed)
        for update_type in ("business_message", "edited_business_message"):
            if update_type in update:
                message = ensure_mapping(update[update_type], update_type)
                parsed = parse_business_message(update, message, update_type=update_type)
                if parsed is not None:
                    messages.append(parsed)
        return tuple(messages)

    def render_reply(self, session: ChannelSession, reply: BotReply) -> ChannelRenderedReply:
        if session.channel not in SUPPORTED_TELEGRAM_CHANNELS:
            raise ValueError(f"unsupported Telegram channel: {session.channel!r}")
        target = telegram_target_from_session(session)
        payload: dict[str, Any] = {
            "schema_version": TELEGRAM_ADAPTER_SCHEMA_VERSION,
            "method": "sendMessage",
            "chat_id": target["chat_id"],
            "text": reply.text,
            "parse_mode": None,
            "disable_web_page_preview": True,
            "requires_approval": reply.requires_approval,
            "live_send_enabled": False,
            "safety": telegram_adapter_safety_contract(),
            "source_session": session.to_json_dict(),
            "reply_metadata": dict(reply.metadata),
            "button_actions": [button.to_json_dict() for button in reply.buttons],
        }
        if target.get("business_connection_id"):
            payload["business_connection_id"] = target["business_connection_id"]
        reply_markup = render_telegram_reply_markup(reply.buttons)
        if reply_markup:
            payload["reply_markup"] = reply_markup
        idempotency_key = stable_telegram_render_idempotency_key(session=session, reply=reply, payload=payload)
        return ChannelRenderedReply(
            channel=session.channel,
            channel_thread_id=session.channel_thread_id,
            payload=payload,
            idempotency_key=idempotency_key,
        )

    def send(self, rendered: ChannelRenderedReply, *, live_send_enabled: bool = False) -> SendResult:
        status = "live_send_disabled" if not live_send_enabled else "live_send_not_implemented"
        error = (
            "Telegram live send is disabled in read-only adapter"
            if not live_send_enabled
            else "Telegram live send requires a future controlled-send stage"
        )
        return SendResult(
            channel=rendered.channel,
            idempotency_key=rendered.idempotency_key,
            sent=False,
            status=status,
            error=error,
            metadata={
                "schema_version": TELEGRAM_ADAPTER_SCHEMA_VERSION,
                "requested_live_send_enabled": bool(live_send_enabled),
                "network_calls": False,
                "telegram_api_called": False,
            },
        )


def parse_bot_message(
    update: Mapping[str, Any],
    message: Mapping[str, Any],
    *,
    update_type: str,
) -> Optional[ChannelMessage]:
    text = message_text(message)
    attachments = telegram_attachments(message)
    if not text and not attachments:
        return None
    chat = ensure_mapping(message.get("chat"), f"{update_type}.chat")
    chat_id = require_telegram_id(chat.get("id"), f"{update_type}.chat.id")
    from_user = mapping_or_none(message.get("from"))
    user_id = str(from_user.get("id")) if from_user and from_user.get("id") is not None else chat_id
    message_id = telegram_message_id(message, update_type=update_type)
    received_at = telegram_datetime(message.get("date"))
    return ChannelMessage(
        channel=TELEGRAM_BOT_CHANNEL,
        channel_message_id=message_id,
        channel_thread_id=chat_id,
        channel_user_id=user_id,
        direction="inbound",
        text=text,
        received_at=received_at,
        attachments=attachments,
        raw_payload={"telegram_update": dict(update)},
        metadata=telegram_message_metadata(update, message, update_type=update_type, channel=TELEGRAM_BOT_CHANNEL),
    )


def parse_business_message(
    update: Mapping[str, Any],
    message: Mapping[str, Any],
    *,
    update_type: str,
) -> Optional[ChannelMessage]:
    text = message_text(message)
    attachments = telegram_attachments(message)
    if not text and not attachments:
        return None
    chat = ensure_mapping(message.get("chat"), f"{update_type}.chat")
    chat_id = require_telegram_id(chat.get("id"), f"{update_type}.chat.id")
    business_connection_id = require_text(
        message.get("business_connection_id") or update.get("business_connection_id"),
        "business_connection_id",
    )
    message_id = telegram_message_id(message, update_type=update_type, prefix=business_connection_id)
    thread_id = f"{business_connection_id}:{chat_id}"
    return ChannelMessage(
        channel=TELEGRAM_BUSINESS_CHANNEL,
        channel_message_id=message_id,
        channel_thread_id=thread_id,
        channel_user_id=chat_id,
        direction="inbound",
        text=text,
        received_at=telegram_datetime(message.get("date")),
        attachments=attachments,
        raw_payload={"telegram_update": dict(update)},
        metadata=telegram_message_metadata(
            update,
            message,
            update_type=update_type,
            channel=TELEGRAM_BUSINESS_CHANNEL,
        )
        | {"business_connection_id": business_connection_id},
    )


def parse_miniapp_message(update: Mapping[str, Any], message: Mapping[str, Any]) -> Optional[ChannelMessage]:
    web_app_data = ensure_mapping(message.get("web_app_data"), "message.web_app_data")
    chat = ensure_mapping(message.get("chat"), "message.chat")
    chat_id = require_telegram_id(chat.get("id"), "message.chat.id")
    from_user = mapping_or_none(message.get("from"))
    user_id = str(from_user.get("id")) if from_user and from_user.get("id") is not None else chat_id
    parsed_data = parse_web_app_data(web_app_data.get("data"))
    update_id = require_text(update.get("update_id"), "update_id")
    return ChannelMessage(
        channel=TELEGRAM_MINIAPP_CHANNEL,
        channel_message_id=f"web_app_data:{update_id}",
        channel_thread_id=chat_id,
        channel_user_id=user_id,
        direction="inbound",
        text=web_app_event_text(web_app_data, parsed_data),
        received_at=telegram_datetime(message.get("date")),
        raw_payload={"telegram_update": dict(update), "web_app_data_parsed": parsed_data},
        metadata=telegram_message_metadata(
            update,
            message,
            update_type="web_app_data",
            channel=TELEGRAM_MINIAPP_CHANNEL,
        )
        | {
            "web_app_button_text": optional_str(web_app_data.get("button_text")),
            "web_app_data": parsed_data,
            "has_web_app_data": True,
        },
    )


def telegram_message_metadata(
    update: Mapping[str, Any],
    message: Mapping[str, Any],
    *,
    update_type: str,
    channel: str,
) -> Mapping[str, Any]:
    chat = mapping_or_none(message.get("chat")) or {}
    from_user = mapping_or_none(message.get("from")) or {}
    metadata = {
        "schema_version": TELEGRAM_ADAPTER_SCHEMA_VERSION,
        "telegram_update_id": optional_str(update.get("update_id")),
        "telegram_update_type": update_type,
        "telegram_channel": channel,
        "telegram_message_id": optional_str(message.get("message_id")),
        "telegram_chat_id": optional_str(chat.get("id")),
        "telegram_chat_type": optional_str(chat.get("type")),
        "telegram_user_id": optional_str(from_user.get("id")),
        "telegram_username": optional_str(from_user.get("username")),
        "telegram_message_thread_id": optional_str(message.get("message_thread_id")),
        "has_attachments": bool(telegram_attachments(message)),
        "attachment_count": len(telegram_attachments(message)),
        "parser_mode": "read_only",
    }
    return {key: value for key, value in metadata.items() if value is not None}


def message_text(message: Mapping[str, Any]) -> str:
    return optional_str(message.get("text")) or optional_str(message.get("caption")) or ""


def telegram_message_id(
    message: Mapping[str, Any],
    *,
    update_type: str,
    prefix: Optional[str] = None,
) -> str:
    message_id = require_text(message.get("message_id"), f"{update_type}.message_id")
    event_id = f"{update_type}:{message_id}" if update_type.startswith("edited_") else message_id
    return f"{prefix}:{event_id}" if prefix else event_id


def telegram_datetime(value: Any) -> datetime:
    if value in (None, ""):
        return datetime.now(timezone.utc)
    try:
        timestamp = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid Telegram date: {value!r}") from exc
    return datetime.fromtimestamp(timestamp, tz=timezone.utc)


def telegram_attachments(message: Mapping[str, Any]) -> tuple[ChannelAttachment, ...]:
    attachments: list[ChannelAttachment] = []
    photo_sizes = message.get("photo")
    if isinstance(photo_sizes, SequenceABC) and not isinstance(photo_sizes, (str, bytes)):
        photo_items = [item for item in photo_sizes if isinstance(item, Mapping) and item.get("file_id")]
        if photo_items:
            largest = photo_items[-1]
            attachments.append(
                ChannelAttachment(
                    kind="photo",
                    uri=telegram_file_uri(largest["file_id"]),
                    size_bytes=int(largest["file_size"]) if largest.get("file_size") is not None else None,
                    metadata={
                        "width": largest.get("width"),
                        "height": largest.get("height"),
                        "all_file_ids": [str(item["file_id"]) for item in photo_items],
                    },
                )
            )
    attachments.extend(single_telegram_attachment(message, "document", kind="document"))
    attachments.extend(single_telegram_attachment(message, "voice", kind="voice"))
    attachments.extend(single_telegram_attachment(message, "audio", kind="audio"))
    attachments.extend(single_telegram_attachment(message, "video", kind="video"))
    attachments.extend(single_telegram_attachment(message, "sticker", kind="sticker"))
    return tuple(attachments)


def single_telegram_attachment(message: Mapping[str, Any], field: str, *, kind: str) -> tuple[ChannelAttachment, ...]:
    payload = mapping_or_none(message.get(field))
    if not payload or not payload.get("file_id"):
        return ()
    return (
        ChannelAttachment(
            kind=kind,
            uri=telegram_file_uri(payload["file_id"]),
            content_type=optional_str(payload.get("mime_type")),
            size_bytes=int(payload["file_size"]) if payload.get("file_size") is not None else None,
            metadata={
                key: value
                for key, value in {
                    "file_name": payload.get("file_name"),
                    "duration": payload.get("duration"),
                    "width": payload.get("width"),
                    "height": payload.get("height"),
                    "emoji": payload.get("emoji"),
                }.items()
                if value is not None
            },
        ),
    )


def telegram_file_uri(file_id: Any) -> str:
    return f"telegram:file:{require_text(file_id, 'file_id')}"


def parse_web_app_data(value: Any) -> Mapping[str, Any]:
    raw = optional_str(value)
    if not raw:
        return {"raw_data": "", "parse_status": "empty"}
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError:
        return {"raw_data": raw, "parse_status": "not_json"}
    if isinstance(decoded, Mapping):
        return dict(decoded) | {"parse_status": "json_object"}
    return {"value": decoded, "parse_status": "json_value"}


def web_app_event_text(web_app_data: Mapping[str, Any], parsed_data: Mapping[str, Any]) -> str:
    button = optional_str(web_app_data.get("button_text"))
    event_name = (
        optional_str(parsed_data.get("action"))
        or optional_str(parsed_data.get("event"))
        or optional_str(parsed_data.get("type"))
        or "web_app_data"
    )
    if button:
        return f"Telegram Mini App event: {button} / {event_name}"
    return f"Telegram Mini App event: {event_name}"


def telegram_target_from_session(session: ChannelSession) -> Mapping[str, str]:
    if session.channel == TELEGRAM_BUSINESS_CHANNEL:
        parts = session.channel_thread_id.split(":", 1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError("telegram_business session thread must be '<business_connection_id>:<chat_id>'")
        return {"business_connection_id": parts[0], "chat_id": parts[1]}
    return {"chat_id": session.channel_thread_id}


def render_telegram_reply_markup(buttons: Sequence[ReplyButton]) -> Optional[Mapping[str, Any]]:
    rows = []
    for button in buttons:
        rendered = {"text": button.label}
        url = optional_str(button.payload.get("url"))
        if button.action == "open_url" and url:
            rendered["url"] = url
        else:
            rendered["callback_data"] = stable_digest(button.to_json_dict())[:48]
        rows.append([rendered])
    if not rows:
        return None
    return {"inline_keyboard": rows}


def stable_telegram_render_idempotency_key(
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
    return f"telegram_rendered_reply:{digest[:32]}"


def telegram_adapter_safety_contract() -> Mapping[str, bool]:
    return {
        "network_calls": False,
        "telegram_api_called": False,
        "live_send": False,
        "write_crm": False,
        "write_tallanto": False,
        "write_runtime_db": False,
        "run_asr": False,
        "run_ra": False,
    }


def ensure_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return value


def mapping_or_none(value: Any) -> Optional[Mapping[str, Any]]:
    return value if isinstance(value, Mapping) else None


def require_telegram_id(value: Any, field_name: str) -> str:
    return require_text(value, field_name)


def optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
