from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from mango_mvp.channels import (
    BotReply,
    ChannelMemoryStore,
    ChannelSession,
    ReplyButton,
    TELEGRAM_BOT_CHANNEL,
    TELEGRAM_BUSINESS_CHANNEL,
    TELEGRAM_MINIAPP_CHANNEL,
    TelegramReadOnlyAdapter,
    build_and_store_channel_draft_preview,
    telegram_adapter_safety_contract,
)


NOW = datetime(2026, 5, 9, 16, 0, tzinfo=timezone.utc)
NOW_TS = int(NOW.timestamp())


def test_parse_regular_telegram_message_update() -> None:
    update = {
        "update_id": 1001,
        "message": {
            "message_id": 42,
            "date": NOW_TS,
            "chat": {"id": 7001, "type": "private"},
            "from": {"id": 9001, "username": "client"},
            "text": "Здравствуйте, сколько стоит курс?",
        },
    }

    messages = TelegramReadOnlyAdapter().parse_inbound(update)

    assert len(messages) == 1
    message = messages[0]
    assert message.channel == TELEGRAM_BOT_CHANNEL
    assert message.channel_message_id == "42"
    assert message.channel_thread_id == "7001"
    assert message.channel_user_id == "9001"
    assert message.text == "Здравствуйте, сколько стоит курс?"
    assert message.received_at == NOW
    assert message.metadata["telegram_update_type"] == "message"
    assert message.metadata["parser_mode"] == "read_only"
    assert message.idempotency_key.startswith("channel_message:telegram_bot:inbound:")


def test_parse_telegram_business_message_update() -> None:
    update = {
        "update_id": 1002,
        "business_message": {
            "business_connection_id": "bc-123",
            "message_id": 77,
            "date": NOW_TS,
            "chat": {"id": 555, "type": "private"},
            "from": {"id": 555, "first_name": "Client"},
            "text": "Хочу записаться на консультацию",
        },
    }

    message = TelegramReadOnlyAdapter().parse_inbound(update)[0]

    assert message.channel == TELEGRAM_BUSINESS_CHANNEL
    assert message.channel_message_id == "bc-123:77"
    assert message.channel_thread_id == "bc-123:555"
    assert message.channel_user_id == "555"
    assert message.metadata["business_connection_id"] == "bc-123"
    assert message.metadata["telegram_update_type"] == "business_message"


def test_parse_miniapp_web_app_data_as_structured_event() -> None:
    update = {
        "update_id": 1003,
        "message": {
            "message_id": 88,
            "date": NOW_TS,
            "chat": {"id": 7001, "type": "private"},
            "from": {"id": 9001},
            "web_app_data": {
                "button_text": "Подобрать курс",
                "data": json.dumps({"action": "course_picker_submit", "grade": 10}, ensure_ascii=False),
            },
        },
    }

    message = TelegramReadOnlyAdapter().parse_inbound(update)[0]

    assert message.channel == TELEGRAM_MINIAPP_CHANNEL
    assert message.channel_message_id == "web_app_data:1003"
    assert message.text == "Telegram Mini App event: Подобрать курс / course_picker_submit"
    assert message.metadata["has_web_app_data"] is True
    assert message.metadata["web_app_data"]["action"] == "course_picker_submit"
    assert message.raw_payload["web_app_data_parsed"]["grade"] == 10


def test_parse_attachment_only_message_without_text() -> None:
    update = {
        "update_id": 1004,
        "message": {
            "message_id": 89,
            "date": NOW_TS,
            "chat": {"id": 7001, "type": "private"},
            "from": {"id": 9001},
            "document": {
                "file_id": "file-doc-1",
                "file_name": "contract.pdf",
                "mime_type": "application/pdf",
                "file_size": 12345,
            },
        },
    }

    message = TelegramReadOnlyAdapter().parse_inbound(update)[0]

    assert message.text == ""
    assert message.attachments[0].kind == "document"
    assert message.attachments[0].uri == "telegram:file:file-doc-1"
    assert message.attachments[0].content_type == "application/pdf"
    assert message.metadata["has_attachments"] is True


def test_parse_unsupported_or_service_update_returns_empty_sequence() -> None:
    adapter = TelegramReadOnlyAdapter()
    unsupported = {"update_id": 1005, "callback_query": {"id": "cb-1"}}
    service_message = {
        "update_id": 1006,
        "message": {
            "message_id": 90,
            "date": NOW_TS,
            "chat": {"id": 7001},
            "from": {"id": 9001},
            "new_chat_member": {"id": 1},
        },
    }

    assert adapter.parse_inbound(unsupported) == ()
    assert adapter.parse_inbound(service_message) == ()


def test_parse_invalid_payload_raises_clear_error() -> None:
    adapter = TelegramReadOnlyAdapter()

    with pytest.raises(ValueError, match="raw_update must be an object"):
        adapter.parse_inbound("not a mapping")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="message.chat must be an object"):
        adapter.parse_inbound({"update_id": 1007, "message": {"message_id": 91, "text": "hi"}})


def test_render_reply_builds_telegram_payload_without_sending() -> None:
    adapter = TelegramReadOnlyAdapter()
    session = ChannelSession(
        channel=TELEGRAM_BOT_CHANNEL,
        channel_thread_id="7001",
        updated_at=NOW,
    )
    reply = BotReply(
        text="Здравствуйте! Передадим вопрос менеджеру.",
        buttons=(ReplyButton(label="Открыть заявку", action="open_url", payload={"url": "https://example.test"}),),
        requires_approval=True,
        safety_flags=("requires_manager_approval",),
    )

    rendered = adapter.render_reply(session, reply)
    repeat = adapter.render_reply(session, reply)

    assert rendered.channel == TELEGRAM_BOT_CHANNEL
    assert rendered.payload["method"] == "sendMessage"
    assert rendered.payload["chat_id"] == "7001"
    assert rendered.payload["live_send_enabled"] is False
    assert rendered.payload["safety"]["telegram_api_called"] is False
    assert rendered.payload["reply_markup"]["inline_keyboard"][0][0]["url"] == "https://example.test"
    assert rendered.idempotency_key == repeat.idempotency_key


def test_render_business_reply_includes_business_connection_id() -> None:
    adapter = TelegramReadOnlyAdapter()
    session = ChannelSession(
        channel=TELEGRAM_BUSINESS_CHANNEL,
        channel_thread_id="bc-123:555",
        updated_at=NOW,
    )

    rendered = adapter.render_reply(session, BotReply(text="Черновик ответа."))

    assert rendered.payload["business_connection_id"] == "bc-123"
    assert rendered.payload["chat_id"] == "555"


def test_send_is_always_blocked_even_when_live_send_requested() -> None:
    adapter = TelegramReadOnlyAdapter()
    session = ChannelSession(channel=TELEGRAM_BOT_CHANNEL, channel_thread_id="7001", updated_at=NOW)
    rendered = adapter.render_reply(session, BotReply(text="Черновик ответа."))

    dry = adapter.send(rendered)
    live_requested = adapter.send(rendered, live_send_enabled=True)

    assert dry.sent is False
    assert dry.status == "live_send_disabled"
    assert dry.metadata["telegram_api_called"] is False
    assert live_requested.sent is False
    assert live_requested.status == "live_send_not_implemented"
    assert "future controlled-send" in (live_requested.error or "")


def test_read_only_telegram_e2e_parse_preview_store_render_and_block_send() -> None:
    adapter = TelegramReadOnlyAdapter()
    store = ChannelMemoryStore()
    update = {
        "update_id": 1008,
        "message": {
            "message_id": 92,
            "date": NOW_TS,
            "chat": {"id": 7001, "type": "private"},
            "from": {"id": 9001},
            "text": "Пожалуйста, перезвоните мне завтра",
        },
    }

    message = adapter.parse_inbound(update)[0]
    preview, store_result = build_and_store_channel_draft_preview(store, message, actor="telegram_adapter_test")
    rendered = adapter.render_reply(preview.session, preview.reply)
    send_result = adapter.send(rendered)

    assert store_result.created is True
    assert store.summary()["drafts"] == 1
    assert store.summary()["actions"] >= 2
    assert rendered.payload["chat_id"] == "7001"
    assert rendered.payload["requires_approval"] is True
    assert send_result.sent is False
    assert send_result.status == "live_send_disabled"


def test_telegram_adapter_safety_contract_blocks_external_effects() -> None:
    safety = telegram_adapter_safety_contract()

    assert safety["network_calls"] is False
    assert safety["live_send"] is False
    assert safety["write_crm"] is False
    assert safety["write_runtime_db"] is False
