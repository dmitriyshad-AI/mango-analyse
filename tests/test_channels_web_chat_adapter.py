from __future__ import annotations

from datetime import datetime, timezone

import pytest

from mango_mvp.channels import (
    BotReply,
    CRM_CHAT_CHANNEL,
    ChannelMemoryStore,
    ChannelSession,
    ReplyButton,
    SITE_CHAT_CHANNEL,
    WebChatReadOnlyAdapter,
    build_and_store_channel_draft_preview,
    web_chat_adapter_safety_contract,
)


NOW = datetime(2026, 5, 9, 17, 0, tzinfo=timezone.utc)
NOW_ISO = NOW.isoformat().replace("+00:00", "Z")


def test_parse_nested_site_chat_payload() -> None:
    payload = {
        "channel": "site_chat",
        "source": "landing_widget",
        "event_id": "evt-1",
        "message": {
            "id": "msg-1",
            "thread_id": "visitor-thread-1",
            "visitor_id": "visitor-1",
            "text": "Здравствуйте, можно подобрать курс?",
            "created_at": NOW_ISO,
            "page_url": "https://example.test/courses",
        },
    }

    messages = WebChatReadOnlyAdapter().parse_inbound(payload)

    assert len(messages) == 1
    message = messages[0]
    assert message.channel == SITE_CHAT_CHANNEL
    assert message.channel_message_id == "msg-1"
    assert message.channel_thread_id == "visitor-thread-1"
    assert message.channel_user_id == "visitor-1"
    assert message.text == "Здравствуйте, можно подобрать курс?"
    assert message.received_at == NOW
    assert message.metadata["source"] == "landing_widget"
    assert message.metadata["page_url"] == "https://example.test/courses"
    assert message.metadata["parser_mode"] == "read_only"


def test_parse_same_site_chat_payload_keeps_message_idempotency_key() -> None:
    payload = {
        "message_id": "msg-stable",
        "thread_id": "visitor-thread-stable",
        "visitor_id": "visitor-stable",
        "text": "Нужна консультация",
        "created_at": NOW_ISO,
    }
    adapter = WebChatReadOnlyAdapter()

    first = adapter.parse_inbound(payload)[0]
    second = adapter.parse_inbound(payload)[0]

    assert first.idempotency_key == second.idempotency_key
    assert first.channel_message_id == second.channel_message_id == "msg-stable"


def test_parse_flat_crm_chat_payload() -> None:
    payload = {
        "channel": "crm_chat",
        "event_type": "message",
        "message_id": "crm-msg-1",
        "conversation_id": "lead-123",
        "contact_id": "contact-55",
        "lead_id": "lead-123",
        "body": "Клиент просит счет и скидку",
        "timestamp": int(NOW.timestamp()),
        "provider": "amocrm_chat",
    }

    message = WebChatReadOnlyAdapter().parse_inbound(payload)[0]

    assert message.channel == CRM_CHAT_CHANNEL
    assert message.channel_message_id == "crm-msg-1"
    assert message.channel_thread_id == "lead-123"
    assert message.channel_user_id == "contact-55"
    assert message.text == "Клиент просит счет и скидку"
    assert message.metadata["crm_lead_id"] == "lead-123"
    assert message.metadata["crm_contact_id"] == "contact-55"
    assert message.metadata["source"] == "amocrm_chat"


def test_parse_attachment_only_site_chat_message() -> None:
    payload = {
        "event_id": "evt-attach",
        "message": {
            "message_id": "msg-attach",
            "session_id": "visitor-thread-1",
            "user_id": "visitor-1",
            "received_at": NOW_ISO,
            "attachments": [
                {
                    "type": "document",
                    "url": "https://files.example.test/contract.pdf",
                    "file_name": "contract.pdf",
                    "mime_type": "application/pdf",
                    "file_size": "2048",
                }
            ],
        },
    }

    message = WebChatReadOnlyAdapter().parse_inbound(payload)[0]

    assert message.channel == SITE_CHAT_CHANNEL
    assert message.text == ""
    assert message.attachments[0].kind == "document"
    assert message.attachments[0].uri == "https://files.example.test/contract.pdf"
    assert message.attachments[0].size_bytes == 2048
    assert message.metadata["has_attachments"] is True


def test_service_or_empty_events_are_ignored() -> None:
    adapter = WebChatReadOnlyAdapter()

    assert adapter.parse_inbound({"channel": "site_chat", "event_type": "typing"}) == ()
    assert adapter.parse_inbound({"channel": "crm_chat", "message": {"type": "read"}}) == ()
    assert adapter.parse_inbound({"channel": "site_chat", "message": {"id": "m1", "thread_id": "t1", "user_id": "u1"}}) == ()


def test_invalid_web_chat_payload_raises_clear_error() -> None:
    adapter = WebChatReadOnlyAdapter()

    with pytest.raises(ValueError, match="raw_update must be an object"):
        adapter.parse_inbound("not a mapping")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unsupported web chat channel"):
        WebChatReadOnlyAdapter(default_channel="unsupported")
    with pytest.raises(ValueError, match="thread_id must not be empty"):
        adapter.parse_inbound({"message_id": "m1", "user_id": "u1", "text": "hello"})


def test_render_site_chat_reply_without_sending() -> None:
    adapter = WebChatReadOnlyAdapter()
    session = ChannelSession(channel=SITE_CHAT_CHANNEL, channel_thread_id="visitor-thread-1", updated_at=NOW)
    reply = BotReply(
        text="Здравствуйте! Передадим вопрос менеджеру.",
        buttons=(ReplyButton(label="Открыть форму", action="open_url", payload={"url": "https://example.test/form"}),),
        requires_approval=True,
        safety_flags=("requires_manager_approval",),
    )

    rendered = adapter.render_reply(session, reply)
    repeat = adapter.render_reply(session, reply)

    assert rendered.channel == SITE_CHAT_CHANNEL
    assert rendered.payload["operation"] == "draft_reply"
    assert rendered.payload["thread_id"] == "visitor-thread-1"
    assert rendered.payload["live_send_enabled"] is False
    assert rendered.payload["buttons"][0]["label"] == "Открыть форму"
    assert rendered.payload["safety"]["chat_api_called"] is False
    assert rendered.idempotency_key == repeat.idempotency_key


def test_render_crm_chat_reply_and_reject_unknown_channel() -> None:
    adapter = WebChatReadOnlyAdapter()
    rendered = adapter.render_reply(
        ChannelSession(channel=CRM_CHAT_CHANNEL, channel_thread_id="lead-123", updated_at=NOW),
        BotReply(text="Черновик ответа."),
    )

    assert rendered.channel == CRM_CHAT_CHANNEL
    assert rendered.payload["thread_id"] == "lead-123"
    with pytest.raises(ValueError, match="unsupported web chat channel"):
        adapter.render_reply(
            ChannelSession(channel="telegram_bot", channel_thread_id="1", updated_at=NOW),
            BotReply(text="Черновик ответа."),
        )


def test_send_is_always_blocked_even_when_live_send_requested() -> None:
    adapter = WebChatReadOnlyAdapter()
    session = ChannelSession(channel=SITE_CHAT_CHANNEL, channel_thread_id="visitor-thread-1", updated_at=NOW)
    rendered = adapter.render_reply(session, BotReply(text="Черновик ответа."))

    dry = adapter.send(rendered)
    live_requested = adapter.send(rendered, live_send_enabled=True)

    assert dry.sent is False
    assert dry.status == "live_send_disabled"
    assert dry.metadata["chat_api_called"] is False
    assert live_requested.sent is False
    assert live_requested.status == "live_send_not_implemented"
    assert "future controlled-send" in (live_requested.error or "")


def test_read_only_web_chat_e2e_parse_preview_store_render_and_block_send() -> None:
    adapter = WebChatReadOnlyAdapter(default_channel=SITE_CHAT_CHANNEL)
    store = ChannelMemoryStore()
    payload = {
        "event_id": "evt-2",
        "message": {
            "message_id": "msg-2",
            "thread_id": "visitor-thread-2",
            "visitor_id": "visitor-2",
            "text": "Пожалуйста, перезвоните завтра вечером",
            "created_at": NOW_ISO,
        },
    }

    message = adapter.parse_inbound(payload)[0]
    preview, store_result = build_and_store_channel_draft_preview(store, message, actor="web_chat_adapter_test")
    rendered = adapter.render_reply(preview.session, preview.reply)
    send_result = adapter.send(rendered)

    assert store_result.created is True
    assert store.summary()["drafts"] == 1
    assert store.summary()["actions"] >= 2
    assert rendered.payload["thread_id"] == "visitor-thread-2"
    assert rendered.payload["requires_approval"] is True
    assert send_result.sent is False
    assert send_result.status == "live_send_disabled"


def test_web_chat_adapter_safety_contract_blocks_external_effects() -> None:
    safety = web_chat_adapter_safety_contract()

    assert safety["network_calls"] is False
    assert safety["live_send"] is False
    assert safety["write_crm"] is False
    assert safety["write_runtime_db"] is False
