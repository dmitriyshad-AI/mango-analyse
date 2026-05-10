from __future__ import annotations

from datetime import datetime, timezone

import pytest

from mango_mvp.channels import (
    BotReply,
    ChannelAttachment,
    ChannelDirection,
    ChannelMessage,
    ChannelSession,
    RecommendedAction,
    ReplyButton,
    SendResult,
    dedupe_channel_messages,
    stable_message_idempotency_key,
)


NOW = datetime(2026, 5, 9, 12, 0, tzinfo=timezone.utc)


def test_channel_message_normalizes_and_builds_stable_key() -> None:
    message = ChannelMessage(
        channel=" Telegram_Bot ",
        channel_message_id=" 42 ",
        channel_thread_id=" chat-1 ",
        channel_user_id=" user-1 ",
        direction=ChannelDirection.INBOUND,
        text="  Нужна консультация по ЕГЭ  ",
        received_at=NOW,
        raw_payload={"update_id": 100},
    )

    assert message.channel == "telegram_bot"
    assert message.text == "Нужна консультация по ЕГЭ"
    assert message.idempotency_key == stable_message_idempotency_key(
        channel="telegram_bot",
        channel_thread_id="chat-1",
        channel_message_id="42",
        direction="inbound",
    )
    assert message.to_json_dict()["idempotency_key"] == message.idempotency_key


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"channel": ""}, "channel must not be empty"),
        ({"channel_message_id": ""}, "channel_message_id must not be empty"),
        ({"channel_thread_id": ""}, "channel_thread_id must not be empty"),
        ({"channel_user_id": ""}, "channel_user_id must not be empty"),
        ({"text": "", "attachments": ()}, "requires text or at least one attachment"),
    ],
)
def test_channel_message_validation(kwargs: dict, error: str) -> None:
    base = {
        "channel": "telegram_bot",
        "channel_message_id": "1",
        "channel_thread_id": "chat",
        "channel_user_id": "user",
        "direction": "inbound",
        "text": "hello",
        "received_at": NOW,
    }
    base.update(kwargs)

    with pytest.raises(ValueError, match=error):
        ChannelMessage(**base)


def test_channel_message_requires_timezone_aware_received_at() -> None:
    with pytest.raises(ValueError, match="received_at must be timezone-aware"):
        ChannelMessage(
            channel="telegram_bot",
            channel_message_id="1",
            channel_thread_id="chat",
            channel_user_id="user",
            direction="inbound",
            text="hello",
            received_at=datetime(2026, 5, 9, 12, 0),
        )


def test_attachment_allows_non_text_message() -> None:
    attachment = ChannelAttachment(kind="document", uri="memory://doc-1", metadata={"name": "contract.pdf"})
    message = ChannelMessage(
        channel="site_chat",
        channel_message_id="m1",
        channel_thread_id="thread",
        channel_user_id="user",
        direction="inbound",
        attachments=(attachment,),
        received_at=NOW,
    )

    assert message.text == ""
    assert message.attachments[0].metadata["name"] == "contract.pdf"


def test_session_key_is_stable_and_state_is_copied() -> None:
    state = {"step": "intro"}
    session = ChannelSession(
        channel="telegram_business",
        channel_thread_id="bc-1:chat-1",
        normalized_customer_id="phone:+79000000000",
        state=state,
        updated_at=NOW,
    )
    state["step"] = "mutated"

    assert session.session_key.startswith("channel_session:telegram_business:")
    assert session.state["step"] == "intro"
    assert session.to_json_dict()["updated_at"] == NOW.isoformat()


def test_session_can_be_created_from_message() -> None:
    message = ChannelMessage(
        channel="site_chat",
        channel_message_id="m1",
        channel_thread_id="thread-1",
        channel_user_id="visitor-1",
        direction="inbound",
        text="Здравствуйте",
        received_at=NOW,
    )

    session = ChannelSession.from_message(
        message,
        normalized_customer_id="visitor:1",
        state={"source": "landing"},
        context_summary="Первое обращение.",
    )

    assert session.channel == "site_chat"
    assert session.channel_thread_id == "thread-1"
    assert session.normalized_customer_id == "visitor:1"
    assert session.updated_at == NOW
    assert session.state["source"] == "landing"


def test_recommended_action_uses_distinct_contract_not_agent_action_model() -> None:
    action = RecommendedAction(
        action_type="draft_client_message",
        target_system="channel",
        entity_type="channel_thread",
        entity_id="telegram:42",
        title="Подготовить ответ",
        summary="Клиент задал вопрос.",
        payload={"thread": "telegram:42", "draft": "Здравствуйте"},
        confidence=0.74,
    )

    assert action.idempotency_key.startswith("recommended_action:draft_client_message:")
    assert action.requires_approval is True
    assert action.to_json_dict()["target_system"] == "channel"


def test_recommended_action_rejects_invalid_confidence() -> None:
    with pytest.raises(ValueError, match="confidence must be between 0 and 1"):
        RecommendedAction(
            action_type="draft_client_message",
            target_system="channel",
            entity_type="channel_thread",
            confidence=1.5,
        )


def test_reply_button_and_bot_reply_are_channel_neutral() -> None:
    action = RecommendedAction(
        action_type="manager_review_required",
        target_system="internal",
        entity_type="channel_thread",
        payload={"reason": "commercial_reply"},
    )
    reply = BotReply(
        text="Черновик ответа подготовлен.",
        buttons=(ReplyButton(label="Одобрить", action="approve_draft", payload={"draft_id": "d1"}),),
        recommended_actions=(action,),
        confidence=0.8,
        safety_flags=("requires_manager_approval",),
    )

    payload = reply.to_json_dict()
    assert payload["requires_approval"] is True
    assert payload["buttons"][0]["action"] == "approve_draft"
    assert payload["recommended_actions"][0]["action_type"] == "manager_review_required"
    assert "telegram" not in payload["text"].lower()


def test_bot_reply_rejects_empty_payload() -> None:
    with pytest.raises(ValueError, match="BotReply requires text, attachment, or recommended action"):
        BotReply(text="")


def test_message_idempotency_changes_by_message_id_and_direction() -> None:
    first = stable_message_idempotency_key(
        channel="telegram_bot",
        channel_thread_id="chat",
        channel_message_id="1",
        direction="inbound",
    )
    same = stable_message_idempotency_key(
        channel="telegram_bot",
        channel_thread_id="chat",
        channel_message_id="1",
        direction="inbound",
    )
    different_message = stable_message_idempotency_key(
        channel="telegram_bot",
        channel_thread_id="chat",
        channel_message_id="2",
        direction="inbound",
    )
    different_direction = stable_message_idempotency_key(
        channel="telegram_bot",
        channel_thread_id="chat",
        channel_message_id="1",
        direction="outbound",
    )

    assert first == same
    assert first != different_message
    assert first != different_direction


def test_dedupe_channel_messages_keeps_first_occurrence() -> None:
    first = ChannelMessage(
        channel="telegram_bot",
        channel_message_id="1",
        channel_thread_id="chat",
        channel_user_id="user",
        direction="inbound",
        text="first",
        received_at=NOW,
    )
    duplicate = ChannelMessage(
        channel="telegram_bot",
        channel_message_id="1",
        channel_thread_id="chat",
        channel_user_id="user",
        direction="inbound",
        text="duplicate text should not win",
        received_at=NOW,
    )
    second = ChannelMessage(
        channel="telegram_bot",
        channel_message_id="2",
        channel_thread_id="chat",
        channel_user_id="user",
        direction="inbound",
        text="second",
        received_at=NOW,
    )

    deduped = dedupe_channel_messages((first, duplicate, second))

    assert deduped == (first, second)


def test_send_result_defaults_to_not_sent() -> None:
    result = SendResult(
        channel="telegram_bot",
        idempotency_key="send:1",
        status="live_send_disabled",
        error="live send is not enabled",
    )

    assert result.sent is False
    assert result.status == "live_send_disabled"
