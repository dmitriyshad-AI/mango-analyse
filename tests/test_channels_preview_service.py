from __future__ import annotations

from datetime import datetime, timezone

import pytest

from mango_mvp.channels import (
    ACTION_CREATE_FOLLOW_UP_TASK,
    ChannelAttachment,
    ChannelDirection,
    ChannelMessage,
    ChannelPreviewService,
    ChannelSession,
    build_channel_draft_preview,
    stable_draft_id,
)


NOW = datetime(2026, 5, 9, 13, 0, tzinfo=timezone.utc)


def inbound_message(text: str = "Сколько стоит подготовка к ЕГЭ?") -> ChannelMessage:
    return ChannelMessage(
        channel="site_chat",
        channel_message_id="msg-1",
        channel_thread_id="thread-1",
        channel_user_id="visitor-1",
        direction=ChannelDirection.INBOUND,
        text=text,
        received_at=NOW,
        raw_payload={"secretish": "not included by default"},
    )


def test_build_preview_creates_draft_reply_and_action_without_sending() -> None:
    message = inbound_message()
    preview = ChannelPreviewService().build_preview(message)

    assert preview.draft_id.startswith("channel_draft:site_chat:")
    assert preview.status == "needs_review"
    assert preview.reply.requires_approval is True
    assert preview.reply.text == "Здравствуйте! Спасибо за вопрос. Уточним детали и вернемся с ответом."
    assert "live_send_disabled" in preview.blocked_reasons
    assert preview.safety["live_send"] is False
    assert preview.safety["network_calls"] is False
    assert preview.safety["write_crm"] is False
    assert preview.reply.recommended_actions[0].action_type == "draft_client_message"
    assert preview.reply.recommended_actions[0].payload["live_send_enabled"] is False
    assert preview.reply.recommended_actions[1].action_type == "request_crm_context"


def test_build_preview_is_idempotent_for_same_message() -> None:
    message = inbound_message()

    first = build_channel_draft_preview(message)
    second = build_channel_draft_preview(message)

    assert first.draft_id == second.draft_id == stable_draft_id(message)
    assert first.idempotency_key == second.idempotency_key
    assert first.reply.recommended_actions[0].idempotency_key == second.reply.recommended_actions[0].idempotency_key


def test_build_preview_changes_id_for_different_message() -> None:
    first_message = inbound_message()
    second_message = ChannelMessage(
        channel="site_chat",
        channel_message_id="msg-2",
        channel_thread_id="thread-1",
        channel_user_id="visitor-1",
        direction="inbound",
        text="Можно записаться?",
        received_at=NOW,
    )

    assert stable_draft_id(first_message) != stable_draft_id(second_message)


def test_build_preview_rejects_non_inbound_message() -> None:
    message = ChannelMessage(
        channel="site_chat",
        channel_message_id="msg-1",
        channel_thread_id="thread-1",
        channel_user_id="manager-1",
        direction="outbound",
        text="Здравствуйте",
        received_at=NOW,
    )

    with pytest.raises(ValueError, match="inbound messages only"):
        build_channel_draft_preview(message)


def test_build_preview_rejects_session_mismatch() -> None:
    message = inbound_message()
    session = ChannelSession(
        channel="site_chat",
        channel_thread_id="another-thread",
        updated_at=NOW,
    )

    with pytest.raises(ValueError, match="session thread must match"):
        build_channel_draft_preview(message, session=session)


def test_build_preview_uses_context_safe_draft_override() -> None:
    message = inbound_message("Хочу поговорить с менеджером")

    preview = build_channel_draft_preview(
        message,
        context={"safe_draft_text": "Здравствуйте! Передадим запрос менеджеру, он вернется с ответом."},
    )

    assert preview.reply.text == "Здравствуйте! Передадим запрос менеджеру, он вернется с ответом."
    assert preview.reply.recommended_actions[0].payload["context_keys"] == ("safe_draft_text",)


def test_build_preview_uses_only_approved_question_catalog_answer() -> None:
    message = inbound_message("Сколько стоит курс?")
    preview = build_channel_draft_preview(
        message,
        context={
            "question_catalog_answer": {
                "approved_for_bot": True,
                "final_approved_answer": "Здравствуйте! Актуальная стоимость указана в утвержденном прайсе.",
                "required_fact_keys": (),
            }
        },
    )

    assert preview.reply.text == "Здравствуйте! Актуальная стоимость указана в утвержденном прайсе."
    assert "question_catalog_context_used" in preview.reply.safety_flags
    assert preview.reply.metadata["question_catalog_context_used"] is True


def test_build_preview_blocks_unapproved_question_catalog_answer() -> None:
    message = inbound_message("Сколько стоит курс?")
    preview = build_channel_draft_preview(
        message,
        context={
            "question_catalog_answer": {
                "approved_for_bot": False,
                "final_approved_answer": "Нельзя использовать без РОПа.",
                "required_fact_keys": (),
            }
        },
    )

    assert preview.reply.text == "Здравствуйте! Спасибо за вопрос. Передадим его менеджеру и вернемся с проверенным ответом."
    assert preview.reply.requires_approval is True


def test_build_preview_includes_follow_up_action_when_message_requests_callback() -> None:
    message = inbound_message("Пожалуйста, перезвоните мне завтра")

    preview = build_channel_draft_preview(message, context={"follow_up_due_at": "2026-05-10"})

    by_type = {action.action_type: action for action in preview.reply.recommended_actions}
    assert ACTION_CREATE_FOLLOW_UP_TASK in by_type
    assert by_type[ACTION_CREATE_FOLLOW_UP_TASK].payload["due_at"] == "2026-05-10"


def test_attachment_only_message_gets_safe_attachment_draft() -> None:
    message = ChannelMessage(
        channel="site_chat",
        channel_message_id="msg-attach",
        channel_thread_id="thread-1",
        channel_user_id="visitor-1",
        direction="inbound",
        attachments=(ChannelAttachment(kind="document", uri="memory://doc-1"),),
        received_at=NOW,
    )

    preview = build_channel_draft_preview(message)

    assert preview.reply.text == "Здравствуйте! Получили вложение. Проверим детали и вернемся с ответом."
    assert preview.safety["llm_calls"] is False


def test_preview_json_excludes_raw_payload_by_default() -> None:
    preview = build_channel_draft_preview(inbound_message())
    payload = preview.to_json_dict()

    assert "raw_payload" not in payload["source_message"]
    assert payload["source_message"]["channel_message_id"] == "msg-1"
    assert payload["reply"]["metadata"]["draft_id"] == preview.draft_id


def test_preview_json_can_include_raw_payload_when_explicitly_requested() -> None:
    preview = build_channel_draft_preview(inbound_message())
    payload = preview.to_json_dict(include_raw_payload=True)

    assert payload["source_message"]["raw_payload"] == {"secretish": "not included by default"}
