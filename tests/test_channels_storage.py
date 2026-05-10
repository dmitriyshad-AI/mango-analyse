from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from mango_mvp.channels import (
    ACTION_STATUS_APPROVED,
    ACTION_STATUS_MOCK_EXECUTED,
    DRAFT_STATUS_APPROVED,
    DRAFT_STATUS_MOCK_SENT,
    DRAFT_STATUS_NEEDS_REVIEW,
    ChannelMemoryStore,
    ChannelMessage,
    ChannelSession,
    build_and_store_channel_draft_preview,
    build_channel_draft_preview,
    channel_store_safety_contract,
)


START = datetime(2026, 5, 9, 15, 0, tzinfo=timezone.utc)


class StepClock:
    def __init__(self) -> None:
        self.value = START

    def __call__(self) -> datetime:
        current = self.value
        self.value = self.value + timedelta(seconds=1)
        return current


def inbound_message(text: str = "Здравствуйте, расскажите про подготовку к ЕГЭ") -> ChannelMessage:
    return ChannelMessage(
        channel="site_chat",
        channel_message_id="msg-1",
        channel_thread_id="thread-1",
        channel_user_id="visitor-1",
        direction="inbound",
        text=text,
        received_at=START,
        raw_payload={"secret": "hidden by default"},
    )


def test_memory_store_upserts_message_idempotently_without_raw_payload_in_snapshot() -> None:
    store = ChannelMemoryStore(clock=StepClock())
    message = inbound_message()

    first = store.upsert_message(message, actor="test")
    second = store.upsert_message(message, actor="test")
    snapshot = store.snapshot()

    assert first.created is True
    assert second.created is False
    assert second.status == "duplicate"
    assert store.get_message(message.idempotency_key) is not None
    assert snapshot["summary"]["messages"] == 1
    assert snapshot["summary"]["history_events"] == 1
    assert "raw_payload" not in snapshot["messages"][0]["message"]
    assert snapshot["safety"]["write_runtime_db"] is False


def test_build_and_store_preview_creates_draft_actions_and_history_idempotently() -> None:
    store = ChannelMemoryStore(clock=StepClock())
    message = inbound_message()

    preview, result = build_and_store_channel_draft_preview(store, message, actor="preview_builder")
    _, duplicate = build_and_store_channel_draft_preview(store, message, actor="preview_builder")

    assert result.created is True
    assert result.message_created is True
    assert result.status == DRAFT_STATUS_NEEDS_REVIEW
    assert result.actions_total == 2
    assert result.actions_created == 2
    assert duplicate.created is False
    assert duplicate.actions_created == 0
    assert store.get_draft(preview.draft_id) is not None
    assert len(store.list_actions(draft_id=preview.draft_id)) == 2
    assert len(store.list_history(session_key=preview.session.session_key)) == 4
    assert len(store.list_history(entity_type="recommended_action")) == 2


def test_draft_lifecycle_requires_approval_before_mock_send_and_keeps_manager_context() -> None:
    store = ChannelMemoryStore(clock=StepClock())
    preview = build_channel_draft_preview(inbound_message())
    store.upsert_preview(preview, actor="preview_builder")

    with pytest.raises(ValueError, match="needs_review -> mock_sent"):
        store.transition_draft(preview.draft_id, DRAFT_STATUS_MOCK_SENT, actor="manager-1")

    approved = store.transition_draft(
        preview.draft_id,
        DRAFT_STATUS_APPROVED,
        actor="manager-1",
        reason="Ответ безопасен",
        manager_context={"review_note": "Можно отправить тестово"},
    )
    mock_sent = store.transition_draft(
        preview.draft_id,
        DRAFT_STATUS_MOCK_SENT,
        actor="manager-1",
        reason="Dry-run отправка для workspace",
    )

    assert approved.approved_by == "manager-1"
    assert approved.manager_context["review_note"] == "Можно отправить тестово"
    assert mock_sent.status == DRAFT_STATUS_MOCK_SENT
    assert mock_sent.manager_context["review_note"] == "Можно отправить тестово"
    status_events = store.list_history(entity_type="channel_draft", entity_id=preview.draft_id)
    assert [event.event_type for event in status_events] == [
        "draft_created",
        "draft_status_changed",
        "draft_status_changed",
    ]
    assert status_events[-1].payload["live_send_executed"] is False


def test_storage_refuses_real_live_send_and_real_action_execution_statuses() -> None:
    store = ChannelMemoryStore(clock=StepClock())
    preview = build_channel_draft_preview(inbound_message())
    store.upsert_preview(preview)
    action = store.list_actions(draft_id=preview.draft_id)[0]

    with pytest.raises(ValueError, match="real live-send"):
        store.transition_draft(preview.draft_id, "sent", actor="manager-1")
    with pytest.raises(ValueError, match="real action execution"):
        store.transition_action(action.idempotency_key, "executed", actor="manager-1")


def test_action_lifecycle_approves_before_mock_execution_and_exports_agent_proposal() -> None:
    store = ChannelMemoryStore(clock=StepClock())
    preview = build_channel_draft_preview(inbound_message("Пожалуйста, перезвоните завтра"))
    store.upsert_preview(preview, actor="preview_builder")
    action = store.list_actions(draft_id=preview.draft_id)[0]

    with pytest.raises(ValueError, match="proposed -> mock_executed"):
        store.transition_action(action.idempotency_key, ACTION_STATUS_MOCK_EXECUTED, actor="manager-1")

    approved = store.transition_action(
        action.idempotency_key,
        ACTION_STATUS_APPROVED,
        actor="manager-1",
        reason="Проверено менеджером",
    )
    executed = store.transition_action(
        action.idempotency_key,
        ACTION_STATUS_MOCK_EXECUTED,
        actor="manager-1",
        reason="Только dry-run",
    )

    assert approved.status == ACTION_STATUS_APPROVED
    assert executed.status == ACTION_STATUS_MOCK_EXECUTED
    exported = executed.to_json_dict()
    assert exported["agent_action_proposal"]["action_type"] == action.action.action_type
    assert exported["agent_action_proposal"]["payload"]["channel_action_policy"]["live_execution_allowed"] is False


def test_store_filters_drafts_actions_and_validates_session_match() -> None:
    store = ChannelMemoryStore(clock=StepClock())
    message = inbound_message()
    wrong_session = ChannelSession(channel="site_chat", channel_thread_id="other-thread", updated_at=START)

    with pytest.raises(ValueError, match="session thread must match"):
        store.upsert_message(message, session=wrong_session)

    preview = build_channel_draft_preview(message)
    store.upsert_preview(preview)

    assert store.list_drafts(status=DRAFT_STATUS_NEEDS_REVIEW) == (store.get_draft(preview.draft_id),)
    assert len(store.list_actions(session_key=preview.session.session_key)) == len(preview.reply.recommended_actions)
    assert store.summary()["draft_status_counts"] == {DRAFT_STATUS_NEEDS_REVIEW: 1}
    assert channel_store_safety_contract()["live_send"] is False
