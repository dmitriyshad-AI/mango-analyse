from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from mango_mvp.channels import (
    ChannelMessage,
    ChannelSQLiteStore,
    build_action_feedback_event,
    build_channel_draft_preview,
    build_channel_signal_decision,
    build_channel_workspace_summary,
)


START = datetime(2026, 5, 11, 2, 0, tzinfo=timezone.utc)


class StepClock:
    def __init__(self) -> None:
        self.value = START

    def __call__(self) -> datetime:
        current = self.value
        self.value = self.value + timedelta(seconds=1)
        return current


def message(message_id: str, thread_id: str, text: str) -> ChannelMessage:
    return ChannelMessage(
        channel="site_chat",
        channel_message_id=message_id,
        channel_thread_id=thread_id,
        channel_user_id=f"visitor-{thread_id}",
        direction="inbound",
        text=text,
        received_at=START,
        raw_payload={"provider_secret": "redacted"},
    )


def test_workspace_summary_builds_operator_inbox_without_mutating_store(tmp_path) -> None:
    store = ChannelSQLiteStore(tmp_path / "channel_product.sqlite", clock=StepClock())
    preview = build_channel_draft_preview(message("msg-1", "thread-hot", "Срочно хочу оплатить сегодня"))
    store.upsert_preview(preview, actor="preview_builder")
    decision = build_channel_signal_decision(message=preview.source_message, preview=preview, context={"priority": "hot"})
    store.record_signal_decision(decision, actor="signal_engine")

    action = store.list_actions(draft_id=preview.draft_id)[0]
    feedback = build_action_feedback_event(
        action,
        "action_dismissed",
        actor="manager-1",
        occurred_at=START,
        decision_id=decision.decision_id,
        reason="Нужно уточнить оффер",
    )
    store.record_feedback_event(feedback)
    before = store.summary()

    workspace = build_channel_workspace_summary(store, created_at=START)
    after = store.summary()
    payload = workspace.to_json_dict()

    assert before == after
    assert payload["metrics"]["sessions"] == 1
    assert payload["metrics"]["needs_review_sessions"] == 1
    assert payload["metrics"]["drafts_needing_review"] == 1
    assert payload["metrics"]["actions_needing_review"] >= 1
    assert payload["metrics"]["hot_leads"] == 1
    assert payload["metrics"]["risk_feedback_events"] == 1
    assert payload["inbox"][0]["status"] == "needs_review"
    assert "hot_lead" in payload["inbox"][0]["reasons"]
    assert payload["safety"]["read_only_workspace"] is True
    assert payload["safety"]["live_send"] is False
    assert payload["safety"]["write_crm"] is False
    assert "redacted" not in str(payload)


def test_workspace_filters_by_session_and_validates_limit(tmp_path) -> None:
    store = ChannelSQLiteStore(tmp_path / "channel_product.sqlite", clock=StepClock())
    first = build_channel_draft_preview(message("msg-1", "thread-1", "Сколько стоит курс?"))
    second = build_channel_draft_preview(message("msg-2", "thread-2", "Здравствуйте"))
    store.upsert_preview(first)
    store.upsert_preview(second)

    filtered = build_channel_workspace_summary(store, session_key=first.session.session_key, limit=1)

    assert len(filtered.inbox) == 1
    assert filtered.inbox[0].session_key == first.session.session_key
    assert filtered.metrics["sessions"] == 1
    with pytest.raises(ValueError, match="between 1 and 500"):
        build_channel_workspace_summary(store, limit=0)
    with pytest.raises(ValueError, match="between 1 and 500"):
        build_channel_workspace_summary(store, limit=501)


def test_workspace_can_read_from_sqlite_read_only_connection(tmp_path) -> None:
    db_path = tmp_path / "channel_product.sqlite"
    writable = ChannelSQLiteStore(db_path, clock=StepClock())
    preview = build_channel_draft_preview(message("msg-1", "thread-1", "Можно записаться?"))
    writable.upsert_preview(preview)
    writable.close()

    readonly = ChannelSQLiteStore.open_read_only(db_path)
    summary = build_channel_workspace_summary(readonly)

    assert summary.metrics["sessions"] == 1
    assert summary.safety["read_only_workspace"] is True
    assert readonly.summary()["messages"] == 1
    readonly.close()
