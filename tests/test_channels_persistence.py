from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from mango_mvp.channels import (
    ACTION_STATUS_APPROVED,
    ACTION_STATUS_MOCK_EXECUTED,
    DRAFT_STATUS_APPROVED,
    DRAFT_STATUS_MOCK_SENT,
    ChannelMessage,
    ChannelSession,
    ChannelSQLiteStore,
    SignalDecision,
    build_action_feedback_event,
    build_channel_draft_preview,
    build_channel_signal_decision,
    build_read_only_lead_outcome_event,
)


START = datetime(2026, 5, 11, 1, 0, tzinfo=timezone.utc)


class StepClock:
    def __init__(self) -> None:
        self.value = START

    def __call__(self) -> datetime:
        current = self.value
        self.value = self.value + timedelta(seconds=1)
        return current


def inbound_message(
    *,
    message_id: str = "msg-1",
    thread_id: str = "thread-1",
    text: str = "Здравствуйте, сколько стоит курс?",
) -> ChannelMessage:
    return ChannelMessage(
        channel="site_chat",
        channel_message_id=message_id,
        channel_thread_id=thread_id,
        channel_user_id=f"user-{thread_id}",
        direction="inbound",
        text=text,
        received_at=START,
        raw_payload={"provider_secret": "must_not_be_stored"},
    )


def test_sqlite_store_bootstraps_reopens_and_keeps_message_idempotency(tmp_path) -> None:
    db_path = tmp_path / "channel_product.sqlite"
    clock = StepClock()
    message = inbound_message()

    store = ChannelSQLiteStore(db_path, clock=clock)
    first = store.upsert_message(message, actor="test")
    store.close()

    reopened = ChannelSQLiteStore(db_path, clock=clock)
    duplicate = reopened.upsert_message(message, actor="test")
    snapshot = reopened.snapshot(include_raw_payload=True)

    assert first.created is True
    assert duplicate.created is False
    assert duplicate.status == "duplicate"
    assert snapshot["summary"]["messages"] == 1
    assert snapshot["summary"]["history_events"] == 1
    assert "raw_payload" not in snapshot["messages"][0]["message"]
    assert "must_not_be_stored" not in str(snapshot)
    assert snapshot["safety"]["write_runtime_db"] is False
    assert snapshot["safety"]["writes_local_channel_product_db"] is True

    second_message = inbound_message(message_id="msg-2", text="Можно записаться сегодня?")
    reopened.upsert_message(second_message, actor="test")
    history_ids = [event.event_id for event in reopened.list_history()]
    assert history_ids[0].startswith("channel_history:00000001:")
    assert history_ids[-1].startswith("channel_history:00000002:")
    reopened.close()


def test_duplicate_message_does_not_mutate_session_before_later_commit(tmp_path) -> None:
    db_path = tmp_path / "channel_product.sqlite"
    store = ChannelSQLiteStore(db_path, clock=StepClock())
    first_message = inbound_message()
    second_message = inbound_message(message_id="msg-2", text="Новый вопрос")
    original_session = ChannelSession.from_message(first_message)
    changed_session = ChannelSession.from_message(
        first_message,
        crm_contact_id="crm-contact-should-not-leak",
        state={"unexpected": "session mutation"},
    )

    store.upsert_message(first_message, session=original_session)
    duplicate = store.upsert_message(first_message, session=changed_session)
    store.upsert_message(second_message)

    persisted = store.get_session(original_session.session_key)
    assert duplicate.status == "duplicate"
    assert persisted is not None
    assert persisted.crm_contact_id is None
    assert persisted.state == {}
    store.close()


def test_sqlite_preview_and_lifecycle_survive_reopen_without_live_statuses(tmp_path) -> None:
    db_path = tmp_path / "channel_product.sqlite"
    store = ChannelSQLiteStore(db_path, clock=StepClock())
    preview = build_channel_draft_preview(inbound_message(text="Срочно хочу оплатить, сколько стоит курс?"))

    created = store.upsert_preview(preview, actor="preview_builder")
    duplicate = store.upsert_preview(preview, actor="preview_builder")
    action = store.list_actions(draft_id=preview.draft_id)[0]

    assert created.created is True
    assert created.actions_created == created.actions_total
    assert created.actions_created >= 1
    assert duplicate.created is False
    assert duplicate.actions_created == 0

    with pytest.raises(ValueError, match="real live-send"):
        store.transition_draft(preview.draft_id, "sent", actor="manager-1")
    with pytest.raises(ValueError, match="proposed -> mock_executed"):
        store.transition_action(action.idempotency_key, ACTION_STATUS_MOCK_EXECUTED, actor="manager-1")

    store.transition_draft(preview.draft_id, DRAFT_STATUS_APPROVED, actor="manager-1", reason="ok")
    store.transition_draft(preview.draft_id, DRAFT_STATUS_MOCK_SENT, actor="manager-1", reason="dry-run only")
    store.transition_action(action.idempotency_key, ACTION_STATUS_APPROVED, actor="manager-1", reason="ok")
    store.transition_action(action.idempotency_key, ACTION_STATUS_MOCK_EXECUTED, actor="manager-1", reason="dry-run only")
    store.close()

    reopened = ChannelSQLiteStore(db_path, clock=StepClock())
    assert reopened.get_draft(preview.draft_id).status == DRAFT_STATUS_MOCK_SENT
    assert reopened.get_action(action.idempotency_key).status == ACTION_STATUS_MOCK_EXECUTED
    assert reopened.summary()["draft_status_counts"] == {DRAFT_STATUS_MOCK_SENT: 1}
    assert reopened.summary()["action_status_counts"][ACTION_STATUS_MOCK_EXECUTED] == 1
    reopened.close()


def test_sqlite_signal_and_feedback_records_are_idempotent_after_reopen(tmp_path) -> None:
    db_path = tmp_path / "channel_product.sqlite"
    store = ChannelSQLiteStore(db_path, clock=StepClock())
    preview = build_channel_draft_preview(inbound_message(text="Срочно хочу оплатить сегодня, перезвоните"))
    store.upsert_preview(preview, actor="preview_builder")
    action = store.list_actions(draft_id=preview.draft_id)[0]
    decision = build_channel_signal_decision(message=preview.source_message, preview=preview, context={"priority": "hot"})
    dirty_decision = SignalDecision(
        decision_id=decision.decision_id,
        message_idempotency_key=decision.message_idempotency_key,
        session_key=decision.session_key,
        signals=decision.signals,
        safe_answer=decision.safe_answer,
        recommended_action_types=decision.recommended_action_types,
        policy_flags=decision.policy_flags,
        blocked_reasons=decision.blocked_reasons,
        created_at=decision.created_at,
        metadata={**decision.metadata, "raw_payload": {"secret": "signal_secret"}},
    )

    signal_result = store.record_signal_decision(dirty_decision, actor="signal_engine")
    feedback = build_action_feedback_event(
        action,
        "action_accepted",
        actor="manager-1",
        occurred_at=START,
        decision_id=decision.decision_id,
    )
    outcome = build_read_only_lead_outcome_event(
        session_key=decision.session_key,
        lead_id="lead-1",
        event_type="lead_won",
        source_system="amocrm",
        actor="amocrm_read_only_import",
        occurred_at=START + timedelta(seconds=1),
        decision_id=decision.decision_id,
    )
    feedback_results = store.record_feedback_many((feedback, outcome))
    store.close()

    reopened = ChannelSQLiteStore(db_path, clock=StepClock())
    duplicate_signal = reopened.record_signal_decision(dirty_decision, actor="signal_engine")
    duplicate_feedback = reopened.record_feedback_event(feedback)
    snapshot = reopened.snapshot()

    assert signal_result.created is True
    assert duplicate_signal.created is False
    assert [item.created for item in feedback_results] == [True, True]
    assert duplicate_feedback.created is False
    assert snapshot["summary"]["signal_decisions"] == 1
    assert snapshot["summary"]["feedback_events"] == 2
    assert snapshot["feedback"]["summary"]["lead_outcomes"]["won"] == 1
    assert snapshot["feedback"]["summary"]["action_review"]["accepted"] == 1
    assert snapshot["safety"]["lead_outcomes_are_read_only_imports"] is True
    assert "must_not_be_stored" not in str(snapshot)
    assert "signal_secret" not in str(snapshot)
    assert "raw_payload" not in str(snapshot["signal_decisions"])
    reopened.close()


def test_sqlite_store_rejects_runtime_paths_and_blocks_read_only_mutations(tmp_path) -> None:
    with pytest.raises(ValueError, match="stable_runtime"):
        ChannelSQLiteStore(tmp_path / "stable_runtime" / "channel.sqlite")
    runtime_dir = tmp_path / "stable_runtime"
    runtime_dir.mkdir()
    runtime_link = tmp_path / "channel_link"
    runtime_link.symlink_to(runtime_dir, target_is_directory=True)
    with pytest.raises(ValueError, match="stable_runtime"):
        ChannelSQLiteStore(runtime_link / "channel.sqlite")
    with pytest.raises(ValueError, match="runtime-looking"):
        ChannelSQLiteStore(tmp_path / "runtime.db")

    db_path = tmp_path / "channel_product.sqlite"
    writable = ChannelSQLiteStore(db_path, clock=StepClock())
    writable.upsert_message(inbound_message())
    writable.close()

    readonly = ChannelSQLiteStore.open_read_only(db_path)
    assert readonly.summary()["messages"] == 1
    with pytest.raises(PermissionError, match="read-only"):
        readonly.upsert_message(inbound_message(message_id="msg-read-only"))
    assert readonly.summary()["messages"] == 1
    readonly.close()
