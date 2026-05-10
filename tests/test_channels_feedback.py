from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from mango_mvp.channels import (
    FEEDBACK_ACTION_ACCEPTED,
    FEEDBACK_CLIENT_NO_REPLY,
    FEEDBACK_CLIENT_REPLIED,
    FEEDBACK_FOLLOW_UP_OVERDUE,
    FEEDBACK_LEAD_LOST,
    FEEDBACK_LEAD_WON,
    FEEDBACK_MANAGER_DRAFT_APPROVED,
    FEEDBACK_MANAGER_DRAFT_EDITED,
    FEEDBACK_ROP_ANSWER_RISKY,
    CRM_CHAT_CHANNEL,
    ChannelFeedbackMemoryStore,
    ChannelMemoryStore,
    ChannelMessage,
    FeedbackEvent,
    WebChatReadOnlyAdapter,
    build_action_feedback_event,
    build_and_store_channel_draft_preview,
    build_channel_draft_preview,
    build_channel_signal_decision,
    build_decision_feedback_event,
    build_feedback_loop_report,
    build_manager_draft_feedback_event,
    build_read_only_lead_outcome_event,
    feedback_loop_safety_contract,
)


START = datetime(2026, 5, 9, 19, 0, tzinfo=timezone.utc)


class StepClock:
    def __init__(self) -> None:
        self.value = START

    def __call__(self) -> datetime:
        current = self.value
        self.value = self.value + timedelta(seconds=1)
        return current


def inbound_message(text: str = "Здравствуйте, сколько стоит курс?") -> ChannelMessage:
    return ChannelMessage(
        channel="site_chat",
        channel_message_id="msg-feedback-1",
        channel_thread_id="thread-feedback-1",
        channel_user_id="visitor-feedback-1",
        direction="inbound",
        text=text,
        received_at=START,
        raw_payload={"secret": "must not be exported by feedback"},
    )


def stored_draft():
    store = ChannelMemoryStore(clock=StepClock())
    preview = build_channel_draft_preview(inbound_message("Срочно хочу оплатить, сколько стоит курс?"))
    store.upsert_preview(preview, actor="preview_builder")
    return store, preview, store.get_draft(preview.draft_id)


def test_feedback_event_validation_blocks_unsafe_or_unlinked_records() -> None:
    with pytest.raises(ValueError, match="require draft_id"):
        FeedbackEvent(
            event_type=FEEDBACK_MANAGER_DRAFT_APPROVED,
            session_key="channel_session:site_chat:test",
            actor="manager-1",
            occurred_at=START,
        )

    with pytest.raises(ValueError, match="requires edited draft text"):
        FeedbackEvent(
            event_type=FEEDBACK_MANAGER_DRAFT_EDITED,
            session_key="channel_session:site_chat:test",
            actor="manager-1",
            draft_id="draft-1",
            occurred_at=START,
        )

    with pytest.raises(ValueError, match="read-only"):
        FeedbackEvent(
            event_type=FEEDBACK_LEAD_WON,
            session_key="channel_session:site_chat:test",
            actor="amocrm_import",
            entity_type="crm_lead",
            entity_id="lead-1",
            source_system="amocrm",
            occurred_at=START,
        )

    with pytest.raises(ValueError, match="raw payload"):
        FeedbackEvent(
            event_type=FEEDBACK_CLIENT_REPLIED,
            session_key="channel_session:site_chat:test",
            actor="system",
            occurred_at=START,
            metadata={"raw_payload": {"provider": "hidden"}},
        )

    with pytest.raises(ValueError, match="external side effect"):
        FeedbackEvent(
            event_type=FEEDBACK_CLIENT_REPLIED,
            session_key="channel_session:site_chat:test",
            actor="system",
            occurred_at=START,
            metadata={"crm_write_executed": True},
        )


def test_feedback_store_records_draft_review_idempotently_without_raw_payload() -> None:
    _, _, draft = stored_draft()
    assert draft is not None
    feedback = ChannelFeedbackMemoryStore(clock=StepClock())
    event = build_manager_draft_feedback_event(
        draft,
        FEEDBACK_MANAGER_DRAFT_APPROVED,
        actor="manager-1",
        occurred_at=START,
        reason="Ответ можно использовать как черновик",
    )

    first = feedback.record_event(event)
    duplicate = feedback.record_event(event)
    snapshot = feedback.snapshot()

    assert first.created is True
    assert duplicate.created is False
    assert duplicate.status == "duplicate"
    assert snapshot["summary"]["manager_review"]["approved"] == 1
    assert snapshot["summary"]["manager_review"]["approval_rate"] == 1.0
    assert "must not be exported by feedback" not in str(snapshot)
    assert snapshot["safety"]["write_crm"] is False


def test_feedback_summary_links_actions_client_followup_rop_and_read_only_lead_outcome() -> None:
    store, preview, draft = stored_draft()
    assert draft is not None
    action = store.list_actions(draft_id=preview.draft_id)[0]
    decision = build_channel_signal_decision(
        message=preview.source_message,
        preview=preview,
        context={"priority": "hot"},
    )
    events = (
        build_action_feedback_event(
            action,
            FEEDBACK_ACTION_ACCEPTED,
            actor="manager-1",
            occurred_at=START,
            decision_id=decision.decision_id,
        ),
        build_decision_feedback_event(
            decision,
            FEEDBACK_CLIENT_REPLIED,
            actor="channel_import",
            occurred_at=START + timedelta(seconds=1),
            value="Клиент уточнил расписание",
        ),
        build_decision_feedback_event(
            decision,
            FEEDBACK_FOLLOW_UP_OVERDUE,
            actor="scheduler",
            occurred_at=START + timedelta(seconds=2),
        ),
        build_decision_feedback_event(
            decision,
            FEEDBACK_ROP_ANSWER_RISKY,
            actor="rop-1",
            occurred_at=START + timedelta(seconds=3),
            score=0.2,
            metadata={"reason": "Нужно проверить коммерческие обещания"},
        ),
        build_read_only_lead_outcome_event(
            session_key=decision.session_key,
            lead_id="lead-100",
            event_type=FEEDBACK_LEAD_WON,
            source_system="amocrm",
            actor="amocrm_read_only_import",
            occurred_at=START + timedelta(seconds=4),
            decision_id=decision.decision_id,
            value="won",
        ),
    )
    feedback = ChannelFeedbackMemoryStore(clock=StepClock())
    feedback.record_many(events)

    summary = feedback.summary(session_key=decision.session_key)
    report = feedback.build_report(decision=decision)

    assert summary["action_review"]["accepted"] == 1
    assert summary["client_engagement"]["reply_rate"] == 1.0
    assert summary["follow_up"]["overdue"] == 1
    assert summary["lead_outcomes"]["won"] == 1
    assert summary["rop_quality"]["risky"] == 1
    assert summary["positive_events"] == 3
    assert summary["risk_events"] == 2
    assert report.decision_id == decision.decision_id
    assert len(report.events) == 5
    assert report.safety["live_send"] is False
    assert report.safety["write_runtime_db"] is False


def test_lead_outcome_factory_requires_read_only_import_and_source_system() -> None:
    event = build_read_only_lead_outcome_event(
        session_key="channel_session:site_chat:test",
        lead_id="lead-1",
        event_type=FEEDBACK_LEAD_LOST,
        source_system="amocrm",
        actor="amocrm_read_only_import",
        occurred_at=START,
        value="lost",
    )

    assert event.imported_read_only is True
    assert event.entity_type == "crm_lead"
    assert event.entity_id == "lead-1"
    assert event.source_system == "amocrm"
    assert event.sentiment_bucket == "risk"

    with pytest.raises(ValueError, match="read-only lead outcome"):
        build_read_only_lead_outcome_event(
            session_key="channel_session:site_chat:test",
            lead_id="lead-1",
            event_type=FEEDBACK_CLIENT_NO_REPLY,
            source_system="amocrm",
            actor="amocrm_read_only_import",
            occurred_at=START,
        )


def test_feedback_report_rejects_mixed_sessions_and_keeps_safety_contract() -> None:
    first = FeedbackEvent(
        event_type=FEEDBACK_CLIENT_REPLIED,
        session_key="channel_session:site_chat:first",
        actor="channel_import",
        occurred_at=START,
    )
    second = FeedbackEvent(
        event_type=FEEDBACK_CLIENT_NO_REPLY,
        session_key="channel_session:site_chat:second",
        actor="scheduler",
        occurred_at=START,
    )

    with pytest.raises(ValueError, match="match session_key"):
        build_feedback_loop_report(
            events=(first, second),
            session_key="channel_session:site_chat:first",
            created_at=START,
        )

    safety = feedback_loop_safety_contract()
    assert safety["network_calls"] is False
    assert safety["live_send"] is False
    assert safety["write_crm"] is False
    assert safety["lead_outcomes_are_read_only_imports"] is True


def test_feedback_store_report_filters_other_decision_events_in_same_session() -> None:
    _, preview, _ = stored_draft()
    decision = build_channel_signal_decision(message=preview.source_message, preview=preview)
    feedback = ChannelFeedbackMemoryStore(clock=StepClock())
    feedback.record_event(
        build_decision_feedback_event(
            decision,
            FEEDBACK_CLIENT_REPLIED,
            actor="channel_import",
            occurred_at=START,
        )
    )
    feedback.record_event(
        FeedbackEvent(
            event_type=FEEDBACK_CLIENT_NO_REPLY,
            session_key=decision.session_key,
            actor="scheduler",
            occurred_at=START + timedelta(seconds=1),
            decision_id="other-decision",
            message_idempotency_key=decision.message_idempotency_key,
        )
    )

    report = feedback.build_report(decision=decision)

    assert len(report.events) == 1
    assert report.events[0].decision_id == decision.decision_id
    assert report.metrics["total_events"] == 1


def test_web_chat_to_signal_to_feedback_loop_e2e_without_live_send_or_crm_write() -> None:
    adapter = WebChatReadOnlyAdapter(default_channel=CRM_CHAT_CHANNEL)
    channel_store = ChannelMemoryStore(clock=StepClock())
    feedback = ChannelFeedbackMemoryStore(clock=StepClock())
    payload = {
        "channel": CRM_CHAT_CHANNEL,
        "message_id": "crm-feedback-1",
        "conversation_id": "lead-feedback-1",
        "contact_id": "contact-feedback-1",
        "body": "Можно счет сегодня? Готов оплатить, нужен менеджер.",
        "timestamp": int(START.timestamp()),
    }

    message = adapter.parse_inbound(payload)[0]
    preview, _ = build_and_store_channel_draft_preview(
        channel_store,
        message,
        actor="crm_chat_preview",
        context={"priority": "hot"},
    )
    decision = build_channel_signal_decision(message=message, preview=preview, context={"priority": "hot"})
    draft = channel_store.get_draft(preview.draft_id)
    action = channel_store.list_actions(draft_id=preview.draft_id)[0]
    assert draft is not None
    feedback.record_many(
        (
            build_manager_draft_feedback_event(
                draft,
                FEEDBACK_MANAGER_DRAFT_APPROVED,
                actor="manager-1",
                occurred_at=START,
                decision_id=decision.decision_id,
            ),
            build_action_feedback_event(
                action,
                FEEDBACK_ACTION_ACCEPTED,
                actor="manager-1",
                occurred_at=START + timedelta(seconds=1),
                decision_id=decision.decision_id,
            ),
            build_decision_feedback_event(
                decision,
                FEEDBACK_CLIENT_NO_REPLY,
                actor="scheduler",
                occurred_at=START + timedelta(seconds=2),
            ),
        )
    )
    rendered = adapter.render_reply(preview.session, preview.reply)
    send_result = adapter.send(rendered)
    report = feedback.build_report(decision=decision)

    assert send_result.sent is False
    assert send_result.metadata["chat_api_called"] is False
    assert report.metrics["manager_review"]["approved"] == 1
    assert report.metrics["action_review"]["accepted"] == 1
    assert report.metrics["client_engagement"]["no_reply"] == 1
    assert report.safety["write_crm"] is False
