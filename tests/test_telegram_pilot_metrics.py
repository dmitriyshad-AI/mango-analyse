from __future__ import annotations

from datetime import datetime, timedelta, timezone

from mango_mvp.channels.contracts import ChannelMessage
from mango_mvp.channels.telegram_pilot_metrics import build_daily_metrics
from mango_mvp.channels.telegram_pilot_store import (
    PILOT_FEEDBACK_NEEDS_EDIT,
    PILOT_FEEDBACK_TOPIC_WRONG,
    PILOT_FEEDBACK_UNSAFE_FACT_ATTEMPT,
    PILOT_FEEDBACK_USEFUL,
    TelegramPilotSQLiteStore,
)


START = datetime(2026, 5, 16, 10, 0, tzinfo=timezone.utc)


class StepClock:
    def __init__(self) -> None:
        self.value = START

    def __call__(self) -> datetime:
        current = self.value
        self.value += timedelta(seconds=2)
        return current


def inbound_message(message_id: str, text: str) -> ChannelMessage:
    return ChannelMessage(
        channel="telegram_bot",
        channel_message_id=message_id,
        channel_thread_id="chat-1",
        channel_user_id="user-1",
        direction="inbound",
        text=text,
        received_at=START,
    )


def create_draft(store: TelegramPilotSQLiteStore, message_id: str, *, safety_flags=(), draft_metadata=None, route="draft_for_manager"):
    return store.upsert_message_context_draft(
        inbound_message(message_id, f"message {message_id}"),
        context={"context_version": "ctx-v1"},
        draft_text=f"draft {message_id}",
        prompt_version="prompt-v1",
        knowledge_base_version="kb-v1",
        safety_flags=safety_flags,
        route=route,
        draft_metadata=draft_metadata or {},
    )


def test_daily_metrics_counts_drafts_and_feedback(tmp_path) -> None:
    store = TelegramPilotSQLiteStore(tmp_path / "telegram_pilot.sqlite", clock=StepClock())
    first = create_draft(store, "msg-1")
    second = create_draft(store, "msg-2")
    store.record_feedback(first.draft_id, PILOT_FEEDBACK_USEFUL, actor="nastya", occurred_at=START + timedelta(minutes=1))
    store.record_feedback(
        second.draft_id,
        PILOT_FEEDBACK_NEEDS_EDIT,
        actor="nastya",
        occurred_at=START + timedelta(minutes=2),
    )

    metrics = build_daily_metrics(store, "2026-05-16")

    assert metrics.incoming_messages == 2
    assert metrics.drafts_created == 2
    assert metrics.useful_drafts == 1
    assert metrics.needs_edit_drafts == 1
    assert metrics.useful_share == 0.5
    assert metrics.verdict == "PASS_WITH_NOTES"
    assert metrics.avg_seconds_to_draft is not None
    store.close()


def test_daily_metrics_flags_unsafe_attempts(tmp_path) -> None:
    store = TelegramPilotSQLiteStore(tmp_path / "telegram_pilot.sqlite", clock=StepClock())
    flagged_by_draft = create_draft(store, "msg-1", safety_flags=("forbidden_fact_attempt",))
    flagged_by_feedback = create_draft(store, "msg-2")
    store.record_feedback(
        flagged_by_feedback.draft_id,
        PILOT_FEEDBACK_UNSAFE_FACT_ATTEMPT,
        actor="nastya",
        occurred_at=START + timedelta(minutes=1),
        metadata={"unsafe_fact_attempted": True},
    )
    store.record_feedback(
        flagged_by_feedback.draft_id,
        PILOT_FEEDBACK_TOPIC_WRONG,
        actor="nastya",
        occurred_at=START + timedelta(minutes=2),
        metadata={"llm_topic_error": True},
    )

    metrics = build_daily_metrics(store, "2026-05-16")

    assert metrics.unsafe_fact_attempts == 2
    assert metrics.verdict == "BLOCKED"
    assert metrics.blocked_reasons == ("unsafe_fact_attempts",)
    assert metrics.topic_errors == 1
    assert flagged_by_draft.draft_id in metrics.unsafe_draft_ids
    assert flagged_by_feedback.draft_id in metrics.unsafe_draft_ids
    assert metrics.topic_error_draft_ids == (flagged_by_feedback.draft_id,)
    store.close()


def test_daily_metrics_counts_funnel_progress(tmp_path) -> None:
    store = TelegramPilotSQLiteStore(tmp_path / "telegram_pilot.sqlite", clock=StepClock())
    create_draft(
        store,
        "msg-1",
        draft_metadata={
            "client_segment": "new_lead",
            "lead_stage": "next_step_offered",
            "next_step_type": "offer_group_check",
        },
        route="bot_answer_self_for_pilot",
    )
    create_draft(
        store,
        "msg-2",
        draft_metadata={
            "client_segment": "known_customer",
            "lead_stage": "manager_handoff",
            "next_step_type": "offer_manager_check",
            "manager_summary": "Бренд: УНПК МФТИ\nЧто проверить: оплату",
        },
        route="manager_only",
    )

    metrics = build_daily_metrics(store, "2026-05-16")

    assert metrics.new_leads == 1
    assert metrics.qualified_leads == 1
    assert metrics.next_step_offered == 2
    assert metrics.manager_handoffs == 1
    store.close()


def test_daily_metrics_counts_reasked_known_data(tmp_path) -> None:
    store = TelegramPilotSQLiteStore(tmp_path / "telegram_pilot.sqlite", clock=StepClock())
    create_draft(
        store,
        "msg-1",
        safety_flags=("asked_known_data_again",),
        draft_metadata={"lead_stage": "qualification_needed", "asked_known_data_again": True},
    )

    metrics = build_daily_metrics(store, "2026-05-16")

    assert metrics.reasked_known_data == 1
    store.close()
