from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from mango_mvp.channels.contracts import ChannelMessage
from mango_mvp.channels.telegram_pilot_store import (
    PILOT_DRAFT_STATUS_MANAGER_MARKED_USEFUL,
    PILOT_DRAFT_STATUS_NEEDS_REVIEW,
    PILOT_FEEDBACK_MANAGER_ONLY,
    PILOT_FEEDBACK_NEEDS_EDIT,
    PILOT_FEEDBACK_USEFUL,
    TelegramPilotSQLiteStore,
)


START = datetime(2026, 5, 16, 9, 0, tzinfo=timezone.utc)


class StepClock:
    def __init__(self) -> None:
        self.value = START

    def __call__(self) -> datetime:
        current = self.value
        self.value += timedelta(seconds=1)
        return current


def inbound_message(message_id: str = "msg-1", text: str = "price question") -> ChannelMessage:
    return ChannelMessage(
        channel="telegram_bot",
        channel_message_id=message_id,
        channel_thread_id="chat-1",
        channel_user_id="user-1",
        direction="inbound",
        text=text,
        received_at=START,
    )


def store_draft(store: TelegramPilotSQLiteStore, message: ChannelMessage):
    return store.upsert_message_context_draft(
        message,
        context={"topic_id": "theme:price", "context_version": "ctx-v1"},
        draft_text="Draft for manager",
        prompt_version="prompt-v1",
        knowledge_base_version="kb-v1",
        topic_id="theme:price",
        route="draft_for_manager",
    )


def test_store_message_context_draft_feedback(tmp_path) -> None:
    store = TelegramPilotSQLiteStore(tmp_path / "telegram_pilot.sqlite", clock=StepClock())
    result = store_draft(store, inbound_message())

    feedback = store.record_feedback(
        result.draft_id,
        PILOT_FEEDBACK_USEFUL,
        actor="nastya",
        reason="usable",
        occurred_at=START + timedelta(minutes=5),
    )
    draft = store.get_draft(result.draft_id)
    summary = store.summary()

    assert result.created is True
    assert result.message_created is True
    assert result.context_created is True
    assert result.status == PILOT_DRAFT_STATUS_NEEDS_REVIEW
    assert feedback.created is True
    assert draft is not None
    assert draft["status"] == PILOT_DRAFT_STATUS_MANAGER_MARKED_USEFUL
    assert draft["prompt_version"] == "prompt-v1"
    assert draft["knowledge_base_version"] == "kb-v1"
    assert summary["messages"] == 1
    assert summary["contexts"] == 1
    assert summary["drafts"] == 1
    assert summary["feedback_events"] == 1
    assert summary["safety"]["live_send"] is False
    store.close()


def test_idempotent_update_does_not_duplicate_draft(tmp_path) -> None:
    store = TelegramPilotSQLiteStore(tmp_path / "telegram_pilot.sqlite", clock=StepClock())
    message = inbound_message()

    first = store_draft(store, message)
    duplicate = store.upsert_message_context_draft(
        message,
        context={"topic_id": "theme:other", "context_version": "ctx-v2"},
        draft_text="Different text must not replace first draft",
        prompt_version="prompt-v2",
        knowledge_base_version="kb-v2",
    )
    draft = store.get_draft(first.draft_id)
    summary = store.summary()

    assert first.created is True
    assert duplicate.created is False
    assert duplicate.draft_id == first.draft_id
    assert draft is not None
    assert draft["draft_text"] == "Draft for manager"
    assert summary["messages"] == 1
    assert summary["contexts"] == 1
    assert summary["drafts"] == 1
    store.close()


def test_store_keeps_funnel_context_and_manager_summary_idempotently(tmp_path) -> None:
    store = TelegramPilotSQLiteStore(tmp_path / "telegram_pilot.sqlite", clock=StepClock())
    message = inbound_message("msg-funnel", "9 класс, физика")

    first = store.upsert_message_context_draft(
        message,
        context={"funnel_state": {"lead_stage": "qualification_needed"}},
        draft_text="Draft for manager",
        prompt_version="prompt-v1",
        knowledge_base_version="kb-v1",
        route="draft_for_manager",
        draft_metadata={
            "funnel_state": {"lead_stage": "qualification_needed"},
            "lead_stage": "qualification_needed",
            "next_step_type": "ask_format",
            "manager_summary": "Бренд: УНПК МФТИ\nЧто нужно проверить: расписание",
        },
    )
    duplicate = store.upsert_message_context_draft(
        message,
        context={"funnel_state": {"lead_stage": "other"}},
        draft_text="Different",
        prompt_version="prompt-v2",
        knowledge_base_version="kb-v2",
        route="manager_only",
        draft_metadata={"lead_stage": "other", "manager_summary": "different"},
    )
    draft = store.get_draft(first.draft_id)

    assert duplicate.created is False
    assert draft is not None
    assert draft["metadata"]["lead_stage"] == "qualification_needed"
    assert draft["metadata"]["next_step_type"] == "ask_format"
    assert draft["metadata"]["manager_summary"].startswith("Бренд: УНПК")
    assert draft["draft_text"] == "Draft for manager"
    store.close()


def test_latest_dialogue_memory_snapshot_returns_latest_for_session(tmp_path) -> None:
    store = TelegramPilotSQLiteStore(tmp_path / "telegram_pilot.sqlite", clock=StepClock())
    session_id = "telegram_public_pilot:foton:123"

    store.upsert_dialogue_memory_snapshot(
        message_key="msg-1",
        session_id=session_id,
        active_brand="foton",
        memory_snapshot={"known_slots": {"grade": "8"}, "schema_version": "dialogue_memory_v2_2026_05_23"},
        created_at=START,
    )
    store.upsert_dialogue_memory_snapshot(
        message_key="msg-2",
        session_id=session_id,
        active_brand="foton",
        memory_snapshot={
            "known_slots": {"grade": "8", "subject": "физика"},
            "schema_version": "dialogue_memory_v2_2026_05_23",
        },
        created_at=START + timedelta(minutes=1),
    )

    latest = store.latest_dialogue_memory_snapshot(session_id=session_id, active_brand="foton")
    other_brand = store.latest_dialogue_memory_snapshot(session_id=session_id, active_brand="unpk")
    store.close()

    assert latest is not None
    assert latest["known_slots"] == {"grade": "8", "subject": "физика"}
    assert other_brand is None


def test_daily_summary_counts_useful_drafts(tmp_path) -> None:
    store = TelegramPilotSQLiteStore(tmp_path / "telegram_pilot.sqlite", clock=StepClock())
    useful = store_draft(store, inbound_message("msg-1", "first"))
    needs_edit = store_draft(store, inbound_message("msg-2", "second"))
    manager_only = store_draft(store, inbound_message("msg-3", "third"))

    store.record_feedback(useful.draft_id, PILOT_FEEDBACK_USEFUL, actor="nastya", occurred_at=START + timedelta(minutes=1))
    store.record_feedback(
        needs_edit.draft_id,
        PILOT_FEEDBACK_NEEDS_EDIT,
        actor="nastya",
        occurred_at=START + timedelta(minutes=2),
    )
    store.record_feedback(
        manager_only.draft_id,
        PILOT_FEEDBACK_MANAGER_ONLY,
        actor="nastya",
        occurred_at=START + timedelta(minutes=3),
    )

    summary = store.daily_summary("2026-05-16")

    assert summary.incoming_messages == 3
    assert summary.drafts_created == 3
    assert summary.useful_drafts == 1
    assert summary.needs_edit_drafts == 1
    assert summary.manager_only_drafts == 1
    assert summary.feedback_events == 3
    assert summary.avg_seconds_to_draft is not None
    store.close()


def test_store_path_rejects_stable_runtime(tmp_path) -> None:
    with pytest.raises(ValueError, match="stable_runtime"):
        TelegramPilotSQLiteStore(tmp_path / "stable_runtime" / "telegram_pilot.sqlite")

    runtime_dir = tmp_path / "stable_runtime"
    runtime_dir.mkdir()
    runtime_link = tmp_path / "pilot_link"
    runtime_link.symlink_to(runtime_dir, target_is_directory=True)
    with pytest.raises(ValueError, match="stable_runtime"):
        TelegramPilotSQLiteStore(runtime_link / "telegram_pilot.sqlite")
