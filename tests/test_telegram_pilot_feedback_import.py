from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

from mango_mvp.channels.contracts import ChannelMessage
from mango_mvp.channels.telegram_pilot_reporting import import_employee_feedback_csv
from mango_mvp.channels.telegram_pilot_store import (
    PILOT_DRAFT_STATUS_MANAGER_MARKED_NEEDS_EDIT,
    PILOT_DRAFT_STATUS_MANAGER_MARKED_USEFUL,
    TelegramPilotSQLiteStore,
)


START = datetime(2026, 5, 23, 12, 0, tzinfo=timezone.utc)


def inbound_message(message_id: str) -> ChannelMessage:
    return ChannelMessage(
        channel="telegram_public_pilot_bot",
        channel_message_id=message_id,
        channel_thread_id="chat-1",
        channel_user_id="user-1",
        direction="inbound",
        text="Какая цена?",
        received_at=START,
    )


def write_feedback_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields = ("draft_id", "human_verdict", "human_comment", "corrected_answer")
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_import_employee_feedback_updates_store_statuses(tmp_path: Path) -> None:
    store = TelegramPilotSQLiteStore(tmp_path / "telegram_pilot.sqlite", clock=lambda: START)
    useful = store.upsert_message_context_draft(
        inbound_message("msg-1"),
        context={},
        draft_text="Полезный ответ",
        prompt_version="prompt",
        knowledge_base_version="kb",
    )
    edit = store.upsert_message_context_draft(
        inbound_message("msg-2"),
        context={},
        draft_text="Нужно поправить",
        prompt_version="prompt",
        knowledge_base_version="kb",
    )
    csv_path = tmp_path / "feedback.csv"
    write_feedback_csv(
        csv_path,
        [
            {"draft_id": useful.draft_id, "human_verdict": "useful", "human_comment": "норм", "corrected_answer": ""},
            {"draft_id": edit.draft_id, "human_verdict": "minor_edit", "human_comment": "тон", "corrected_answer": "Лучше так"},
            {"draft_id": "", "human_verdict": "useful", "human_comment": "skip", "corrected_answer": ""},
        ],
    )

    summary = import_employee_feedback_csv(store, csv_path, actor="nastya")
    useful_draft = store.get_draft(useful.draft_id)
    edit_draft = store.get_draft(edit.draft_id)

    assert summary.imported == 2
    assert summary.skipped == 1
    assert useful_draft is not None
    assert useful_draft["status"] == PILOT_DRAFT_STATUS_MANAGER_MARKED_USEFUL
    assert edit_draft is not None
    assert edit_draft["status"] == PILOT_DRAFT_STATUS_MANAGER_MARKED_NEEDS_EDIT
    assert len(store.list_feedback_events(day="2026-05-23")) == 2
    store.close()


def test_import_employee_feedback_reports_unknown_draft(tmp_path: Path) -> None:
    store = TelegramPilotSQLiteStore(tmp_path / "telegram_pilot.sqlite", clock=lambda: START)
    csv_path = tmp_path / "feedback.csv"
    write_feedback_csv(
        csv_path,
        [{"draft_id": "missing", "human_verdict": "unsafe", "human_comment": "bad", "corrected_answer": ""}],
    )

    summary = import_employee_feedback_csv(store, csv_path)

    assert summary.imported == 0
    assert summary.skipped == 1
    assert summary.errors
    store.close()
