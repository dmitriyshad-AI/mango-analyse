from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from mango_mvp.channels.contracts import ChannelMessage
from mango_mvp.channels.telegram_pilot_reporting import build_pilot_daily_report
from mango_mvp.channels.telegram_pilot_store import TelegramPilotSQLiteStore


START = datetime(2026, 5, 23, 9, 0, tzinfo=timezone.utc)


class StepClock:
    def __init__(self) -> None:
        self.value = START

    def __call__(self) -> datetime:
        current = self.value
        self.value += timedelta(seconds=3)
        return current


def inbound_message(message_id: str, text: str) -> ChannelMessage:
    return ChannelMessage(
        channel="telegram_public_pilot_bot",
        channel_message_id=message_id,
        channel_thread_id="79092009933",
        channel_user_id="79092009933",
        direction="inbound",
        text=text,
        received_at=START,
        metadata={"brand": "unpk"},
    )


def test_daily_journal_report_writes_full_artifact_pack_and_masks_private_data(tmp_path: Path) -> None:
    store = TelegramPilotSQLiteStore(tmp_path / "telegram_pilot.sqlite", clock=StepClock())
    result = store.upsert_message_context_draft(
        inbound_message("msg-1", "9 класс, физика. Сколько стоит?"),
        context={
            "active_brand": "unpk",
            "known_client_fields": {"phone": "79092009933", "student_name": "Колосов Даниил"},
            "known_dialog_fields": {"grade": "9", "subject": "физика"},
            "confirmed_facts": {"fact:unpk:installment": "В УНПК можно платить помесячно, за семестр или за год."},
            "missing_facts": [],
        },
        draft_text="В УНПК можно платить помесячно, за семестр или за год. При оплате за семестр действует скидка 10%, за год - 14%.",
        prompt_version="telegram_public_pilot:gpt-5.5:high",
        knowledge_base_version="kb-v6.3",
        topic_id="theme:006_installment",
        route="bot_answer_self_for_pilot",
        safety_flags=("manager_approval_required", "no_auto_send"),
        draft_metadata={
            "brand": "unpk",
            "latency_seconds": 4.2,
            "client_send_executed": True,
            "known_slots": {"grade": "9", "subject": "физика"},
            "missing_slots": [],
            "funnel_state": {
                "lead_stage": "next_step_offered",
                "filled_slots": {"grade": "9", "subject": "физика"},
                "missing_slots": [],
                "next_step_type": "offer_group_check",
                "semantic_flags": ["class_and_subject_known"],
            },
            "llm_result": {
                "message_type": "question",
                "risk_level": "low",
                "route": "bot_answer_self_for_pilot",
                "topic_id": "theme:006_installment",
                "safety_flags": ["manager_approval_required", "no_auto_send"],
                "missing_facts": [],
                "context_used": ["confirmed_facts"],
            },
        },
    )
    store.record_feedback(result.draft_id, "manager_marked_useful", actor="nastya", reason="ok", occurred_at=START)

    out_dir = tmp_path / "report"
    report = build_pilot_daily_report(store, "2026-05-23", out_dir=out_dir)

    expected = {
        "pilot_messages.csv",
        "pilot_messages.jsonl",
        "pilot_summary.json",
        "pilot_summary.md",
        "semantic_review_queue.csv",
        "regression_candidates.csv",
        "employee_review_sheet.csv",
        "p0_incidents.csv",
        "known_data_reask_cases.csv",
        "template_or_generic_cases.csv",
        "facts_used_summary.csv",
        "implementation_notes.md",
        "semantic_review.md",
    }
    assert expected.issubset(set(report["files"]))
    csv_text = (out_dir / "pilot_messages.csv").read_text(encoding="utf-8")
    assert "79092009933" not in csv_text
    assert "[PHONE_MASKED]" in csv_text
    assert "gpt-5.5" in csv_text
    assert "kb-v6.3" in csv_text

    with (out_dir / "pilot_messages.csv").open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert rows[0]["draft_id"] == result.draft_id
    assert rows[0]["sent_to_client"] == "true"
    assert rows[0]["asked_known_data_again"] == "false"

    summary = json.loads((out_dir / "pilot_summary.json").read_text(encoding="utf-8"))
    assert summary["messages_total"] == 1
    assert summary["autonomous_answers"] == 1
    assert summary["formal_passed"] is True
    store.close()


def test_store_persists_dialogue_memory_snapshot(tmp_path: Path) -> None:
    store = TelegramPilotSQLiteStore(tmp_path / "telegram_pilot.sqlite", clock=StepClock())
    result = store.upsert_message_context_draft(
        inbound_message("msg-memory", "8 класс, физика онлайн. Это цена сейчас?"),
        context={"active_brand": "foton"},
        draft_text="Да, это текущая цена.",
        prompt_version="telegram_public_pilot:gpt-5.5:high",
        knowledge_base_version="kb-v6.3",
        topic_id="theme:001_pricing",
        route="draft_for_manager",
    )

    write = store.upsert_dialogue_memory_snapshot(
        message_key=result.message_key,
        session_id="telegram_public_pilot:foton:1",
        active_brand="foton",
        memory_snapshot={
            "schema_version": "dialogue_memory_v1_2026_05_23",
            "session_id": "telegram_public_pilot:foton:1",
            "active_brand": "foton",
            "known_slots": {"grade": "8", "subject": "физика", "format": "онлайн"},
        },
        created_at=START,
    )

    assert write.created is True
    snapshots = store.list_dialogue_memory_snapshots(day="2026-05-23")
    assert snapshots[0]["memory_snapshot"]["known_slots"]["grade"] == "8"
    assert snapshots[0]["active_brand"] == "foton"
    store.close()


def test_daily_journal_report_routes_known_data_reask_to_queue(tmp_path: Path) -> None:
    store = TelegramPilotSQLiteStore(tmp_path / "telegram_pilot.sqlite", clock=StepClock())
    store.upsert_message_context_draft(
        inbound_message("msg-2", "Меня зовут Анна, ребёнок Даниил, 9 класс."),
        context={
            "active_brand": "unpk",
            "known_client_fields": {"student_name": "Даниил", "phone": "79092009933"},
            "known_dialog_fields": {"grade": "9"},
        },
        draft_text="Напишите, пожалуйста, ФИО ребёнка, телефон и какой класс у ребёнка.",
        prompt_version="telegram_public_pilot:gpt-5.5:high",
        knowledge_base_version="kb-v6.3",
        topic_id="theme:016_program",
        route="draft_for_manager",
    )

    out_dir = tmp_path / "report"
    build_pilot_daily_report(store, "2026-05-23", out_dir=out_dir)

    reask = (out_dir / "known_data_reask_cases.csv").read_text(encoding="utf-8")
    assert "asked_known_data_again" in reask
    assert "student_name" in reask
    assert "phone" in reask
    store.close()
