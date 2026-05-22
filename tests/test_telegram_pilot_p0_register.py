from __future__ import annotations

import csv

import pytest

from mango_mvp.channels.telegram_pilot_p0_register import (
    append_p0_register_record,
    build_p0_register_record,
    p0_trigger,
)


def test_p0_register_detects_refund_topic_and_blocks_autonomous_attempt(tmp_path) -> None:
    record = build_p0_register_record(
        brand="unpk",
        chat_id=123,
        input_text="Как вернуть деньги?",
        answer_text="Передам вопрос ответственному сотруднику.",
        topic_id="theme:009_refund",
        route="bot_answer_self_for_pilot",
        safety_flags=("high_risk_manager_only",),
        client_send_executed=True,
    )

    assert record is not None
    assert record.severity == "P0_AUTONOMOUS_ROUTE_ATTEMPT"
    assert record.risk_code == "p0_refund_or_training_refusal"

    path = append_p0_register_record(tmp_path / "p0.csv", record)
    with path.open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert rows[0]["brand"] == "unpk"
    assert rows[0]["severity"] == "P0_AUTONOMOUS_ROUTE_ATTEMPT"


def test_p0_register_detects_legal_text_without_topic() -> None:
    assert p0_trigger(topic_id="theme:001_pricing", input_text="Пойду в прокуратуру", safety_flags=()) == "input_text"


def test_p0_register_path_rejects_stable_runtime(tmp_path) -> None:
    record = build_p0_register_record(
        brand="foton",
        chat_id=1,
        input_text="Хочу подать жалобу",
        answer_text="Передам ответственному сотруднику.",
        topic_id="theme:019b_negative_feedback",
        route="manager_only",
        client_send_executed=True,
    )
    assert record is not None
    with pytest.raises(ValueError, match="stable_runtime"):
        append_p0_register_record(tmp_path / "stable_runtime" / "p0.csv", record)
