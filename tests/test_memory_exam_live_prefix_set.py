from __future__ import annotations

from pathlib import Path

from scripts import build_memory_exam_live_prefix_set as builder


def test_live_prefix_turns_use_next_manager_reply_and_skip_missing_target() -> None:
    messages = [
        _message("09:00", "inbound", "Первый вопрос"),
        _message("09:01", "outbound", "Первый ответ менеджера"),
        _message("09:02", "inbound", "Второй вопрос"),
        _message("09:03", "outbound", "Второй ответ менеджера"),
        _message("09:04", "inbound", "Третий вопрос без ответа"),
    ]

    turns = builder.build_prefix_turns(messages, max_turns=5)

    assert [turn["client_message"] for turn in turns] == ["Первый вопрос", "Второй вопрос"]
    assert [turn["reference_manager_reply"] for turn in turns] == [
        "Первый ответ менеджера",
        "Второй ответ менеджера",
    ]
    assert turns[-1]["client_stop"] is True


def test_live_prefix_builder_scrubs_pii_from_replay_text() -> None:
    messages = [
        _message("09:00", "inbound", "Телефон +7 999 123-45-67 и test@example.com"),
        _message("09:01", "outbound", "Ответ на customer:abcdef1234567890"),
    ]

    turns = builder.build_prefix_turns(messages, max_turns=1)

    assert "[телефон скрыт]" in turns[0]["client_message"]
    assert "[email скрыт]" in turns[0]["client_message"]
    assert "[служебный id скрыт]" in turns[0]["reference_manager_reply"]


def test_generated_commands_use_requested_db_limit_and_quote_paths() -> None:
    command = builder.render_micro_commands(
        Path("local data/scenarios.jsonl"),
        Path("local data/replay.jsonl"),
        Path("local data/out"),
        timeline_db=Path("/tmp/prod snapshot/customer_timeline.sqlite"),
        limit=3,
    )

    assert "--limit 3" in command
    assert "TELEGRAM_BOT_SAFE_CRM_CONTEXT_DB='/tmp/prod snapshot/customer_timeline.sqlite'" in command
    assert "--scenarios 'local data/scenarios.jsonl'" in command
    assert "--out-dir 'local data/out/micro_on'" in command


def _message(event_at: str, direction: str, text: str) -> builder.Message:
    return builder.Message(
        source_system="telegram_history",
        event_id=f"event-{event_at}-{direction}",
        source_id=f"source-{event_at}-{direction}",
        customer_id="customer:test",
        brand="unpk",
        dialog_key="dialog-1",
        event_at=f"2026-07-10T{event_at}:00+00:00",
        direction=direction,
        text=builder.scrub_text(text),
        allowed_context_items=1,
    )
