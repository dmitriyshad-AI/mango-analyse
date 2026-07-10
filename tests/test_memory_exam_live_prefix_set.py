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
    assert "TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1" in command
    assert "TELEGRAM_BOT_SAFE_MEMORY_STEP_GUARD=1" in command
    assert "--semantic-verifier-mode codex" in command
    assert "--bot-max-attempts 1" in command
    assert "--disable-bot-cache" in command
    assert "--semantic-verifier-mode off" not in command


def test_select_dialogs_honors_brand_targets_before_source_targets() -> None:
    candidates = [
        {
            "dialog_id": f"dialog-{brand}-{index}",
            "brand": brand,
            "source_system": "telegram_history",
            "allowed_context_items": 3,
            "start_at": f"2026-07-10T00:0{index}:00+00:00",
        }
        for brand in ("unpk", "foton")
        for index in range(3)
    ]

    selected = builder.select_dialogs(
        candidates,
        limit=5,
        source_targets={"telegram_history": 5},
        brand_targets={"foton": 2, "unpk": 3},
    )

    assert sum(row["brand"] == "foton" for row in selected) == 2
    assert sum(row["brand"] == "unpk" for row in selected) == 3


def test_runtime_visible_context_explicitly_allows_builder_customer_id(monkeypatch, tmp_path: Path) -> None:
    db = tmp_path / "timeline.sqlite"
    db.write_bytes(b"placeholder")
    captured = {}

    def fake_context(**kwargs):
        captured.update(kwargs)
        return {"timeline_context": {"bot_context": {"items": [{"chunk_type": "mango_call_summary"}]}}}

    monkeypatch.setattr(builder, "build_bot_safe_crm_context", fake_context)

    count = builder.runtime_visible_context_items(db, customer_id="customer:test", brand="foton")

    assert count == 1
    assert captured["allow_explicit_customer_id"] is True


def test_require_memory_rejects_dialog_without_runtime_visible_call(monkeypatch, tmp_path: Path) -> None:
    messages = [
        _message("09:00", "inbound", "Первый вопрос"),
        _message("09:01", "outbound", "Первый ответ"),
        _message("09:02", "inbound", "Второй вопрос"),
        _message("09:03", "outbound", "Второй ответ"),
        _message("09:04", "inbound", "Третий вопрос"),
        _message("09:05", "outbound", "Третий ответ"),
    ]
    monkeypatch.setattr(
        builder,
        "runtime_visible_context",
        lambda *_args, **_kwargs: ({"chunk_type": "email_message"},),
    )

    candidates = builder.build_dialog_candidates(
        builder.group_messages(messages),
        timeline_db=tmp_path / "timeline.sqlite",
        min_inbound=3,
        max_turns_per_dialog=4,
        require_memory=True,
    )

    assert candidates == []


def test_report_counts_runtime_visible_call_memory(tmp_path: Path) -> None:
    selected = [
        {
            "source_system": "telegram_history",
            "allowed_context_items": 2,
            "allowed_call_context_items": 1,
            "visible_context_by_type": {"mango_call_summary": 1, "email_message": 1},
            "stored_allowed_context_items": 3,
            "turns": [{"client_message": "Вопрос", "reference_manager_reply": "Ответ"}],
        }
    ]

    report = builder.build_report(
        [],
        selected,
        selected,
        db_path=tmp_path / "timeline.sqlite",
        scenario_path=tmp_path / "scenarios.jsonl",
        replay_path=tmp_path / "replay.jsonl",
    )

    assert report["selected_with_call_memory"] == 1
    assert report["selected_without_call_memory"] == 0
    assert report["selected_visible_context_by_type"]["mango_call_summary"] == 1


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
