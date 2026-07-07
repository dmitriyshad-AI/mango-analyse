from __future__ import annotations

import json
from pathlib import Path

from mango_mvp.replay_exam.models import BotReplayResult, ReplayCase
from mango_mvp.replay_exam.runner import load_cases, run_replay_exam, write_replay_outputs


def test_replay_runner_parallelizes_by_dialog_and_reports_zero_client_llm(tmp_path: Path) -> None:
    cases = [
        ReplayCase("d1", "p", "c1", "d1#1", "foton", "Привет", "Ответ"),
        ReplayCase("d1", "p", "c1", "d1#2", "foton", "Цена?", "Ответ"),
        ReplayCase("d2", "p", "c2", "d2#1", "unpk", "ЛВШ?", "Ответ"),
    ]

    def provider(case: ReplayCase, context: dict[str, object]) -> BotReplayResult:
        assert context["sends_client_replies"] is False
        assert context["crm_context"] == {}
        return BotReplayResult(route="draft_for_manager", bot_text=f"Ответ: {case.client_message}", safety_flags=("draft_only",))

    rows = run_replay_exam(cases, provider, parallel_dialogs=2)
    write_replay_outputs(tmp_path, rows)

    summary = json.loads((tmp_path / "replay_summary.json").read_text(encoding="utf-8"))
    assert summary["turns"] == 3
    assert summary["llm_calls"]["client"] == 0
    assert (tmp_path / "replay_results.jsonl").exists()


def test_replay_runner_threads_dialogue_memory_between_turns() -> None:
    cases = [
        ReplayCase("d", "p", "c", "d#1", "foton", "Нужна физика", "Ответ", turn_index=1),
        ReplayCase("d", "p", "c", "d#2", "foton", "А расписание?", "Ответ", turn_index=2),
    ]
    memory_turn_counts: list[int] = []

    def provider(case: ReplayCase, context: dict[str, object]) -> BotReplayResult:
        memory = context.get("dialogue_memory") if isinstance(context.get("dialogue_memory"), dict) else {}
        memory_turn_counts.append(len(memory.get("turns") or ()))
        return BotReplayResult(route="bot_answer_self_for_pilot", bot_text=f"Ответ: {case.client_message}")

    run_replay_exam(cases, provider, parallel_dialogs=1)

    assert memory_turn_counts[0] == 0
    assert memory_turn_counts[1] >= 1


def test_replay_runner_exports_memory_snapshots_for_full_tests() -> None:
    cases = [
        ReplayCase("d", "p", "c", "d#1", "foton", "Нужна физика", "Ответ", turn_index=1),
        ReplayCase("d", "p", "c", "d#2", "foton", "А расписание?", "Ответ", turn_index=2),
    ]

    def provider(case: ReplayCase, context: dict[str, object]) -> BotReplayResult:
        return BotReplayResult(route="bot_answer_self_for_pilot", bot_text=f"Ответ: {case.client_message}")

    rows = run_replay_exam(cases, provider, parallel_dialogs=1)

    assert rows[0]["memory_snapshot"]["known_slots"] == {}
    assert rows[0]["memory_snapshot_after"]["schema_version"] == "wappi_replay_memory_snapshot_v1"
    assert rows[1]["memory_snapshot"]["schema_version"] == "wappi_replay_memory_snapshot_v1"
    assert set(rows[1]["memory_snapshot"]) == {"schema_version", "known_slots", "do_not_reask", "p0_latch"}


def test_replay_runner_allows_numbers_from_current_turn_retrieved_facts() -> None:
    cases = [ReplayCase("d", "p", "c", "d#1", "foton", "Сколько стоит?", "")]

    def provider(case: ReplayCase, context: dict[str, object]) -> BotReplayResult:
        return BotReplayResult(
            route="bot_answer_self_for_pilot",
            bot_text="Стоимость курса — 12 345 ₽.",
            metadata={
                "dialogue_contract_pipeline": {
                    "retrieved_facts": {
                        "price.current": {"client_safe_text": "Стоимость курса — 12 345 ₽."}
                    }
                }
            },
        )

    rows = run_replay_exam(cases, provider, parallel_dialogs=1)

    assert rows[0]["machine_gate"]["passed"] is True
    assert rows[0]["machine_gate"]["new_numbers"] == []
    assert rows[0]["machine_gate"]["client_safe_numbers_count"] >= 1


def test_replay_runner_allows_numbers_from_retrieved_fact_string_values() -> None:
    cases = [ReplayCase("d", "p", "c", "d#1", "unpk", "Когда старт?", "")]

    def provider(case: ReplayCase, context: dict[str, object]) -> BotReplayResult:
        return BotReplayResult(
            route="bot_answer_self_for_pilot",
            bot_text="Старт группы — 13.09.2026.",
            metadata={
                "direct_path": {
                    "retrieved_facts": {
                        "schedule.group_start": "УНПК: старт группы — 13.09.2026."
                    }
                }
            },
        )

    rows = run_replay_exam(cases, provider, parallel_dialogs=1)

    assert rows[0]["machine_gate"]["passed"] is True
    assert rows[0]["machine_gate"]["new_numbers"] == []


def test_replay_runner_allows_time_range_components_from_retrieved_fact() -> None:
    cases = [ReplayCase("d", "p", "c", "d#1", "unpk", "Когда занятие?", "")]

    def provider(case: ReplayCase, context: dict[str, object]) -> BotReplayResult:
        return BotReplayResult(
            route="bot_answer_self_for_pilot",
            bot_text="Занятие идёт 14:30–16:30.",
            metadata={
                "direct_path": {
                    "retrieved_facts": {
                        "schedule.group_time": "УНПК: олимпиадная группа — суббота 14:30–16:30."
                    }
                }
            },
        )

    rows = run_replay_exam(cases, provider, parallel_dialogs=1)

    assert rows[0]["machine_gate"]["passed"] is True
    assert rows[0]["machine_gate"]["new_numbers"] == []


def test_replay_runner_does_not_allow_numbers_from_internal_only_fact_text() -> None:
    cases = [ReplayCase("d", "p", "c", "d#1", "unpk", "Сколько стоит?", "")]

    def provider(case: ReplayCase, context: dict[str, object]) -> BotReplayResult:
        return BotReplayResult(
            route="bot_answer_self_for_pilot",
            bot_text="Внутренняя цена — 12 345 ₽.",
            metadata={
                "direct_path": {
                    "retrieved_facts": {
                        "price.internal": {"internal_only_text": "Внутренняя цена — 12 345 ₽."}
                    }
                }
            },
        )

    rows = run_replay_exam(cases, provider, parallel_dialogs=1)

    assert rows[0]["machine_gate"]["passed"] is False
    assert rows[0]["machine_gate"]["new_numbers"] == ["12345"]


def test_replay_runner_does_not_allow_numbers_from_manager_reference_or_raw_response() -> None:
    cases = [ReplayCase("d", "p", "c", "d#1", "foton", "Сколько стоит?", "Менеджер: 12 345 ₽.")]

    def provider(case: ReplayCase, context: dict[str, object]) -> BotReplayResult:
        return BotReplayResult(
            route="bot_answer_self_for_pilot",
            bot_text="Стоимость курса — 12 345 ₽.",
            metadata={
                "replay_raw_response": "Стоимость курса — 12 345 ₽.",
                "manager_reference": "Менеджер: 12 345 ₽.",
                "dialogue_contract_pipeline": {"retrieved_facts": {}},
            },
        )

    rows = run_replay_exam(cases, provider, parallel_dialogs=1)

    assert rows[0]["machine_gate"]["passed"] is False
    assert "new_number_unverified" in rows[0]["machine_gate"]["flags"]
    assert rows[0]["machine_gate"]["new_numbers"] == ["12345"]
    assert rows[0]["machine_gate"]["client_safe_numbers_count"] == 0


def test_replay_runner_does_not_allow_numbers_from_previous_turn_retrieved_facts() -> None:
    cases = [ReplayCase("d", "p", "c", "d#2", "foton", "А сейчас сколько?", "")]

    def provider(case: ReplayCase, context: dict[str, object]) -> BotReplayResult:
        return BotReplayResult(
            route="bot_answer_self_for_pilot",
            bot_text="Стоимость сейчас — 12 345 ₽.",
            metadata={
                "previous_turn": {
                    "retrieved_facts": {
                        "old.price": {"client_safe_text": "Стоимость в прошлом ответе — 12 345 ₽."}
                    }
                },
                "direct_path": {"retrieved_facts": {}},
            },
        )

    rows = run_replay_exam(cases, provider, parallel_dialogs=1)

    assert rows[0]["machine_gate"]["passed"] is False
    assert rows[0]["machine_gate"]["new_numbers"] == ["12345"]
    assert rows[0]["machine_gate"]["client_safe_numbers_count"] == 0


def test_load_cases_reads_scrubbed_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(
        json.dumps(
            {
                "dialog_id": "d",
                "profile_id": "p",
                "chat_id": "c",
                "turn_id": "d#1",
                "brand": "foton",
                "client_message": "Вопрос",
                "manager_reference": "Ответ",
                "prefix_messages": [
                    {"from_me": False, "text": "Ранний ход", "ts_masked": "masked+000000s"}
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    case = load_cases(path)[0]
    assert case.turn_id == "d#1"
    assert case.prefix_messages[0].text == "Ранний ход"


def test_load_cases_reads_v4_whitelist_prefix_messages(tmp_path: Path) -> None:
    path = tmp_path / "cases_v4.jsonl"
    path.write_text(
        json.dumps(
            {
                "schema_version": "wappi_replay_case_v4",
                "exam_id": "exam-1",
                "contour": "foton-main",
                "dialog_key_masked": "dialog-mask",
                "turn_index": 2,
                "client_message": "Вопрос",
                "manager_reference": "Ответ",
                "prefix_messages": [
                    {"from_me": False, "text": "Первый клиент", "ts_masked": "masked+000000s"},
                    {"from_me": True, "text": "Первый ответ", "ts_masked": "masked+000030s"},
                ],
                "segment": "chat_only",
                "meta": {"ts_masked": "masked+000060s"},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    case = load_cases(path)[0]

    assert case.dialog_id == "dialog-mask"
    assert case.turn_id == "exam-1"
    assert case.turn_index == 2
    assert case.contour == "foton-main"
    assert case.brand == "foton"
    assert [message.from_me for message in case.prefix_messages] == [False, True]
    assert [message.text for message in case.prefix_messages] == ["Первый клиент", "Первый ответ"]
    assert [message.ts_masked for message in case.prefix_messages] == ["masked+000000s", "masked+000030s"]
    assert all(message.raw == {} for message in case.prefix_messages)


def test_load_cases_rejects_prefix_messages_outside_v4_whitelist(tmp_path: Path) -> None:
    path = tmp_path / "bad_cases.jsonl"
    path.write_text(
        json.dumps(
            {
                "schema_version": "wappi_replay_case_v4",
                "dialog_key_masked": "dialog-mask",
                "turn_index": 1,
                "brand": "foton",
                "client_message": "Вопрос",
                "manager_reference": "Ответ",
                "prefix_messages": [{"from_me": False, "text": "Ранний ход", "ts_masked": "masked+000000s", "raw": {"phone": "79001234567"}}],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    import pytest

    with pytest.raises(ValueError, match="exact keys"):
        load_cases(path)


def test_load_cases_rejects_missing_from_me_in_prefix_message(tmp_path: Path) -> None:
    path = tmp_path / "bad_cases.jsonl"
    path.write_text(
        json.dumps(
            {
                "schema_version": "wappi_replay_case_v4",
                "dialog_key_masked": "dialog-mask",
                "turn_index": 1,
                "brand": "foton",
                "client_message": "Вопрос",
                "manager_reference": "Ответ",
                "prefix_messages": [{"text": "Ранний ход", "ts_masked": "masked+000000s"}],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    import pytest

    with pytest.raises(ValueError, match="exact keys"):
        load_cases(path)


def test_replay_runner_sorts_cases_by_numeric_turn_index() -> None:
    calls: list[str] = []
    cases = [
        ReplayCase("d", "p", "c", "d#10", "foton", "Десятый", "Ответ", turn_index=10),
        ReplayCase("d", "p", "c", "d#2", "foton", "Второй", "Ответ", turn_index=2),
        ReplayCase("d", "p", "c", "d#1", "foton", "Первый", "Ответ", turn_index=1),
    ]

    def provider(case: ReplayCase, context: dict[str, object]) -> BotReplayResult:
        calls.append(case.turn_id)
        return BotReplayResult(route="draft_for_manager", bot_text=f"Ответ: {case.client_message}")

    rows = run_replay_exam(cases, provider, parallel_dialogs=1)

    assert calls == ["d#1", "d#2", "d#10"]
    assert [row["turn_id"] for row in rows] == ["d#1", "d#2", "d#10"]
