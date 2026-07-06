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
                    {
                        "profile_id": "p",
                        "chat_id": "c",
                        "message_id": "m0",
                        "text": "Ранний ход",
                        "timestamp": 1,
                        "from_me": False,
                    }
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
