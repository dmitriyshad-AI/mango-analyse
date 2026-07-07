from __future__ import annotations

import json
from pathlib import Path

from mango_mvp.replay_exam.judge import build_balanced_replay_judge_payloads, build_replay_judge_payload, serialize_judge_payload
from mango_mvp.replay_exam.judge_executor import (
    build_replay_judge_requests,
    execute_replay_judge_requests,
    write_replay_judge_payloads,
)
from mango_mvp.replay_exam.models import BotReplayResult, ReplayCase, ReplayMessage


def test_replay_judge_payload_keeps_hidden_key_outside_payload() -> None:
    case = ReplayCase(
        dialog_id="d",
        profile_id="p",
        chat_id="c",
        turn_id="d#1",
        brand="foton",
        client_message="Нужна физика",
        manager_reference="Добрый день",
        turn_index=1,
        contour="foton",
        dialog_key_masked="dialog-mask",
        prefix_messages=(ReplayMessage("p", "c", "m0", "Здравствуйте", 1, False, ts_masked="masked_000000s"),),
    )
    payload = build_replay_judge_payload(
        case,
        BotReplayResult(route="draft_for_manager", bot_text="B"),
        BotReplayResult(route="draft_for_manager", bot_text="ON"),
        seed="s",
    )
    serialized = serialize_judge_payload(payload)
    assert payload.hidden_key["answer_a"] in {"baseline", "candidate"}
    assert "hidden_key" not in serialized
    assert "manager_reference" not in serialized
    assert "manager_reference" not in json.dumps(payload.payload.get("answer_a"), ensure_ascii=False)
    assert "manager_reference" not in json.dumps(payload.payload.get("answer_b"), ensure_ascii=False)
    assert "route" not in json.dumps(payload.payload.get("answer_a"), ensure_ascii=False)
    assert "route" not in json.dumps(payload.payload.get("answer_b"), ensure_ascii=False)
    assert "SECRET" not in serialized
    assert "prefix_messages" in serialized
    assert "client_safe_facts_digest" in serialized
    assert "replay_judge_v1" in serialized


def test_replay_judge_balances_ab_exactly_for_even_set() -> None:
    rows = []
    for index in range(4):
        case = ReplayCase("d", "p", "c", f"d#{index}", "foton", "Вопрос", "Ответ")
        rows.append(
            (
                case,
                BotReplayResult(route="draft_for_manager", bot_text="B"),
                BotReplayResult(route="draft_for_manager", bot_text="ON"),
            )
        )

    payloads = build_balanced_replay_judge_payloads(rows, seed="s")

    assert [payload.hidden_key["answer_a"] for payload in payloads].count("baseline") == 2
    assert [payload.hidden_key["answer_a"] for payload in payloads].count("candidate") == 2


def test_replay_judge_executor_filters_clean_chat_only_and_caps_calls(tmp_path: Path) -> None:
    cases = [
        ReplayCase("d1", "p", "c", "exam-2", "foton", "Вопрос 2", "Менеджер 2", segment="chat_only"),
        ReplayCase("d1", "p", "c", "exam-1", "foton", "Вопрос 1", "Менеджер 1", segment="chat_only"),
        ReplayCase("d2", "p", "c", "exam-3", "foton", "Вопрос 3", "Менеджер 3", segment="external_context"),
    ]
    rows = [
        {
            "turn_id": "exam-2",
            "route": "bot_answer_self_for_pilot",
            "bot_text": "Бот 2",
            "safety_flags": [],
            "machine_gate": {"passed": True},
        },
        {
            "turn_id": "exam-1",
            "route": "bot_answer_self_for_pilot",
            "bot_text": "Бот 1",
            "safety_flags": [],
            "machine_gate": {"passed": True},
        },
        {
            "turn_id": "exam-3",
            "route": "bot_answer_self_for_pilot",
            "bot_text": "Бот 3",
            "safety_flags": [],
            "machine_gate": {"passed": True},
        },
    ]

    requests = build_replay_judge_requests(cases, rows, max_judge_calls=1)
    write_replay_judge_payloads(tmp_path, requests)

    payload_lines = (tmp_path / "judge_payloads.jsonl").read_text(encoding="utf-8").splitlines()
    key_lines = (tmp_path / "judge_key.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(payload_lines) == 1
    assert json.loads(payload_lines[0])["exam_id"] == "exam-1"
    assert "hidden_key" not in payload_lines[0]
    assert json.loads(key_lines[0])["hidden_key"]["answer_a"] in {"baseline", "candidate"}


def test_replay_judge_executor_writes_results_without_hidden_key(tmp_path: Path) -> None:
    case = ReplayCase("d", "p", "c", "exam-1", "foton", "Вопрос", "Менеджер", segment="chat_only")
    row = {
        "turn_id": "exam-1",
        "route": "draft_for_manager",
        "bot_text": "Бот",
        "safety_flags": [],
        "machine_gate": {"passed": True},
    }
    requests = build_replay_judge_requests([case], [row], max_judge_calls=1)

    execute_replay_judge_requests(
        tmp_path,
        requests,
        runner=lambda payload: {"winner": "answer_a", "bot_send_as_is": False, "flags": ["needs_edit"], "reason": "короче"},
    )

    result_text = (tmp_path / "judge_results.jsonl").read_text(encoding="utf-8")
    summary = json.loads((tmp_path / "judge_summary.json").read_text(encoding="utf-8"))
    assert "hidden_key" not in result_text
    assert json.loads(result_text)["result"]["flags"] == ["needs_edit"]
    assert summary["calls"] == 1
