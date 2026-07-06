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
