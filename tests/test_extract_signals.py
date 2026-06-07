from __future__ import annotations

import json
from pathlib import Path

from scripts.extract_signals import compare_summaries, summarize_transcripts


def _write_transcripts(path: Path, dialogs: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in dialogs) + "\n", encoding="utf-8")


def test_extract_signals_counts_routes_handoff_with_facts_and_gates(tmp_path: Path) -> None:
    transcripts = tmp_path / "dynamic_dialog_transcripts.jsonl"
    _write_transcripts(
        transcripts,
        [
            {
                "dialog_id": "d1",
                "brand": "foton",
                "judge_result": {
                    "hard_gates_passed": False,
                    "violated_gates": ["brand_leak", "p0_not_to_manager"],
                },
                "turns": [
                    {
                        "turn": 1,
                        "client_message": "сколько стоит?",
                        "bot_text": "Менеджер уточнит.",
                        "bot_route": "draft_for_manager",
                        "bot_dialogue_contract_pipeline": {
                            "fallback_reason": "no_fact_or_unverified",
                            "retrieved_fact_keys": ["prices.current"],
                            "handoff_trace": {"layer": "guardchain", "fallback_reason": "no_fact_or_unverified"},
                        },
                        "tone_metric": {"tone_score": 31},
                    },
                    {
                        "turn": 2,
                        "client_message": "спасибо",
                        "bot_text": "Рада помочь.",
                        "bot_route": "bot_answer_self",
                        "tone_metric": {"tone_score": 82},
                    },
                ],
            }
        ],
    )

    summary = summarize_transcripts(transcripts)

    assert summary["dialogs"] == 1
    assert summary["turns"] == 2
    assert summary["routes"]["draft_for_manager"] == 1
    assert summary["fallback_reasons"]["no_fact_or_unverified"] == 1
    assert summary["handoff_trace_layers"]["guardchain"] == 1
    assert summary["handoff_with_facts"] == 1
    assert summary["tone_score_avg"] == 56.5
    assert summary["hard_gate_failures"] == 1
    assert summary["brand_gate_failures"] == 1
    assert summary["p0_gate_failures"] == 1


def test_extract_signals_compare_reports_deltas(tmp_path: Path) -> None:
    left = tmp_path / "left.jsonl"
    right = tmp_path / "right.jsonl"
    _write_transcripts(
        left,
        [
            {
                "dialog_id": "d1",
                "brand": "unpk",
                "judge_result": {"hard_gates_passed": True, "violated_gates": []},
                "turns": [
                    {
                        "turn": 1,
                        "bot_route": "draft_for_manager",
                        "bot_dialogue_contract_pipeline": {"retrieved_fact_keys": ["schedule.group"]},
                        "tone_score": 40,
                    }
                ],
            }
        ],
    )
    _write_transcripts(
        right,
        [
            {
                "dialog_id": "d1",
                "brand": "unpk",
                "judge_result": {"hard_gates_passed": True, "violated_gates": []},
                "turns": [{"turn": 1, "bot_route": "bot_answer_self", "tone_score": 70}],
            }
        ],
    )

    delta = compare_summaries(summarize_transcripts(left), summarize_transcripts(right))

    assert delta["handoff_with_facts_delta"] == -1
    assert delta["routes_delta"]["draft_for_manager"] == -1
    assert delta["routes_delta"]["bot_answer_self"] == 1
    assert delta["tone_score_avg_delta"] == 30
