from __future__ import annotations

import json
from pathlib import Path

from scripts.report_adr003_partial_answer_policy_shadow import build_report


def _write_json(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


def _partial_case(dialog_id: str, *, turn: int = 1) -> dict:
    return {
        "dialog_id": dialog_id,
        "turn": turn,
        "client_excerpt": "секретный клиентский текст не должен попасть в policy shadow",
        "route": "draft_for_manager",
        "requested_action": "answer_question",
        "risk_class": "missing_facts",
        "answerability": "manager_only",
        "must_handoff": True,
        "partial_answer_shadow": {"status": "draft_partial_shadow_candidate"},
        "kb_support": {"proven_parts": ["product_existence"], "missing_slots": [], "uncovered_categories": []},
    }


def _queue_example(dialog_id: str, *, turn: int = 1, **overrides: object) -> dict:
    data = {
        "dialog_id": dialog_id,
        "turn": turn,
        "route": "draft_for_manager",
        "requested_action": "answer_question",
        "danger_adjacent": False,
        "next_autonomy_workstream": "fix_proof_axis_alignment",
        "source_alignment_status": "aligned_covers_missing_fact_axis",
        "text_readiness_status": "",
        "shadow_text_renderer_status": "",
        "direct_quote_forbidden": False,
        "template_registry_status": "not_required",
    }
    data.update(overrides)
    return data


def test_blocks_danger_adjacent_candidate(tmp_path: Path) -> None:
    partial = _write_json(tmp_path / "partial.json", {"partial_cases": [_partial_case("p0_like")]})
    queue = _write_json(
        tmp_path / "queue.json",
        {
            "real_lever_analysis": {
                "examples": [
                    _queue_example("p0_like", danger_adjacent=True, next_autonomy_workstream="danger_adjacent_do_not_lower")
                ]
            }
        },
    )

    report = build_report(partial_report=partial, queue_report=queue)

    assert report["totals"]["partial_draft_candidates_input"] == 1
    assert report["totals"]["policy_shadow_candidates"] == 0
    assert report["totals"]["blocked_danger_adjacent"] == 1
    assert report["cases"][0]["policy_status"] == "blocked_danger_adjacent"
    assert report["cases"][0]["active_behavior_allowed"] is False


def test_blocks_source_axis_mismatch_candidate(tmp_path: Path) -> None:
    partial = _write_json(tmp_path / "partial.json", {"partial_cases": [_partial_case("axis_mismatch")]})
    queue = _write_json(
        tmp_path / "queue.json",
        {
            "real_lever_analysis": {
                "examples": [
                    _queue_example(
                        "axis_mismatch",
                        source_alignment_status="blocked_source_axis_mismatch",
                        source_alignment_blockers=["source_axis_mismatch"],
                    )
                ]
            }
        },
    )

    report = build_report(partial_report=partial, queue_report=queue)

    assert report["totals"]["policy_shadow_candidates"] == 0
    assert report["totals"]["blocked_source_axis_mismatch"] == 1
    assert report["cases"][0]["policy_status"] == "blocked_source_axis_mismatch"


def test_reports_owner_review_candidate_without_exporting_text(tmp_path: Path) -> None:
    partial = _write_json(tmp_path / "partial.json", {"partial_cases": [_partial_case("clean")]})
    queue = _write_json(
        tmp_path / "queue.json",
        {"real_lever_analysis": {"examples": [_queue_example("clean")]}},
    )

    report = build_report(partial_report=partial, queue_report=queue)

    assert report["totals"]["policy_shadow_candidates"] == 1
    case = report["cases"][0]
    assert case["policy_status"] == "policy_shadow_candidate_requires_owner_review"
    assert case["active_behavior_allowed"] is False
    assert case["generated_text_exported"] is False
    assert case["customer_text_exported"] is False
    assert "client_excerpt" not in case
    assert "секретный клиентский текст" not in json.dumps(report, ensure_ascii=False)
