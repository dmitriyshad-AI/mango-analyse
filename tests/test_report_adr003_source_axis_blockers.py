from __future__ import annotations

import json
from pathlib import Path

from scripts.report_adr003_source_axis_blockers import build_report


def _write_json(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


def _queue_report(*rows: dict) -> dict:
    return {"real_lever_analysis": {"current_handoff_queue": {"examples": list(rows)}}}


def _row(dialog_id: str, **overrides: object) -> dict:
    data = {
        "dialog_id": dialog_id,
        "turn": 1,
        "route": "draft_for_manager",
        "requested_action": "answer_question",
        "next_autonomy_workstream": "fix_proof_axis_alignment",
        "danger_adjacent": False,
        "proof_reconciliation_status": "would_reconcile_to_safe_reference",
        "proof_reconciliation_exact_fact_keys": ["fact.one"],
        "source_alignment_status": "blocked_source_axis_mismatch",
        "source_alignment_blockers": ["source_axis_mismatch"],
        "shadow_text_renderer_status": "blocked_source_axis_mismatch",
    }
    data.update(overrides)
    return data


def test_counts_source_axis_and_danger_without_text(tmp_path: Path) -> None:
    path = _write_json(
        tmp_path / "queue.json",
        _queue_report(
            _row("axis"),
            _row("danger", danger_adjacent=True, next_autonomy_workstream="danger_adjacent_do_not_lower"),
        ),
    )

    report = build_report(queue_report=path)

    assert report["totals"]["current_handoff_rows"] == 2
    assert report["totals"]["source_axis_blocked"] == 2
    assert report["totals"]["source_axis_policy_primary"] == 1
    assert report["totals"]["danger_adjacent"] == 1
    assert report["totals"]["route_only_review_candidates"] == 0
    assert report["acceptance"]["active_readiness"] == "no_go"
    assert "client_excerpt" not in json.dumps(report, ensure_ascii=False)
    assert "bot_excerpt" not in json.dumps(report, ensure_ascii=False)


def test_route_only_candidate_is_review_only(tmp_path: Path) -> None:
    path = _write_json(
        tmp_path / "queue.json",
        _queue_report(
            _row(
                "route_only",
                next_autonomy_workstream="route_only_ack_status_candidate_review",
                source_alignment_status="",
                shadow_text_renderer_status="",
            )
        ),
    )

    report = build_report(queue_report=path)
    case = report["cases"][0]

    assert report["totals"]["route_only_review_candidates"] == 1
    assert case["policy_status"] == "route_only_review_candidate"
    assert case["active_behavior_allowed"] is False
    assert case["client_text_exported"] is False
    assert case["bot_text_exported"] is False


def test_ignores_non_handoff_rows(tmp_path: Path) -> None:
    path = _write_json(
        tmp_path / "queue.json",
        _queue_report(_row("already_self", route="bot_answer_self_for_pilot")),
    )

    report = build_report(queue_report=path)

    assert report["totals"]["current_handoff_rows"] == 0
    assert report["cases"] == []
