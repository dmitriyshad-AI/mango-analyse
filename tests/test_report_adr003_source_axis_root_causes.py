from __future__ import annotations

import json
from pathlib import Path

from scripts.report_adr003_source_axis_root_causes import build_report


def _write_json(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


def _fact_gap_case(dialog_id: str, **overrides: object) -> dict:
    data = {
        "dialog_id": dialog_id,
        "turn": 1,
        "route": "draft_for_manager",
        "requested_action": "answer_question",
        "next_autonomy_workstream": "fix_proof_axis_alignment",
        "root_cause": "proof_axis_mismatch",
        "missing_fact_categories": ["boarding_food", "class_grade"],
        "client_excerpt": "не экспортировать",
        "bot_excerpt": "тоже не экспортировать",
        "kb_support": {
            "missing_slots": ["grade"],
            "proven_parts": ["product_existence"],
            "product_check_status": "exists",
            "product_check_reason": "exact_product_existence_fact",
            "product_check_exact_fact_keys": ["fact.one"],
            "platform_fact_count": 0,
            "price_fact_count": 0,
        },
    }
    data.update(overrides)
    return data


def _source_case(dialog_id: str, **overrides: object) -> dict:
    data = {
        "dialog_id": dialog_id,
        "turn": 1,
        "policy_status": "blocked_source_axis_mismatch",
        "next_autonomy_workstream": "fix_proof_axis_alignment",
    }
    data.update(overrides)
    return data


def test_classifies_missing_required_slot_without_text(tmp_path: Path) -> None:
    fact_gap = _write_json(tmp_path / "gap.json", {"cases": [_fact_gap_case("missing_slot")]})
    source = _write_json(tmp_path / "source.json", {"cases": [_source_case("missing_slot")]})

    report = build_report(fact_gap_report=fact_gap, source_axis_report=source)

    assert report["totals"]["missing_required_slot"] == 1
    assert report["totals"]["route_only_active_candidates"] == 0
    assert report["cases"][0]["root_cause"] == "missing_required_slot_partial_policy_needed"
    payload = json.dumps(report, ensure_ascii=False)
    assert "не экспортировать" not in payload
    assert "client_excerpt" not in payload
    assert "bot_excerpt" not in payload


def test_classifies_platform_axis_taxonomy_gap_under_manager_only(tmp_path: Path) -> None:
    fact_gap = _write_json(
        tmp_path / "gap.json",
        {
            "cases": [
                _fact_gap_case(
                    "platform",
                    route="manager_only",
                    missing_fact_categories=["platform_current"],
                    kb_support={
                        "missing_slots": [],
                        "proven_parts": ["platform_current"],
                        "product_check_status": "needs_slot",
                        "platform_fact_count": 3,
                        "price_fact_count": 0,
                    },
                )
            ]
        },
    )
    source = _write_json(
        tmp_path / "source.json",
        {"cases": [_source_case("platform", policy_status="blocked_manager_only_route")]},
    )

    report = build_report(fact_gap_report=fact_gap, source_axis_report=source)

    assert report["totals"]["platform_axis_taxonomy_gap"] == 1
    assert report["totals"]["manager_only_policy"] == 1
    assert report["cases"][0]["root_cause"] == "manager_only_with_platform_axis_taxonomy_gap"
    assert report["cases"][0]["active_behavior_allowed"] is False


def test_classifies_danger_first(tmp_path: Path) -> None:
    fact_gap = _write_json(
        tmp_path / "gap.json",
        {"cases": [_fact_gap_case("danger", next_autonomy_workstream="danger_adjacent_do_not_lower")]},
    )
    source = _write_json(tmp_path / "source.json", {"cases": [_source_case("danger")]})

    report = build_report(fact_gap_report=fact_gap, source_axis_report=source)

    assert report["totals"]["danger_adjacent"] == 1
    assert report["cases"][0]["root_cause"] == "danger_adjacent_do_not_lower"
