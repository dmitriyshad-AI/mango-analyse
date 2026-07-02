from __future__ import annotations

import json
from pathlib import Path

from scripts import report_adr003_action_gate_counterfactual_proof as report


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _fact_gap(tmp_path: Path, cases: list[dict]) -> Path:
    path = tmp_path / "fact_gap.json"
    _write_json(path, {"cases": cases})
    return path


def _transcripts(tmp_path: Path, dialogs: list[dict]) -> Path:
    path = tmp_path / "transcripts.jsonl"
    _write_jsonl(path, dialogs)
    return path


def _kb(tmp_path: Path, facts: list[dict]) -> Path:
    path = tmp_path / "kb.json"
    _write_json(path, {"facts": facts})
    return path


def _gap_case(
    dialog_id: str,
    *,
    route: str = "manager_only",
    root_cause: str = "frame_action_blocks_existence_proof",
    missing_facts: list[str] | None = None,
) -> dict:
    return {
        "dialog_id": dialog_id,
        "missing_facts": missing_facts or ["наличие программ для ребёнка после 5-го класса"],
        "root_cause": root_cause,
        "route": route,
        "turn": 1,
    }


def _dialog(
    dialog_id: str,
    *,
    route: str = "manager_only",
    action: str = "check_availability",
    risk_class: str = "manager_action",
    answerability: str = "manager_only",
    must_handoff: bool = True,
    missing_facts: list[str] | None = None,
) -> dict:
    return {
        "brand": "unpk",
        "dialog_id": dialog_id,
        "turns": [
            {
                "bot_missing_facts": missing_facts or ["наличие программ для ребёнка после 5-го класса"],
                "bot_route": route,
                "bot_semantic_frame": {
                    "answerability": answerability,
                    "confidence": 0.91,
                    "deal_stage": "research",
                    "must_handoff": must_handoff,
                    "payment_readiness": "none",
                    "requested_action": action,
                    "requested_product": {
                        "brand": "unpk",
                        "format": "",
                        "grade": "5 класс",
                        "program_kind": "летняя школа",
                        "raw_text": "подходит после 5 класса",
                        "subject": "",
                    },
                    "risk_class": risk_class,
                },
                "bot_semantic_frame_existence_proof_shadow": {
                    "exact_fact_keys": [],
                    "reason": "requested_action_not_answer_question",
                    "status": "blocked",
                },
                "client_message": "Подойдёт после 5 класса?",
                "turn": 1,
            }
        ],
    }


def _fact(*, fact_key: str = "lvsh.classes") -> dict:
    return {
        "allowed_for_client_answer": True,
        "brand": "unpk",
        "client_safe_text": "УНПК: ЛВШ рассчитана на 5-10 классы.",
        "fact_id": f"fact:unpk:{fact_key}",
        "fact_key": fact_key,
        "fact_type": "course_parameter",
        "forbidden_for_client": False,
        "internal_only": False,
        "product": "camp",
        "program_kind": "летняя школа",
        "structured_value": {"classes_raw": "5-10", "valid_until": "2026-12-31"},
        "valid_until": "2026-12-31",
    }


def _build(tmp_path: Path, *, cases: list[dict], dialogs: list[dict], facts: list[dict]) -> dict:
    return report.build_report(
        fact_gap_report=_fact_gap(tmp_path, cases),
        transcripts=_transcripts(tmp_path, dialogs),
        kb_snapshot=_kb(tmp_path, facts),
    )


def test_action_only_stays_blocked_but_safe_reference_gets_exact_proof(tmp_path: Path) -> None:
    result = _build(
        tmp_path,
        cases=[_gap_case("age")],
        dialogs=[_dialog("age")],
        facts=[_fact()],
    )

    case = result["cases"][0]
    assert case["action_only_counterfactual"]["status"] == "blocked"
    assert case["action_only_counterfactual"]["reason"] == "protected_handoff_frame"
    assert case["safe_reference_counterfactual"]["status"] == "exists"
    assert case["counterfactual_status"] == "safe_reference_exact_proof_report_only"
    assert case["active_behavior_allowed"] is False
    assert result["totals"]["new_active_candidates"] == 0


def test_residual_live_availability_keeps_safe_reference_no_go(tmp_path: Path) -> None:
    result = _build(
        tmp_path,
        cases=[_gap_case("live", missing_facts=["актуальное наличие мест", "наличие программ для 5 класса"])],
        dialogs=[_dialog("live", missing_facts=["актуальное наличие мест", "наличие программ для 5 класса"])],
        facts=[_fact()],
    )

    case = result["cases"][0]
    assert case["safe_reference_counterfactual"]["status"] == "exists"
    assert case["counterfactual_status"] == "safe_reference_exact_proof_but_residual_hard_missing"
    assert "live_availability" in case["residual_missing_categories_after_existence"]
    assert "residual_hard_missing_axes" in case["why_not_active"]


def test_action_only_exact_proof_with_residual_axes_is_not_success(tmp_path: Path) -> None:
    result = _build(
        tmp_path,
        cases=[
            _gap_case(
                "axis",
                route="draft_for_manager",
                root_cause="proof_axis_mismatch",
                missing_facts=["подтвержденные даты, локация и наличие питания для 5 класса"],
            )
        ],
        dialogs=[
            _dialog(
                "axis",
                route="draft_for_manager",
                action="answer_question",
                risk_class="missing_facts",
                answerability="manager_only",
                missing_facts=["подтвержденные даты, локация и наличие питания для 5 класса"],
            )
        ],
        facts=[_fact()],
    )

    case = result["cases"][0]
    assert case["action_only_counterfactual"]["status"] == "exists"
    assert case["counterfactual_status"] == "action_only_exact_proof_but_residual_hard_missing"
    assert "dates_schedule" in case["residual_missing_categories_after_existence"]
    assert "location_address" in case["residual_missing_categories_after_existence"]


def test_danger_adjacent_case_is_negative_control(tmp_path: Path) -> None:
    result = _build(
        tmp_path,
        cases=[_gap_case("danger", root_cause="danger_adjacent_do_not_lower")],
        dialogs=[_dialog("danger", risk_class="p0", action="handoff_manager")],
        facts=[_fact()],
    )

    case = result["cases"][0]
    assert case["counterfactual_status"] == "negative_control_preserved"
    assert case["active_behavior_allowed"] is False
    assert result["totals"]["negative_controls_preserved_total"] == 1


def test_no_exact_fact_remains_no_go(tmp_path: Path) -> None:
    result = _build(
        tmp_path,
        cases=[_gap_case("missing")],
        dialogs=[_dialog("missing")],
        facts=[],
    )

    case = result["cases"][0]
    assert case["safe_reference_counterfactual"]["status"] == "blocked"
    assert case["counterfactual_status"] == "safe_reference_no_exact_proof"
