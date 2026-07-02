from __future__ import annotations

import json
from pathlib import Path

from scripts import report_adr003_manager_only_exact_proof_root_cause as report


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _gold(dialog_id: str) -> dict:
    return {
        "dialog_id": dialog_id,
        "turn": 1,
        "expected": {
            "must_handoff": False,
            "risk_class": "safe",
            "answerability": "answer_self",
            "requested_action": "answer_question",
        },
        "notes": "safe reference: course existence/format without live seats",
    }


def _fact() -> dict:
    return {
        "brand": "unpk",
        "fact_key": "lvsh_mendeleevo_2026.directions.fizmat.classes",
        "fact_type": "course_parameter",
        "product": "lvsh_mendeleevo_2026",
        "program_kind": "camp_lvsh",
        "client_safe_text": "УНПК: ЛВШ Менделеево для физико-математического направления рассчитана на 5-10 классы.",
        "allowed_for_client_answer": True,
        "forbidden_for_client": False,
        "internal_only": False,
        "valid_until": "2026-08-31",
        "structured_value": {"classes_raw": "5-10", "raw_value": "5-10", "valid_until": "2026-08-31"},
    }


def _turn(
    *,
    route: str = "manager_only",
    frame: dict | None = None,
    retrieval: dict | None = None,
    plan: dict | None = None,
    contract: dict | None = None,
) -> dict:
    semantic_frame = {
        "risk_class": "safe",
        "answerability": "answer_self",
        "requested_action": "answer_question",
        "must_handoff": False,
        "confidence": 0.86,
        "requested_product": {
            "brand": "unpk",
            "subject": "",
            "grade": "5",
            "format": "",
            "program_kind": "летняя школа",
            "raw_text": "ребёнок закончил 5 класс",
        },
    }
    semantic_frame.update(frame or {})
    return {
        "turn": 1,
        "client_message": "Ребёнок закончил 5 класс",
        "bot_text": "Передам менеджеру.",
        "bot_route": route,
        "bot_message_type": "context_update",
        "bot_reason_class": "policy_permission",
        "bot_safety_flags": ["manager_approval_required", "no_auto_send", "message_type_context_update"],
        "bot_missing_facts": ["актуальное наличие мест"],
        "bot_semantic_frame": semantic_frame,
        "bot_semantic_frame_self_answer_shadow": {
            "reason": "route_not_draft_for_manager",
            "guards": {"freshness": {"reason": "no_exact_fact_keys"}},
        },
        "bot_fact_retrieval_trace": retrieval
        if retrieval is not None
        else {"candidate_count": 0, "selected_exact_ids": [], "selected_adjacent_ids": [], "mode": "off"},
        "bot_conversation_intent_plan": plan
        if plan is not None
        else {
            "primary_intent": "general_consultation",
            "topic_id": "service:S5_general_consultation",
            "route_bias": "draft_for_manager",
            "product_scope": "",
            "fact_scope": "",
            "direct_question": "",
            "required_fact_keys": [],
            "known_slots": {"grade": "5"},
        },
        "bot_answer_contract": contract
        if contract is not None
        else {
            "route": "draft_for_manager",
            "route_bias": "draft_for_manager",
            "route_reason": "help_then_one_question",
            "required_fact_keys": [],
        },
    }


def _build(tmp_path: Path, *, turn: dict) -> dict:
    transcripts = tmp_path / "transcripts.jsonl"
    gold = tmp_path / "gold.jsonl"
    kb = tmp_path / "kb.json"
    _write_jsonl(transcripts, [{"dialog_id": "d1", "brand": "unpk", "turns": [turn]}])
    _write_jsonl(gold, [_gold("d1")])
    _write_json(kb, {"facts": [_fact()]})
    return report.build_report(transcripts=transcripts, gold=gold, kb_snapshot=kb)


def test_manager_only_exact_proof_reports_runtime_retrieval_gap(tmp_path: Path) -> None:
    result = _build(tmp_path, turn=_turn())

    assert result["totals"]["manager_only_exact_proof_rows"] == 1
    assert result["totals"]["runtime_exact_proof_missing"] == 1
    case = result["cases"][0]
    assert case["source_fact_key"] == "lvsh_mendeleevo_2026.directions.fizmat.classes"
    assert case["runtime_candidate_count"] == 0
    assert "runtime_retrieval_missed_exact_fact" in case["root_cause_codes"]
    assert "runtime_retrieval_zero_candidates" in case["root_cause_codes"]
    assert "conversation_plan_no_product_scope" in case["root_cause_codes"]
    assert "answer_contract_no_required_fact_keys" in case["root_cause_codes"]
    assert "route_locked_manager_only" in case["root_cause_codes"]
    assert result["acceptance"]["active_readiness"] == "no_go"


def test_frame_manager_action_is_separate_root_cause(tmp_path: Path) -> None:
    result = _build(
        tmp_path,
        turn=_turn(
            frame={
                "risk_class": "manager_action",
                "answerability": "manager_only",
                "requested_action": "check_availability",
                "must_handoff": True,
                "confidence": 0.93,
            }
        ),
    )

    case = result["cases"][0]
    assert "frame_marks_manager_action" in case["root_cause_codes"]
    assert "frame_action_not_safe_reference" in case["root_cause_codes"]
    assert "frame_confidence_below_threshold" not in case["root_cause_codes"]


def test_runtime_exact_key_present_removes_retrieval_gap(tmp_path: Path) -> None:
    result = _build(
        tmp_path,
        turn=_turn(
            retrieval={
                "candidate_count": 1,
                "selected_exact_ids": ["lvsh_mendeleevo_2026.directions.fizmat.classes"],
                "selected_adjacent_ids": [],
                "mode": "wide",
            },
            plan={"product_scope": "camp", "direct_question": "есть ли ЛВШ для 5 класса", "required_fact_keys": ["lvsh_mendeleevo_2026.directions.fizmat.classes"]},
            contract={"route": "draft_for_manager", "route_bias": "draft_for_manager", "route_reason": "help_then_one_question", "required_fact_keys": ["lvsh_mendeleevo_2026.directions.fizmat.classes"]},
        ),
    )

    case = result["cases"][0]
    assert "runtime_retrieval_missed_exact_fact" not in case["root_cause_codes"]
    assert "runtime_retrieval_zero_candidates" not in case["root_cause_codes"]
    assert "conversation_plan_no_product_scope" not in case["root_cause_codes"]
    assert "answer_contract_no_required_fact_keys" not in case["root_cause_codes"]


def test_markdown_is_redacted_from_client_text(tmp_path: Path) -> None:
    result = _build(tmp_path, turn=_turn())
    rendered = report.render_markdown(result)

    assert "Ребёнок закончил 5 класс" not in rendered
    assert "Передам менеджеру" not in rendered
    assert "lvsh_mendeleevo_2026.directions.fizmat.classes" in rendered
