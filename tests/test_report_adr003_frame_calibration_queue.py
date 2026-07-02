from __future__ import annotations

import json
from pathlib import Path

from scripts import report_adr003_frame_calibration_queue as report


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _gold(
    dialog_id: str,
    *,
    must_handoff: bool = False,
    requested_action: str = "answer_question",
    notes: str = "safe reference: course existence/format without live seats",
) -> dict:
    return {
        "dialog_id": dialog_id,
        "turn": 1,
        "expected": {
            "must_handoff": must_handoff,
            "risk_class": "safe" if not must_handoff else "payment_dispute",
            "answerability": "answer_self" if not must_handoff else "manager_only",
            "requested_action": requested_action,
        },
        "review_label": "frame_too_cautious" if not must_handoff else "frame_too_confident",
        "notes": notes,
    }


def _turn(
    dialog_id: str,
    *,
    route: str = "manager_only",
    message_type: str = "context_update",
    missing_facts: list[str] | None = None,
    frame: dict | None = None,
    proof_reconciliation: dict | None = None,
) -> dict:
    semantic_frame = {
        "risk_class": "manager_action",
        "answerability": "manager_only",
        "requested_action": "check_availability",
        "must_handoff": True,
        "confidence": 0.92,
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
        "dialog_id": dialog_id,
        "brand": "unpk",
        "turns": [
            {
                "turn": 1,
                "brand": "unpk",
                "client_message": "Ребёнок закончил 5 класс",
                "bot_text": "Передам менеджеру.",
                "bot_route": route,
                "bot_message_type": message_type,
                "bot_safety_flags": ["manager_approval_required", "no_auto_send", f"message_type_{message_type}"],
                "bot_missing_facts": missing_facts if missing_facts is not None else ["актуальное наличие мест"],
                "bot_semantic_frame": semantic_frame,
                "bot_semantic_frame_proof_reconciliation_shadow": proof_reconciliation or {},
                "bot_semantic_frame_self_answer_shadow": {
                    "status": "blocked",
                    "reason": "route_not_draft_for_manager",
                    "self_class": "safe_reference",
                    "guards": {
                        "actual_p0": False,
                        "has_missing_facts": bool(missing_facts),
                        "blocking_flags": [],
                        "freshness": {"ok": False, "exact_fact_count": 0, "fresh_client_safe_count": 0},
                    },
                },
            }
        ],
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


def _build(tmp_path: Path, dialogs: list[dict], gold_rows: list[dict]) -> dict:
    transcripts = tmp_path / "transcripts.jsonl"
    gold = tmp_path / "gold.jsonl"
    kb = tmp_path / "kb.json"
    _write_jsonl(transcripts, dialogs)
    _write_jsonl(gold, gold_rows)
    _write_json(kb, {"facts": [_fact()]})
    return report.build_report(
        transcripts=transcripts,
        gold=gold,
        kb_snapshot=kb,
        as_of_date=report._parse_date("2026-07-02"),
    )


def test_existence_vs_availability_goes_to_frame_calibration_not_active(tmp_path: Path) -> None:
    result = _build(tmp_path, [_turn("d1")], [_gold("d1")])

    assert result["totals"]["true_frame_too_cautious"] == 1
    assert result["real_lever_analysis"]["totals"]["too_cautious_total"] == 1
    assert result["real_lever_analysis"]["totals"]["fact_assertion_required"] == 1
    assert result["real_lever_analysis"]["totals"]["clean_route_only_discussion"] == 0
    assert result["real_lever_analysis"]["totals"]["stable_existence_as_check_availability"] == 1
    assert result["real_lever_analysis"]["totals"]["stable_existence_as_enroll"] == 0
    assert result["real_lever_analysis"]["scope_confusion"]["count"] == 1
    assert result["real_lever_analysis"]["by_frame_requested_action"] == {"check_availability": 1}
    assert result["real_lever_analysis"]["by_lever_class"] == {"fact_assertion_required": 1}
    example = result["real_lever_analysis"]["examples"][0]
    assert example["user_scope"] == "stable_existence_format"
    assert example["frame_scope"] == "live_availability_or_enroll"
    assert example["scope_confusion"] is True
    assert result["workstreams"]["semanticframe_existence_vs_availability"]["count"] == 1
    assert result["workstreams"]["retrieval_delivery_runtime_missing_exact_proof"]["count"] == 1
    assert result["workstreams"]["conversation_plan_scope_missing"]["count"] == 1
    assert result["workstreams"]["policy_manager_only_exact_proof"]["count"] == 1
    assert result["workstreams"]["policy_context_update_exact_proof"]["count"] == 1
    assert result["acceptance"]["active_readiness"] == "no_go"
    item = result["workstreams"]["semanticframe_existence_vs_availability"]["examples"][0]
    assert item["active_allowed"] is False
    assert item["active_block_reason"] == "frame_confuses_safe_reference_with_live_availability_or_enrollment"
    assert "requested_action" in item["calibration_target"]


def test_factless_ack_status_is_reported_separately(tmp_path: Path) -> None:
    result = _build(
        tmp_path,
        [
            _turn(
                "ack",
                route="draft_for_manager",
                message_type="context_update",
                missing_facts=[],
                frame={
                    "risk_class": "safe",
                    "answerability": "answer_self",
                    "requested_action": "acknowledge_status",
                    "must_handoff": True,
                    "confidence": 0.94,
                },
            )
        ],
        [_gold("ack", notes="harmless ack/status, no fact assertion")],
    )

    real = result["real_lever_analysis"]
    assert real["totals"]["too_cautious_total"] == 1
    assert real["totals"]["factless_ack_status"] == 1
    assert real["totals"]["fact_assertion_required"] == 0
    assert real["totals"]["clean_route_only_discussion"] == 1
    assert real["by_lever_class"] == {"clean_factless_ack_status_discussion": 1}


def test_true_enroll_booking_request_is_negative_control(tmp_path: Path) -> None:
    result = _build(
        tmp_path,
        [
            _turn(
                "book",
                route="manager_only",
                message_type="question",
                frame={
                    "risk_class": "manager_action",
                    "answerability": "manager_only",
                    "requested_action": "enroll",
                    "must_handoff": True,
                    "confidence": 0.96,
                },
            )
        ],
        [_gold("book", must_handoff=True, requested_action="enroll", notes="true enrollment/booking request")],
    )

    real = result["real_lever_analysis"]
    assert real["totals"]["too_cautious_total"] == 0
    assert real["totals"]["true_live_availability_negative_controls"] == 1
    assert real["totals"]["true_enroll_booking_negative_controls"] == 1
    negative = real["negative_controls"][0]
    assert negative["expected_action"] == "enroll"
    assert negative["expected_scope"] == "live_availability_or_enroll"
    assert negative["active_block_reason"] == "negative_control_true_live_or_operational_request"


def test_manual_too_cautious_label_is_separate_from_true_frame_error(tmp_path: Path) -> None:
    result = _build(
        tmp_path,
        [
            _turn(
                "d1",
                route="bot_answer_self_for_pilot",
                message_type="question",
                missing_facts=[],
                frame={
                    "risk_class": "safe",
                    "answerability": "answer_self",
                    "requested_action": "answer_question",
                    "must_handoff": False,
                    "confidence": 0.95,
                },
            )
        ],
        [_gold("d1", notes="safe reference: price/format")],
    )

    assert result["totals"]["manual_too_cautious_labels"] == 1
    assert result["totals"]["true_frame_too_cautious"] == 0
    assert result["real_lever_analysis"]["totals"]["too_cautious_total"] == 0
    assert result["workstreams"]["already_self_no_active_leverage"]["count"] == 1
    item = result["workstreams"]["already_self_no_active_leverage"]["examples"][0]
    assert item["active_allowed"] is False
    assert item["active_block_reason"] == "current_route_already_self_no_route_leverage"


def test_proof_reconciliation_shadow_is_reported_but_not_active_go(tmp_path: Path) -> None:
    result = _build(
        tmp_path,
        [
            _turn(
                "d1",
                route="draft_for_manager",
                message_type="question",
                missing_facts=["platform.current"],
                frame={
                    "risk_class": "missing_facts",
                    "answerability": "manager_only",
                    "requested_action": "answer_question",
                    "must_handoff": True,
                    "confidence": 0.92,
                },
                proof_reconciliation={
                    "status": "would_reconcile_to_safe_reference",
                    "reason": "fresh_proof_contradicts_missing_facts_frame",
                    "route_before": "draft_for_manager",
                    "exact_fact_keys": ["online_platform.levels.1"],
                    "frame_before": {
                        "risk_class": "missing_facts",
                        "answerability": "manager_only",
                        "requested_action": "answer_question",
                        "must_handoff": True,
                    },
                },
            )
        ],
        [_gold("d1", notes="safe reference: platform/format without live seats")],
    )

    assert result["totals"]["proof_reconciliation_would_reconcile"] == 1
    assert result["real_lever_analysis"]["totals"]["proof_reconciliation_would_reconcile"] == 1
    assert result["real_lever_analysis"]["totals"]["proof_reconciliation_current_handoff"] == 1
    assert result["workstreams"]["semanticframe_proof_reconciliation_candidate"]["count"] == 1
    assert result["proof_reconciliation"]["would_reconcile"] == 1
    assert result["proof_reconciliation"]["examples"][0]["exact_fact_keys"] == ["online_platform.levels.1"]
    assert result["acceptance"]["active_readiness"] == "no_go"


def test_low_confidence_safe_frame_is_calibration_work(tmp_path: Path) -> None:
    result = _build(
        tmp_path,
        [
            _turn(
                "d1",
                message_type="question",
                missing_facts=[],
                frame={
                    "risk_class": "safe",
                    "answerability": "answer_self",
                    "requested_action": "answer_question",
                    "must_handoff": False,
                    "confidence": 0.86,
                },
            )
        ],
        [_gold("d1", notes="safe reference: course existence/format without live seats")],
    )

    assert result["totals"]["true_frame_too_cautious"] == 0
    assert result["workstreams"]["semanticframe_low_confidence"]["count"] == 1
    assert result["workstreams"]["policy_manager_only_exact_proof"]["count"] == 1


def test_too_confident_handoff_gold_is_measurement_review(tmp_path: Path) -> None:
    result = _build(
        tmp_path,
        [
            _turn(
                "p0",
                route="manager_only",
                frame={
                    "risk_class": "safe",
                    "answerability": "answer_self",
                    "requested_action": "answer_question",
                    "must_handoff": False,
                    "confidence": 0.95,
                },
            )
        ],
        [_gold("p0", must_handoff=True, notes="payment dispute")],
    )

    assert result["totals"]["true_frame_too_confident"] == 1
    assert result["workstreams"]["measurement_review_unclear"]["count"] == 1
